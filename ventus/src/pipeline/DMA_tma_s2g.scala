
/*
 * Descriptor-addressed tensor shared-to-global DMA datapath.
 *
 * The fallback path is intentionally conservative: it supports 4-byte tensor
 * elements, linear element iteration, optional shared-memory swizzle, and a
 * full shared read -> TLB -> L2 Put sequence per element.
 *
 * The row-level line-task path handles CUDA-style FP32 tensor swizzle,
 * interleave, regular stride, OOB-suppress, and ordinary no-permute rows by
 * emitting coalesced S2GLineTask requests into the bulk S2G backend. The RTL
 * intentionally does not carry a separate multi-row linear-span fusion path.
 */
package pipeline

import chisel3._
import chisel3.util._
import config.config.Parameters
import L1Cache.{DCacheMemReq_p, DCacheMemRsp}
import mmu.{L1TlbReq, L1TlbRsp, SV32}
import top.parameters._

class DmaTensorS2G(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val from_fifo = Flipped(DecoupledIO(new vExeData))
    val line_task = DecoupledIO(new S2GLineTask)
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
    val shared_rsp = Flipped(DecoupledIO(new DmaSharedRsp))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
    val to_l2cache = DecoupledIO(new DCacheMemReq_p)
    val from_l2cache = Flipped(DecoupledIO(new DCacheMemRsp))
    val inst_complete = DecoupledIO(new DmaCompletion)
    val perfEnable = Input(Bool())
    val perfReset = Input(Bool())
    val perf = if (PMU_DMA_S2G) Some(Output(new S2GPerfCounters)) else None
  })

  val lineOffsetBits = log2Ceil(l2cacheline)
  val wordOffsetBits = log2Ceil(dma_aligned_bulk)
  val wordIdxWidth = log2Ceil(dcache_BlockWords).max(1)
  val sharedSetIdxHi = log2Ceil(sharedmem_depth) + dcache_BlockOffsetBits + dcache_WordOffsetBits - 1
  val sharedSetIdxLo = dcache_BlockOffsetBits + dcache_WordOffsetBits
  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 2, "DMA tensor S2G source encoding needs at least 3 low bits")

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, lineOffsetBits), 0.U(lineOffsetBits.W))

  def minUInt(a: UInt, b: UInt): UInt = Mux(a < b, a, b)

  def sharedSetIdx(addr: UInt): UInt =
    addr(sharedSetIdxHi, sharedSetIdxLo)

  def sharedBlockOffset(addr: UInt): UInt =
    addr(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits)

  def swizzleSharedAddr(logicalAddr: UInt, baseAddr: UInt, mode: UInt, rowLow: UInt): UInt = {
    val rel = logicalAddr - baseAddr
    val rel32 = Cat(rel(xLen - 1, 5), rel(4) ^ rowLow(0), rel(3, 0))
    val rel64 = Cat(rel(xLen - 1, 6), rel(5, 4) ^ rowLow(1, 0), rel(3, 0))
    val rel128 = Cat(rel(xLen - 1, 7), rel(6, 4) ^ rowLow(2, 0), rel(3, 0))
    baseAddr + MuxLookup(mode, rel)(Seq(
      1.U -> rel32,
      2.U -> rel64,
      3.U -> rel128
    ))
  }

  def tensorDataWidth(dataType: UInt): UInt = Mux(
    dataType === DataType.UINT8 || dataType === DataType.INT8, 1.U,
    Mux(
      dataType === DataType.UINT16 || dataType === DataType.INT16 ||
        dataType === DataType.FLOAT16 || dataType === DataType.BFLOAT16, 2.U,
      Mux(
        dataType === DataType.UINT32 || dataType === DataType.INT32 ||
          dataType === DataType.FLOAT32, 4.U,
        8.U
      )
    )
  )

  val tensorS2GStates = Enum(31)
  val s_idle = tensorS2GStates(0)
  val s_desc_tlb_req = tensorS2GStates(1)
  val s_desc_tlb_rsp = tensorS2GStates(2)
  val s_desc_l2_req = tensorS2GStates(3)
  val s_desc_l2_rsp = tensorS2GStates(4)
  val s_addr_init = tensorS2GStates(5)
  val s_addr_pitch = tensorS2GStates(6)
  val s_addr_slice = tensorS2GStates(7)
  val s_addr_dim1 = tensorS2GStates(8)
  val s_addr_dim2 = tensorS2GStates(9)
  val s_addr_dim3 = tensorS2GStates(10)
  val s_addr_dim4 = tensorS2GStates(11)
  val s_setup = tensorS2GStates(12)
  val s_setup_stride_1 = tensorS2GStates(13)
  val s_prepare = tensorS2GStates(14)
  val s_prepare_addr = tensorS2GStates(15)
  val s_check = tensorS2GStates(16)
  val s_shared_req = tensorS2GStates(17)
  val s_shared_rsp = tensorS2GStates(18)
  val s_tlb_req = tensorS2GStates(19)
  val s_tlb_rsp = tensorS2GStates(20)
  val s_l2_req = tensorS2GStates(21)
  val s_l2_rsp = tensorS2GStates(22)
  val s_complete = tensorS2GStates(23)
  val s_addr_mul_start = tensorS2GStates(24)
  val s_addr_mul_wait = tensorS2GStates(25)
  val s_setup_mul_start = tensorS2GStates(26)
  val s_setup_mul_wait = tensorS2GStates(27)
  val s_line_prepare = tensorS2GStates(28)
  val s_line_shared_req = tensorS2GStates(29)
  val s_line_advance = tensorS2GStates(30)
  val state = RegInit(s_idle)

  val sharedSrcReg = RegInit(0.U(xLen.W))
  val descPtrReg = RegInit(0.U(xLen.W))
  val widReg = RegInit(0.U(depth_warp.W))
  val groupReg = RegInit(0.U(log2Ceil(dma_group_entries).W))
  val asidReg = RegInit(0.U(SV32.asidLen.W))
  val pAddrReg = RegInit(0.U(SV32.paLen.W))
  val currentIdxReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val rawOutDimReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val outDimReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val outStrideBytesReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedRowStrideReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val globalStrideBytesReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val globalOffsetPartReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val globalLogicalOffsetReg = RegInit(0.U(xLen.W))
  val linearGlobalBaseReg = RegInit(0.U(xLen.W))
  val sharedOffsetPartReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedRowPartReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedLogicalOffsetReg = RegInit(0.U(xLen.W))
  val sharedRowReg = RegInit(0.U(xLen.W))
  val currentValidReg = RegInit(false.B)
  val currentLastReg = RegInit(false.B)
  val fastCoalesceReg = RegInit(false.B)
  val currentIdxNextReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val currentAdvanceDimOHReg = RegInit(VecInit(Seq.fill(5)(false.B)))
  val globalDim0OffsetStepReg = RegInit(0.U(xLen.W))
  val globalOffsetPartNextReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedOffsetPartNextReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedRowPartNextReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val globalReqLineBaseReg = RegInit(0.U(xLen.W))
  val globalReqWordReg = RegInit(0.U((lineOffsetBits - wordOffsetBits).W))
  val sharedSetReg = RegInit(0.U((sharedSetIdxHi - sharedSetIdxLo + 1).W))
  val sharedBlockReg = RegInit(0.U(dcache_BlockOffsetBits.W))
  val sharedDataReg = Reg(UInt(xLen.W))
  val lineRowOffsetReg = RegInit(0.U(xLen.W))
  val lineRowBytesReg = RegInit(0.U(xLen.W))
  val lineGlobalBaseReg = RegInit(0.U(xLen.W))
  val lineSharedBaseReg = RegInit(0.U(xLen.W))
  val lineChunkBytesReg = RegInit(0.U(xLen.W))
  val lineTaskFirstReg = RegInit(false.B)
  val descWordsReg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))
  val descBoxAddressReg = RegInit(0.U(xLen.W))
  val descAddrAccumReg = RegInit(0.U(xLen.W))
  val descAddrSliceStrideReg = RegInit(0.U(xLen.W))
  val descAddrOpReg = RegInit(0.U(3.W))
  val setupMulOpReg = RegInit(0.U(3.W))
  val tmaSetupMul = Module(new TmaMul32Unit)

  // Small descriptor-line cache. Tensor S2G kernels usually reuse the same
  // tensor map across many stages, so this removes repeated descriptor TLB/L2
  // setup without growing storage with tile size.
  require(tma_desc_cache_entries > 0, "Tensor S2G descriptor cache must have at least one entry")
  val descCacheValid = RegInit(VecInit(Seq.fill(tma_desc_cache_entries)(false.B)))
  val descCacheLine = RegInit(VecInit(Seq.fill(tma_desc_cache_entries)(0.U(xLen.W))))
  val descCacheData = Reg(Vec(tma_desc_cache_entries, Vec(dcache_BlockWords, UInt(xLen.W))))
  val descCacheIdxWidth = log2Ceil(tma_desc_cache_entries).max(1)
  val descCacheReplace = RegInit(0.U(descCacheIdxWidth.W))
  val incomingDescLine = alignToL2Line(io.from_fifo.bits.in1(0))
  val incomingDescCacheHitVec = VecInit((0 until tma_desc_cache_entries).map { i =>
    descCacheValid(i) && descCacheLine(i) === incomingDescLine
  })
  val incomingDescCacheHit = incomingDescCacheHitVec.asUInt.orR
  val incomingDescCacheWords = Wire(Vec(dcache_BlockWords, UInt(xLen.W)))
  for (w <- 0 until dcache_BlockWords) {
    incomingDescCacheWords(w) := Mux1H(
      incomingDescCacheHitVec,
      (0 until tma_desc_cache_entries).map(i => descCacheData(i)(w))
    )
  }

  val perfCycle = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfAckIssueCycle = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfInstIssued = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfLineIssued = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfPutFull = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfPutPart = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfBytesWritten = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfSharedReadReq = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfSharedReadRsp = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfTlbReq = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfAckCount = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfAckLatencySum = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfLineFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfReadEntryFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfAckTagFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None

  val coordsReg = Reg(Vec(5, UInt(xLen.W)))
  val coords = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { i => coords(i) := coordsReg(i) }

  val tvars = Wire(new TensorVars)
  val descControl = descWordsReg(1)
  val descByteStride = Wire(Vec(5, UInt(xLen.W)))
  val outDim = Wire(Vec(5, UInt(xLen.W)))
  val currentIdx = Wire(Vec(5, UInt(xLen.W)))
  val currentCoord = Wire(Vec(5, UInt(xLen.W)))

  descByteStride(0) := Mux(descWordsReg(9) === 0.U, tensorDataWidth(descControl(3, 0)), descWordsReg(9))
  (1 until 5).foreach { i => descByteStride(i) := descWordsReg(9 + i) }
  (0 until 5).foreach { i =>
    tvars.boxDim(i) := descWordsReg(14 + i)
    tvars.elementStrides(i) := Mux(descWordsReg(19 + i) === 0.U, 1.U, descWordsReg(19 + i))
    tvars.globalDim(i) := descWordsReg(4 + i)
  }
  tvars.interleaveMode := descControl(9, 8)
  tvars.swizzleMode := descControl(11, 10)
  tvars.L2promotion := descControl(13, 12)
  tvars.oobfill := descControl(14).asUInt
  tvars.dataType := descControl(3, 0)
  tvars.tensorRank := descControl(6, 4)
  tvars.globalAddress := descWordsReg(2)
  tvars.globalStrides(0) := descByteStride(1)
  tvars.globalStrides(1) := descByteStride(2)
  tvars.globalStrides(2) := descByteStride(3)
  tvars.globalStrides(3) := descByteStride(4)
  tvars.globalStrides(4) := 0.U
  tvars.datawidth := dma_aligned_bulk.U

  val descAddrOpPitch = 0.U(3.W)
  val descAddrOpSlice = 1.U(3.W)
  val descAddrOpDim1 = 2.U(3.W)
  val descAddrOpDim2 = 3.U(3.W)
  val descAddrOpDim3 = 4.U(3.W)
  val descAddrOpDim4 = 5.U(3.W)

  val setupMulOpByte2 = 0.U(3.W)
  val setupMulOpByte3 = 1.U(3.W)
  val setupMulOpRow3 = 2.U(3.W)
  val setupMulOpByte4 = 3.U(3.W)
  val setupMulOpRow4 = 4.U(3.W)

  val descAddrMulA = WireDefault(0.U(xLen.W))
  val descAddrMulB = WireDefault(0.U(xLen.W))
  switch(descAddrOpReg) {
    is(descAddrOpPitch) {
      descAddrMulA := MuxLookup(tvars.tensorRank, 0.U(xLen.W))(Seq(
        3.U -> descByteStride(1),
        4.U -> descByteStride(2),
        5.U -> descByteStride(3)
      ))
      descAddrMulB := MuxLookup(tvars.tensorRank, 0.U(xLen.W))(Seq(
        3.U -> tvars.globalDim(1),
        4.U -> tvars.globalDim(2),
        5.U -> tvars.globalDim(3)
      ))
    }
    is(descAddrOpSlice) {
      descAddrMulA := Mux(tvars.interleaveMode === 1.U, coords(0) >> 2, coords(0) >> 3)
      descAddrMulB := descAddrSliceStrideReg
    }
    is(descAddrOpDim1) {
      descAddrMulA := coords(1)
      descAddrMulB := descByteStride(1)
    }
    is(descAddrOpDim2) {
      descAddrMulA := coords(2)
      descAddrMulB := descByteStride(2)
    }
    is(descAddrOpDim3) {
      descAddrMulA := coords(3)
      descAddrMulB := descByteStride(3)
    }
    is(descAddrOpDim4) {
      descAddrMulA := coords(4)
      descAddrMulB := descByteStride(4)
    }
  }

  val rawOutDim = Wire(Vec(5, UInt(xLen.W)))
  rawOutDim(0) := tvars.boxDim(0)
  (1 until 5).foreach { i =>
    rawOutDim(i) := Mux(
      tvars.tensorRank > i.U,
      Mux(tvars.elementStrides(i) <= 1.U,
        tvars.boxDim(i),
        TmaPow2Math.ceilDivPow2(tvars.boxDim(i), tvars.elementStrides(i))),
      1.U)
  }
  (0 until 5).foreach { i =>
    val activeDim = tvars.tensorRank > i.U
    val coordInBounds = coords(i) < tvars.globalDim(i)
    val elemsToDimEnd = tvars.globalDim(i) - coords(i)
    val maxValidElems = Mux(
      tvars.elementStrides(i) <= 1.U,
      elemsToDimEnd,
      TmaPow2Math.ceilDivPow2(elemsToDimEnd, tvars.elementStrides(i))
    )
    outDim(i) := Mux(
      activeDim,
      Mux(coordInBounds, minUInt(rawOutDim(i), maxValidElems), 0.U),
      1.U)
  }

  (0 until 5).foreach { i => currentIdx(i) := currentIdxReg(i) }

  val currentLastDim = Wire(Vec(5, Bool()))
  (0 until 5).foreach { i =>
    currentLastDim(i) := (tvars.tensorRank <= i.U) || ((currentIdxReg(i) + 1.U) >= outDimReg(i))
  }
  val currentIdxNext = Wire(Vec(5, UInt(xLen.W)))
  val currentIdxCarry = Wire(Vec(6, Bool()))
  currentIdxCarry(0) := true.B
  (0 until 5).foreach { i =>
    val activeDim = tvars.tensorRank > i.U
    currentIdxNext(i) := currentIdxReg(i)
    when(currentIdxCarry(i) && activeDim) {
      currentIdxNext(i) := Mux(currentLastDim(i), 0.U, currentIdxReg(i) + 1.U)
    }
    currentIdxCarry(i + 1) := currentIdxCarry(i) && currentLastDim(i)
  }
  val currentLast = currentIdxCarry(5)
  val currentAdvanceDimOH = Wire(Vec(5, Bool()))
  (0 until 5).foreach { i =>
    currentAdvanceDimOH(i) := currentIdxCarry(i) && (tvars.tensorRank > i.U) && !currentLastDim(i)
  }
  val outDimZeroSetup = (0 until 5).map(i => outDim(i) === 0.U).reduce(_ || _)
  val tensorFastStrideOk = (0 until 5).map(i => tvars.elementStrides(i) === 1.U).reduce(_ && _)
  val tensorOriginalInBounds = (0 until 5).map { i =>
    (tvars.tensorRank <= i.U) || ((coords(i) + rawOutDim(i)) <= tvars.globalDim(i))
  }.reduce(_ && _)
  val tensorNoPermute = tvars.interleaveMode === 0.U && tvars.swizzleMode === 0.U
  val tensorNoInterleave = tvars.interleaveMode === 0.U
  val tensorInterleaveCoalesceOk =
    (tvars.interleaveMode =/= 0.U) && tensorFastStrideOk && tensorOriginalInBounds
  val tensorLineCoalesce =
    tensorDataWidth(tvars.dataType) === dma_aligned_bulk.U &&
      (tvars.interleaveMode =/= 3.U) &&
      (tensorNoInterleave || tensorInterleaveCoalesceOk) &&
      !outDimZeroSetup

  val rowLastDim = Wire(Vec(5, Bool()))
  rowLastDim(0) := true.B
  (1 until 5).foreach { i =>
    val activeDim = tvars.tensorRank > i.U
    rowLastDim(i) := !activeDim || ((currentIdxReg(i) + 1.U) >= outDimReg(i))
  }
  val rowIdxNext = Wire(Vec(5, UInt(xLen.W)))
  val rowIdxCarry = Wire(Vec(6, Bool()))
  rowIdxNext(0) := 0.U
  rowIdxCarry(0) := false.B
  rowIdxCarry(1) := true.B
  (1 until 5).foreach { i =>
    val activeDim = tvars.tensorRank > i.U
    rowIdxNext(i) := currentIdxReg(i)
    when(rowIdxCarry(i) && activeDim) {
      rowIdxNext(i) := Mux(rowLastDim(i), 0.U, currentIdxReg(i) + 1.U)
    }
    rowIdxCarry(i + 1) := rowIdxCarry(i) && rowLastDim(i)
  }
  val rowLast = rowIdxCarry(5)
  val rowAdvanceDimOH = Wire(Vec(5, Bool()))
  rowAdvanceDimOH(0) := false.B
  (1 until 5).foreach { i =>
    rowAdvanceDimOH(i) := rowIdxCarry(i) && (tvars.tensorRank > i.U) && !rowLastDim(i)
  }

  val setupMulA = WireDefault(0.U(xLen.W))
  val setupMulB = WireDefault(0.U(xLen.W))
  switch(setupMulOpReg) {
    is(setupMulOpByte2) {
      setupMulA := outStrideBytesReg(1)
      setupMulB := rawOutDimReg(1)
    }
    is(setupMulOpByte3) {
      setupMulA := outStrideBytesReg(2)
      setupMulB := rawOutDimReg(2)
    }
    is(setupMulOpRow3) {
      setupMulA := sharedRowStrideReg(2)
      setupMulB := rawOutDimReg(2)
    }
    is(setupMulOpByte4) {
      setupMulA := outStrideBytesReg(3)
      setupMulB := rawOutDimReg(3)
    }
    is(setupMulOpRow4) {
      setupMulA := sharedRowStrideReg(3)
      setupMulB := rawOutDimReg(3)
    }
  }

  val tmaSetupMulA = Mux(state === s_addr_mul_start, descAddrMulA, setupMulA)
  val tmaSetupMulB = Mux(state === s_addr_mul_start, descAddrMulB, setupMulB)
  tmaSetupMul.io.in.valid := state === s_addr_mul_start || state === s_setup_mul_start
  tmaSetupMul.io.in.bits.a := tmaSetupMulA
  tmaSetupMul.io.in.bits.b := tmaSetupMulB
  tmaSetupMul.io.out.ready := state === s_addr_mul_wait || state === s_setup_mul_wait

  currentCoord(0) := coords(0) + TmaPow2Math.scaleByPow2(currentIdx(0), tvars.elementStrides(0))
  currentCoord(1) := coords(1) + TmaPow2Math.scaleByPow2(currentIdx(1), tvars.elementStrides(1))
  currentCoord(2) := coords(2) + TmaPow2Math.scaleByPow2(currentIdx(2), tvars.elementStrides(2))
  currentCoord(3) := coords(3) + TmaPow2Math.scaleByPow2(currentIdx(3), tvars.elementStrides(3))
  currentCoord(4) := coords(4) + TmaPow2Math.scaleByPow2(currentIdx(4), tvars.elementStrides(4))

  val currentValidExpr0 = currentCoord(0) < tvars.globalDim(0)
  var currentValidExpr = currentValidExpr0
  (1 until 5).foreach { i =>
    currentValidExpr = currentValidExpr && ((tvars.tensorRank <= i.U) || (currentCoord(i) < tvars.globalDim(i)))
  }
  val currentValid = Wire(Bool())
  currentValid := currentValidExpr

  val nextCoord0ForOffset = coords(0) + TmaPow2Math.scaleByPow2(currentIdxNext(0), tvars.elementStrides(0))
  val globalInterleaveDim0DeltaBytes = TmaPow2Math.interleaveDim0DeltaBytes(
    currentCoord(0),
    nextCoord0ForOffset,
    descAddrSliceStrideReg,
    tvars.interleaveMode
  )
  val globalDim0OffsetStep = Mux(
    tvars.interleaveMode === 0.U,
    globalStrideBytesReg(0),
    globalInterleaveDim0DeltaBytes
  )

  val globalOffsetPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  val sharedOffsetPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  val sharedRowPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { i =>
    globalOffsetPartPrepared(i) := globalOffsetPartReg(i)
    sharedOffsetPartPrepared(i) := sharedOffsetPartReg(i)
    sharedRowPartPrepared(i) := sharedRowPartReg(i)
  }
  when(currentAdvanceDimOHReg(0)) {
    globalOffsetPartPrepared(0) := globalOffsetPartReg(0) + globalDim0OffsetStepReg
    sharedOffsetPartPrepared(0) := sharedOffsetPartReg(0) + outStrideBytesReg(0)
  }.elsewhen(currentAdvanceDimOHReg(1)) {
    globalOffsetPartPrepared(0) := 0.U
    globalOffsetPartPrepared(1) := globalOffsetPartReg(1) + globalStrideBytesReg(1)
    sharedOffsetPartPrepared(0) := 0.U
    sharedOffsetPartPrepared(1) := sharedOffsetPartReg(1) + outStrideBytesReg(1)
    sharedRowPartPrepared(1) := sharedRowPartReg(1) + sharedRowStrideReg(1)
  }.elsewhen(currentAdvanceDimOHReg(2)) {
    (0 until 2).foreach { i =>
      globalOffsetPartPrepared(i) := 0.U
      sharedOffsetPartPrepared(i) := 0.U
      sharedRowPartPrepared(i) := 0.U
    }
    globalOffsetPartPrepared(2) := globalOffsetPartReg(2) + globalStrideBytesReg(2)
    sharedOffsetPartPrepared(2) := sharedOffsetPartReg(2) + outStrideBytesReg(2)
    sharedRowPartPrepared(2) := sharedRowPartReg(2) + sharedRowStrideReg(2)
  }.elsewhen(currentAdvanceDimOHReg(3)) {
    (0 until 3).foreach { i =>
      globalOffsetPartPrepared(i) := 0.U
      sharedOffsetPartPrepared(i) := 0.U
      sharedRowPartPrepared(i) := 0.U
    }
    globalOffsetPartPrepared(3) := globalOffsetPartReg(3) + globalStrideBytesReg(3)
    sharedOffsetPartPrepared(3) := sharedOffsetPartReg(3) + outStrideBytesReg(3)
    sharedRowPartPrepared(3) := sharedRowPartReg(3) + sharedRowStrideReg(3)
  }.elsewhen(currentAdvanceDimOHReg(4)) {
    (0 until 4).foreach { i =>
      globalOffsetPartPrepared(i) := 0.U
      sharedOffsetPartPrepared(i) := 0.U
      sharedRowPartPrepared(i) := 0.U
    }
    globalOffsetPartPrepared(4) := globalOffsetPartReg(4) + globalStrideBytesReg(4)
    sharedOffsetPartPrepared(4) := sharedOffsetPartReg(4) + outStrideBytesReg(4)
    sharedRowPartPrepared(4) := sharedRowPartReg(4) + sharedRowStrideReg(4)
  }

  val rowGlobalOffsetPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  val rowSharedOffsetPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  val rowSharedRowPartPrepared = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { i =>
    rowGlobalOffsetPartPrepared(i) := globalOffsetPartReg(i)
    rowSharedOffsetPartPrepared(i) := sharedOffsetPartReg(i)
    rowSharedRowPartPrepared(i) := sharedRowPartReg(i)
  }
  when(rowAdvanceDimOH(1)) {
    rowGlobalOffsetPartPrepared(0) := 0.U
    rowGlobalOffsetPartPrepared(1) := globalOffsetPartReg(1) + globalStrideBytesReg(1)
    rowSharedOffsetPartPrepared(0) := 0.U
    rowSharedOffsetPartPrepared(1) := sharedOffsetPartReg(1) + outStrideBytesReg(1)
    rowSharedRowPartPrepared(1) := sharedRowPartReg(1) + sharedRowStrideReg(1)
  }.elsewhen(rowAdvanceDimOH(2)) {
    (0 until 2).foreach { i =>
      rowGlobalOffsetPartPrepared(i) := 0.U
      rowSharedOffsetPartPrepared(i) := 0.U
      rowSharedRowPartPrepared(i) := 0.U
    }
    rowGlobalOffsetPartPrepared(2) := globalOffsetPartReg(2) + globalStrideBytesReg(2)
    rowSharedOffsetPartPrepared(2) := sharedOffsetPartReg(2) + outStrideBytesReg(2)
    rowSharedRowPartPrepared(2) := sharedRowPartReg(2) + sharedRowStrideReg(2)
  }.elsewhen(rowAdvanceDimOH(3)) {
    (0 until 3).foreach { i =>
      rowGlobalOffsetPartPrepared(i) := 0.U
      rowSharedOffsetPartPrepared(i) := 0.U
      rowSharedRowPartPrepared(i) := 0.U
    }
    rowGlobalOffsetPartPrepared(3) := globalOffsetPartReg(3) + globalStrideBytesReg(3)
    rowSharedOffsetPartPrepared(3) := sharedOffsetPartReg(3) + outStrideBytesReg(3)
    rowSharedRowPartPrepared(3) := sharedRowPartReg(3) + sharedRowStrideReg(3)
  }.elsewhen(rowAdvanceDimOH(4)) {
    (0 until 4).foreach { i =>
      rowGlobalOffsetPartPrepared(i) := 0.U
      rowSharedOffsetPartPrepared(i) := 0.U
      rowSharedRowPartPrepared(i) := 0.U
    }
    rowGlobalOffsetPartPrepared(4) := globalOffsetPartReg(4) + globalStrideBytesReg(4)
    rowSharedOffsetPartPrepared(4) := sharedOffsetPartReg(4) + outStrideBytesReg(4)
    rowSharedRowPartPrepared(4) := sharedRowPartReg(4) + sharedRowStrideReg(4)
  }

  val globalLogicalOffset = globalOffsetPartReg.reduce(_ + _)
  val sharedLogicalOffset = sharedOffsetPartReg.reduce(_ + _)
  val sharedRow = sharedRowPartReg.reduce(_ + _)

  val sharedCurrentAddr = swizzleSharedAddr(sharedSrcReg + sharedLogicalOffsetReg, sharedSrcReg, tvars.swizzleMode, sharedRowReg)
  val globalCurrentAddr = linearGlobalBaseReg + globalLogicalOffsetReg

  val globalReqLineBase = alignToL2Line(globalCurrentAddr)
  val globalReqWord = globalCurrentAddr(lineOffsetBits - 1, wordOffsetBits)
  val sharedSet = sharedSetIdx(sharedCurrentAddr)
  val sharedBlock = sharedBlockOffset(sharedCurrentAddr)

  val tensorDim0StrideLine =
    fastCoalesceReg &&
      tensorNoInterleave &&
      tvars.elementStrides(0) =/= 1.U
  val tensorSwizzleWideLine =
    fastCoalesceReg &&
      tensorNoInterleave &&
      tvars.swizzleMode =/= 0.U &&
      !tensorDim0StrideLine
  val lineDstOffset = Mux(
    tensorDim0StrideLine,
    TmaPow2Math.scaleByPow2(lineRowOffsetReg, tvars.elementStrides(0)),
    lineRowOffsetReg
  )
  val lineSrcCur = lineSharedBaseReg + lineRowOffsetReg
  val lineDstCur = lineGlobalBaseReg + lineDstOffset
  val lineBytesLeft = lineRowBytesReg - lineRowOffsetReg
  val lineRowBytesDefault = (outDimReg(0) << log2Ceil(dma_aligned_bulk))(xLen - 1, 0)
  val tensorInterleaveLine = fastCoalesceReg && (tvars.interleaveMode =/= 0.U)

  val interleaveGroupElems = Mux(tvars.interleaveMode === 1.U, 4.U(xLen.W), 8.U(xLen.W))
  val interleaveCoord0 = coords(0) + currentIdxReg(0)
  val interleaveInSlice = Mux(
    tvars.interleaveMode === 1.U,
    interleaveCoord0(1, 0).pad(xLen),
    interleaveCoord0(2, 0).pad(xLen)
  )
  val interleaveElemsToSlice = interleaveGroupElems - interleaveInSlice
  val interleaveElemsLeft = outDimReg(0) - currentIdxReg(0)
  val interleaveSegmentElems = minUInt(interleaveElemsLeft, interleaveElemsToSlice)
  val interleaveSegmentBytes = (interleaveSegmentElems << log2Ceil(dma_aligned_bulk))(xLen - 1, 0)
  val interleaveSegmentIdx0Next = currentIdxReg(0) + interleaveSegmentElems
  val interleaveDim0Done = interleaveSegmentIdx0Next >= outDimReg(0)
  val interleaveSegmentLast = interleaveDim0Done && rowLast

  val lineSwizzleSpanBytes = MuxLookup(tvars.swizzleMode, l2cacheline.U(xLen.W))(Seq(
    1.U -> 32.U(xLen.W),
    2.U -> 64.U(xLen.W),
    3.U -> 128.U(xLen.W)
  ))
  val lineSrcRel = lineSrcCur - sharedSrcReg
  val lineSwizzleSpanOffset = MuxLookup(tvars.swizzleMode, 0.U(xLen.W))(Seq(
    1.U -> lineSrcRel(4, 0).pad(xLen),
    2.U -> lineSrcRel(5, 0).pad(xLen),
    3.U -> lineSrcRel(6, 0).pad(xLen)
  ))
  val lineBytesToSwizzleSpan = Mux(
    tvars.swizzleMode === 0.U || tensorSwizzleWideLine,
    l2cacheline.U(xLen.W),
    lineSwizzleSpanBytes - lineSwizzleSpanOffset
  )
  val lineBytesToDstLine = l2cacheline.U - lineDstCur(lineOffsetBits - 1, 0)
  val lineWordsToDstLine = dcache_BlockWords.U(xLen.W) - lineDstCur(lineOffsetBits - 1, wordOffsetBits)
  val lineStrideElemsToDstLine = TmaPow2Math.ceilDivPow2(lineWordsToDstLine, tvars.elementStrides(0))
  val lineStrideBytesToDstLine = (lineStrideElemsToDstLine << log2Ceil(dma_aligned_bulk))(xLen - 1, 0)
  val lineBytesToSrcLine = l2cacheline.U - lineSrcCur(lineOffsetBits - 1, 0)
  val lineBytesPerSharedReq = (numgroupshared * dma_aligned_bulk).U
  val lineDstLimitBytes = Mux(tensorDim0StrideLine, lineStrideBytesToDstLine, lineBytesToDstLine)
  val lineChunkBytes = minUInt(lineBytesLeft,
    minUInt(lineDstLimitBytes,
      minUInt(lineBytesToSrcLine, minUInt(lineBytesToSwizzleSpan, lineBytesPerSharedReq))))

  val fullWordMask = Fill(BytesOfWord, 1.U)
  val l2ReqIsElemPut = state === s_l2_req
  val l2ReqIsDataPut = l2ReqIsElemPut

  val descSource = ((2 << 1) | 1).U(l1cache_sourceBits.W)
  val tensorAckSource = Cat(
    0.U(log2Ceil(max_dma_tag).W),
    0.U(log2Ceil(max_dma_inst).W),
    4.U(dmaSourceLowBits.W)
  )

  val lineTaskRowOffsetNext = lineRowOffsetReg + lineChunkBytes
  val lineTaskLast = lineTaskRowOffsetNext >= lineRowBytesReg && currentLastReg

  io.line_task.valid := state === s_line_shared_req
  io.line_task.bits.wid := widReg
  io.line_task.bits.group := groupReg
  io.line_task.bits.asid := asidReg
  io.line_task.bits.src := lineSrcCur
  io.line_task.bits.dst := lineDstCur
  io.line_task.bits.bytes := lineChunkBytes
  io.line_task.bits.dstWordStride := Mux(tensorDim0StrideLine, tvars.elementStrides(0), 1.U)
  io.line_task.bits.swizzleMode := Mux(fastCoalesceReg, tvars.swizzleMode, 0.U)
  io.line_task.bits.swizzleBase := sharedSrcReg
  io.line_task.bits.swizzleRow := sharedRowPartReg.reduce(_ + _)(2, 0)
  io.line_task.bits.earlyRelease :=
    fastCoalesceReg && tensorNoPermute && !tensorDim0StrideLine
  io.line_task.bits.first := lineTaskFirstReg
  io.line_task.bits.last := lineTaskLast

  io.from_fifo.ready := state === s_idle
  io.shared_req.valid := state === s_shared_req
  io.shared_req.bits.instrId := 1.U
  io.shared_req.bits.isWrite := false.B
  io.shared_req.bits.setIdx := sharedSetReg
  (0 until numgroupshared).foreach { lane =>
    val elemActive = if (lane == 0) true.B else false.B
    val elemMask = if (lane == 0) fullWordMask else 0.U
    io.shared_req.bits.perLaneAddr(lane).activeMask :=
      elemActive
    io.shared_req.bits.perLaneAddr(lane).blockOffset :=
      sharedBlockReg
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H :=
      elemMask
    io.shared_req.bits.data(lane) := 0.U
  }

  io.shared_rsp.ready := state === s_shared_rsp

  io.to_l2TLB.valid := state === s_desc_tlb_req || state === s_tlb_req
  io.to_l2TLB.bits.vaddr := Mux(state === s_desc_tlb_req || state === s_desc_tlb_rsp || state === s_desc_l2_req || state === s_desc_l2_rsp,
    alignToL2Line(descPtrReg),
    globalReqLineBaseReg)
  io.to_l2TLB.bits.asid := asidReg
  io.from_l2TLB.ready := state === s_tlb_rsp || state === s_desc_tlb_rsp

  val l2Mask = Wire(Vec(dcache_BlockWords, UInt(BytesOfWord.W)))
  val l2Data = Wire(Vec(dcache_BlockWords, UInt(xLen.W)))
  (0 until dcache_BlockWords).foreach { i =>
    l2Mask(i) := 0.U
    l2Data(i) := 0.U
  }
  when(state === s_l2_req) {
    l2Mask(globalReqWordReg) := Fill(BytesOfWord, 1.U)
    l2Data(globalReqWordReg) := sharedDataReg
  }.otherwise {
    (0 until dcache_BlockWords).foreach { i =>
      l2Mask(i) := Fill(BytesOfWord, 1.U)
    }
  }
  io.to_l2cache.valid := state === s_l2_req || state === s_desc_l2_req
  io.to_l2cache.bits.a_opcode := Mux(
    state === s_desc_l2_req,
    4.U,
    1.U
  )
  io.to_l2cache.bits.a_source := Mux(l2ReqIsDataPut, tensorAckSource, descSource)
  io.to_l2cache.bits.a_addr.foreach(_ := pAddrReg)
  io.to_l2cache.bits.a_mask := l2Mask
  io.to_l2cache.bits.a_data := l2Data
  io.to_l2cache.bits.a_param := 0.U
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)

  io.from_l2cache.ready := state === s_desc_l2_rsp || state === s_l2_rsp
  io.inst_complete.valid := state === s_complete
  io.inst_complete.bits.wid := widReg
  io.inst_complete.bits.group := groupReg
  io.inst_complete.bits.is_s2g := true.B

  when(state === s_idle && io.from_fifo.fire) {
    sharedSrcReg := io.from_fifo.bits.in3(0)
    descPtrReg := io.from_fifo.bits.in1(0)
    widReg := io.from_fifo.bits.ctrl.wid
    groupReg := io.from_fifo.bits.ctrl.dma_group
    asidReg := io.from_fifo.bits.ctrl.asid.getOrElse(0.U)
    when(!reset.asBool) {
      assert(io.from_fifo.bits.ctrl.funct === 4.U,
        "DMA tensor S2G datapath received a non-tensor-S2G instruction")
      assert(io.from_fifo.bits.in1(0)(log2Ceil(l2cacheline) - 1, 0) === 0.U,
        "DMA tensor S2G requires 128B-aligned descriptor pointer")
      assert(io.from_fifo.bits.in3(0)(wordOffsetBits - 1, 0) === 0.U,
        "DMA tensor S2G requires 4-byte aligned shared source pointer")
    }
    for (i <- 0 until 5) {
      coordsReg(i) := io.from_fifo.bits.in2(i)
    }
    (0 until 5).foreach { i =>
      currentIdxReg(i) := 0.U
      currentIdxNextReg(i) := 0.U
      currentAdvanceDimOHReg(i) := false.B
      globalOffsetPartReg(i) := 0.U
      globalOffsetPartNextReg(i) := 0.U
      sharedOffsetPartReg(i) := 0.U
      sharedOffsetPartNextReg(i) := 0.U
      sharedRowPartReg(i) := 0.U
      sharedRowPartNextReg(i) := 0.U
    }
    globalDim0OffsetStepReg := 0.U
    globalLogicalOffsetReg := 0.U
    sharedLogicalOffsetReg := 0.U
    sharedRowReg := 0.U
    currentValidReg := false.B
    currentLastReg := false.B
    fastCoalesceReg := false.B
    descBoxAddressReg := 0.U
    descAddrAccumReg := 0.U
    descAddrSliceStrideReg := 0.U
    descAddrOpReg := descAddrOpPitch
    setupMulOpReg := setupMulOpByte2
    globalReqLineBaseReg := 0.U
    globalReqWordReg := 0.U
    sharedSetReg := 0.U
    sharedBlockReg := 0.U
    lineRowOffsetReg := 0.U
    lineRowBytesReg := 0.U
    lineGlobalBaseReg := 0.U
    lineSharedBaseReg := 0.U
    lineChunkBytesReg := 0.U
    lineTaskFirstReg := true.B
    for (i <- 0 until 32) {
      descWordsReg(i) := 0.U
    }
    when(incomingDescCacheHit) {
      descWordsReg := incomingDescCacheWords
    }
    state := Mux(incomingDescCacheHit, s_addr_init, s_desc_tlb_req)
  }

  val descCacheWrite = state === s_desc_l2_rsp && io.from_l2cache.fire
  val descCacheWriteLine = alignToL2Line(descPtrReg)
  val descCacheWriteHitVec = VecInit((0 until tma_desc_cache_entries).map { i =>
    descCacheValid(i) && descCacheLine(i) === descCacheWriteLine
  })
  val descCacheWriteHit = descCacheWriteHitVec.asUInt.orR
  val descCacheInvalidVec = VecInit((0 until tma_desc_cache_entries).map(i => !descCacheValid(i)))
  val descCacheHasInvalid = descCacheInvalidVec.asUInt.orR
  val descCacheVictim = Mux(descCacheHasInvalid, PriorityEncoder(descCacheInvalidVec), descCacheReplace)
  val descCacheWriteIdx = Mux(descCacheWriteHit, PriorityEncoder(descCacheWriteHitVec), descCacheVictim)

  when(state === s_desc_l2_rsp && io.from_l2cache.fire) {
    descWordsReg := io.from_l2cache.bits.d_data
  }
  when(descCacheWrite) {
    for (i <- 0 until tma_desc_cache_entries) {
      when(descCacheWriteIdx === i.U) {
        descCacheValid(i) := true.B
        descCacheLine(i) := descCacheWriteLine
        descCacheData(i) := io.from_l2cache.bits.d_data
      }
    }
    when(!descCacheWriteHit) {
      if (tma_desc_cache_entries > 1) {
        descCacheReplace := Mux(
          descCacheWriteIdx === (tma_desc_cache_entries - 1).U,
          0.U,
          descCacheWriteIdx + 1.U
        )
      }
    }
  }

  switch(state) {
    is(s_desc_tlb_req) {
      when(io.to_l2TLB.fire) { state := s_desc_tlb_rsp }
    }
    is(s_desc_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        pAddrReg := io.from_l2TLB.bits.paddr
        state := s_desc_l2_req
      }
    }
    is(s_desc_l2_req) {
      when(io.to_l2cache.fire) { state := s_desc_l2_rsp }
    }
    is(s_desc_l2_rsp) {
      when(io.from_l2cache.fire) { state := s_addr_init }
    }
    is(s_addr_init) {
      val cInSlice = Mux(
        tvars.interleaveMode === 1.U,
        coords(0)(1, 0).pad(xLen),
        coords(0)(2, 0).pad(xLen)
      )
      val linearStart =
        descWordsReg(2) + ((coords(0) << log2Ceil(dma_aligned_bulk))(xLen - 1, 0))
      val interleaveStart =
        descWordsReg(2) + ((cInSlice << log2Ceil(dma_aligned_bulk))(xLen - 1, 0))
      descAddrAccumReg := Mux(tvars.interleaveMode === 0.U, linearStart, interleaveStart)
      descAddrSliceStrideReg := Mux(tvars.interleaveMode === 1.U, 16.U(xLen.W), 32.U(xLen.W))
      descAddrOpReg := descAddrOpPitch
      state := s_addr_mul_start
    }
    is(s_addr_mul_start) {
      when(tmaSetupMul.io.in.fire) { state := s_addr_mul_wait }
    }
    is(s_addr_mul_wait) {
      when(tmaSetupMul.io.out.fire) {
        val product = tmaSetupMul.io.out.bits
        switch(descAddrOpReg) {
          is(descAddrOpPitch) {
            when(tvars.interleaveMode =/= 0.U && tvars.tensorRank >= 3.U) {
              descAddrSliceStrideReg := product
            }
            descAddrOpReg := descAddrOpSlice
            state := s_addr_mul_start
          }
          is(descAddrOpSlice) {
            when(tvars.interleaveMode =/= 0.U) {
              descAddrAccumReg := descAddrAccumReg + product
            }
            descAddrOpReg := descAddrOpDim1
            state := s_addr_mul_start
          }
          is(descAddrOpDim1) {
            descAddrAccumReg := descAddrAccumReg + product
            descAddrOpReg := descAddrOpDim2
            state := s_addr_mul_start
          }
          is(descAddrOpDim2) {
            descAddrAccumReg := descAddrAccumReg + product
            descAddrOpReg := descAddrOpDim3
            state := s_addr_mul_start
          }
          is(descAddrOpDim3) {
            descAddrAccumReg := descAddrAccumReg + product
            descAddrOpReg := descAddrOpDim4
            state := s_addr_mul_start
          }
          is(descAddrOpDim4) {
            descBoxAddressReg := descAddrAccumReg + product
            state := s_setup
          }
        }
      }
    }
    is(s_setup) {
      when(!reset.asBool) {
        assert(tvars.interleaveMode =/= 3.U,
          "DMA tensor S2G interleaveMode=3 is reserved")
        when(tvars.interleaveMode =/= 0.U) {
          assert(tvars.tensorRank >= 3.U,
            "DMA tensor S2G interleave requires rank >= 3")
        }
        assert(tensorDataWidth(tvars.dataType) === dma_aligned_bulk.U,
          "DMA tensor S2G v0 supports only 4-byte elements")
        (0 until 5).foreach { d =>
          assert(TmaPow2Math.isSupportedStride(tvars.elementStrides(d)),
            "DMA tensor S2G v0 supports only power-of-two element strides up to 32")
        }
      }
      (0 until 5).foreach { i =>
        rawOutDimReg(i) := rawOutDim(i)
        outDimReg(i) := outDim(i)
        outStrideBytesReg(i) := 0.U
        sharedRowStrideReg(i) := 0.U
        globalStrideBytesReg(i) := 0.U
        currentIdxNextReg(i) := 0.U
        currentAdvanceDimOHReg(i) := false.B
        globalOffsetPartReg(i) := 0.U
        globalOffsetPartNextReg(i) := 0.U
        sharedOffsetPartReg(i) := 0.U
        sharedOffsetPartNextReg(i) := 0.U
        sharedRowPartReg(i) := 0.U
        sharedRowPartNextReg(i) := 0.U
      }
      globalLogicalOffsetReg := 0.U
      linearGlobalBaseReg := descBoxAddressReg
      fastCoalesceReg := tensorLineCoalesce
      outStrideBytesReg(0) := dma_aligned_bulk.U
      sharedRowStrideReg(1) := 1.U
      globalStrideBytesReg(0) := TmaPow2Math.scaleByDataWidth(tvars.elementStrides(0), tvars.datawidth)
      (1 until 5).foreach { i =>
        globalStrideBytesReg(i) := TmaPow2Math.scaleByPow2(descByteStride(i), tvars.elementStrides(i))
      }
      state := Mux(outDimZeroSetup, s_complete, s_setup_stride_1)
    }
    is(s_setup_stride_1) {
      outStrideBytesReg(1) := (rawOutDimReg(0) << log2Ceil(dma_aligned_bulk))(xLen - 1, 0)
      sharedRowStrideReg(2) := rawOutDimReg(1)
      setupMulOpReg := setupMulOpByte2
      state := s_setup_mul_start
    }
    is(s_setup_mul_start) {
      when(tmaSetupMul.io.in.fire) { state := s_setup_mul_wait }
    }
    is(s_setup_mul_wait) {
      when(tmaSetupMul.io.out.fire) {
        val product = tmaSetupMul.io.out.bits
        switch(setupMulOpReg) {
          is(setupMulOpByte2) {
            outStrideBytesReg(2) := product
            setupMulOpReg := setupMulOpByte3
            state := s_setup_mul_start
          }
          is(setupMulOpByte3) {
            outStrideBytesReg(3) := product
            setupMulOpReg := setupMulOpRow3
            state := s_setup_mul_start
          }
          is(setupMulOpRow3) {
            sharedRowStrideReg(3) := product
            setupMulOpReg := setupMulOpByte4
            state := s_setup_mul_start
          }
          is(setupMulOpByte4) {
            outStrideBytesReg(4) := product
            setupMulOpReg := setupMulOpRow4
            state := s_setup_mul_start
          }
          is(setupMulOpRow4) {
            sharedRowStrideReg(4) := product
            state := Mux(fastCoalesceReg, s_line_prepare, s_prepare)
          }
        }
      }
    }
    is(s_prepare) {
      (0 until 5).foreach { i =>
        currentIdxNextReg(i) := currentIdxNext(i)
        currentAdvanceDimOHReg(i) := currentAdvanceDimOH(i)
      }
      globalDim0OffsetStepReg := globalDim0OffsetStep
      globalLogicalOffsetReg := globalLogicalOffset
      sharedLogicalOffsetReg := sharedLogicalOffset
      sharedRowReg := sharedRow
      currentValidReg := currentValid
      currentLastReg := currentLast
      state := s_prepare_addr
    }
    is(s_prepare_addr) {
      globalReqLineBaseReg := globalReqLineBase
      globalReqWordReg := globalReqWord
      sharedSetReg := sharedSet
      sharedBlockReg := sharedBlock
      (0 until 5).foreach { i =>
        globalOffsetPartNextReg(i) := globalOffsetPartPrepared(i)
        sharedOffsetPartNextReg(i) := sharedOffsetPartPrepared(i)
        sharedRowPartNextReg(i) := sharedRowPartPrepared(i)
      }
      state := s_check
    }
    is(s_check) {
      when(currentValidReg) {
        state := s_shared_req
      }.elsewhen(currentLastReg) {
        state := s_complete
      }.otherwise {
        (0 until 5).foreach { i =>
          currentIdxReg(i) := currentIdxNextReg(i)
          globalOffsetPartReg(i) := globalOffsetPartNextReg(i)
          sharedOffsetPartReg(i) := sharedOffsetPartNextReg(i)
          sharedRowPartReg(i) := sharedRowPartNextReg(i)
        }
        state := s_prepare
      }
    }
    is(s_shared_req) {
      when(io.shared_req.fire) { state := s_shared_rsp }
    }
    is(s_shared_rsp) {
      when(io.shared_rsp.fire) {
        when(!reset.asBool) {
          assert(!io.shared_rsp.bits.isWrite,
            "DMA tensor S2G shared response must be a read response")
        }
        sharedDataReg := io.shared_rsp.bits.data(0)
        state := s_tlb_req
      }
    }
    is(s_tlb_req) {
      when(io.to_l2TLB.fire) {
        state := s_tlb_rsp
      }
    }
    is(s_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        pAddrReg := io.from_l2TLB.bits.paddr
        state := s_l2_req
      }
    }
    is(s_l2_req) {
      when(io.to_l2cache.fire) { state := s_l2_rsp }
    }
    is(s_l2_rsp) {
      when(io.from_l2cache.fire) {
        when(!reset.asBool) {
          assert(io.from_l2cache.bits.d_opcode === 0.U,
            "DMA tensor S2G L2 response must be AccessAck")
        }
        when(currentLastReg) {
          state := s_complete
        }.otherwise {
          (0 until 5).foreach { i =>
            currentIdxReg(i) := currentIdxNextReg(i)
            globalOffsetPartReg(i) := globalOffsetPartNextReg(i)
            sharedOffsetPartReg(i) := sharedOffsetPartNextReg(i)
            sharedRowPartReg(i) := sharedRowPartNextReg(i)
          }
          state := s_prepare
        }
      }
    }
    is(s_line_prepare) {
      (0 until 5).foreach { i =>
        currentIdxNextReg(i) := rowIdxNext(i)
        currentAdvanceDimOHReg(i) := rowAdvanceDimOH(i)
      }
      lineRowOffsetReg := 0.U
      lineRowBytesReg := Mux(
        tensorInterleaveLine,
        interleaveSegmentBytes,
        lineRowBytesDefault
      )
      lineGlobalBaseReg := linearGlobalBaseReg + globalOffsetPartReg.reduce(_ + _)
      lineSharedBaseReg := sharedSrcReg + sharedOffsetPartReg.reduce(_ + _)
      currentLastReg := Mux(
        tensorInterleaveLine,
        interleaveSegmentLast,
        rowLast
      )
      state := s_line_shared_req
    }
    is(s_line_shared_req) {
      when(io.line_task.fire) {
        lineChunkBytesReg := lineChunkBytes
        lineTaskFirstReg := false.B
        state := s_line_advance
      }
    }
    is(s_line_advance) {
      val rowOffsetNext = lineRowOffsetReg + lineChunkBytesReg
      val interleaveAdvanceElems = lineRowBytesReg >> log2Ceil(dma_aligned_bulk)
      val interleaveAdvanceIdx0Next = currentIdxReg(0) + interleaveAdvanceElems
      val interleaveAdvanceDim0Done = interleaveAdvanceIdx0Next >= outDimReg(0)
      val interleaveAdvanceCoord0Next = coords(0) + interleaveAdvanceIdx0Next
      val interleaveAdvanceDim0Delta = TmaPow2Math.interleaveDim0DeltaBytes(
        coords(0) + currentIdxReg(0),
        interleaveAdvanceCoord0Next,
        descAddrSliceStrideReg,
        tvars.interleaveMode
      )
      when(rowOffsetNext < lineRowBytesReg) {
        lineRowOffsetReg := rowOffsetNext
        state := s_line_shared_req
      }.elsewhen(currentLastReg) {
        state := s_idle
      }.elsewhen(tensorInterleaveLine && !interleaveAdvanceDim0Done) {
        currentIdxReg(0) := interleaveAdvanceIdx0Next
        globalOffsetPartReg(0) := globalOffsetPartReg(0) + interleaveAdvanceDim0Delta
        sharedOffsetPartReg(0) := sharedOffsetPartReg(0) + lineRowBytesReg
        state := s_line_prepare
      }.otherwise {
        (0 until 5).foreach { i =>
          currentIdxReg(i) := currentIdxNextReg(i)
          globalOffsetPartReg(i) := rowGlobalOffsetPartPrepared(i)
          sharedOffsetPartReg(i) := rowSharedOffsetPartPrepared(i)
          sharedRowPartReg(i) := rowSharedRowPartPrepared(i)
        }
        state := s_line_prepare
      }
    }
    is(s_complete) {
      when(io.inst_complete.fire) { state := s_idle }
    }
  }

  if (PMU_DMA_S2G) {
    val tensorInputFire = io.from_fifo.fire
    val tensorSharedReqFire = io.shared_req.fire && state === s_shared_req
    val tensorSharedRspFire = io.shared_rsp.fire && state === s_shared_rsp
    val tensorTlbReqFire = io.to_l2TLB.fire && state === s_tlb_req
    val tensorDataPutFire = io.to_l2cache.fire && l2ReqIsDataPut
    val tensorAckFire = io.from_l2cache.fire && state === s_l2_rsp
    val tensorPutBytes = PopCount(l2Mask.asUInt).pad(64)
    if (PMU_DMA_S2G_DETAIL) {
      val tensorAckLatency = perfCycle.get - perfAckIssueCycle.get
      when(io.perfReset) {
        perfCycle.get := 0.U
        perfAckIssueCycle.get := 0.U
        perfAckLatencySum.get := 0.U
        perfLineFullStallCycles.get := 0.U
        perfReadEntryFullStallCycles.get := 0.U
        perfAckTagFullStallCycles.get := 0.U
      }.elsewhen(io.perfEnable) {
        perfCycle.get := perfCycle.get + 1.U
        when(tensorDataPutFire) {
          perfAckIssueCycle.get := perfCycle.get
        }
        when(tensorAckFire) {
          perfAckLatencySum.get := perfAckLatencySum.get + tensorAckLatency
        }
      }
    }

    when(io.perfReset) {
      perfInstIssued.get := 0.U
      perfLineIssued.get := 0.U
      perfPutFull.get := 0.U
      perfPutPart.get := 0.U
      perfBytesWritten.get := 0.U
      perfSharedReadReq.get := 0.U
      perfSharedReadRsp.get := 0.U
      perfTlbReq.get := 0.U
      perfAckCount.get := 0.U
    }.elsewhen(io.perfEnable) {
      when(tensorInputFire) {
        perfInstIssued.get := perfInstIssued.get + 1.U
      }
      when(tensorDataPutFire) {
        perfLineIssued.get := perfLineIssued.get + 1.U
        perfPutPart.get := perfPutPart.get + 1.U
        perfBytesWritten.get := perfBytesWritten.get + tensorPutBytes
      }
      when(tensorSharedReqFire) {
        perfSharedReadReq.get := perfSharedReadReq.get + 1.U
      }
      when(tensorSharedRspFire) {
        perfSharedReadRsp.get := perfSharedReadRsp.get + 1.U
      }
      when(tensorTlbReqFire) {
        perfTlbReq.get := perfTlbReq.get + 1.U
      }
      when(tensorAckFire) {
        perfAckCount.get := perfAckCount.get + 1.U
      }
    }

    io.perf.get.instIssued := perfInstIssued.get
    io.perf.get.lineIssued := perfLineIssued.get
    io.perf.get.putFull := perfPutFull.get
    io.perf.get.putPart := perfPutPart.get
    io.perf.get.bytesWritten := perfBytesWritten.get
    io.perf.get.sharedReadReq := perfSharedReadReq.get
    io.perf.get.sharedReadRsp := perfSharedReadRsp.get
    io.perf.get.tlbReq := perfTlbReq.get
    io.perf.get.ackCount := perfAckCount.get
    if (PMU_DMA_S2G_DETAIL) {
      io.perf.get.ackLatencySum := perfAckLatencySum.get
      io.perf.get.lineFullStallCycles := perfLineFullStallCycles.get
      io.perf.get.readEntryFullStallCycles := perfReadEntryFullStallCycles.get
      io.perf.get.ackTagFullStallCycles := perfAckTagFullStallCycles.get
    } else {
      io.perf.get.ackLatencySum := 0.U
      io.perf.get.lineFullStallCycles := 0.U
      io.perf.get.readEntryFullStallCycles := 0.U
      io.perf.get.ackTagFullStallCycles := 0.U
    }
  }
}
