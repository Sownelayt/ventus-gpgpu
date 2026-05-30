
/*
 * Descriptor-addressed tensor shared-to-global DMA datapath.
 *
 * v0 is intentionally conservative: the RTL supports 4-byte tensor elements,
 * linear element iteration, optional shared-memory swizzle, and a full shared
 * read -> TLB -> L2 Put sequence per element.
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
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
    val shared_rsp = Flipped(DecoupledIO(new DmaSharedRsp))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
    val to_l2cache = DecoupledIO(new DCacheMemReq_p)
    val from_l2cache = Flipped(DecoupledIO(new DCacheMemRsp))
    val inst_complete = DecoupledIO(UInt(depth_warp.W))
  })

  val lineOffsetBits = log2Ceil(l2cacheline)
  val wordOffsetBits = log2Ceil(dma_aligned_bulk)
  val sharedSetIdxHi = log2Ceil(sharedmem_depth) + dcache_BlockOffsetBits + dcache_WordOffsetBits - 1
  val sharedSetIdxLo = dcache_BlockOffsetBits + dcache_WordOffsetBits
  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 2, "DMA tensor S2G source encoding needs at least 3 low bits")

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, lineOffsetBits), 0.U(lineOffsetBits.W))

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

  val tensorS2GStates = Enum(28)
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
  val state = RegInit(s_idle)

  val sharedSrcReg = RegInit(0.U(xLen.W))
  val descPtrReg = RegInit(0.U(xLen.W))
  val widReg = RegInit(0.U(depth_warp.W))
  val asidReg = RegInit(0.U(SV32.asidLen.W))
  val pAddrReg = RegInit(0.U(SV32.paLen.W))
  val currentIdxReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
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
  val descWordsReg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))
  val descBoxAddressReg = RegInit(0.U(xLen.W))
  val descAddrAccumReg = RegInit(0.U(xLen.W))
  val descAddrSliceStrideReg = RegInit(0.U(xLen.W))
  val descAddrOpReg = RegInit(0.U(3.W))
  val setupMulOpReg = RegInit(0.U(3.W))
  val tmaSetupMul = Module(new TmaMul32Unit)

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

  outDim(0) := tvars.boxDim(0)
  (1 until 5).foreach { i =>
    outDim(i) := Mux(
      tvars.tensorRank > i.U,
      Mux(tvars.elementStrides(i) <= 1.U,
        tvars.boxDim(i),
        TmaPow2Math.ceilDivPow2(tvars.boxDim(i), tvars.elementStrides(i))),
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

  val setupMulA = WireDefault(0.U(xLen.W))
  val setupMulB = WireDefault(0.U(xLen.W))
  switch(setupMulOpReg) {
    is(setupMulOpByte2) {
      setupMulA := outStrideBytesReg(1)
      setupMulB := outDimReg(1)
    }
    is(setupMulOpByte3) {
      setupMulA := outStrideBytesReg(2)
      setupMulB := outDimReg(2)
    }
    is(setupMulOpRow3) {
      setupMulA := sharedRowStrideReg(2)
      setupMulB := outDimReg(2)
    }
    is(setupMulOpByte4) {
      setupMulA := outStrideBytesReg(3)
      setupMulB := outDimReg(3)
    }
    is(setupMulOpRow4) {
      setupMulA := sharedRowStrideReg(3)
      setupMulB := outDimReg(3)
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

  val globalLogicalOffset = globalOffsetPartReg.reduce(_ + _)
  val sharedLogicalOffset = sharedOffsetPartReg.reduce(_ + _)
  val sharedRow = sharedRowPartReg.reduce(_ + _)

  val sharedCurrentAddr = swizzleSharedAddr(sharedSrcReg + sharedLogicalOffsetReg, sharedSrcReg, tvars.swizzleMode, sharedRowReg)
  val globalCurrentAddr = linearGlobalBaseReg + globalLogicalOffsetReg

  val globalReqLineBase = alignToL2Line(globalCurrentAddr)
  val globalReqWord = globalCurrentAddr(lineOffsetBits - 1, wordOffsetBits)
  val sharedSet = sharedSetIdx(sharedCurrentAddr)
  val sharedBlock = sharedBlockOffset(sharedCurrentAddr)

  val descSource = ((2 << 1) | 1).U(l1cache_sourceBits.W)
  val tensorAckSource = Cat(
    0.U(log2Ceil(max_dma_tag).W),
    0.U(log2Ceil(max_dma_inst).W),
    4.U(dmaSourceLowBits.W)
  )

  io.from_fifo.ready := state === s_idle
  io.shared_req.valid := state === s_shared_req
  io.shared_req.bits.instrId := 1.U
  io.shared_req.bits.isWrite := false.B
  io.shared_req.bits.setIdx := sharedSetReg
  (0 until numgroupshared).foreach { lane =>
    io.shared_req.bits.perLaneAddr(lane).activeMask := (if (lane == 0) true.B else false.B)
    io.shared_req.bits.perLaneAddr(lane).blockOffset := sharedBlockReg
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H := (if (lane == 0) Fill(BytesOfWord, 1.U) else 0.U)
    io.shared_req.bits.data(lane) := 0.U
  }

  io.shared_rsp.ready := state === s_shared_rsp

  io.to_l2TLB.valid := state === s_tlb_req || state === s_desc_tlb_req
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
  io.to_l2cache.bits.a_opcode := Mux(state === s_l2_req, 1.U, 4.U)
  io.to_l2cache.bits.a_source := Mux(state === s_l2_req, tensorAckSource, descSource)
  io.to_l2cache.bits.a_addr.foreach(_ := pAddrReg)
  io.to_l2cache.bits.a_mask := l2Mask
  io.to_l2cache.bits.a_data := l2Data
  io.to_l2cache.bits.a_param := 0.U
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)

  io.from_l2cache.ready := state === s_desc_l2_rsp || state === s_l2_rsp
  io.inst_complete.valid := state === s_complete
  io.inst_complete.bits := widReg

  when(state === s_idle && io.from_fifo.fire) {
    sharedSrcReg := io.from_fifo.bits.in3(0)
    descPtrReg := io.from_fifo.bits.in1(0)
    widReg := io.from_fifo.bits.ctrl.wid
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
    descBoxAddressReg := 0.U
    descAddrAccumReg := 0.U
    descAddrSliceStrideReg := 0.U
    descAddrOpReg := descAddrOpPitch
    setupMulOpReg := setupMulOpByte2
    for (i <- 0 until 32) {
      descWordsReg(i) := 0.U
    }
    state := s_desc_tlb_req
  }

  when(state === s_desc_l2_rsp && io.from_l2cache.fire) {
    descWordsReg := io.from_l2cache.bits.d_data
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
      outStrideBytesReg(0) := dma_aligned_bulk.U
      sharedRowStrideReg(1) := 1.U
      globalStrideBytesReg(0) := TmaPow2Math.scaleByDataWidth(tvars.elementStrides(0), tvars.datawidth)
      (1 until 5).foreach { i =>
        globalStrideBytesReg(i) := TmaPow2Math.scaleByPow2(descByteStride(i), tvars.elementStrides(i))
      }
      state := Mux(outDimZeroSetup, s_complete, s_setup_stride_1)
    }
    is(s_setup_stride_1) {
      outStrideBytesReg(1) := (outDimReg(0) << log2Ceil(dma_aligned_bulk))(xLen - 1, 0)
      sharedRowStrideReg(2) := outDimReg(1)
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
            state := s_prepare
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
      when(io.to_l2TLB.fire) { state := s_tlb_rsp }
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
    is(s_complete) {
      when(io.inst_complete.fire) { state := s_idle }
    }
  }
}
