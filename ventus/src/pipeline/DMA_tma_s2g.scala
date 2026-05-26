
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

  val (s_idle :: s_desc_tlb_req :: s_desc_tlb_rsp :: s_desc_l2_req :: s_desc_l2_rsp ::
    s_setup :: s_check :: s_shared_req :: s_shared_rsp :: s_tlb_req :: s_tlb_rsp ::
    s_l2_req :: s_l2_rsp :: s_complete :: Nil) = Enum(14)
  val state = RegInit(s_idle)

  val sharedSrcReg = RegInit(0.U(xLen.W))
  val descPtrReg = RegInit(0.U(xLen.W))
  val widReg = RegInit(0.U(depth_warp.W))
  val asidReg = RegInit(0.U(SV32.asidLen.W))
  val pAddrReg = RegInit(0.U(SV32.paLen.W))
  val currentIdxReg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val sharedDataReg = Reg(UInt(xLen.W))
  val descWordsReg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))

  val coordsReg = Reg(Vec(5, UInt(xLen.W)))
  val coords = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { i => coords(i) := coordsReg(i) }

  val tvars = Wire(new TensorVars)
  val descControl = descWordsReg(1)
  val descByteStride = Wire(Vec(5, UInt(xLen.W)))
  val descBoxAddress = Wire(UInt(xLen.W))
  val outDim = Wire(Vec(5, UInt(xLen.W)))
  val outStrideBytes = Wire(Vec(5, UInt(xLen.W)))
  val outRowStride = Wire(Vec(5, UInt(xLen.W)))
  val currentIdx = Wire(Vec(5, UInt(xLen.W)))
  val currentCoord = Wire(Vec(5, UInt(xLen.W)))

  descByteStride(0) := Mux(descWordsReg(9) === 0.U, tensorDataWidth(descControl(3, 0)), descWordsReg(9))
  (1 until 5).foreach { i => descByteStride(i) := descWordsReg(9 + i) }
  val descGlobalDim = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { i => descGlobalDim(i) := descWordsReg(4 + i) }

  def tensorGlobalAddress(
      base: UInt,
      logicalCoord: Vec[UInt],
      rank: UInt,
      globalDim: Vec[UInt],
      byteStride: Vec[UInt],
      datawidth: UInt,
      interleaveMode: UInt
  ): UInt = {
    val linear = base +
      logicalCoord(0) * byteStride(0) +
      logicalCoord(1) * byteStride(1) +
      logicalCoord(2) * byteStride(2) +
      logicalCoord(3) * byteStride(3) +
      logicalCoord(4) * byteStride(4)

    val sliceBytes = Mux(interleaveMode === 1.U, 16.U(xLen.W), 32.U(xLen.W))
    // Tensor S2G v0 only accepts 4-byte elements, checked in s_setup.
    val cSlice = Mux(interleaveMode === 1.U, logicalCoord(0) >> 2, logicalCoord(0) >> 3)
    val cInSlice = Mux(
      interleaveMode === 1.U,
      logicalCoord(0)(1, 0).pad(xLen),
      logicalCoord(0)(2, 0).pad(xLen)
    )
    val cSliceStride = Wire(UInt(xLen.W))
    cSliceStride := sliceBytes
    when(rank === 3.U) { cSliceStride := byteStride(1) * globalDim(1) }
      .elsewhen(rank === 4.U) { cSliceStride := byteStride(2) * globalDim(2) }
      .elsewhen(rank === 5.U) { cSliceStride := byteStride(3) * globalDim(3) }

    val interleaved = base +
      cInSlice * dma_aligned_bulk.U(xLen.W) +
      cSlice * cSliceStride +
      logicalCoord(1) * byteStride(1) +
      logicalCoord(2) * byteStride(2) +
      logicalCoord(3) * byteStride(3) +
      logicalCoord(4) * byteStride(4)

    Mux(interleaveMode === 0.U, linear, interleaved)
  }

  descBoxAddress := tensorGlobalAddress(
    descWordsReg(2),
    coords,
    descControl(6, 4),
    descGlobalDim,
    descByteStride,
    tensorDataWidth(descControl(3, 0)),
    descControl(9, 8)
  )

  tvars.BoxAddress := descBoxAddress
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
  tvars.datawidth := tensorDataWidth(tvars.dataType)

  outDim(0) := tvars.boxDim(0)
  (1 until 5).foreach { i =>
    outDim(i) := Mux(
      tvars.tensorRank > i.U,
      Mux(tvars.elementStrides(i) <= 1.U,
        tvars.boxDim(i),
        (tvars.boxDim(i) + tvars.elementStrides(i) - 1.U) / tvars.elementStrides(i)),
      1.U)
  }

  outStrideBytes(0) := tvars.datawidth
  (1 until 5).foreach { i => outStrideBytes(i) := outStrideBytes(i - 1) * outDim(i - 1) }

  outRowStride(0) := 0.U
  outRowStride(1) := 1.U
  (2 until 5).foreach { i => outRowStride(i) := outRowStride(i - 1) * outDim(i - 1) }

  (0 until 5).foreach { i => currentIdx(i) := currentIdxReg(i) }

  val currentLastDim = Wire(Vec(5, Bool()))
  (0 until 5).foreach { i =>
    currentLastDim(i) := (tvars.tensorRank <= i.U) || ((currentIdxReg(i) + 1.U) >= outDim(i))
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
  val outDimZero = (0 until 5).map(i => outDim(i) === 0.U).reduce(_ || _)

  currentCoord(0) := coords(0) + currentIdx(0) * tvars.elementStrides(0)
  currentCoord(1) := coords(1) + currentIdx(1) * tvars.elementStrides(1)
  currentCoord(2) := coords(2) + currentIdx(2) * tvars.elementStrides(2)
  currentCoord(3) := coords(3) + currentIdx(3) * tvars.elementStrides(3)
  currentCoord(4) := coords(4) + currentIdx(4) * tvars.elementStrides(4)

  val currentValidExpr0 = currentCoord(0) < tvars.globalDim(0)
  var currentValidExpr = currentValidExpr0
  (1 until 5).foreach { i =>
    currentValidExpr = currentValidExpr && ((tvars.tensorRank <= i.U) || (currentCoord(i) < tvars.globalDim(i)))
  }
  val currentValid = Wire(Bool())
  currentValid := currentValidExpr

  val sharedLogicalOffset = Wire(UInt(xLen.W))
  val sharedRow = Wire(UInt(xLen.W))
  val sharedRowStride = Wire(Vec(5, UInt(xLen.W)))
  sharedRowStride(0) := 0.U
  sharedRowStride(1) := 1.U
  (2 until 5).foreach { i => sharedRowStride(i) := sharedRowStride(i - 1) * outDim(i - 1) }
  sharedLogicalOffset :=
    currentIdx(0) * outStrideBytes(0) +
      currentIdx(1) * outStrideBytes(1) +
      currentIdx(2) * outStrideBytes(2) +
      currentIdx(3) * outStrideBytes(3) +
      currentIdx(4) * outStrideBytes(4)
  sharedRow :=
    currentIdx(1) * sharedRowStride(1) +
      currentIdx(2) * sharedRowStride(2) +
      currentIdx(3) * sharedRowStride(3) +
      currentIdx(4) * sharedRowStride(4)

  val sharedCurrentAddr = swizzleSharedAddr(sharedSrcReg + sharedLogicalOffset, sharedSrcReg, tvars.swizzleMode, sharedRow)
  val globalCurrentAddr = tensorGlobalAddress(
    tvars.globalAddress,
    currentCoord,
    tvars.tensorRank,
    tvars.globalDim,
    descByteStride,
    tvars.datawidth,
    tvars.interleaveMode
  )

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
  io.shared_req.bits.setIdx := sharedSet
  (0 until numgroupshared).foreach { lane =>
    io.shared_req.bits.perLaneAddr(lane).activeMask := (if (lane == 0) true.B else false.B)
    io.shared_req.bits.perLaneAddr(lane).blockOffset := sharedBlock
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H := (if (lane == 0) Fill(BytesOfWord, 1.U) else 0.U)
    io.shared_req.bits.data(lane) := 0.U
  }

  io.shared_rsp.ready := state === s_shared_rsp

  io.to_l2TLB.valid := state === s_tlb_req || state === s_desc_tlb_req
  io.to_l2TLB.bits.vaddr := Mux(state === s_desc_tlb_req || state === s_desc_tlb_rsp || state === s_desc_l2_req || state === s_desc_l2_rsp,
    alignToL2Line(descPtrReg),
    globalReqLineBase)
  io.to_l2TLB.bits.asid := asidReg
  io.from_l2TLB.ready := state === s_tlb_rsp || state === s_desc_tlb_rsp

  val l2Mask = Wire(Vec(dcache_BlockWords, UInt(BytesOfWord.W)))
  val l2Data = Wire(Vec(dcache_BlockWords, UInt(xLen.W)))
  (0 until dcache_BlockWords).foreach { i =>
    l2Mask(i) := 0.U
    l2Data(i) := 0.U
  }
  when(state === s_l2_req) {
    l2Mask(globalReqWord) := Fill(BytesOfWord, 1.U)
    l2Data(globalReqWord) := sharedDataReg
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
    (0 until 5).foreach { i => currentIdxReg(i) := 0.U }
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
      when(io.from_l2cache.fire) { state := s_setup }
    }
    is(s_setup) {
      when(!reset.asBool) {
        assert(tvars.interleaveMode =/= 3.U,
          "DMA tensor S2G interleaveMode=3 is reserved")
        when(tvars.interleaveMode =/= 0.U) {
          assert(tvars.tensorRank >= 3.U,
            "DMA tensor S2G interleave requires rank >= 3")
        }
        assert(tvars.datawidth === dma_aligned_bulk.U,
          "DMA tensor S2G v0 supports only 4-byte elements")
      }
      state := Mux(outDimZero, s_complete, s_check)
    }
    is(s_check) {
      when(currentValid) {
        state := s_shared_req
      }.elsewhen(currentLast) {
        state := s_complete
      }.otherwise {
        (0 until 5).foreach { i => currentIdxReg(i) := currentIdxNext(i) }
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
        when(currentLast) {
          state := s_complete
        }.otherwise {
          (0 until 5).foreach { i => currentIdxReg(i) := currentIdxNext(i) }
          state := s_check
        }
      }
    }
    is(s_complete) {
      when(io.inst_complete.fire) { state := s_idle }
    }
  }
}
