/*
 * Shared-memory to global-memory DMA datapath for CP_ASYNC_BULK_S2G.
 *
 * First implementation is intentionally serial: one S2G instruction owns the
 * path, each chunk reads a contiguous shared-memory span, translates the global
 * destination line, issues one L2 Put, and completes only after the L2 ack.
 */
package pipeline

import chisel3._
import chisel3.util._
import config.config.Parameters
import L1Cache.{DCacheMemReq_p, DCacheMemRsp}
import mmu.{L1TlbReq, L1TlbRsp, SV32}
import top.parameters._

class DmaSharedRsp extends Bundle {
  val instrId = UInt(log2Up(lsu_nMshrEntry).W)
  val isWrite = Bool()
  val data = Vec(num_thread, UInt(xLen.W))
  val activeMask = Vec(num_thread, Bool())
}

class DmaS2G(implicit p: Parameters) extends Module {
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
  require(dmaSourceLowBits > 1, "DMA S2G source encoding needs a distinct non-meta low-bit pattern")

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, lineOffsetBits), 0.U(lineOffsetBits.W))

  def sharedSetIdx(addr: UInt): UInt =
    addr(sharedSetIdxHi, sharedSetIdxLo)

  def sharedBlockOffset(addr: UInt): UInt =
    addr(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits)

  def minUInt(a: UInt, b: UInt): UInt = Mux(a < b, a, b)

  val (s_idle :: s_shared_req :: s_shared_rsp :: s_tlb_req :: s_tlb_rsp ::
    s_l2_req :: s_l2_rsp :: s_complete :: Nil) = Enum(8)
  val state = RegInit(s_idle)

  val srcReg = RegInit(0.U(xLen.W))
  val dstReg = RegInit(0.U(xLen.W))
  val sizeReg = RegInit(0.U(xLen.W))
  val offsetReg = RegInit(0.U(xLen.W))
  val widReg = RegInit(0.U(depth_warp.W))
  val asidReg = RegInit(0.U(SV32.asidLen.W))
  val pAddrReg = RegInit(0.U(SV32.paLen.W))
  val chunkBytesReg = RegInit(0.U(xLen.W))
  val pendingReadMaskReg = RegInit(0.U(numgroupshared.W))
  val laneL2WordReg = Reg(Vec(numgroupshared, UInt(log2Ceil(dcache_BlockWords).W)))
  val laneByteMaskReg = Reg(Vec(numgroupshared, UInt(BytesOfWord.W)))
  val l2MaskReg = Reg(Vec(dcache_BlockWords, UInt(BytesOfWord.W)))
  val l2DataReg = Reg(Vec(dcache_BlockWords, UInt(xLen.W)))

  val curSrc = srcReg + offsetReg
  val curDst = dstReg + offsetReg
  val dstLineBase = alignToL2Line(curDst)
  val bytesLeft = sizeReg - offsetReg
  val bytesToDstLine = l2cacheline.U - curDst(lineOffsetBits - 1, 0)
  val bytesToSrcLine = l2cacheline.U - curSrc(lineOffsetBits - 1, 0)
  val bytesPerSharedReq = (numgroupshared * dma_aligned_bulk).U
  val chunkBytes = minUInt(bytesLeft, minUInt(bytesToDstLine, minUInt(bytesToSrcLine, bytesPerSharedReq)))
  val dstStartWord = curDst(lineOffsetBits - 1, wordOffsetBits)

  val laneByteMask = Wire(Vec(numgroupshared, UInt(BytesOfWord.W)))
  val laneActive = Wire(Vec(numgroupshared, Bool()))
  for (lane <- 0 until numgroupshared) {
    val maskBits = Wire(Vec(BytesOfWord, Bool()))
    for (byte <- 0 until BytesOfWord) {
      maskBits(byte) := (lane.U * dma_aligned_bulk.U + byte.U) < chunkBytes
    }
    laneByteMask(lane) := maskBits.asUInt
    laneActive(lane) := laneByteMask(lane).orR
  }
  val laneMask = laneActive.asUInt

  val sharedReqAddr = Wire(Vec(numgroupshared, UInt(xLen.W)))
  for (lane <- 0 until numgroupshared) {
    sharedReqAddr(lane) := curSrc + (lane.U << wordOffsetBits)
  }

  io.from_fifo.ready := state === s_idle

  io.shared_req.valid := state === s_shared_req
  io.shared_req.bits.instrId := 0.U
  io.shared_req.bits.isWrite := false.B
  io.shared_req.bits.setIdx := sharedSetIdx(curSrc)
  for (lane <- 0 until numgroupshared) {
    io.shared_req.bits.perLaneAddr(lane).activeMask := laneActive(lane)
    io.shared_req.bits.perLaneAddr(lane).blockOffset := sharedBlockOffset(sharedReqAddr(lane))
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H := Fill(BytesOfWord, 1.U)
    io.shared_req.bits.data(lane) := 0.U
  }

  io.shared_rsp.ready := state === s_shared_rsp

  io.to_l2TLB.valid := state === s_tlb_req
  io.to_l2TLB.bits.vaddr := dstLineBase
  io.to_l2TLB.bits.asid := asidReg
  io.from_l2TLB.ready := state === s_tlb_rsp

  val fullWordMask = Fill(BytesOfWord, 1.U)
  val isFullLinePut = l2MaskReg.map(_ === fullWordMask).reduce(_ && _)
  val s2gSource = Cat(
    0.U(log2Ceil(max_dma_tag).W),
    0.U(log2Ceil(max_dma_inst).W),
    2.U(dmaSourceLowBits.W)
  )

  io.to_l2cache.valid := state === s_l2_req
  io.to_l2cache.bits.a_opcode := Mux(isFullLinePut, 0.U, 1.U)
  io.to_l2cache.bits.a_param := 0.U
  io.to_l2cache.bits.a_source := s2gSource
  io.to_l2cache.bits.a_addr.foreach(_ := pAddrReg)
  io.to_l2cache.bits.a_data := l2DataReg
  io.to_l2cache.bits.a_mask := l2MaskReg
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)

  io.from_l2cache.ready := state === s_l2_rsp

  io.inst_complete.valid := state === s_complete
  io.inst_complete.bits := widReg

  when(state === s_idle && io.from_fifo.fire) {
    srcReg := io.from_fifo.bits.in1(0)
    dstReg := io.from_fifo.bits.in3(0)
    sizeReg := io.from_fifo.bits.in2(0)
    offsetReg := 0.U
    widReg := io.from_fifo.bits.ctrl.wid
    asidReg := io.from_fifo.bits.ctrl.asid.getOrElse(0.U)
    when(!reset.asBool) {
      assert(io.from_fifo.bits.ctrl.funct === 3.U,
        "DMA S2G datapath received a non-S2G instruction")
      assert(io.from_fifo.bits.in1(0)(wordOffsetBits - 1, 0) === 0.U &&
             io.from_fifo.bits.in3(0)(wordOffsetBits - 1, 0) === 0.U &&
             io.from_fifo.bits.in2(0)(wordOffsetBits - 1, 0) === 0.U,
        "DMA S2G requires 4-byte aligned src, dst, and size")
    }
    state := Mux(io.from_fifo.bits.in2(0) === 0.U, s_complete, s_shared_req)
  }

  when(state === s_shared_req && io.shared_req.fire) {
    chunkBytesReg := chunkBytes
    pendingReadMaskReg := laneMask
    for (word <- 0 until dcache_BlockWords) {
      l2MaskReg(word) := 0.U
      l2DataReg(word) := 0.U
    }
    for (lane <- 0 until numgroupshared) {
      val l2Word = dstStartWord + lane.U
      laneL2WordReg(lane) := l2Word(log2Ceil(dcache_BlockWords) - 1, 0)
      laneByteMaskReg(lane) := laneByteMask(lane)
    }
    state := s_shared_rsp
  }

  when(state === s_shared_rsp && io.shared_rsp.fire) {
    val rspMask = io.shared_rsp.bits.activeMask.asUInt
    val pendingNext = pendingReadMaskReg & ~rspMask
    when(!reset.asBool) {
      assert(!io.shared_rsp.bits.isWrite,
        "DMA S2G shared response must be a read response")
      assert((rspMask & ~pendingReadMaskReg) === 0.U,
        "DMA S2G shared response returned lanes that were not pending")
    }
    for (lane <- 0 until numgroupshared) {
      when(io.shared_rsp.bits.activeMask(lane)) {
        l2DataReg(laneL2WordReg(lane)) := io.shared_rsp.bits.data(lane)
        l2MaskReg(laneL2WordReg(lane)) := laneByteMaskReg(lane)
      }
    }
    pendingReadMaskReg := pendingNext
    when(pendingNext === 0.U) {
      state := s_tlb_req
    }
  }

  when(state === s_tlb_req && io.to_l2TLB.fire) {
    state := s_tlb_rsp
  }

  when(state === s_tlb_rsp && io.from_l2TLB.fire) {
    pAddrReg := io.from_l2TLB.bits.paddr
    state := s_l2_req
  }

  when(state === s_l2_req && io.to_l2cache.fire) {
    state := s_l2_rsp
  }

  when(state === s_l2_rsp && io.from_l2cache.fire) {
    when(!reset.asBool) {
      assert(io.from_l2cache.bits.d_opcode === 0.U,
        "DMA S2G L2 response must be AccessAck")
    }
    when(offsetReg + chunkBytesReg >= sizeReg) {
      state := s_complete
    }.otherwise {
      offsetReg := offsetReg + chunkBytesReg
      state := s_shared_req
    }
  }

  when(state === s_complete && io.inst_complete.fire) {
    state := s_idle
  }
}
