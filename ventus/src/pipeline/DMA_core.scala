/*
 * DMA Core for Ventus GPGPU
 *
 * Current scope:
 *
 *   Implemented here:  Tensor DMA address generation, OOB fill, element
 *                      stride/subbox iteration, 32B/64B/128B swizzle,
 *                      descriptor-addressed TMA, tensor-map prefetch, bulk
 *                      shared -> global writeback, and tensor shared -> global
 *                      writeback.
 *   Not started:       Im2col, TC FP16/BF16.
 */
package pipeline

import chisel3._
import chisel3.util._
import top.parameters._
import L1Cache.{DCacheMemReq_p, DCacheMemRsp}
import config.config.Parameters
import mmu.{L1TlbReq, L1TlbRsp, SV32}

// DataType enumeration for tensor element types.
//
// Must match the encoding in spike/riscv/insns/cp_async_tensor.h and the
// `dataType` field used by testcases host code:
//
//   0  UINT8   1B    1  UINT16  2B    2  UINT32  4B    3  INT8    1B
//   4  INT16   2B    5  INT32   4B    6  FP32    4B    7  FP16    2B
//   8  BF16    2B    9  UINT64  8B    10 INT64   8B    11 FP64    8B
//
object DataType {
  val UINT8        = 0.U(4.W)
  val UINT16       = 1.U(4.W)
  val UINT32       = 2.U(4.W)
  val INT8         = 3.U(4.W)
  val INT16        = 4.U(4.W)
  val INT32        = 5.U(4.W)
  val FLOAT32      = 6.U(4.W)
  val FLOAT16      = 7.U(4.W)
  val BFLOAT16     = 8.U(4.W)
  val UINT64       = 9.U(4.W)
  val INT64        = 10.U(4.W)
  val FLOAT64      = 11.U(4.W)
}

object TmaPow2Math {
  def isSupportedStride(stride: UInt): Bool =
    stride === 1.U || stride === 2.U || stride === 4.U || stride === 8.U ||
      stride === 16.U || stride === 32.U

  def scaleByPow2(value: UInt, factor: UInt): UInt = {
    MuxLookup(factor, value)(Seq(
      1.U  -> value,
      2.U  -> ((value << 1)(xLen - 1, 0)),
      4.U  -> ((value << 2)(xLen - 1, 0)),
      8.U  -> ((value << 3)(xLen - 1, 0)),
      16.U -> ((value << 4)(xLen - 1, 0)),
      32.U -> ((value << 5)(xLen - 1, 0))
    ))
  }

  def scaleByDataWidth(value: UInt, datawidth: UInt): UInt = {
    MuxLookup(datawidth, value)(Seq(
      1.U -> value,
      2.U -> ((value << 1)(xLen - 1, 0)),
      4.U -> ((value << 2)(xLen - 1, 0)),
      8.U -> ((value << 3)(xLen - 1, 0))
    ))
  }

  def ceilDivPow2(value: UInt, stride: UInt): UInt = {
    MuxLookup(stride, value)(Seq(
      1.U  -> value,
      2.U  -> ((value + 1.U) >> 1),
      4.U  -> ((value + 3.U) >> 2),
      8.U  -> ((value + 7.U) >> 3),
      16.U -> ((value + 15.U) >> 4),
      32.U -> ((value + 31.U) >> 5)
    ))
  }

  def scaleBySmallFactor(value: UInt, factor: UInt): UInt = {
    val value2 = (value << 1)(xLen - 1, 0)
    val value3 = (value2 + value)(xLen - 1, 0)
    val value4 = (value << 2)(xLen - 1, 0)
    val value5 = (value4 + value)(xLen - 1, 0)
    val value6 = (value4 + value2)(xLen - 1, 0)
    val value7 = (value4 + value2 + value)(xLen - 1, 0)
    val value8 = (value << 3)(xLen - 1, 0)
    MuxLookup(factor, 0.U(xLen.W))(Seq(
      0.U -> 0.U(xLen.W),
      1.U -> value,
      2.U -> value2,
      3.U -> value3,
      4.U -> value4,
      5.U -> value5,
      6.U -> value6,
      7.U -> value7,
      8.U -> value8
    ))
  }//a simple multiply implement

  def interleaveDim0DeltaBytes(
      currentCoord: UInt,
      nextCoord: UInt,
      sliceStride: UInt,
      interleaveMode: UInt
  ): UInt = {
    val currentInSlice = Mux(interleaveMode === 1.U, currentCoord(1, 0).pad(xLen), currentCoord(2, 0).pad(xLen))
    val nextInSlice = Mux(interleaveMode === 1.U, nextCoord(1, 0).pad(xLen), nextCoord(2, 0).pad(xLen))
    val inSliceDeltaBytes = ((nextInSlice - currentInSlice) << 2)(xLen - 1, 0)
    val currSlice = Mux(interleaveMode === 1.U, currentCoord >> 2, currentCoord >> 3)
    val nextSlice = Mux(interleaveMode === 1.U, nextCoord >> 2, nextCoord >> 3)
    val sliceDeltaBytes = scaleBySmallFactor(sliceStride, nextSlice - currSlice)
    (sliceDeltaBytes + inSliceDeltaBytes)(xLen - 1, 0)
  }

  def elementOffsetBytes(index: UInt, stride: UInt, datawidth: UInt): UInt =
    scaleByDataWidth(scaleByPow2(index, stride), datawidth)

  def rowSpanBytes(count: UInt, stride: UInt, datawidth: UInt): UInt =
    Mux(count === 0.U, 0.U, elementOffsetBytes(count - 1.U, stride, datawidth) + datawidth)
}

class TmaMul32Unit extends Module {
  val io = IO(new Bundle {
    val in = Flipped(DecoupledIO(new Bundle {
      val a = UInt(xLen.W)
      val b = UInt(xLen.W)
    }))
    val out = DecoupledIO(UInt(xLen.W))
  })

  val sIdle :: sCalc :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val p0Reg = RegInit(0.U(xLen.W))
  val p1LoReg = RegInit(0.U(16.W))
  val p2LoReg = RegInit(0.U(16.W))
  val resultReg = RegInit(0.U(xLen.W))

  val aLo = io.in.bits.a(15, 0)
  val aHi = io.in.bits.a(31, 16)
  val bLo = io.in.bits.b(15, 0)
  val bHi = io.in.bits.b(31, 16)
  val p0 = aLo * bLo
  val p1 = aLo * bHi
  val p2 = aHi * bLo

  io.in.ready := state === sIdle
  io.out.valid := state === sDone
  io.out.bits := resultReg

  when(io.in.fire) {
    p0Reg := p0
    p1LoReg := p1(15, 0)
    p2LoReg := p2(15, 0)
    state := sCalc
  }.elsewhen(state === sCalc) {
    val mid = p1LoReg +& p2LoReg
    val upper = p0Reg(31, 16) + mid(15, 0)
    resultReg := Cat(upper(15, 0), p0Reg(15, 0))
    state := sDone
  }.elsewhen(io.out.fire) {
    state := sIdle
  }
}

// Tensor descriptor bundle for descriptor-addressed CP_ASYNC_TENSOR_G2S (funct=2)
class TensorVars extends Bundle {
  val boxDim         = Vec(5, UInt(xLen.W))
  val elementStrides = Vec(5, UInt(xLen.W))
  val interleaveMode = UInt(log2Ceil(3).W)
  val swizzleMode    = UInt(log2Ceil(4).W)
  val L2promotion    = UInt(log2Ceil(4).W)
  val oobfill        = UInt(log2Ceil(2).W)
  val datawidth      = UInt(4.W)  //  4 bits to represent 8
  val dataType       = UInt(log2Ceil(13).W)
  val tensorRank     = UInt(log2Ceil(5).W)
  val globalAddress  = UInt(xLen.W)
  val globalDim      = Vec(5, UInt(xLen.W))
  val globalStrides  = Vec(5, UInt(xLen.W))
}

// Saved register state for address calculation
class DmaRegSave extends Bundle {
  val in1 = Vec(num_thread, UInt(xLen.W))
  val in2 = Vec(num_thread, UInt(xLen.W))
  val in3 = Vec(num_thread, UInt(xLen.W))
  val address = UInt(xLen.W)
  val ctrl = new CtrlSigs()
}

// Decoded DMA instruction info stored in temp memory
class vExeDataDMA extends Bundle {
  val src = UInt(xLen.W)
  val dst = UInt(xLen.W)
  val funct = UInt(4.W)
  val copysize = UInt(xLen.W)
  val srcsize = UInt(xLen.W)
  val wid = UInt(depth_warp.W)
  val tensorvars = new TensorVars  // only valid when funct=2
}

// Tag info for each L2 cacheline request
class DmaCachelineInfo extends Bundle {
  val tag = UInt(xLen.W)
  val inst_index = UInt(log2Ceil(max_dma_inst).W)
  val box_dim0_start    = UInt(xLen.W)
  val tensor_dim0_start = UInt(xLen.W)
  val tensor_high_dim_valid = Bool()
  val shared_row_base   = UInt(xLen.W)
  val swizzle_row_low   = UInt(3.W)
  val tensor_copy       = Bool()
  val tensor_interleave = Bool()
  val tensor_elem_valid = Bool()
  val tensor_line_oob   = Bool()
  val tensor_elem_addr  = UInt(xLen.W)
  val shared_elem_addr  = UInt(xLen.W)
}

// Combined L2 response + tag info
class DmaL2CachelineInfo(implicit p: Parameters) extends Bundle {
  val base = new DCacheMemRsp
  val cacheline_info = new DmaCachelineInfo
}

// Output from Temp_mem to Addrcalc_shared
class DmaTempOutput extends Bundle {
  val entry_index = UInt(log2Ceil(max_dma_inst).W)
  val mask = Vec(numgroupshared, Bool())
  val data = Vec(numgroupshared, UInt((dma_aligned_bulk * BitsOfByte).W))
  val tag = UInt(xLen.W)
  val box_dim0_start = UInt(xLen.W)
  val shared_row_base = UInt(xLen.W)
  val swizzle_row_low = UInt(3.W)
  val tensor_interleave = Bool()
  val shared_elem_addr = UInt(xLen.W)
  val inst_src = UInt(xLen.W)
  val inst_dst = UInt(xLen.W)
  val inst_funct = UInt(4.W)
  val tensor_datawidth = UInt(4.W)
  val tensor_element_stride0 = UInt(xLen.W)
  val tensor_swizzle_mode = UInt(log2Ceil(4).W)
}

// ============================================================
// AddrCalc_l2cache: decode DMA instruction, generate L2 requests
// ============================================================
class AddrCalc_l2cache(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val from_fifo = Flipped(DecoupledIO(new vExeData))
    val to_tempmem_inst = DecoupledIO(new vExeDataDMA)
    val to_tempmem_tag = DecoupledIO(new DmaCachelineInfo)
    val inst_mem_index = Input(UInt(log2Ceil(max_dma_inst).W))
    val tag_mem_index = Input(UInt(log2Ceil(max_dma_tag).W))
    val to_l2cache = DecoupledIO(new DCacheMemReq_p)
    val from_l2cache_meta = Flipped(DecoupledIO(new DCacheMemRsp))
    val tag_reuse_hit = Input(Bool())
    val meta_complete = DecoupledIO(UInt(depth_warp.W))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
  })

  val addrCalcStates = Enum(19)
  val s_idle = addrCalcStates(0)
  val s_prefetch_tlb_req = addrCalcStates(1)
  val s_prefetch_tlb_rsp = addrCalcStates(2)
  val s_prefetch_l2cache = addrCalcStates(3)
  val s_desc_tlb_req = addrCalcStates(4)
  val s_desc_tlb_rsp = addrCalcStates(5)
  val s_desc_l2cache = addrCalcStates(6)
  val s_desc_rsp = addrCalcStates(7)
  val s_tensor_addr_init = addrCalcStates(8)
  val s_addr_mul_start = addrCalcStates(9)
  val s_addr_mul_wait = addrCalcStates(10)
  val s_tensor_setup = addrCalcStates(11)
  val s_setup_mul_start = addrCalcStates(12)
  val s_setup_mul_wait = addrCalcStates(13)
  val s_save = addrCalcStates(14)
  val s_l2cache_tag = addrCalcStates(15)
  val s_tlb_req = addrCalcStates(16)
  val s_tlb_rsp = addrCalcStates(17)
  val s_l2cache = addrCalcStates(18)
  val state = RegInit(s_idle)
  val reg_save = Reg(new DmaRegSave)
  val inst_mem_index_reg = RegInit(0.U(log2Up(max_dma_inst).W))
  val tag_mem_index_reg = RegInit(0.U(log2Ceil(max_dma_tag).W))
  val p_addr_reg = Reg(UInt(SV32.paLen.W))
  val desc_ptr_reg = RegInit(0.U(xLen.W))
  val meta_vaddr_reg = RegInit(0.U(xLen.W))
  val desc_words_reg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))
  val dyn_words_reg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))

  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 0, "DMA source encoding needs a spare low bit for metadata responses")
  require(tma_desc_cache_entries > 0, "TMA descriptor cache must have at least one entry")
  require(tma_prefetch_slots > 0, "TMA prefetch response sink must have at least one slot")
  val dmaMetaKindBits = 2
  val dmaMetaSlotShift = dmaMetaKindBits + 1
  val prefetchSlotIdxWidth = log2Ceil(tma_prefetch_slots).max(1)
  require(dmaMetaSlotShift + prefetchSlotIdxWidth <= l1cache_sourceBits,
    "DMA metadata source encoding needs enough bits for prefetch slots")
  val dmaMetaKindPrefetch = 0.U(dmaMetaKindBits.W)
  val dmaMetaKindDesc = 1.U(dmaMetaKindBits.W)
  def dmaMetaSource(kind: Int): UInt =
    ((kind << 1) | 1).U(l1cache_sourceBits.W)
  def dmaPrefetchSource(slot: UInt): UInt = {
    val src = Wire(UInt(l1cache_sourceBits.W))
    src := ((slot << dmaMetaSlotShift).asUInt | dmaMetaSource(0))(l1cache_sourceBits - 1, 0)
    src
  }
  val dmaDescSource = dmaMetaSource(1)

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))

  val prefetchCompleteQ = Module(new Queue(UInt(depth_warp.W), tma_prefetch_slots))
  val prefetchSlotValid = RegInit(VecInit(Seq.fill(tma_prefetch_slots)(false.B)))
  val prefetchSlotWid = RegInit(VecInit(Seq.fill(tma_prefetch_slots)(0.U(depth_warp.W))))
  val prefetchSlotLine = RegInit(VecInit(Seq.fill(tma_prefetch_slots)(0.U(xLen.W))))
  val prefetchSlotFreeVec = VecInit((0 until tma_prefetch_slots).map(i => !prefetchSlotValid(i)))
  val prefetchSlotAvailable = prefetchSlotFreeVec.asUInt.orR
  val prefetchAllocSlot = PriorityEncoder(prefetchSlotFreeVec)

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

  // ---- Tensor iteration state ----
  import DataType._
  val tensor_dim_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_base_global_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_global_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_offset_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_shared_offset_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_stride_bytes_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_shared_stride_bytes_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_row_base_reg = RegInit(0.U(xLen.W))
  val tensor_shared_row_base_reg = RegInit(0.U(xLen.W))
  val desc_box_address_reg = RegInit(0.U(xLen.W))
  val desc_addr_accum_reg = RegInit(0.U(xLen.W))
  val desc_addr_slice_stride_reg = RegInit(0.U(xLen.W))
  val desc_addr_op_reg = RegInit(0.U(3.W))
  val tensor_setup_mul_op_reg = RegInit(0.U(2.W))
  val tensor_setup_box_elements_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim0_offset_bytes_reg = RegInit(0.U(xLen.W))
  val tensor_row_span_bytes_reg = RegInit(0.U(xLen.W))
  val tensor_copy_size_reg = RegInit(0.U(xLen.W))
  val tensor_swizzle_row_low_reg = RegInit(0.U(3.W))
  val tensor_rank_reg = RegInit(0.U(3.W))
  val tensor_interleave_mode_reg = RegInit(false.B)
  val tensor_interleave_sel_reg = RegInit(0.U(2.W))
  val tensor_box_dim_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_element_stride_reg = RegInit(VecInit(Seq.fill(5)(1.U(xLen.W))))
  val tensor_global_dim_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_can_advance_limit_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_pos_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_global_pos_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_dim_offset_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_shared_offset_next = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach(x => tensor_dim_pos_next(x) := tensor_dim_pos_reg(x))
  (0 until 5).foreach(x => tensor_global_pos_next(x) := tensor_global_pos_reg(x))
  (0 until 5).foreach(x => tensor_dim_offset_next(x) := tensor_dim_offset_reg(x))
  (0 until 5).foreach(x => tensor_shared_offset_next(x) := tensor_shared_offset_reg(x))

  // ---- TensorVars extraction from descriptor memory ----
  val tvars = Wire(new TensorVars)
  val desc_control = desc_words_reg(1)
  def tensorDataWidth(dataType: UInt): UInt = Mux(
    dataType === UINT8 || dataType === INT8, 1.U,
    Mux(
      dataType === UINT16 || dataType === INT16 ||
        dataType === FLOAT16 || dataType === BFLOAT16, 2.U,
      Mux(
        dataType === UINT32 || dataType === INT32 ||
          dataType === FLOAT32, 4.U,
        8.U  // UINT64 / INT64 / FLOAT64
      )
    )
  )
  val desc_byte_stride = Wire(Vec(5, UInt(xLen.W)))
  desc_byte_stride(0) := Mux(desc_words_reg(9) === 0.U, tensorDataWidth(desc_control(3, 0)), desc_words_reg(9))
  (1 until 5).foreach { x => desc_byte_stride(x) := desc_words_reg(9 + x) }
  val desc_coords = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { x => desc_coords(x) := dyn_words_reg(x) }
  val tensor_copy_mode = reg_save.ctrl.funct === 2.U
  val tensor_interleave_mode = tensor_copy_mode && (tvars.interleaveMode =/= 0.U)
  (0 until 5).foreach { x =>
    tvars.boxDim(x) := desc_words_reg(14 + x)
    tvars.elementStrides(x) := Mux(desc_words_reg(19 + x) === 0.U, 1.U, desc_words_reg(19 + x))
  }
  tvars.interleaveMode := desc_control(9, 8)
  tvars.swizzleMode    := desc_control(11, 10)
  tvars.L2promotion    := desc_control(13, 12)
  tvars.oobfill        := desc_control(14).asUInt
  tvars.dataType       := desc_control(3, 0)
  tvars.tensorRank     := desc_control(6, 4)
  tvars.globalAddress  := desc_words_reg(2)
  (0 until 5).foreach { x =>
    tvars.globalDim(x) := desc_words_reg(4 + x)
  }
  tvars.globalStrides(0) := desc_byte_stride(1)
  tvars.globalStrides(1) := desc_byte_stride(2)
  tvars.globalStrides(2) := desc_byte_stride(3)
  tvars.globalStrides(3) := desc_byte_stride(4)
  tvars.globalStrides(4) := 0.U

  // datawidth derived from dataType — encoding matches spike cp_async_tensor.h.
  tvars.datawidth := tensorDataWidth(tvars.dataType)


  val descAddrOpPitch = 0.U(3.W)
  val descAddrOpSlice = 1.U(3.W)
  val descAddrOpDim1 = 2.U(3.W)
  val descAddrOpDim2 = 3.U(3.W)
  val descAddrOpDim3 = 4.U(3.W)
  val descAddrOpDim4 = 5.U(3.W)
  val setupMulOpShared2 = 0.U(2.W)
  val setupMulOpShared3 = 1.U(2.W)
  val setupMulOpShared4 = 2.U(2.W)
  val setupMulOpCopy5 = 3.U(2.W)

  val descAddrMulA = WireDefault(0.U(xLen.W))
  val descAddrMulB = WireDefault(0.U(xLen.W))
  switch(desc_addr_op_reg) {
    is(descAddrOpPitch) {
      descAddrMulA := MuxLookup(tvars.tensorRank, 0.U(xLen.W))(Seq(
        3.U -> desc_byte_stride(1),
        4.U -> desc_byte_stride(2),
        5.U -> desc_byte_stride(3)
      ))
      descAddrMulB := MuxLookup(tvars.tensorRank, 0.U(xLen.W))(Seq(
        3.U -> tvars.globalDim(1),
        4.U -> tvars.globalDim(2),
        5.U -> tvars.globalDim(3)
      ))
    }
    is(descAddrOpSlice) {
      descAddrMulA := Mux(tvars.interleaveMode === 1.U, desc_coords(0) >> 2, desc_coords(0) >> 3)
      descAddrMulB := desc_addr_slice_stride_reg
    }
    is(descAddrOpDim1) {
      descAddrMulA := desc_coords(1)
      descAddrMulB := desc_byte_stride(1)
    }
    is(descAddrOpDim2) {
      descAddrMulA := desc_coords(2)
      descAddrMulB := desc_byte_stride(2)
    }
    is(descAddrOpDim3) {
      descAddrMulA := desc_coords(3)
      descAddrMulB := desc_byte_stride(3)
    }
    is(descAddrOpDim4) {
      descAddrMulA := desc_coords(4)
      descAddrMulB := desc_byte_stride(4)
    }
  }

  val setupMulA = WireDefault(0.U(xLen.W))
  val setupMulB = WireDefault(0.U(xLen.W))
  switch(tensor_setup_mul_op_reg) {
    is(setupMulOpShared2) {
      setupMulA := tensor_shared_stride_bytes_reg(1)
      setupMulB := tensor_setup_box_elements_reg(1)
    }
    is(setupMulOpShared3) {
      setupMulA := tensor_shared_stride_bytes_reg(2)
      setupMulB := tensor_setup_box_elements_reg(2)
    }
    is(setupMulOpShared4) {
      setupMulA := tensor_shared_stride_bytes_reg(3)
      setupMulB := tensor_setup_box_elements_reg(3)
    }
    is(setupMulOpCopy5) {
      setupMulA := tensor_shared_stride_bytes_reg(4)
      setupMulB := tensor_setup_box_elements_reg(4)
    }
  }

  val tmaAddrSetupMul = Module(new TmaMul32Unit)
  tmaAddrSetupMul.io.in.valid := state === s_addr_mul_start || state === s_setup_mul_start
  tmaAddrSetupMul.io.in.bits.a := Mux(state === s_addr_mul_start, descAddrMulA, setupMulA)
  tmaAddrSetupMul.io.in.bits.b := Mux(state === s_addr_mul_start, descAddrMulB, setupMulB)
  tmaAddrSetupMul.io.out.ready := state === s_addr_mul_wait || state === s_setup_mul_wait

  // Tensor setup still decodes the descriptor once. The cacheline issue path
  // below then advances with registered offsets and adders instead of repeated
  // per-cacheline div/mod/multiply chains.
  val box_elements_num_setup = Wire(Vec(5, UInt(xLen.W)))
  box_elements_num_setup(0) := tvars.boxDim(0)
  (1 until 5).foreach { x =>
    box_elements_num_setup(x) := Mux(
      tvars.elementStrides(x) <= 1.U,
      tvars.boxDim(x),
      TmaPow2Math.ceilDivPow2(tvars.boxDim(x), tvars.elementStrides(x))
    )
  }

  val tensor_dim_stride_bytes_setup = Wire(Vec(5, UInt(xLen.W)))
  tensor_dim_stride_bytes_setup(0) := TmaPow2Math.scaleByDataWidth(tvars.elementStrides(0), tvars.datawidth)
  (1 until 5).foreach { x =>
    tensor_dim_stride_bytes_setup(x) := TmaPow2Math.scaleByPow2(tvars.globalStrides(x - 1), tvars.elementStrides(x))
  }


  // The tensor box base is computed directly from desc_coords above, so keep the
  // original logical coordinates instead of reconstructing them with div/mod.
  val box_dim0_offset_bytes_setup = TmaPow2Math.scaleByDataWidth(desc_coords(0), desc_byte_stride(0))
  val box_offset_elems_setup = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { x => box_offset_elems_setup(x) := desc_coords(x) }

  val tensor_interleave_active = tensor_copy_mode && tensor_interleave_mode_reg
  val tensor_high_dim_valid_terms = (1 until 5).map { d =>
    (tensor_rank_reg <= d.U) || (tensor_global_pos_reg(d) < tensor_global_dim_reg(d))
  }
  val tensor_high_dim_valid = tensor_high_dim_valid_terms.reduce(_ && _)

  val box_dim0_start = tensor_row_base_reg
  val tensor_dim0_row_start = tensor_row_base_reg - tensor_dim0_offset_bytes_reg
  val box_dim0_end = tensor_row_base_reg + tensor_row_span_bytes_reg

  val tensor_can_advance_dim = Wire(Vec(5, Bool()))
  tensor_can_advance_dim(0) := tensor_interleave_active &&
    (tensor_dim_pos_reg(0) < tensor_dim_can_advance_limit_reg(0))
  (1 until 5).foreach { d =>
    tensor_can_advance_dim(d) :=
      (tensor_rank_reg > d.U) &&
      (tensor_dim_pos_reg(d) < tensor_dim_can_advance_limit_reg(d))
  }

  val tensor_advance_dim_oh = Wire(Vec(5, Bool()))
  tensor_advance_dim_oh(0) := tensor_can_advance_dim(0)
  tensor_advance_dim_oh(1) := !tensor_can_advance_dim(0) && tensor_can_advance_dim(1)
  tensor_advance_dim_oh(2) := !tensor_can_advance_dim(0) && !tensor_can_advance_dim(1) && tensor_can_advance_dim(2)
  tensor_advance_dim_oh(3) := !tensor_can_advance_dim(0) && !tensor_can_advance_dim(1) && !tensor_can_advance_dim(2) && tensor_can_advance_dim(3)
  tensor_advance_dim_oh(4) := !tensor_can_advance_dim(0) && !tensor_can_advance_dim(1) && !tensor_can_advance_dim(2) && !tensor_can_advance_dim(3) && tensor_can_advance_dim(4)
  val tensor_has_next_row = tensor_advance_dim_oh.asUInt.orR
  val tensor_interleave_dim0_delta_bytes = TmaPow2Math.interleaveDim0DeltaBytes(
    tensor_global_pos_reg(0),
    tensor_global_pos_next(0),
    desc_addr_slice_stride_reg,
    tensor_interleave_sel_reg
  )
  val tensor_dim0_offset_step = Mux(
    tensor_interleave_active,
    tensor_interleave_dim0_delta_bytes,
    tensor_dim_stride_bytes_reg(0)
  )

  when(tensor_advance_dim_oh(0)) {
    tensor_dim_pos_next(0) := tensor_dim_pos_reg(0) + tensor_element_stride_reg(0)
    tensor_global_pos_next(0) := tensor_global_pos_reg(0) + tensor_element_stride_reg(0)
    tensor_dim_offset_next(0) := tensor_dim_offset_reg(0) + tensor_dim0_offset_step
    tensor_shared_offset_next(0) := tensor_shared_offset_reg(0) + tensor_shared_stride_bytes_reg(0)
  }.elsewhen(tensor_advance_dim_oh(1)) {
    tensor_dim_pos_next(0) := 0.U
    tensor_global_pos_next(0) := tensor_base_global_pos_reg(0)
    tensor_dim_offset_next(0) := 0.U
    tensor_shared_offset_next(0) := 0.U
    tensor_dim_pos_next(1) := tensor_dim_pos_reg(1) + tensor_element_stride_reg(1)
    tensor_global_pos_next(1) := tensor_global_pos_reg(1) + tensor_element_stride_reg(1)
    tensor_dim_offset_next(1) := tensor_dim_offset_reg(1) + tensor_dim_stride_bytes_reg(1)
    tensor_shared_offset_next(1) := tensor_shared_offset_reg(1) + tensor_shared_stride_bytes_reg(1)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    (0 to 1).foreach { d =>
      tensor_dim_pos_next(d) := 0.U
      tensor_global_pos_next(d) := tensor_base_global_pos_reg(d)
      tensor_dim_offset_next(d) := 0.U
      tensor_shared_offset_next(d) := 0.U
    }
    tensor_dim_pos_next(2) := tensor_dim_pos_reg(2) + tensor_element_stride_reg(2)
    tensor_global_pos_next(2) := tensor_global_pos_reg(2) + tensor_element_stride_reg(2)
    tensor_dim_offset_next(2) := tensor_dim_offset_reg(2) + tensor_dim_stride_bytes_reg(2)
    tensor_shared_offset_next(2) := tensor_shared_offset_reg(2) + tensor_shared_stride_bytes_reg(2)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    (0 to 2).foreach { d =>
      tensor_dim_pos_next(d) := 0.U
      tensor_global_pos_next(d) := tensor_base_global_pos_reg(d)
      tensor_dim_offset_next(d) := 0.U
      tensor_shared_offset_next(d) := 0.U
    }
    tensor_dim_pos_next(3) := tensor_dim_pos_reg(3) + tensor_element_stride_reg(3)
    tensor_global_pos_next(3) := tensor_global_pos_reg(3) + tensor_element_stride_reg(3)
    tensor_dim_offset_next(3) := tensor_dim_offset_reg(3) + tensor_dim_stride_bytes_reg(3)
    tensor_shared_offset_next(3) := tensor_shared_offset_reg(3) + tensor_shared_stride_bytes_reg(3)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    (0 to 3).foreach { d =>
      tensor_dim_pos_next(d) := 0.U
      tensor_global_pos_next(d) := tensor_base_global_pos_reg(d)
      tensor_dim_offset_next(d) := 0.U
      tensor_shared_offset_next(d) := 0.U
    }
    tensor_dim_pos_next(4) := tensor_dim_pos_reg(4) + tensor_element_stride_reg(4)
    tensor_global_pos_next(4) := tensor_global_pos_reg(4) + tensor_element_stride_reg(4)
    tensor_dim_offset_next(4) := tensor_dim_offset_reg(4) + tensor_dim_stride_bytes_reg(4)
    tensor_shared_offset_next(4) := tensor_shared_offset_reg(4) + tensor_shared_stride_bytes_reg(4)
  }

  val tensor_linear_row_delta = WireDefault(0.U(xLen.W))
  when(tensor_advance_dim_oh(1)) {
    tensor_linear_row_delta := tensor_dim_stride_bytes_reg(1)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    tensor_linear_row_delta := tensor_dim_stride_bytes_reg(2) - tensor_dim_offset_reg(1)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    tensor_linear_row_delta := tensor_dim_stride_bytes_reg(3) - tensor_dim_offset_reg(1) - tensor_dim_offset_reg(2)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    tensor_linear_row_delta := tensor_dim_stride_bytes_reg(4) - tensor_dim_offset_reg(1) -
      tensor_dim_offset_reg(2) - tensor_dim_offset_reg(3)
  }
  val tensor_interleave_row_delta = WireDefault(0.U(xLen.W))
  when(tensor_advance_dim_oh(0)) {
    tensor_interleave_row_delta := tensor_dim0_offset_step
  }.elsewhen(tensor_advance_dim_oh(1)) {
    tensor_interleave_row_delta := tensor_dim_stride_bytes_reg(1) - tensor_dim_offset_reg(0)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    tensor_interleave_row_delta := tensor_dim_stride_bytes_reg(2) - tensor_dim_offset_reg(0) - tensor_dim_offset_reg(1)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    tensor_interleave_row_delta := tensor_dim_stride_bytes_reg(3) - tensor_dim_offset_reg(0) -
      tensor_dim_offset_reg(1) - tensor_dim_offset_reg(2)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    tensor_interleave_row_delta := tensor_dim_stride_bytes_reg(4) - tensor_dim_offset_reg(0) -
      tensor_dim_offset_reg(1) - tensor_dim_offset_reg(2) - tensor_dim_offset_reg(3)
  }
  val tensor_next_row_base = tensor_row_base_reg + Mux(
    tensor_interleave_active,
    tensor_interleave_row_delta,
    tensor_linear_row_delta
  )

  val tensor_linear_shared_row_delta = WireDefault(0.U(xLen.W))
  when(tensor_advance_dim_oh(1)) {
    tensor_linear_shared_row_delta := tensor_shared_stride_bytes_reg(1)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    tensor_linear_shared_row_delta := tensor_shared_stride_bytes_reg(2) - tensor_shared_offset_reg(1)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    tensor_linear_shared_row_delta := tensor_shared_stride_bytes_reg(3) - tensor_shared_offset_reg(1) - tensor_shared_offset_reg(2)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    tensor_linear_shared_row_delta := tensor_shared_stride_bytes_reg(4) - tensor_shared_offset_reg(1) -
      tensor_shared_offset_reg(2) - tensor_shared_offset_reg(3)
  }
  val tensor_interleave_shared_row_delta = WireDefault(0.U(xLen.W))
  when(tensor_advance_dim_oh(0)) {
    tensor_interleave_shared_row_delta := tensor_shared_stride_bytes_reg(0)
  }.elsewhen(tensor_advance_dim_oh(1)) {
    tensor_interleave_shared_row_delta := tensor_shared_stride_bytes_reg(1) - tensor_shared_offset_reg(0)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    tensor_interleave_shared_row_delta := tensor_shared_stride_bytes_reg(2) - tensor_shared_offset_reg(0) - tensor_shared_offset_reg(1)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    tensor_interleave_shared_row_delta := tensor_shared_stride_bytes_reg(3) - tensor_shared_offset_reg(0) -
      tensor_shared_offset_reg(1) - tensor_shared_offset_reg(2)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    tensor_interleave_shared_row_delta := tensor_shared_stride_bytes_reg(4) - tensor_shared_offset_reg(0) -
      tensor_shared_offset_reg(1) - tensor_shared_offset_reg(2) - tensor_shared_offset_reg(3)
  }
  val tensor_next_shared_row_base = tensor_shared_row_base_reg + Mux(
    tensor_interleave_active,
    tensor_interleave_shared_row_delta,
    tensor_linear_shared_row_delta
  )

  val tensor_row_has_next_line = !tensor_interleave_active && ((reg_save.address + l2cacheline.U) < box_dim0_end)
  val address_next = Wire(UInt(xLen.W))
  address_next := reg_save.address + l2cacheline.U
  when(tensor_copy_mode) {
    when(tensor_row_has_next_line) {
      address_next := reg_save.address + l2cacheline.U
    }.elsewhen(tensor_has_next_row) {
      address_next := alignToL2Line(tensor_next_row_base)
    }
  }

  // Aligned next cacheline boundary
  val next_cacheline = Wire(UInt(xLen.W))
  next_cacheline := Cat(reg_save.address(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W)) + l2cacheline.U

  // Complete when next cacheline >= src + srcsize (linear) or highest dim done (tensor)
  val complete_address = Wire(Bool())
  complete_address := Mux(tensor_copy_mode,
    !tensor_row_has_next_line && !tensor_has_next_row,
    next_cacheline >= (reg_save.in1(0) + reg_save.in2(0)))

  // Temp inst store interface
  io.to_tempmem_inst.valid := state === s_save
  io.to_tempmem_inst.bits.src := Mux(tensor_copy_mode, tvars.globalAddress, reg_save.in1(0))
  io.to_tempmem_inst.bits.srcsize := Mux(tensor_copy_mode, tvars.datawidth, reg_save.in2(0))
  io.to_tempmem_inst.bits.dst := reg_save.in3(0)
  io.to_tempmem_inst.bits.wid := reg_save.ctrl.wid
  io.to_tempmem_inst.bits.funct := Mux(tensor_copy_mode, 2.U, reg_save.ctrl.funct)
  io.to_tempmem_inst.bits.tensorvars := tvars
  io.to_tempmem_inst.bits.copysize := 0.U
  switch(reg_save.ctrl.funct) {
    is(0.U) { io.to_tempmem_inst.bits.copysize := 4.U << reg_save.ctrl.copysize }
    is(1.U) { io.to_tempmem_inst.bits.copysize := reg_save.in2(0) }
    is(2.U) { io.to_tempmem_inst.bits.copysize := tensor_copy_size_reg }
  }

  // Temp tag store interface
  val tensor_tag_line_base = Cat(reg_save.address(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))
  val tensor_tag_line_end = tensor_tag_line_base + l2cacheline.U
  val tensor_dim0_valid_end = tensor_dim0_row_start +
    TmaPow2Math.scaleByDataWidth(tensor_global_dim_reg(0), tvars.datawidth)
  val tensor_interleave_elem_valid = tensor_high_dim_valid &&
    (tensor_global_pos_reg(0) < tensor_global_dim_reg(0))
  val tensor_noninterleave_line_valid = tensor_high_dim_valid &&
    (tensor_tag_line_base < tensor_dim0_valid_end) &&
    (tensor_tag_line_end > tensor_dim0_row_start)
  val tensor_line_has_valid_data = Mux(
    tensor_interleave_active,
    tensor_interleave_elem_valid,
    tensor_noninterleave_line_valid
  )
  io.to_tempmem_tag.valid := state === s_l2cache_tag
  io.to_tempmem_tag.bits.tag := tensor_tag_line_base
  io.to_tempmem_tag.bits.inst_index := inst_mem_index_reg
  io.to_tempmem_tag.bits.box_dim0_start := box_dim0_start
  io.to_tempmem_tag.bits.tensor_dim0_start := tensor_dim0_row_start
  io.to_tempmem_tag.bits.tensor_high_dim_valid := tensor_high_dim_valid
  io.to_tempmem_tag.bits.shared_row_base := tensor_shared_row_base_reg
  io.to_tempmem_tag.bits.swizzle_row_low := tensor_swizzle_row_low_reg
  io.to_tempmem_tag.bits.tensor_copy := tensor_copy_mode
  io.to_tempmem_tag.bits.tensor_interleave := tensor_interleave_active
  io.to_tempmem_tag.bits.tensor_elem_valid := tensor_interleave_elem_valid
  io.to_tempmem_tag.bits.tensor_line_oob := tensor_copy_mode && !tensor_line_has_valid_data
  io.to_tempmem_tag.bits.tensor_elem_addr := tensor_row_base_reg
  io.to_tempmem_tag.bits.shared_elem_addr := tensor_shared_row_base_reg

  // TLB request
  val aligned_vaddr = alignToL2Line(reg_save.address)
  val aligned_meta_vaddr = alignToL2Line(meta_vaddr_reg)
  val meta_tlb_req_state =
    state === s_prefetch_tlb_req || state === s_desc_tlb_req
  val meta_tlb_rsp_state =
    state === s_prefetch_tlb_rsp || state === s_desc_tlb_rsp
  val meta_l2cache_state =
    state === s_prefetch_l2cache || state === s_desc_l2cache
  io.to_l2TLB.valid := state === s_tlb_req || meta_tlb_req_state
  io.to_l2TLB.bits.vaddr := Mux(meta_tlb_req_state, aligned_meta_vaddr, aligned_vaddr)
  io.to_l2TLB.bits.asid := reg_save.ctrl.asid.getOrElse(0.U)

  // TLB response
  io.from_l2TLB.ready := state === s_tlb_rsp || meta_tlb_rsp_state

  // L2 cache request — uses translated physical address
  io.to_l2cache.valid := state === s_l2cache ||
    state === s_desc_l2cache ||
    (state === s_prefetch_l2cache && prefetchSlotAvailable)
  io.to_l2cache.bits.a_opcode := 4.U // Get
  val metaReqSource = MuxCase(dmaDescSource, Seq(
    (state === s_prefetch_l2cache) -> dmaPrefetchSource(prefetchAllocSlot)
  ))
  io.to_l2cache.bits.a_source := Mux(
    meta_l2cache_state,
    metaReqSource,
    Cat(tag_mem_index_reg, inst_mem_index_reg, 0.U(dmaSourceLowBits.W))
  )
  io.to_l2cache.bits.a_addr.foreach(_ := p_addr_reg)
  io.to_l2cache.bits.a_mask := VecInit(Seq.fill(dcache_BlockWords)(Fill(BytesOfWord, 1.U)))
  io.to_l2cache.bits.a_data := VecInit(Seq.fill(dcache_BlockWords)(0.U(xLen.W)))
  io.to_l2cache.bits.a_param := 0.U
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)

  val metaRspKind = io.from_l2cache_meta.bits.d_source(2, 1)
  val metaRspIsPrefetch = io.from_l2cache_meta.bits.d_source(0) && metaRspKind === dmaMetaKindPrefetch
  val metaRspIsDesc = io.from_l2cache_meta.bits.d_source(0) && metaRspKind === dmaMetaKindDesc
  val prefetchRspSlot = io.from_l2cache_meta.bits.d_source(
    dmaMetaSlotShift + prefetchSlotIdxWidth - 1,
    dmaMetaSlotShift)
  val prefetchRspSlotValid = prefetchSlotValid(prefetchRspSlot)
  prefetchCompleteQ.io.enq.valid := io.from_l2cache_meta.valid && metaRspIsPrefetch && prefetchRspSlotValid
  prefetchCompleteQ.io.enq.bits := prefetchSlotWid(prefetchRspSlot)
  io.from_l2cache_meta.ready := MuxCase(false.B, Seq(
    metaRspIsPrefetch -> (prefetchRspSlotValid && prefetchCompleteQ.io.enq.ready),
    metaRspIsDesc -> (state === s_desc_rsp)
  ))
  val prefetchRspFire = io.from_l2cache_meta.fire && metaRspIsPrefetch
  val descRspFire = io.from_l2cache_meta.fire && metaRspIsDesc
  io.meta_complete.valid := prefetchCompleteQ.io.deq.valid
  io.meta_complete.bits := prefetchCompleteQ.io.deq.bits
  prefetchCompleteQ.io.deq.ready := io.meta_complete.ready

  io.from_fifo.ready := state === s_idle

  // TLB timeout watchdog
  val tlb_wait_cnt = RegInit(0.U(16.W))
  when(state === s_tlb_req || state === s_tlb_rsp || meta_tlb_req_state || meta_tlb_rsp_state) {
    tlb_wait_cnt := tlb_wait_cnt + 1.U
  }.otherwise {
    tlb_wait_cnt := 0.U
  }
  when(!reset.asBool) {
    assert(tlb_wait_cnt < 1024.U, "DMA TLB response timeout: possible deadlock")
  }

  // TLB page offset consistency assertion
  when(!reset.asBool && (state === s_tlb_rsp || meta_tlb_rsp_state) && io.from_l2TLB.fire) {
    val tlb_check_vaddr = Mux(meta_tlb_rsp_state, aligned_meta_vaddr, aligned_vaddr)
    assert(io.from_l2TLB.bits.paddr(SV32.offsetLen - 1, 0) ===
           tlb_check_vaddr(SV32.offsetLen - 1, 0),
      "TLB paddr page offset mismatch with vaddr")
  }

  when(state === s_prefetch_l2cache && io.to_l2cache.fire) {
    prefetchSlotValid(prefetchAllocSlot) := true.B
    prefetchSlotWid(prefetchAllocSlot) := reg_save.ctrl.wid
    prefetchSlotLine(prefetchAllocSlot) := aligned_meta_vaddr
  }

  val prefetchRspLine = Wire(UInt(xLen.W))
  prefetchRspLine := Mux1H(
    (0 until tma_prefetch_slots).map(i => prefetchRspSlot === i.U),
    (0 until tma_prefetch_slots).map(i => prefetchSlotLine(i))
  )
  when(prefetchRspFire) {
    prefetchSlotValid(prefetchRspSlot) := false.B
  }

  val descCacheWrite = prefetchRspFire || descRspFire
  val descCacheWriteLine = Mux(prefetchRspFire, prefetchRspLine, alignToL2Line(desc_ptr_reg))
  val descCacheWriteHitVec = VecInit((0 until tma_desc_cache_entries).map { i =>
    descCacheValid(i) && descCacheLine(i) === descCacheWriteLine
  })
  val descCacheWriteHit = descCacheWriteHitVec.asUInt.orR
  val descCacheInvalidVec = VecInit((0 until tma_desc_cache_entries).map(i => !descCacheValid(i)))
  val descCacheHasInvalid = descCacheInvalidVec.asUInt.orR
  val descCacheVictim = Mux(descCacheHasInvalid, PriorityEncoder(descCacheInvalidVec), descCacheReplace)
  val descCacheWriteIdx = Mux(descCacheWriteHit, PriorityEncoder(descCacheWriteHitVec), descCacheVictim)
  when(descCacheWrite) {
    for (i <- 0 until tma_desc_cache_entries) {
      when(descCacheWriteIdx === i.U) {
        descCacheValid(i) := true.B
        descCacheLine(i) := descCacheWriteLine
        descCacheData(i) := io.from_l2cache_meta.bits.d_data
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

  // FSM
  switch(state) {
    is(s_idle) {
      when(io.from_fifo.fire) {
        when(io.from_fifo.bits.ctrl.funct === 5.U) {
          state := s_prefetch_tlb_req
        }.elsewhen(io.from_fifo.bits.ctrl.funct === 2.U) {
          when(incomingDescCacheHit) {
            state := s_tensor_addr_init
          }.otherwise {
            state := s_desc_tlb_req
          }
        }.otherwise {
          state := s_save
        }
      }
    }
    is(s_prefetch_tlb_req) {
      when(io.to_l2TLB.fire) { state := s_prefetch_tlb_rsp }
    }
    is(s_prefetch_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        p_addr_reg := io.from_l2TLB.bits.paddr
        state := s_prefetch_l2cache
      }
    }
    is(s_prefetch_l2cache) {
      when(io.to_l2cache.fire) { state := s_idle }
    }
    is(s_desc_tlb_req) {
      when(io.to_l2TLB.fire) { state := s_desc_tlb_rsp }
    }
    is(s_desc_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        p_addr_reg := io.from_l2TLB.bits.paddr
        state := s_desc_l2cache
      }
    }
    is(s_desc_l2cache) {
      when(io.to_l2cache.fire) { state := s_desc_rsp }
    }
    is(s_desc_rsp) {
      when(descRspFire) {
        state := s_tensor_addr_init
      }
    }
    is(s_tensor_addr_init) {
      state := s_addr_mul_start
    }
    is(s_addr_mul_start) {
      when(tmaAddrSetupMul.io.in.fire) { state := s_addr_mul_wait }
    }
    is(s_addr_mul_wait) {
      when(tmaAddrSetupMul.io.out.fire) {
        when(desc_addr_op_reg === descAddrOpDim4) { state := s_tensor_setup }
          .otherwise { state := s_addr_mul_start }
      }
    }
    is(s_tensor_setup) {
      val setupHasZero = (0 until 5).map { d =>
        (tvars.tensorRank > d.U) && (box_elements_num_setup(d) === 0.U)
      }.reduce(_ || _)
      when(setupHasZero || tvars.tensorRank <= 1.U) { state := s_save }
        .otherwise { state := s_setup_mul_start }
    }
    is(s_setup_mul_start) {
      when(tmaAddrSetupMul.io.in.fire) { state := s_setup_mul_wait }
    }
    is(s_setup_mul_wait) {
      when(tmaAddrSetupMul.io.out.fire) {
        val setupDone =
          (tensor_setup_mul_op_reg === setupMulOpShared2 && tensor_rank_reg <= 2.U) ||
          (tensor_setup_mul_op_reg === setupMulOpShared3 && tensor_rank_reg <= 3.U) ||
          (tensor_setup_mul_op_reg === setupMulOpShared4 && tensor_rank_reg <= 4.U) ||
          (tensor_setup_mul_op_reg === setupMulOpCopy5)
        when(setupDone) { state := s_save }
          .otherwise { state := s_setup_mul_start }
      }
    }
    is(s_save) {
      when(io.to_tempmem_inst.fire) { state := s_l2cache_tag }
    }
    is(s_l2cache_tag) {
      when(io.to_tempmem_tag.fire) {
        when(io.tag_reuse_hit) {
          when(complete_address) { state := s_idle }
            .otherwise { state := s_l2cache_tag }
        }.otherwise {
          state := s_tlb_req
        }
      }
    }
    is(s_tlb_req) {
      when(io.to_l2TLB.fire) { state := s_tlb_rsp }
    }
    is(s_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        p_addr_reg := io.from_l2TLB.bits.paddr
        state := s_l2cache
      }
    }
    is(s_l2cache) {
      when(io.to_l2cache.fire) {
        when(complete_address) { state := s_idle }
        .otherwise { state := s_l2cache_tag }
      }
    }
  }

  // FSM operations
  switch(state) {
    is(s_idle) {
      when(io.from_fifo.fire) {
        reg_save.in1 := io.from_fifo.bits.in1
        reg_save.in2 := io.from_fifo.bits.in2
        reg_save.in3 := io.from_fifo.bits.in3
        desc_ptr_reg := io.from_fifo.bits.in1(0)
        dyn_words_reg := io.from_fifo.bits.in2
        meta_vaddr_reg := io.from_fifo.bits.in1(0)
        when(io.from_fifo.bits.ctrl.funct === 2.U && incomingDescCacheHit) {
          desc_words_reg := incomingDescCacheWords
        }
        // Descriptor tensor starts from the descriptor pointer. Tensor setup
        // fills the real box base after descriptor fetch and VRS2 coords decode.
        reg_save.address := alignToL2Line(io.from_fifo.bits.in1(0))
        reg_save.ctrl := io.from_fifo.bits.ctrl
      }
    }
    is(s_desc_rsp) {
      when(descRspFire) {
        desc_words_reg := io.from_l2cache_meta.bits.d_data
      }
    }
    is(s_tensor_addr_init) {
      val cInSlice = Mux(
        tvars.interleaveMode === 1.U,
        desc_coords(0)(1, 0).pad(xLen),
        desc_coords(0)(2, 0).pad(xLen)
      )
      val linearStart = desc_words_reg(2) + TmaPow2Math.scaleByDataWidth(desc_coords(0), tvars.datawidth)
      val interleaveStart = desc_words_reg(2) + ((cInSlice << log2Ceil(dma_aligned_bulk))(xLen - 1, 0))
      desc_addr_accum_reg := Mux(tvars.interleaveMode === 0.U, linearStart, interleaveStart)
      desc_addr_slice_stride_reg := Mux(tvars.interleaveMode === 1.U, 16.U(xLen.W), 32.U(xLen.W))
      desc_addr_op_reg := Mux(tvars.interleaveMode === 0.U, descAddrOpDim1, descAddrOpPitch)
      tensor_setup_mul_op_reg := setupMulOpShared2
    }
    is(s_addr_mul_wait) {
      when(tmaAddrSetupMul.io.out.fire) {
        val product = tmaAddrSetupMul.io.out.bits
        switch(desc_addr_op_reg) {
          is(descAddrOpPitch) {
            when(tvars.interleaveMode =/= 0.U && tvars.tensorRank >= 3.U) {
              desc_addr_slice_stride_reg := product
            }
            desc_addr_op_reg := descAddrOpSlice
          }
          is(descAddrOpSlice) {
            when(tvars.interleaveMode =/= 0.U) {
              desc_addr_accum_reg := desc_addr_accum_reg + product
            }
            desc_addr_op_reg := descAddrOpDim1
          }
          is(descAddrOpDim1) {
            desc_addr_accum_reg := desc_addr_accum_reg + product
            desc_addr_op_reg := descAddrOpDim2
          }
          is(descAddrOpDim2) {
            desc_addr_accum_reg := desc_addr_accum_reg + product
            desc_addr_op_reg := descAddrOpDim3
          }
          is(descAddrOpDim3) {
            desc_addr_accum_reg := desc_addr_accum_reg + product
            desc_addr_op_reg := descAddrOpDim4
          }
          is(descAddrOpDim4) {
            desc_box_address_reg := desc_addr_accum_reg + product
          }
        }
      }
    }
    is(s_tensor_setup) {
      when(!reset.asBool && tensor_copy_mode) {
        assert(tvars.interleaveMode =/= 3.U,
          "DMA TMA interleaveMode=3 is reserved")
        when(tvars.interleaveMode =/= 0.U) {
          assert(tvars.tensorRank >= 3.U,
            "DMA TMA interleave requires rank >= 3")
          assert(tvars.datawidth === dma_aligned_bulk.U,
            "DMA TMA interleave v0 supports only 4-byte elements")
        }
        (0 until 5).foreach { d =>
          assert(TmaPow2Math.isSupportedStride(tvars.elementStrides(d)),
            "DMA TMA v0 supports only power-of-two element strides up to 32")
        }
        when(tvars.elementStrides(0) > 1.U) {
          assert(tvars.datawidth === dma_aligned_bulk.U,
            "DMA TMA dim0 elementStride v0 supports only 4-byte elements")
        }
        assert(desc_ptr_reg(log2Ceil(l2cacheline) - 1, 0) === 0.U,
          "DMA descriptor TMA requires 128B-aligned descriptor in v0")
      }
      val setupHasZero = (0 until 5).map { d =>
        (tvars.tensorRank > d.U) && (box_elements_num_setup(d) === 0.U)
      }.reduce(_ || _)
      val tensor_shared_stride1_setup = TmaPow2Math.scaleByDataWidth(tvars.boxDim(0), tvars.datawidth)
      (0 until 5).foreach { x =>
        tensor_dim_pos_reg(x) := 0.U
        tensor_base_global_pos_reg(x) := box_offset_elems_setup(x)
        tensor_global_pos_reg(x) := box_offset_elems_setup(x)
        tensor_dim_offset_reg(x) := 0.U
        tensor_shared_offset_reg(x) := 0.U
        tensor_dim_stride_bytes_reg(x) := tensor_dim_stride_bytes_setup(x)
        tensor_shared_stride_bytes_reg(x) := 0.U
        tensor_box_dim_reg(x) := tvars.boxDim(x)
        tensor_element_stride_reg(x) := tvars.elementStrides(x)
        tensor_global_dim_reg(x) := tvars.globalDim(x)
        tensor_dim_can_advance_limit_reg(x) := Mux(
          tvars.boxDim(x) > tvars.elementStrides(x),
          tvars.boxDim(x) - tvars.elementStrides(x),
          0.U
        )
        tensor_setup_box_elements_reg(x) := box_elements_num_setup(x)
      }
      tensor_rank_reg := tvars.tensorRank
      tensor_interleave_mode_reg := tensor_interleave_mode
      tensor_interleave_sel_reg := tvars.interleaveMode
      tensor_shared_stride_bytes_reg(0) := tvars.datawidth
      tensor_shared_stride_bytes_reg(1) := tensor_shared_stride1_setup
      tensor_setup_mul_op_reg := setupMulOpShared2
      tensor_copy_size_reg := Mux(setupHasZero, 0.U, tensor_shared_stride1_setup)
      tensor_row_base_reg := desc_box_address_reg
      reg_save.address := alignToL2Line(desc_box_address_reg)
      tensor_shared_row_base_reg := reg_save.in3(0)
      tensor_dim0_offset_bytes_reg := Mux(tensor_interleave_mode, 0.U, box_dim0_offset_bytes_setup)
      tensor_row_span_bytes_reg := Mux(
        tensor_interleave_mode,
        tvars.datawidth,
        TmaPow2Math.rowSpanBytes(tvars.boxDim(0), tvars.elementStrides(0), tvars.datawidth)
      )
      tensor_swizzle_row_low_reg := 0.U
    }
    is(s_setup_mul_wait) {
      when(tmaAddrSetupMul.io.out.fire) {
        val product = tmaAddrSetupMul.io.out.bits
        switch(tensor_setup_mul_op_reg) {
          is(setupMulOpShared2) {
            tensor_shared_stride_bytes_reg(2) := product
            when(tensor_rank_reg <= 2.U) {
              tensor_copy_size_reg := product
            }.otherwise {
              tensor_setup_mul_op_reg := setupMulOpShared3
            }
          }
          is(setupMulOpShared3) {
            tensor_shared_stride_bytes_reg(3) := product
            when(tensor_rank_reg <= 3.U) {
              tensor_copy_size_reg := product
            }.otherwise {
              tensor_setup_mul_op_reg := setupMulOpShared4
            }
          }
          is(setupMulOpShared4) {
            tensor_shared_stride_bytes_reg(4) := product
            when(tensor_rank_reg <= 4.U) {
              tensor_copy_size_reg := product
            }.otherwise {
              tensor_setup_mul_op_reg := setupMulOpCopy5
            }
          }
          is(setupMulOpCopy5) {
            tensor_copy_size_reg := product
          }
        }
      }
    }
    is(s_save) {
      when(io.to_tempmem_inst.fire) {
        inst_mem_index_reg := io.inst_mem_index
      }
    }
    is(s_l2cache_tag) {
      when(io.to_tempmem_tag.fire) {
        tag_mem_index_reg := io.tag_mem_index
        when(io.tag_reuse_hit) {
          when(!complete_address) {
            reg_save.address := address_next
            when(tensor_copy_mode && !tensor_row_has_next_line) {
              (0 until 5).foreach { x =>
                tensor_dim_pos_reg(x) := tensor_dim_pos_next(x)
                tensor_global_pos_reg(x) := tensor_global_pos_next(x)
                tensor_dim_offset_reg(x) := tensor_dim_offset_next(x)
                tensor_shared_offset_reg(x) := tensor_shared_offset_next(x)
              }
              when(tensor_has_next_row) {
                tensor_row_base_reg := tensor_next_row_base
                tensor_shared_row_base_reg := tensor_next_shared_row_base
                when(!tensor_interleave_active || !tensor_advance_dim_oh(0)) {
                  tensor_swizzle_row_low_reg := tensor_swizzle_row_low_reg + 1.U
                }
              }
            }
          }
          when(complete_address) {
            (0 until 5).foreach { x =>
              tensor_dim_pos_reg(x) := 0.U
              tensor_base_global_pos_reg(x) := 0.U
              tensor_global_pos_reg(x) := 0.U
              tensor_dim_offset_reg(x) := 0.U
              tensor_shared_offset_reg(x) := 0.U
            }
            tensor_row_base_reg := 0.U
            tensor_shared_row_base_reg := 0.U
            tensor_dim0_offset_bytes_reg := 0.U
            tensor_row_span_bytes_reg := 0.U
            tensor_copy_size_reg := 0.U
            tensor_swizzle_row_low_reg := 0.U
            tensor_rank_reg := 0.U
            tensor_interleave_mode_reg := false.B
            tensor_interleave_sel_reg := 0.U
            (0 until 5).foreach { x =>
              tensor_box_dim_reg(x) := 0.U
              tensor_element_stride_reg(x) := 1.U
              tensor_global_dim_reg(x) := 0.U
              tensor_dim_can_advance_limit_reg(x) := 0.U
            }
            desc_addr_slice_stride_reg := 0.U
          }
        }
      }
    }
    is(s_l2cache) {
      when(io.to_l2cache.fire) {
        when(!complete_address) {
          reg_save.address := address_next
          when(tensor_copy_mode && !tensor_row_has_next_line) {
            (0 until 5).foreach { x =>
              tensor_dim_pos_reg(x) := tensor_dim_pos_next(x)
              tensor_global_pos_reg(x) := tensor_global_pos_next(x)
              tensor_dim_offset_reg(x) := tensor_dim_offset_next(x)
              tensor_shared_offset_reg(x) := tensor_shared_offset_next(x)
            }
            when(tensor_has_next_row) {
              tensor_row_base_reg := tensor_next_row_base
              tensor_shared_row_base_reg := tensor_next_shared_row_base
              when(!tensor_interleave_active || !tensor_advance_dim_oh(0)) {
                tensor_swizzle_row_low_reg := tensor_swizzle_row_low_reg + 1.U
              }
            }
          }
        }
        // B2 fix: clear tensor iterators on completion to avoid polluting next instruction.
        when(complete_address) {
          (0 until 5).foreach { x =>
            tensor_dim_pos_reg(x) := 0.U
            tensor_base_global_pos_reg(x) := 0.U
            tensor_global_pos_reg(x) := 0.U
            tensor_dim_offset_reg(x) := 0.U
            tensor_shared_offset_reg(x) := 0.U
          }
          tensor_row_base_reg := 0.U
          tensor_shared_row_base_reg := 0.U
          tensor_dim0_offset_bytes_reg := 0.U
          tensor_row_span_bytes_reg := 0.U
          tensor_copy_size_reg := 0.U
          tensor_swizzle_row_low_reg := 0.U
          tensor_rank_reg := 0.U
          tensor_interleave_mode_reg := false.B
          tensor_interleave_sel_reg := 0.U
          (0 until 5).foreach { x =>
            tensor_box_dim_reg(x) := 0.U
            tensor_element_stride_reg(x) := 1.U
            tensor_global_dim_reg(x) := 0.U
            tensor_dim_can_advance_limit_reg(x) := 0.U
          }
          desc_addr_slice_stride_reg := 0.U
        }
      }
    }
  }

}

// ============================================================
// Temp_mem: buffer L2 responses, manage inst/tag/data entries
// ============================================================
class Temp_mem(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val from_addr = Flipped(DecoupledIO(new vExeDataDMA))
    val from_addr_tag = Flipped(DecoupledIO(new DmaCachelineInfo))
    val inst_mem_index = Output(UInt(log2Ceil(max_dma_inst).W))
    val tag_mem_index = Output(UInt(log2Ceil(max_dma_tag).W))
    val from_l2cache = Flipped(DecoupledIO(new DCacheMemRsp))
    val from_shared = Flipped(DecoupledIO(new DCacheCoreRsp_np))
    val from_addr_tag_reuse = Output(Bool())
    val to_shared = DecoupledIO(new DmaTempOutput)
    val inst_complete = DecoupledIO(UInt(32.W))
  })

  val datamem = Mem(max_l2cacheline, new DmaL2CachelineInfo)
  val instmem = Mem(max_dma_inst, new vExeDataDMA)
  val tagmem = Mem(max_dma_tag, new DmaCachelineInfo)

  val from_l2cache_all = Wire(new DmaL2CachelineInfo)
  from_l2cache_all.base := io.from_l2cache.bits
  val tagmem_read_entry = tagmem.read(
    io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag))
  )
  from_l2cache_all.cacheline_info.tag := tagmem_read_entry.tag
  from_l2cache_all.cacheline_info.inst_index := tagmem_read_entry.inst_index
  from_l2cache_all.cacheline_info.tensor_dim0_start := tagmem_read_entry.tensor_dim0_start
  from_l2cache_all.cacheline_info.box_dim0_start := tagmem_read_entry.box_dim0_start
  from_l2cache_all.cacheline_info.tensor_high_dim_valid := tagmem_read_entry.tensor_high_dim_valid
  from_l2cache_all.cacheline_info.shared_row_base := tagmem_read_entry.shared_row_base
  from_l2cache_all.cacheline_info.swizzle_row_low := tagmem_read_entry.swizzle_row_low
  from_l2cache_all.cacheline_info.tensor_copy := tagmem_read_entry.tensor_copy
  from_l2cache_all.cacheline_info.tensor_interleave := tagmem_read_entry.tensor_interleave
  from_l2cache_all.cacheline_info.tensor_elem_valid := tagmem_read_entry.tensor_elem_valid
  from_l2cache_all.cacheline_info.tensor_line_oob := tagmem_read_entry.tensor_line_oob
  from_l2cache_all.cacheline_info.tensor_elem_addr := tagmem_read_entry.tensor_elem_addr
  from_l2cache_all.cacheline_info.shared_elem_addr := tagmem_read_entry.shared_elem_addr

  val finish_cnt = RegInit(VecInit(Seq.fill(max_dma_inst)(1.U(xLen.W))))
  val used_inst = RegInit(0.U(max_dma_inst.W))
  val used_tag = RegInit(0.U(max_dma_tag.W))
  val used_cache = RegInit(0.U(max_l2cacheline.W))
  val complete = Wire(Vec(max_dma_inst, Bool()))
  (0 until max_dma_inst).foreach(x => complete(x) := finish_cnt(x) === 0.U)

  val entry_index_reg = RegInit(VecInit(Seq.fill(max_l2cacheline)(0.U(log2Ceil(max_dma_inst).W))))
  val valid_inst_entry = Mux(used_inst.andR, 0.U, PriorityEncoder(~used_inst))
  val valid_data_entry = Mux(used_cache.andR, 0.U, PriorityEncoder(~used_cache))
  val valid_tag_entry = Mux(used_tag.andR, 0.U, PriorityEncoder(~used_tag))
  val complete_inst_entry = Mux(complete.asUInt.orR, PriorityEncoder(complete), 0.U)

  val current_inst_entry_index = io.from_l2cache.bits.d_source(
    l1cache_sourceBits - 1 - log2Ceil(max_dma_tag),
    l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  )
  val current_inst_entry_index_reg = RegInit(0.U(log2Ceil(max_dma_inst).W))

  val tensorReuseEntries = 4
  val tensorReuseValid = RegInit(VecInit(Seq.fill(tensorReuseEntries)(false.B)))
  val tensorReuseTag = RegInit(VecInit(Seq.fill(tensorReuseEntries)(0.U(xLen.W))))
  val tensorReuseRsp = Reg(Vec(tensorReuseEntries, Vec(dcache_BlockWords, UInt(xLen.W))))
  val tensorReuseReplace = RegInit(0.U(log2Ceil(tensorReuseEntries).W))
  val tensorPendingValid = RegInit(VecInit(Seq.fill(max_dma_tag)(false.B)))
  val tensorPendingTag = RegInit(VecInit(Seq.fill(max_dma_tag)(0.U(xLen.W))))

  // Mask for slicing L2 cacheline into shared-mem groups
  val mask_l2cache = RegInit(VecInit(Seq.fill(numgroupl2cache)(false.B)))
  val mask_index =
    if (numgroupshared == numgroupl2cache) 0.U(log2Ceil(numgroupl2cache).W)
    else PriorityEncoder(mask_l2cache)
  val mask_l2cache_next = Wire(Vec(numgroupl2cache, Bool()))
  if (numgroupshared == numgroupl2cache) {
    mask_l2cache_next := VecInit(Seq.fill(numgroupl2cache)(false.B))
  } else {
    (0 until numgroupl2cache).foreach(x =>
      mask_l2cache_next(x) := mask_l2cache(x) && !(x.U >= mask_index && x.U < mask_index + numgroupshared.U)
    )
  }
  val mask_shared = Wire(Vec(numgroupshared, Bool()))
  if (numgroupshared == numgroupl2cache) {
    (0 until numgroupshared).foreach(x => mask_shared(x) := mask_l2cache(x))
  } else {
    (0 until numgroupshared).foreach(x =>
      when(x.U + mask_index < numgroupl2cache.U) { mask_shared(x) := mask_l2cache(x.U + mask_index) }
      .otherwise { mask_shared(x) := false.B }
    )
  }

  val output_inst = Reg(new vExeDataDMA)
  val output_data = Reg(new DmaL2CachelineInfo)
  val output_data_entry = Reg(UInt(log2Ceil(max_l2cacheline).W))
  val raw_data_4byte = Wire(Vec(numgroupl2cache, UInt((dma_aligned_bulk * BitsOfByte).W)))
  (0 until numgroupl2cache).foreach(x =>
    raw_data_4byte(x) := output_data.base.d_data.asUInt((x + 1) * dma_aligned_bulk * BitsOfByte - 1, x * dma_aligned_bulk * BitsOfByte)
  )
  val output_data_4byte = Reg(Vec(numgroupl2cache, UInt((dma_aligned_bulk * BitsOfByte).W)))

  val tag_wire = output_data.cacheline_info.tag

  // OOB fill is generated per 4-byte shared lane. This avoids four full-width
  // temporary cacheline images and removes the datawidth mux from the data RAM
  // writeback path.
  import DataType._
  val t_box_dim0_start = output_data.cacheline_info.box_dim0_start
  val t_tensor_dim0_start = output_data.cacheline_info.tensor_dim0_start
  val t_tensor_high_dim_valid = output_data.cacheline_info.tensor_high_dim_valid
  val t_boxDim0       = output_inst.tensorvars.boxDim(0)
  val t_globalDim0    = output_inst.tensorvars.globalDim(0)
  val t_elementStride0 = output_inst.tensorvars.elementStrides(0)
  val t_datawidth     = output_inst.tensorvars.datawidth
  val t_dataType      = output_inst.tensorvars.dataType
  val t_oobfill       = output_inst.tensorvars.oobfill
  val t_dim0_row_span_bytes = TmaPow2Math.rowSpanBytes(t_boxDim0, t_elementStride0, t_datawidth)
  val t_dim0_end      = t_box_dim0_start + t_dim0_row_span_bytes
  val t_tensor_dim0_end = t_tensor_dim0_start + TmaPow2Math.scaleByDataWidth(t_globalDim0, t_datawidth)
  val t_supports_dim0_gather = t_datawidth === dma_aligned_bulk.U && t_elementStride0 > 1.U

  def dim0GatherSelectedByStride(diffBytes: UInt, strideElems: UInt): Bool = {
    MuxLookup(strideElems, true.B)(Seq(
      2.U  -> !diffBytes(2),
      4.U  -> !diffBytes(3, 2).orR,
      8.U  -> !diffBytes(4, 2).orR,
      16.U -> !diffBytes(5, 2).orR,
      32.U -> !diffBytes(6, 2).orR
    ))
  }

  val t_group_dim0_selected = Wire(Vec(numgroupl2cache, Bool()))
  (0 until numgroupl2cache).foreach { g =>
    val groupAddr = tag_wire + (g * dma_aligned_bulk).U
    val diffBytes = groupAddr - t_box_dim0_start
    t_group_dim0_selected(g) := !t_supports_dim0_gather ||
      dim0GatherSelectedByStride(diffBytes, t_elementStride0)
  }

  def tensorDim0ElemValid(elemAddr: UInt, group: Int): Bool = {
    val in_row_span = (elemAddr >= t_box_dim0_start) && (elemAddr < t_dim0_end)
    val in_tensor_row = (elemAddr >= t_tensor_dim0_start) && (elemAddr < t_tensor_dim0_end)
    in_row_span && in_tensor_row && t_tensor_high_dim_valid && t_group_dim0_selected(group)
  }

  val tensor_data_next_4byte = Wire(Vec(numgroupl2cache, UInt((dma_aligned_bulk * BitsOfByte).W)))
  val fill8 = 0.U(8.W)
  val fill16 = Mux(t_dataType === UINT16, 0.U(16.W),
    Mux(t_oobfill.asBool, Fill(16, 1.U), 0.U(16.W)))
  val is32Int = (t_dataType === UINT32) || (t_dataType === INT32)
  val fill32 = Mux(is32Int, 0.U(32.W),
    Mux(t_oobfill.asBool, Fill(32, 1.U), 0.U(32.W)))
  val is64Int = (t_dataType === UINT64) || (t_dataType === INT64)
  val fill64Word = Mux(is64Int, 0.U(32.W),
    Mux(t_oobfill.asBool, Fill(32, 1.U), 0.U(32.W)))
  (0 until numgroupl2cache).foreach { g =>
    val groupBase = tag_wire + (g * dma_aligned_bulk).U
    val raw = raw_data_4byte(g)
    val byte0 = Mux(tensorDim0ElemValid(groupBase, g), raw(7, 0), fill8)
    val byte1 = Mux(tensorDim0ElemValid(groupBase + 1.U, g), raw(15, 8), fill8)
    val byte2 = Mux(tensorDim0ElemValid(groupBase + 2.U, g), raw(23, 16), fill8)
    val byte3 = Mux(tensorDim0ElemValid(groupBase + 3.U, g), raw(31, 24), fill8)
    val half0 = Mux(tensorDim0ElemValid(groupBase, g), raw(15, 0), fill16)
    val half1 = Mux(tensorDim0ElemValid(groupBase + 2.U, g), raw(31, 16), fill16)
    val wordValid = tensorDim0ElemValid(groupBase, g)
    val elem8Base = tag_wire + ((g / 2) * 8).U
    val doubleWordValid = tensorDim0ElemValid(elem8Base, g)
    tensor_data_next_4byte(g) := MuxLookup(t_datawidth, raw)(Seq(
      1.U -> Cat(byte3, byte2, byte1, byte0),
      2.U -> Cat(half1, half0),
      4.U -> Mux(wordValid, raw, fill32),
      8.U -> Mux(doubleWordValid, raw, fill64Word)
    ))
  }

  val output_data_next_4byte = Wire(Vec(numgroupl2cache, UInt((dma_aligned_bulk * BitsOfByte).W)))
  output_data_next_4byte := raw_data_4byte
  when(output_inst.funct === 2.U) {
    when(output_data.cacheline_info.tensor_interleave) {
      val elemAddr = output_data.cacheline_info.tensor_elem_addr
      (0 until numgroupl2cache).foreach { g =>
        val hitElem = (tag_wire + (g * dma_aligned_bulk).U) === elemAddr
        output_data_next_4byte(g) := Mux(
          hitElem && !output_data.cacheline_info.tensor_elem_valid,
          fill32,
          raw_data_4byte(g)
        )
      }
    }.otherwise {
      output_data_next_4byte := tensor_data_next_4byte
    }
  }


  val s_idle :: s_getdata :: s_shared :: s_shared1 :: s_reset :: Nil = Enum(5)
  val state = RegInit(s_idle)

  val nonInterleaveTensorTag = io.from_addr_tag.bits.tensor_copy &&
    !io.from_addr_tag.bits.tensor_interleave
  val tensorReuseHitVec = VecInit((0 until tensorReuseEntries).map { i =>
    nonInterleaveTensorTag &&
      tensorReuseValid(i) &&
      tensorReuseTag(i) === io.from_addr_tag.bits.tag
  })
  val tensorReuseHit = tensorReuseHitVec.asUInt.orR
  val tensorReuseHitIdx = PriorityEncoder(tensorReuseHitVec.asUInt)
  val tensorPendingHitVec = VecInit((0 until max_dma_tag).map { i =>
    tensorPendingValid(i) &&
      tensorPendingTag(i) === io.from_addr_tag.bits.tag
  })
  val tensorPendingHit = nonInterleaveTensorTag &&
    tensorPendingHitVec.asUInt.orR
  val tensorReuseReady = tensorReuseHit && !used_cache.andR && !io.from_l2cache.fire &&
    state =/= s_shared1 && state =/= s_reset
  val tensorFillTag = io.from_addr_tag.bits.tensor_copy && io.from_addr_tag.bits.tensor_line_oob
  val tensorFillReady = tensorFillTag && !used_cache.andR && !io.from_l2cache.fire &&
    state =/= s_shared1 && state =/= s_reset
  val tensorReuseFire = io.from_addr_tag.fire && tensorReuseHit
  val tensorFillFire = io.from_addr_tag.fire && tensorFillTag
  val tensorBypassDataFire = tensorReuseFire || tensorFillFire
  val normalTagFire = io.from_addr_tag.fire && !tensorReuseHit && !tensorFillTag
  val tensorReplayLine = Wire(new DmaL2CachelineInfo)
  tensorReplayLine.base.d_opcode := 0.U
  tensorReplayLine.base.d_param := 0.U
  tensorReplayLine.base.d_source := 0.U
  tensorReplayLine.base.d_addr := 0.U
  tensorReplayLine.base.d_data := tensorReuseRsp(tensorReuseHitIdx)
  tensorReplayLine.cacheline_info := io.from_addr_tag.bits

  val tensorFillLine = Wire(new DmaL2CachelineInfo)
  tensorFillLine.base.d_opcode := 0.U
  tensorFillLine.base.d_param := 0.U
  tensorFillLine.base.d_source := 0.U
  tensorFillLine.base.d_addr := 0.U
  tensorFillLine.base.d_data := VecInit(Seq.fill(dcache_BlockWords)(0.U(xLen.W)))
  tensorFillLine.cacheline_info := io.from_addr_tag.bits

  val dataWriteValid = io.from_l2cache.fire || tensorReuseFire || tensorFillFire
  val dataWriteBits = Mux(tensorFillFire, tensorFillLine,
    Mux(tensorReuseFire, tensorReplayLine, from_l2cache_all))
  val tagWriteValid = normalTagFire

  io.from_l2cache.ready := !used_cache.andR && (state === s_idle || state === s_getdata)
  io.from_shared.ready := !io.from_addr.fire && state =/= s_reset
  io.from_addr.ready := state === s_idle && !used_inst.andR
  // Non-interleave TMA rows often share one L2 cacheline. Reuse returned lines,
  // stall only if the same line is still pending, and otherwise allow different
  // lines to stay outstanding. Interleave remains serialized because it issues
  // many element-sized requests to the same line.
  val interleaveTagSerialBusy = (state =/= s_idle) || used_tag.orR || used_cache.orR
  val normalTagReady = !used_tag.andR && !io.from_l2cache.fire && !tensorPendingHit &&
    !(io.from_addr_tag.bits.tensor_interleave && interleaveTagSerialBusy)
  io.from_addr_tag.ready := Mux(tensorFillTag, tensorFillReady,
    Mux(tensorReuseHit, tensorReuseReady, normalTagReady))
  io.from_addr_tag_reuse := tensorReuseHit || tensorFillTag

  // Design invariant assertions: entry allocation must never fire when slots are full
  when(!reset.asBool) {
    assert(!(io.from_addr.fire && used_inst.andR),
      "DMA Temp_mem: inst entry allocated when all slots full")
    assert(!(normalTagFire && used_tag.andR),
      "DMA Temp_mem: tag entry allocated when all tag slots full")
    assert(!(io.from_l2cache.fire && used_cache.andR),
      "DMA Temp_mem: data entry allocated when all cache slots full")
    assert(!(io.from_l2cache.fire && tensorBypassDataFire),
      "DMA Temp_mem: L2 response and tensor bypass both attempted a data write")
  }
  io.to_shared.valid := state === s_shared
  io.inst_complete.valid := state === s_reset
  io.inst_mem_index := Mux(io.from_addr.fire, valid_inst_entry, 0.U)
  io.tag_mem_index := Mux(io.from_addr_tag.fire, valid_tag_entry, 0.U)

  // finish_cnt update on inst arrival
  when(io.from_addr.fire) {
    when(!(io.from_addr.bits.funct === 2.U &&
        io.from_addr.bits.tensorvars.interleaveMode === 0.U)) {
      tensorReuseValid := VecInit(Seq.fill(tensorReuseEntries)(false.B))
    }
    // BULK copysize must be aligned to dma_aligned_bulk (4 bytes)
    when(!reset.asBool && io.from_addr.bits.funct === 1.U) {
      assert(io.from_addr.bits.copysize(log2Ceil(dma_aligned_bulk) - 1, 0) === 0.U,
        "DMA BULK copysize must be aligned to dma_aligned_bulk bytes")
    }
    switch(io.from_addr.bits.funct) {
      is(0.U) {
        val group_start = io.from_addr.bits.src(xLen - 1, 2)
        val group_end = (io.from_addr.bits.src + io.from_addr.bits.copysize)(xLen - 1, 2)
        when((io.from_addr.bits.src + io.from_addr.bits.copysize)(1, 0) === 0.U) {
          finish_cnt(valid_inst_entry) := group_end - group_start
        }.otherwise {
          finish_cnt(valid_inst_entry) := 1.U + (group_end - group_start)
        }
      }
      is(1.U) {
        finish_cnt(valid_inst_entry) := io.from_addr.bits.copysize >> log2Ceil(dma_aligned_bulk)
      }
      is(2.U) {
        finish_cnt(valid_inst_entry) :=
          (io.from_addr.bits.copysize + (dma_aligned_bulk - 1).U) >> log2Ceil(dma_aligned_bulk)
      }
    }
  }
  when(io.from_shared.fire) {
    val rsp_instr_idx = io.from_shared.bits.instrId
    val rsp_lane_count = PopCount(io.from_shared.bits.activeMask)
    finish_cnt(rsp_instr_idx) := finish_cnt(rsp_instr_idx) - rsp_lane_count
    when(!reset.asBool) {
      assert(finish_cnt(rsp_instr_idx) >= rsp_lane_count,
        "DMA finish_cnt underflow: decrement exceeds remaining count")
    }
  }

  // State machine
  switch(state) {
    is(s_idle) {
      when(io.to_shared.ready && used_cache.orR && !io.from_l2cache.fire) { state := s_getdata }
      .elsewhen(complete.asUInt.orR) { state := s_reset }
    }
    is(s_getdata) { state := s_shared }
    is(s_shared) {
      when(io.to_shared.fire) { state := s_shared1 }
    }
    is(s_shared1) {
      when(mask_l2cache.asUInt === 0.U) { state := s_idle }
      .otherwise { state := s_shared }
    }
    is(s_reset) {
      when(io.inst_complete.fire) { state := s_idle }
    }
  }

  // State operations
  switch(state) {
    is(s_idle) {
      mask_l2cache := VecInit(Seq.fill(numgroupl2cache)(false.B))
      when(io.from_addr.fire) {
        used_inst := used_inst.bitSet(valid_inst_entry, true.B)
        instmem.write(valid_inst_entry, io.from_addr.bits)
      }
      when(io.from_l2cache.fire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        used_tag := used_tag.bitSet(io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag)), false.B)
        entry_index_reg(valid_data_entry) := current_inst_entry_index
        when(tagmem_read_entry.tensor_copy && !tagmem_read_entry.tensor_interleave) {
          val reuseHitVec = VecInit((0 until tensorReuseEntries).map { i =>
            tensorReuseValid(i) && tensorReuseTag(i) === tagmem_read_entry.tag
          })
          val reuseInvalidVec = VecInit((0 until tensorReuseEntries).map(i => !tensorReuseValid(i)))
          val reuseWriteIdx = Mux(
            reuseHitVec.asUInt.orR,
            PriorityEncoder(reuseHitVec.asUInt),
            Mux(reuseInvalidVec.asUInt.orR, PriorityEncoder(reuseInvalidVec.asUInt), tensorReuseReplace)
          )
          tensorReuseValid(reuseWriteIdx) := true.B
          tensorReuseTag(reuseWriteIdx) := tagmem_read_entry.tag
          tensorReuseRsp(reuseWriteIdx) := io.from_l2cache.bits.d_data
          when(!reuseHitVec.asUInt.orR && !reuseInvalidVec.asUInt.orR) {
            tensorReuseReplace := Mux(
              tensorReuseReplace === (tensorReuseEntries - 1).U,
              0.U,
              tensorReuseReplace + 1.U
            )
          }
          tensorPendingValid(io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag))) := false.B
        }
      }
      when(tensorBypassDataFire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        entry_index_reg(valid_data_entry) := io.from_addr_tag.bits.inst_index
      }.elsewhen(normalTagFire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        when(nonInterleaveTensorTag) {
          tensorPendingValid(valid_tag_entry) := true.B
          tensorPendingTag(valid_tag_entry) := io.from_addr_tag.bits.tag
        }
      }
      when(io.to_shared.ready && used_cache.orR && !io.from_l2cache.fire) {
        output_data := datamem.read(PriorityEncoder(used_cache))
        output_inst := instmem.read(entry_index_reg(PriorityEncoder(used_cache)))
        current_inst_entry_index_reg := entry_index_reg(PriorityEncoder(used_cache))
        output_data_entry := PriorityEncoder(used_cache)
      }
      when(complete.asUInt.orR) {
        output_inst := instmem(PriorityEncoder(complete.asUInt))
      }
    }
    is(s_getdata) {
      // Mark valid groups in the cacheline
      when(output_inst.funct === 2.U) {
        when(output_data.cacheline_info.tensor_interleave) {
          val elemAddr = output_data.cacheline_info.tensor_elem_addr
          (0 until numgroupl2cache).foreach { x =>
            val addr_start = tag_wire + (x.U * dma_aligned_bulk.U)
            mask_l2cache(x) := addr_start === elemAddr
          }
          output_data_4byte := output_data_next_4byte
        }.otherwise {
          // Tensor mode: mask based on dim0 gather selection within the current row.
          val box_start = output_data.cacheline_info.box_dim0_start
          val datawidth = output_inst.tensorvars.datawidth
          val elementStride0 = output_inst.tensorvars.elementStrides(0)
          val dim0_row_span_bytes = TmaPow2Math.rowSpanBytes(
            output_inst.tensorvars.boxDim(0),
            elementStride0,
            datawidth
          )
          val box_end = box_start + dim0_row_span_bytes
          (0 until numgroupl2cache).foreach { x =>
            val addr_start = tag_wire + (x.U * dma_aligned_bulk.U)
            val in_row_span = (addr_start >= box_start) && (addr_start < box_end)
            val is_selected = Mux(
              elementStride0 <= 1.U,
              true.B,
              t_group_dim0_selected(x),
            )
            mask_l2cache(x) := in_row_span && is_selected
          }
          output_data_4byte := output_data_next_4byte
        }
      }.otherwise {
        // Linear copy: mark valid groups by [src, src+copysize)
        (0 until numgroupl2cache).foreach { x =>
          val addr_start = tag_wire + (x.U * dma_aligned_bulk.U)
          val condition = (addr_start >= output_inst.src) && (addr_start < (output_inst.src + output_inst.copysize))
          mask_l2cache(x) := condition
        }
        output_data_4byte := output_data_next_4byte
      }
      when(io.from_l2cache.fire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        used_tag := used_tag.bitSet(io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag)), false.B)
        entry_index_reg(valid_data_entry) := current_inst_entry_index
        when(tagmem_read_entry.tensor_copy && !tagmem_read_entry.tensor_interleave) {
          val reuseHitVec = VecInit((0 until tensorReuseEntries).map { i =>
            tensorReuseValid(i) && tensorReuseTag(i) === tagmem_read_entry.tag
          })
          val reuseInvalidVec = VecInit((0 until tensorReuseEntries).map(i => !tensorReuseValid(i)))
          val reuseWriteIdx = Mux(
            reuseHitVec.asUInt.orR,
            PriorityEncoder(reuseHitVec.asUInt),
            Mux(reuseInvalidVec.asUInt.orR, PriorityEncoder(reuseInvalidVec.asUInt), tensorReuseReplace)
          )
          tensorReuseValid(reuseWriteIdx) := true.B
          tensorReuseTag(reuseWriteIdx) := tagmem_read_entry.tag
          tensorReuseRsp(reuseWriteIdx) := io.from_l2cache.bits.d_data
          when(!reuseHitVec.asUInt.orR && !reuseInvalidVec.asUInt.orR) {
            tensorReuseReplace := Mux(
              tensorReuseReplace === (tensorReuseEntries - 1).U,
              0.U,
              tensorReuseReplace + 1.U
            )
          }
          tensorPendingValid(io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag))) := false.B
        }
      }
      when(tensorBypassDataFire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        entry_index_reg(valid_data_entry) := io.from_addr_tag.bits.inst_index
      }.elsewhen(normalTagFire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        when(nonInterleaveTensorTag) {
          tensorPendingValid(valid_tag_entry) := true.B
          tensorPendingTag(valid_tag_entry) := io.from_addr_tag.bits.tag
        }
      }
    }
    is(s_shared) {
      when(io.to_shared.fire) {
        mask_l2cache := mask_l2cache_next
      }
      when(tensorBypassDataFire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        entry_index_reg(valid_data_entry) := io.from_addr_tag.bits.inst_index
      }.elsewhen(normalTagFire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        when(nonInterleaveTensorTag) {
          tensorPendingValid(valid_tag_entry) := true.B
          tensorPendingTag(valid_tag_entry) := io.from_addr_tag.bits.tag
        }
      }
    }
    is(s_shared1) {
      when(mask_l2cache.asUInt === 0.U) {
        used_cache := used_cache.bitSet(output_data_entry, false.B)
        mask_l2cache := VecInit(Seq.fill(numgroupl2cache)(false.B))
        current_inst_entry_index_reg := 0.U
      }
      when(normalTagFire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        when(nonInterleaveTensorTag) {
          tensorPendingValid(valid_tag_entry) := true.B
          tensorPendingTag(valid_tag_entry) := io.from_addr_tag.bits.tag
        }
      }
    }
    is(s_reset) {
      used_inst := used_inst.bitSet(complete_inst_entry, false.B)
      finish_cnt(PriorityEncoder(complete)) := 1.U
      when(!(output_inst.funct === 2.U &&
          output_inst.tensorvars.interleaveMode === 0.U)) {
        tensorReuseValid := VecInit(Seq.fill(tensorReuseEntries)(false.B))
      }
      when(normalTagFire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        when(nonInterleaveTensorTag) {
          tensorPendingValid(valid_tag_entry) := true.B
          tensorPendingTag(valid_tag_entry) := io.from_addr_tag.bits.tag
        }
      }
    }
  }

  when(dataWriteValid) {
    datamem.write(valid_data_entry, dataWriteBits)
  }
  when(tagWriteValid) {
    tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
  }

  // Output to shared
  io.to_shared.bits.entry_index := current_inst_entry_index_reg
  io.to_shared.bits.mask := mask_shared
  if (numgroupshared == numgroupl2cache) {
    (0 until numgroupshared).foreach(x =>
      io.to_shared.bits.data(x) := output_data_4byte(x)
    )
  } else {
    (0 until numgroupshared).foreach(x =>
      when(x.U + mask_index < numgroupl2cache.U) {
        io.to_shared.bits.data(x) := output_data_4byte(x.U + mask_index)
      }.otherwise {
        io.to_shared.bits.data(x) := 0.U
      }
    )
  }
  io.to_shared.bits.inst_src := output_inst.src
  io.to_shared.bits.inst_dst := output_inst.dst
  io.to_shared.bits.inst_funct := output_inst.funct
  io.to_shared.bits.tensor_datawidth := output_inst.tensorvars.datawidth
  io.to_shared.bits.tensor_element_stride0 := output_inst.tensorvars.elementStrides(0)
  io.to_shared.bits.tensor_swizzle_mode := output_inst.tensorvars.swizzleMode
  io.to_shared.bits.tag := {
    if (numgroupshared == numgroupl2cache) tag_wire
    else tag_wire + mask_index * dma_aligned_bulk.U
  }
  io.to_shared.bits.box_dim0_start := output_data.cacheline_info.box_dim0_start
  io.to_shared.bits.shared_row_base := output_data.cacheline_info.shared_row_base
  io.to_shared.bits.swizzle_row_low := output_data.cacheline_info.swizzle_row_low
  io.to_shared.bits.tensor_interleave := output_data.cacheline_info.tensor_interleave
  io.to_shared.bits.shared_elem_addr := output_data.cacheline_info.shared_elem_addr
  io.inst_complete.bits := output_inst.wid

}

// ============================================================
// Addrcalc_shared: convert temp output to shared memory requests
// ============================================================
class Addrcalc_shared extends Module {
  val io = IO(new Bundle {
    val from_temp = Flipped(DecoupledIO(new DmaTempOutput))
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
  })

  val s_idle :: s_prepare :: s_send :: Nil = Enum(3)
  val state = RegInit(s_idle)
  val reg_save = Reg(new DmaTempOutput)

  val sharedSetIdxBits = log2Ceil(sharedmem_depth)
  val sharedSetIdxHi = sharedSetIdxBits + dcache_BlockOffsetBits + dcache_WordOffsetBits - 1
  val sharedSetIdxLo = dcache_BlockOffsetBits + dcache_WordOffsetBits
  val sharedTagBits = xLen - sharedSetIdxBits - dcache_BlockOffsetBits - dcache_WordOffsetBits

  val setIdxReg = RegInit(0.U(sharedSetIdxBits.W))
  val currentTagReg = RegInit(0.U(sharedTagBits.W))
  val laneTagReg = Reg(Vec(numgroupshared, UInt(sharedTagBits.W)))
  val laneSetReg = Reg(Vec(numgroupshared, UInt(sharedSetIdxBits.W)))
  val laneBlockOffsetReg = Reg(Vec(numgroupshared, UInt(dcache_BlockOffsetBits.W)))
  val blockOffsetReg = Reg(Vec(numgroupshared, UInt(dcache_BlockOffsetBits.W)))
  val currentMaskReg = RegInit(VecInit(Seq.fill(numgroupshared)(false.B)))
  val maskNextReg = RegInit(VecInit(Seq.fill(numgroupshared)(false.B)))

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

  def sharedLaneAddr(input: DmaTempOutput): Vec[UInt] = {
    val result = Wire(Vec(numgroupshared, UInt(xLen.W)))
    val linearBase = input.tag - input.inst_src + input.inst_dst
    (0 until numgroupshared).foreach { x =>
      result(x) := linearBase + (x.U << log2Ceil(dma_aligned_bulk))
    }
    when(input.inst_funct === 2.U) {
      when(input.tensor_interleave) {
        val interleaveAddr = swizzleSharedAddr(
          input.shared_elem_addr,
          input.inst_dst,
          input.tensor_swizzle_mode,
          input.swizzle_row_low
        )
        (0 until numgroupshared).foreach { x => result(x) := interleaveAddr }
      }.otherwise {
        val supportsPackedDim0Gather =
          (input.tensor_datawidth === dma_aligned_bulk.U) && (input.tensor_element_stride0 > 1.U)
        (0 until numgroupshared).foreach { x =>
          val srcDim0OffsetBytes = input.tag - input.box_dim0_start +
            (x.U << log2Ceil(dma_aligned_bulk))
          val srcDim0OffsetElems = srcDim0OffsetBytes >> log2Ceil(dma_aligned_bulk)
          val packedElemIndex = MuxLookup(input.tensor_element_stride0, srcDim0OffsetElems)(Seq(
            2.U  -> (srcDim0OffsetElems >> 1),
            4.U  -> (srcDim0OffsetElems >> 2),
            8.U  -> (srcDim0OffsetElems >> 3),
            16.U -> (srcDim0OffsetElems >> 4),
            32.U -> (srcDim0OffsetElems >> 5)
          ))
          val packedDim0OffsetBytes = Mux(
            supportsPackedDim0Gather,
            packedElemIndex << log2Ceil(dma_aligned_bulk),
            srcDim0OffsetBytes,
          )
          val logicalAddr = input.shared_row_base + packedDim0OffsetBytes
          result(x) := swizzleSharedAddr(
            logicalAddr,
            input.inst_dst,
            input.tensor_swizzle_mode,
            input.swizzle_row_low
          )
        }
      }
    }
    result
  }

  def firstActiveTag(mask: UInt, tags: Vec[UInt]): UInt =
    Mux(mask.orR, tags(PriorityEncoder(mask)), 0.U)

  def firstActiveSet(mask: UInt, sets: Vec[UInt]): UInt =
    Mux(mask.orR, sets(PriorityEncoder(mask)), 0.U)

  val inputAddr = sharedLaneAddr(io.from_temp.bits)
  val inputLaneTag = Wire(Vec(numgroupshared, UInt(sharedTagBits.W)))
  val inputLaneSet = Wire(Vec(numgroupshared, UInt(sharedSetIdxBits.W)))
  val inputLaneBlockOffset = Wire(Vec(numgroupshared, UInt(dcache_BlockOffsetBits.W)))
  (0 until numgroupshared).foreach { x =>
    inputLaneTag(x) := inputAddr(x)(xLen - 1, sharedSetIdxHi + 1)
    inputLaneSet(x) := inputAddr(x)(sharedSetIdxHi, sharedSetIdxLo)
    inputLaneBlockOffset(x) :=
      inputAddr(x)(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits)
  }
  val inputActiveMaskBits = io.from_temp.bits.mask.asUInt
  val nextMaskBits = maskNextReg.asUInt

  val current_mask = Wire(Vec(numgroupshared, Bool()))
  (0 until numgroupshared).foreach(x =>
    current_mask(x) := reg_save.mask(x) &&
      (laneTagReg(x) === currentTagReg) &&
      (laneSetReg(x) === setIdxReg)
  )
  val mask_next = Wire(Vec(numgroupshared, Bool()))
  (0 until numgroupshared).foreach(x =>
    mask_next(x) := reg_save.mask(x) && !current_mask(x)
  )

  io.from_temp.ready := state === s_idle
  io.shared_req.valid := state === s_send
  io.shared_req.bits.instrId := reg_save.entry_index
  io.shared_req.bits.isWrite := true.B
  io.shared_req.bits.setIdx := setIdxReg
  (0 until numgroupshared).foreach(x => {
    io.shared_req.bits.perLaneAddr(x).blockOffset := blockOffsetReg(x)
    io.shared_req.bits.perLaneAddr(x).wordOffset1H := Fill(BytesOfWord, 1.U)
    io.shared_req.bits.perLaneAddr(x).activeMask := currentMaskReg(x)
    io.shared_req.bits.data(x) := reg_save.data(x)
  })

  switch(state) {
    is(s_idle) {
      when(io.from_temp.fire) {
        reg_save := io.from_temp.bits
        (0 until numgroupshared).foreach { x =>
          laneTagReg(x) := inputLaneTag(x)
          laneSetReg(x) := inputLaneSet(x)
          laneBlockOffsetReg(x) := inputLaneBlockOffset(x)
        }
        currentTagReg := firstActiveTag(inputActiveMaskBits, inputLaneTag)
        setIdxReg := firstActiveSet(inputActiveMaskBits, inputLaneSet)
        state := Mux(io.from_temp.bits.mask.asUInt.orR, s_prepare, s_idle)
      }
    }
    is(s_prepare) {
      (0 until numgroupshared).foreach { x =>
        blockOffsetReg(x) := laneBlockOffsetReg(x)
        currentMaskReg(x) := current_mask(x)
      }
      maskNextReg := mask_next
      state := s_send
    }
    is(s_send) {
      when(io.shared_req.fire) {
        when(maskNextReg.asUInt === 0.U) {
          state := s_idle
        }.otherwise {
          reg_save.mask := maskNextReg
          currentTagReg := firstActiveTag(nextMaskBits, laneTagReg)
          setIdxReg := firstActiveSet(nextMaskBits, laneSetReg)
          state := s_prepare
        }
      }
    }
  }
}

// ============================================================
// DMA_core: top-level DMA module
// ============================================================
class DMA_core(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val dma_req = Flipped(DecoupledIO(new vExeData))
    val dma_cache_rsp = Flipped(DecoupledIO(new DCacheMemRsp))
    val shared_rsp = Flipped(DecoupledIO(new DmaSharedRsp))
    val dma_cache_req = DecoupledIO(new DCacheMemReq_p)
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
    val fence_end_dma = DecoupledIO(UInt(depth_warp.W))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
  })

  // Input FIFO
  val InputFIFO = Module(new Queue(new vExeData, entries = 1, pipe = true))
  InputFIFO.io.enq <> io.dma_req

  val inputIsS2GBulk = InputFIFO.io.deq.bits.ctrl.funct === 3.U
  val inputIsTensorS2G = InputFIFO.io.deq.bits.ctrl.funct === 4.U

  // Existing G2S/TMA address calculator
  val addrCalc_l2cache = Module(new AddrCalc_l2cache)
  addrCalc_l2cache.io.from_fifo.valid := InputFIFO.io.deq.valid && !inputIsS2GBulk && !inputIsTensorS2G
  addrCalc_l2cache.io.from_fifo.bits := InputFIFO.io.deq.bits

  // S2G datapaths
  val dmaS2G = Module(new DmaS2G)
  dmaS2G.io.from_fifo.valid := InputFIFO.io.deq.valid && inputIsS2GBulk
  dmaS2G.io.from_fifo.bits := InputFIFO.io.deq.bits

  val dmaTensorS2G = Module(new DmaTensorS2G)
  dmaTensorS2G.io.from_fifo.valid := InputFIFO.io.deq.valid && inputIsTensorS2G
  dmaTensorS2G.io.from_fifo.bits := InputFIFO.io.deq.bits

  InputFIFO.io.deq.ready := Mux(inputIsS2GBulk,
    dmaS2G.io.from_fifo.ready,
    Mux(inputIsTensorS2G, dmaTensorS2G.io.from_fifo.ready, addrCalc_l2cache.io.from_fifo.ready))

  // L2 request arbitration: G2S/TMA, bulk S2G, and tensor S2G share the DMA L2 port.
  val dmaL2ReqArb = Module(new Arbiter(new DCacheMemReq_p, 3))
  dmaL2ReqArb.io.in(0) <> addrCalc_l2cache.io.to_l2cache
  dmaL2ReqArb.io.in(1) <> dmaS2G.io.to_l2cache
  dmaL2ReqArb.io.in(2) <> dmaTensorS2G.io.to_l2cache
  io.dma_cache_req <> dmaL2ReqArb.io.out

  // TLB arbitration. The TLB response has no source tag, so keep only one
  // outstanding DMA TLB request across all DMA directions.
  val dmaTlbReqArb = Module(new Arbiter(new L1TlbReq(SV32), 3))
  dmaTlbReqArb.io.in(0) <> addrCalc_l2cache.io.to_l2TLB
  dmaTlbReqArb.io.in(1) <> dmaS2G.io.to_l2TLB
  dmaTlbReqArb.io.in(2) <> dmaTensorS2G.io.to_l2TLB
  val tlbBusy = RegInit(false.B)
  val tlbOwner = RegInit(0.U(2.W))
  io.to_l2TLB.valid := dmaTlbReqArb.io.out.valid && !tlbBusy
  io.to_l2TLB.bits := dmaTlbReqArb.io.out.bits
  dmaTlbReqArb.io.out.ready := io.to_l2TLB.ready && !tlbBusy
  when(io.to_l2TLB.fire) {
    tlbBusy := true.B
    tlbOwner := dmaTlbReqArb.io.chosen
  }
  addrCalc_l2cache.io.from_l2TLB.valid := io.from_l2TLB.valid && tlbBusy && tlbOwner === 0.U
  addrCalc_l2cache.io.from_l2TLB.bits := io.from_l2TLB.bits
  dmaS2G.io.from_l2TLB.valid := io.from_l2TLB.valid && tlbBusy && tlbOwner === 1.U
  dmaS2G.io.from_l2TLB.bits := io.from_l2TLB.bits
  dmaTensorS2G.io.from_l2TLB.valid := io.from_l2TLB.valid && tlbBusy && tlbOwner === 2.U
  dmaTensorS2G.io.from_l2TLB.bits := io.from_l2TLB.bits
  io.from_l2TLB.ready := Mux(tlbBusy && tlbOwner === 0.U,
    addrCalc_l2cache.io.from_l2TLB.ready,
    Mux(tlbBusy && tlbOwner === 1.U,
      dmaS2G.io.from_l2TLB.ready,
      Mux(tlbBusy && tlbOwner === 2.U, dmaTensorS2G.io.from_l2TLB.ready, false.B)))
  when(io.from_l2TLB.fire) {
    tlbBusy := false.B
  }

  // Temporary memory
  val tempmem = Module(new Temp_mem)
  tempmem.io.from_addr <> addrCalc_l2cache.io.to_tempmem_inst
  tempmem.io.from_addr_tag <> addrCalc_l2cache.io.to_tempmem_tag
  addrCalc_l2cache.io.tag_reuse_hit := tempmem.io.from_addr_tag_reuse
  addrCalc_l2cache.io.inst_mem_index := tempmem.io.inst_mem_index
  addrCalc_l2cache.io.tag_mem_index := tempmem.io.tag_mem_index

  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 2, "DMA source encoding needs enough low bits for S2G response demux")

  val dmaRspIsMeta = io.dma_cache_rsp.bits.d_source(0)
  val dmaRspMetaKind = io.dma_cache_rsp.bits.d_source(2, 1)
  val dmaRspLowBits = io.dma_cache_rsp.bits.d_source(dmaSourceLowBits - 1, 0)
  val dmaRspIsMetaPrefetch = dmaRspIsMeta && dmaRspMetaKind === 0.U
  val dmaRspIsMetaDesc = dmaRspIsMeta && dmaRspMetaKind === 1.U
  val dmaRspIsMetaTensor = dmaRspIsMeta && dmaRspMetaKind === 2.U
  val dmaRspIsBulkS2GAck = !dmaRspIsMeta && io.dma_cache_rsp.bits.d_opcode === 0.U && dmaRspLowBits === 2.U
  val dmaRspIsTensorS2GAck = !dmaRspIsMeta && io.dma_cache_rsp.bits.d_opcode === 0.U && dmaRspLowBits === 4.U

  addrCalc_l2cache.io.from_l2cache_meta.valid := io.dma_cache_rsp.valid && (dmaRspIsMetaPrefetch || dmaRspIsMetaDesc)
  addrCalc_l2cache.io.from_l2cache_meta.bits := io.dma_cache_rsp.bits
  dmaTensorS2G.io.from_l2cache.valid := io.dma_cache_rsp.valid && (dmaRspIsMetaTensor || dmaRspIsTensorS2GAck)
  dmaTensorS2G.io.from_l2cache.bits := io.dma_cache_rsp.bits
  dmaS2G.io.from_l2cache.valid := io.dma_cache_rsp.valid && dmaRspIsBulkS2GAck
  dmaS2G.io.from_l2cache.bits := io.dma_cache_rsp.bits
  tempmem.io.from_l2cache.valid := io.dma_cache_rsp.valid && !dmaRspIsMeta && !dmaRspIsBulkS2GAck && !dmaRspIsTensorS2GAck
  tempmem.io.from_l2cache.bits := io.dma_cache_rsp.bits
  io.dma_cache_rsp.ready := Mux(dmaRspIsMetaPrefetch || dmaRspIsMetaDesc,
    addrCalc_l2cache.io.from_l2cache_meta.ready,
    Mux(dmaRspIsMetaTensor || dmaRspIsTensorS2GAck,
      dmaTensorS2G.io.from_l2cache.ready,
      Mux(dmaRspIsBulkS2GAck, dmaS2G.io.from_l2cache.ready, tempmem.io.from_l2cache.ready)))

  tempmem.io.from_shared.valid := io.shared_rsp.valid && io.shared_rsp.bits.isWrite
  tempmem.io.from_shared.bits.instrId := io.shared_rsp.bits.instrId
  tempmem.io.from_shared.bits.data := io.shared_rsp.bits.data
  tempmem.io.from_shared.bits.activeMask := io.shared_rsp.bits.activeMask
  dmaS2G.io.shared_rsp.valid := io.shared_rsp.valid && !io.shared_rsp.bits.isWrite && io.shared_rsp.bits.instrId === 0.U
  dmaS2G.io.shared_rsp.bits := io.shared_rsp.bits
  dmaTensorS2G.io.shared_rsp.valid := io.shared_rsp.valid && !io.shared_rsp.bits.isWrite && io.shared_rsp.bits.instrId === 1.U
  dmaTensorS2G.io.shared_rsp.bits := io.shared_rsp.bits
  io.shared_rsp.ready := Mux(io.shared_rsp.bits.isWrite,
    tempmem.io.from_shared.ready,
    Mux(io.shared_rsp.bits.instrId === 0.U,
      dmaS2G.io.shared_rsp.ready,
      Mux(io.shared_rsp.bits.instrId === 1.U, dmaTensorS2G.io.shared_rsp.ready, false.B)))

  val dmaCompleteArb = Module(new Arbiter(UInt(depth_warp.W), 4))
  dmaCompleteArb.io.in(0) <> tempmem.io.inst_complete
  dmaCompleteArb.io.in(1) <> addrCalc_l2cache.io.meta_complete
  dmaCompleteArb.io.in(2) <> dmaS2G.io.inst_complete
  dmaCompleteArb.io.in(3) <> dmaTensorS2G.io.inst_complete
  io.fence_end_dma <> dmaCompleteArb.io.out

  // Shared memory request arbitration: G2S writes and both S2G reads share LDS.
  val addrCalc_shared = Module(new Addrcalc_shared)
  addrCalc_shared.io.from_temp <> tempmem.io.to_shared
  val dmaSharedReqArb = Module(new Arbiter(new ShareMemCoreReq_np, 3))
  dmaSharedReqArb.io.in(0) <> addrCalc_shared.io.shared_req
  dmaSharedReqArb.io.in(1) <> dmaS2G.io.shared_req
  dmaSharedReqArb.io.in(2) <> dmaTensorS2G.io.shared_req
  io.shared_req <> dmaSharedReqArb.io.out


}
