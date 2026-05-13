/*
 * DMA Core for Ventus GPGPU
 *
 * Current scope:
 *
 *   Implemented here:  Tensor DMA address generation, OOB fill, element
 *                      stride/subbox iteration, and 32B/64B/128B swizzle.
 *   Not started:       Im2col, shared -> global writeback, descriptor
 *                      prefetch, TC FP16/BF16.
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
// `dataType` field used by pocl/examples/tma_matrix_test host code:
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

// Tensor descriptor bundle for CP_ASYNC_TENSOR (funct=3)
class TensorVars extends Bundle {
  val BoxAddress     = UInt(xLen.W)
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
  val tensorvars = new TensorVars  // only valid when funct=3
}

// Tag info for each L2 cacheline request
class DmaCachelineInfo extends Bundle {
  val tag = UInt(xLen.W)
  val tensor_dim_step   = Vec(5, UInt(xLen.W))
  val box_dim0_start    = UInt(xLen.W)
  val tensor_dim0_start = UInt(xLen.W)
  val tensor_high_dim_valid = Bool()
  val shared_row_base   = UInt(xLen.W)
  val dim0_stride_bytes = UInt(xLen.W)
  val swizzle_row_low   = UInt(3.W)
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
  val cacheline_info = new DmaCachelineInfo
  val instinfo = new vExeDataDMA
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
    val meta_complete = DecoupledIO(UInt(depth_warp.W))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
  })

  val (s_idle :: s_prefetch_tlb_req :: s_prefetch_tlb_rsp ::
    s_prefetch_l2cache :: s_prefetch_rsp :: s_prefetch_done ::
    s_desc_tlb_req :: s_desc_tlb_rsp :: s_desc_l2cache :: s_desc_rsp ::
    s_dyn_tlb_req :: s_dyn_tlb_rsp :: s_dyn_l2cache :: s_dyn_rsp ::
    s_tensor_setup :: s_save :: s_l2cache_tag :: s_tlb_req :: s_tlb_rsp ::
    s_l2cache :: Nil) = Enum(20)
  val state = RegInit(s_idle)
  val reg_save = Reg(new DmaRegSave)
  val inst_mem_index_reg = RegInit(0.U(log2Up(max_dma_inst).W))
  val tag_mem_index_reg = RegInit(0.U(log2Ceil(max_dma_tag).W))
  val p_addr_reg = Reg(UInt(SV32.paLen.W))
  val desc_ptr_reg = RegInit(0.U(xLen.W))
  val dyn_ptr_reg = RegInit(0.U(xLen.W))
  val meta_vaddr_reg = RegInit(0.U(xLen.W))
  val desc_words_reg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))
  val dyn_words_reg = RegInit(VecInit(Seq.fill(32)(0.U(xLen.W))))

  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 0, "DMA source encoding needs a spare low bit for metadata responses")
  val dmaMetaSource = 1.U(l1cache_sourceBits.W)

  // ---- Tensor iteration state ----
  import DataType._
  val tensor_dim_step_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_base_global_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_global_pos_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_offset_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_shared_offset_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_dim_stride_bytes_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_shared_stride_bytes_reg = RegInit(VecInit(Seq.fill(5)(0.U(xLen.W))))
  val tensor_row_base_reg = RegInit(0.U(xLen.W))
  val tensor_shared_row_base_reg = RegInit(0.U(xLen.W))
  val tensor_dim0_offset_bytes_reg = RegInit(0.U(xLen.W))
  val tensor_row_span_bytes_reg = RegInit(0.U(xLen.W))
  val tensor_copy_size_reg = RegInit(0.U(xLen.W))
  val tensor_swizzle_row_low_reg = RegInit(0.U(3.W))
  val tensor_dim_step_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_dim_pos_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_global_pos_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_dim_offset_next = Wire(Vec(5, UInt(xLen.W)))
  val tensor_shared_offset_next = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach(x => tensor_dim_step_next(x) := tensor_dim_step_reg(x))
  (0 until 5).foreach(x => tensor_dim_pos_next(x) := tensor_dim_pos_reg(x))
  (0 until 5).foreach(x => tensor_global_pos_next(x) := tensor_global_pos_reg(x))
  (0 until 5).foreach(x => tensor_dim_offset_next(x) := tensor_dim_offset_reg(x))
  (0 until 5).foreach(x => tensor_shared_offset_next(x) := tensor_shared_offset_reg(x))

  // ---- TensorVars extraction from legacy VGPRs or descriptor memory ----
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
  (0 until 5).foreach { x => desc_coords(x) := Mux(dyn_ptr_reg === 0.U, 0.U, dyn_words_reg(x)) }
  val desc_box_address =
    desc_words_reg(2) +
      desc_coords(0) * desc_byte_stride(0) +
      desc_coords(1) * desc_byte_stride(1) +
      desc_coords(2) * desc_byte_stride(2) +
      desc_coords(3) * desc_byte_stride(3) +
      desc_coords(4) * desc_byte_stride(4)

  val desc_mode = reg_save.ctrl.funct === 6.U
  val tensor_copy_mode = (reg_save.ctrl.funct === 3.U) || desc_mode
  when(desc_mode) {
    tvars.BoxAddress := desc_box_address
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
  }.otherwise {
    tvars.BoxAddress := reg_save.in2(0)
    (0 until 5).foreach { x =>
      tvars.boxDim(x) := reg_save.in2(1 + x)
      tvars.elementStrides(x) := reg_save.in2(6 + x)
    }
    tvars.interleaveMode := reg_save.in2(11)(log2Ceil(3) - 1, 0)
    tvars.swizzleMode    := reg_save.in2(12)(log2Ceil(4) - 1, 0)
    tvars.L2promotion    := reg_save.in2(13)(log2Ceil(4) - 1, 0)
    tvars.oobfill        := reg_save.in2(14)(log2Ceil(2) - 1, 0)
    tvars.dataType       := reg_save.in1(0)(log2Ceil(13) - 1, 0)
    tvars.tensorRank     := reg_save.in1(1)(log2Ceil(5) - 1, 0)
    tvars.globalAddress  := reg_save.in1(2)
    (0 until 5).foreach { x =>
      tvars.globalDim(x) := reg_save.in1(3 + x)
      tvars.globalStrides(x) := reg_save.in1(8 + x)
    }
  }

  // datawidth derived from dataType — encoding matches spike cp_async_tensor.h.
  tvars.datawidth := tensorDataWidth(tvars.dataType)

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))

  // Tensor setup still decodes the descriptor once. The cacheline issue path
  // below then advances with registered offsets and adders instead of repeated
  // per-cacheline div/mod/multiply chains.
  val box_elements_num_setup = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { x =>
    box_elements_num_setup(x) := Mux(
      tvars.elementStrides(x) === 0.U,
      0.U,
      (tvars.boxDim(x) + tvars.elementStrides(x) - 1.U) / tvars.elementStrides(x)
    )
  }
  box_elements_num_setup(0) := tvars.boxDim(0)

  val tensor_dim_stride_bytes_setup = Wire(Vec(5, UInt(xLen.W)))
  tensor_dim_stride_bytes_setup(0) := tvars.elementStrides(0) * tvars.datawidth
  (1 until 5).foreach { x =>
    tensor_dim_stride_bytes_setup(x) := tvars.globalStrides(x - 1) * tvars.elementStrides(x)
  }

  val tensor_shared_stride_bytes_setup = Wire(Vec(5, UInt(xLen.W)))
  tensor_shared_stride_bytes_setup(0) := tvars.datawidth
  tensor_shared_stride_bytes_setup(1) := tvars.boxDim(0) * tvars.datawidth
  tensor_shared_stride_bytes_setup(2) := tensor_shared_stride_bytes_setup(1) * box_elements_num_setup(1)
  tensor_shared_stride_bytes_setup(3) := tensor_shared_stride_bytes_setup(2) * box_elements_num_setup(2)
  tensor_shared_stride_bytes_setup(4) := tensor_shared_stride_bytes_setup(3) * box_elements_num_setup(3)

  val Tensorcopysize = Wire(UInt(xLen.W))
  Tensorcopysize := 0.U
  switch(tvars.tensorRank) {
    is(1.U) { Tensorcopysize := tensor_shared_stride_bytes_setup(1) }
    is(2.U) { Tensorcopysize := tensor_shared_stride_bytes_setup(1) * box_elements_num_setup(1) }
    is(3.U) { Tensorcopysize := tensor_shared_stride_bytes_setup(2) * box_elements_num_setup(2) }
    is(4.U) { Tensorcopysize := tensor_shared_stride_bytes_setup(3) * box_elements_num_setup(3) }
    is(5.U) { Tensorcopysize := tensor_shared_stride_bytes_setup(4) * box_elements_num_setup(4) }
  }

  val box_to_global_offset = tvars.BoxAddress - tvars.globalAddress
  val box_dim0_offset_bytes_setup = Wire(UInt(xLen.W))
  box_dim0_offset_bytes_setup := box_to_global_offset
  when(tvars.tensorRank > 1.U && tvars.globalStrides(0) =/= 0.U) {
    box_dim0_offset_bytes_setup := box_to_global_offset % tvars.globalStrides(0)
  }

  val box_offset_elems_setup = Wire(Vec(5, UInt(xLen.W)))
  (0 until 5).foreach { x => box_offset_elems_setup(x) := 0.U }

  val box_offset_rem4 = box_to_global_offset
  val box_offset_rem3 = Wire(UInt(xLen.W))
  box_offset_rem3 := box_offset_rem4
  when(tvars.tensorRank > 4.U && tvars.globalStrides(3) =/= 0.U) {
    box_offset_elems_setup(4) := box_offset_rem4 / tvars.globalStrides(3)
    box_offset_rem3 := box_offset_rem4 % tvars.globalStrides(3)
  }

  val box_offset_rem2 = Wire(UInt(xLen.W))
  box_offset_rem2 := box_offset_rem3
  when(tvars.tensorRank > 3.U && tvars.globalStrides(2) =/= 0.U) {
    box_offset_elems_setup(3) := box_offset_rem3 / tvars.globalStrides(2)
    box_offset_rem2 := box_offset_rem3 % tvars.globalStrides(2)
  }

  val box_offset_rem1 = Wire(UInt(xLen.W))
  box_offset_rem1 := box_offset_rem2
  when(tvars.tensorRank > 2.U && tvars.globalStrides(1) =/= 0.U) {
    box_offset_elems_setup(2) := box_offset_rem2 / tvars.globalStrides(1)
    box_offset_rem1 := box_offset_rem2 % tvars.globalStrides(1)
  }

  val box_offset_rem0 = Wire(UInt(xLen.W))
  box_offset_rem0 := box_offset_rem1
  when(tvars.tensorRank > 1.U && tvars.globalStrides(0) =/= 0.U) {
    box_offset_elems_setup(1) := box_offset_rem1 / tvars.globalStrides(0)
    box_offset_rem0 := box_offset_rem1 % tvars.globalStrides(0)
  }
  when(tvars.datawidth =/= 0.U) {
    box_offset_elems_setup(0) := box_offset_rem0 / tvars.datawidth
  }

  val tensor_high_dim_valid_terms = (1 until 5).map { d =>
    (tvars.tensorRank <= d.U) || (tensor_global_pos_reg(d) < tvars.globalDim(d))
  }
  val tensor_high_dim_valid = tensor_high_dim_valid_terms.reduce(_ && _)

  val box_dim0_start = tensor_row_base_reg
  val tensor_dim0_row_start = tensor_row_base_reg - tensor_dim0_offset_bytes_reg
  val box_dim0_end = tensor_row_base_reg + tensor_row_span_bytes_reg

  val tensor_can_advance_dim = Wire(Vec(5, Bool()))
  tensor_can_advance_dim(0) := false.B
  (1 until 5).foreach { d =>
    tensor_can_advance_dim(d) :=
      (tvars.tensorRank > d.U) &&
      ((tensor_dim_pos_reg(d) + tvars.elementStrides(d)) < tvars.boxDim(d))
  }

  val tensor_advance_dim_oh = Wire(Vec(5, Bool()))
  tensor_advance_dim_oh(0) := false.B
  tensor_advance_dim_oh(1) := tensor_can_advance_dim(1)
  tensor_advance_dim_oh(2) := !tensor_can_advance_dim(1) && tensor_can_advance_dim(2)
  tensor_advance_dim_oh(3) := !tensor_can_advance_dim(1) && !tensor_can_advance_dim(2) && tensor_can_advance_dim(3)
  tensor_advance_dim_oh(4) := !tensor_can_advance_dim(1) && !tensor_can_advance_dim(2) && !tensor_can_advance_dim(3) && tensor_can_advance_dim(4)
  val tensor_has_next_row = tensor_advance_dim_oh.asUInt.orR

  when(tensor_advance_dim_oh(1)) {
    tensor_dim_step_next(1) := tensor_dim_step_reg(1) + 1.U
    tensor_dim_pos_next(1) := tensor_dim_pos_reg(1) + tvars.elementStrides(1)
    tensor_global_pos_next(1) := tensor_global_pos_reg(1) + tvars.elementStrides(1)
    tensor_dim_offset_next(1) := tensor_dim_offset_reg(1) + tensor_dim_stride_bytes_reg(1)
    tensor_shared_offset_next(1) := tensor_shared_offset_reg(1) + tensor_shared_stride_bytes_reg(1)
  }.elsewhen(tensor_advance_dim_oh(2)) {
    tensor_dim_step_next(1) := 0.U
    tensor_dim_pos_next(1) := 0.U
    tensor_global_pos_next(1) := tensor_base_global_pos_reg(1)
    tensor_dim_offset_next(1) := 0.U
    tensor_shared_offset_next(1) := 0.U
    tensor_dim_step_next(2) := tensor_dim_step_reg(2) + 1.U
    tensor_dim_pos_next(2) := tensor_dim_pos_reg(2) + tvars.elementStrides(2)
    tensor_global_pos_next(2) := tensor_global_pos_reg(2) + tvars.elementStrides(2)
    tensor_dim_offset_next(2) := tensor_dim_offset_reg(2) + tensor_dim_stride_bytes_reg(2)
    tensor_shared_offset_next(2) := tensor_shared_offset_reg(2) + tensor_shared_stride_bytes_reg(2)
  }.elsewhen(tensor_advance_dim_oh(3)) {
    (1 to 2).foreach { d =>
      tensor_dim_step_next(d) := 0.U
      tensor_dim_pos_next(d) := 0.U
      tensor_global_pos_next(d) := tensor_base_global_pos_reg(d)
      tensor_dim_offset_next(d) := 0.U
      tensor_shared_offset_next(d) := 0.U
    }
    tensor_dim_step_next(3) := tensor_dim_step_reg(3) + 1.U
    tensor_dim_pos_next(3) := tensor_dim_pos_reg(3) + tvars.elementStrides(3)
    tensor_global_pos_next(3) := tensor_global_pos_reg(3) + tvars.elementStrides(3)
    tensor_dim_offset_next(3) := tensor_dim_offset_reg(3) + tensor_dim_stride_bytes_reg(3)
    tensor_shared_offset_next(3) := tensor_shared_offset_reg(3) + tensor_shared_stride_bytes_reg(3)
  }.elsewhen(tensor_advance_dim_oh(4)) {
    (1 to 3).foreach { d =>
      tensor_dim_step_next(d) := 0.U
      tensor_dim_pos_next(d) := 0.U
      tensor_global_pos_next(d) := tensor_base_global_pos_reg(d)
      tensor_dim_offset_next(d) := 0.U
      tensor_shared_offset_next(d) := 0.U
    }
    tensor_dim_step_next(4) := tensor_dim_step_reg(4) + 1.U
    tensor_dim_pos_next(4) := tensor_dim_pos_reg(4) + tvars.elementStrides(4)
    tensor_global_pos_next(4) := tensor_global_pos_reg(4) + tvars.elementStrides(4)
    tensor_dim_offset_next(4) := tensor_dim_offset_reg(4) + tensor_dim_stride_bytes_reg(4)
    tensor_shared_offset_next(4) := tensor_shared_offset_reg(4) + tensor_shared_stride_bytes_reg(4)
  }

  val tensor_next_row_base =
    tvars.BoxAddress + tensor_dim_offset_next(1) + tensor_dim_offset_next(2) +
      tensor_dim_offset_next(3) + tensor_dim_offset_next(4)
  val tensor_next_shared_row_base =
    reg_save.in3(0) + tensor_shared_offset_next(1) + tensor_shared_offset_next(2) +
      tensor_shared_offset_next(3) + tensor_shared_offset_next(4)

  val tensor_row_has_next_line = (reg_save.address + l2cacheline.U) < box_dim0_end
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
  io.to_tempmem_inst.bits.funct := Mux(tensor_copy_mode, 3.U, reg_save.ctrl.funct)
  io.to_tempmem_inst.bits.tensorvars := tvars
  io.to_tempmem_inst.bits.copysize := 0.U
  switch(reg_save.ctrl.funct) {
    is(0.U) { io.to_tempmem_inst.bits.copysize := 4.U << reg_save.ctrl.copysize }
    is(1.U) { io.to_tempmem_inst.bits.copysize := reg_save.in2(0) }
    is(3.U) { io.to_tempmem_inst.bits.copysize := tensor_copy_size_reg }
    is(6.U) { io.to_tempmem_inst.bits.copysize := tensor_copy_size_reg }
  }

  // Temp tag store interface
  io.to_tempmem_tag.valid := state === s_l2cache_tag
  io.to_tempmem_tag.bits.tag := Cat(reg_save.address(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))
  io.to_tempmem_tag.bits.box_dim0_start := box_dim0_start
  io.to_tempmem_tag.bits.tensor_dim0_start := tensor_dim0_row_start
  io.to_tempmem_tag.bits.tensor_high_dim_valid := tensor_high_dim_valid
  io.to_tempmem_tag.bits.shared_row_base := tensor_shared_row_base_reg
  io.to_tempmem_tag.bits.dim0_stride_bytes := tensor_dim_stride_bytes_reg(0)
  io.to_tempmem_tag.bits.swizzle_row_low := tensor_swizzle_row_low_reg
  (0 until 5).foreach { x =>
    io.to_tempmem_tag.bits.tensor_dim_step(x) := tensor_dim_step_reg(x)
  }

  // TLB request
  val aligned_vaddr = Cat(reg_save.address(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))
  val aligned_meta_vaddr = Cat(meta_vaddr_reg(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W))
  val meta_tlb_req_state =
    state === s_prefetch_tlb_req || state === s_desc_tlb_req || state === s_dyn_tlb_req
  val meta_tlb_rsp_state =
    state === s_prefetch_tlb_rsp || state === s_desc_tlb_rsp || state === s_dyn_tlb_rsp
  val meta_l2cache_state =
    state === s_prefetch_l2cache || state === s_desc_l2cache || state === s_dyn_l2cache
  val meta_rsp_state =
    state === s_prefetch_rsp || state === s_desc_rsp || state === s_dyn_rsp
  io.to_l2TLB.valid := state === s_tlb_req || meta_tlb_req_state
  io.to_l2TLB.bits.vaddr := Mux(meta_tlb_req_state, aligned_meta_vaddr, aligned_vaddr)
  io.to_l2TLB.bits.asid := reg_save.ctrl.asid.getOrElse(0.U)

  // TLB response
  io.from_l2TLB.ready := state === s_tlb_rsp || meta_tlb_rsp_state

  // L2 cache request — uses translated physical address
  io.to_l2cache.valid := state === s_l2cache || meta_l2cache_state
  io.to_l2cache.bits.a_opcode := 4.U // Get
  io.to_l2cache.bits.a_source := Mux(
    meta_l2cache_state,
    dmaMetaSource,
    Cat(tag_mem_index_reg, inst_mem_index_reg, 0.U(dmaSourceLowBits.W))
  )
  io.to_l2cache.bits.a_addr.foreach(_ := p_addr_reg)
  io.to_l2cache.bits.a_mask := VecInit(Seq.fill(dcache_BlockWords)(Fill(BytesOfWord, 1.U)))
  io.to_l2cache.bits.a_data := VecInit(Seq.fill(dcache_BlockWords)(0.U(xLen.W)))
  io.to_l2cache.bits.a_param := 0.U
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)
  io.from_l2cache_meta.ready := meta_rsp_state
  io.meta_complete.valid := state === s_prefetch_done
  io.meta_complete.bits := reg_save.ctrl.wid

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
    val tlb_check_vaddr = Mux(meta_tlb_rsp_state, aligned_meta_vaddr, reg_save.address)
    assert(io.from_l2TLB.bits.paddr(SV32.offsetLen - 1, 0) ===
           tlb_check_vaddr(SV32.offsetLen - 1, 0),
      "TLB paddr page offset mismatch with vaddr")
  }

  // FSM
  switch(state) {
    is(s_idle) {
      when(io.from_fifo.fire) {
        when(io.from_fifo.bits.ctrl.funct === 5.U) {
          state := s_prefetch_tlb_req
        }.elsewhen(io.from_fifo.bits.ctrl.funct === 6.U) {
          state := s_desc_tlb_req
        }.elsewhen(io.from_fifo.bits.ctrl.funct === 3.U) {
          state := s_tensor_setup
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
      when(io.to_l2cache.fire) { state := s_prefetch_rsp }
    }
    is(s_prefetch_rsp) {
      when(io.from_l2cache_meta.fire) { state := s_prefetch_done }
    }
    is(s_prefetch_done) {
      when(io.meta_complete.fire) { state := s_idle }
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
      when(io.from_l2cache_meta.fire) {
        when(dyn_ptr_reg === 0.U) {
          state := s_tensor_setup
        }.otherwise {
          state := s_dyn_tlb_req
        }
      }
    }
    is(s_dyn_tlb_req) {
      when(io.to_l2TLB.fire) { state := s_dyn_tlb_rsp }
    }
    is(s_dyn_tlb_rsp) {
      when(io.from_l2TLB.fire) {
        p_addr_reg := io.from_l2TLB.bits.paddr
        state := s_dyn_l2cache
      }
    }
    is(s_dyn_l2cache) {
      when(io.to_l2cache.fire) { state := s_dyn_rsp }
    }
    is(s_dyn_rsp) {
      when(io.from_l2cache_meta.fire) { state := s_tensor_setup }
    }
    is(s_tensor_setup) {
      state := s_save
    }
    is(s_save) {
      when(io.to_tempmem_inst.fire) { state := s_l2cache_tag }
    }
    is(s_l2cache_tag) {
      when(io.to_tempmem_tag.fire) { state := s_tlb_req }
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
        dyn_ptr_reg := io.from_fifo.bits.in2(0)
        meta_vaddr_reg := io.from_fifo.bits.in1(0)
        // Legacy tensor: initial address from BoxAddress (in2(0)); linear: from src (in1(0)).
        // Descriptor tensor fills the real BoxAddress after its descriptor/dynamic fetch.
        reg_save.address := Mux(io.from_fifo.bits.ctrl.funct === 3.U,
          Cat(io.from_fifo.bits.in2(0)(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W)),
          Cat(io.from_fifo.bits.in1(0)(xLen - 1, xLen - addr_tag_bits), 0.U((xLen - addr_tag_bits).W)))
        reg_save.ctrl := io.from_fifo.bits.ctrl
      }
    }
    is(s_desc_rsp) {
      when(io.from_l2cache_meta.fire) {
        desc_words_reg := io.from_l2cache_meta.bits.d_data
        when(dyn_ptr_reg === 0.U) {
          dyn_words_reg := VecInit(Seq.fill(32)(0.U(xLen.W)))
        }.otherwise {
          meta_vaddr_reg := dyn_ptr_reg
        }
      }
    }
    is(s_dyn_rsp) {
      when(io.from_l2cache_meta.fire) {
        dyn_words_reg := io.from_l2cache_meta.bits.d_data
      }
    }
    is(s_tensor_setup) {
      when(!reset.asBool && tensor_copy_mode) {
        assert(tvars.interleaveMode === 0.U,
          "DMA TMA interleaveMode is not supported in RTL")
        when(desc_mode) {
          assert(desc_ptr_reg(log2Ceil(l2cacheline) - 1, 0) === 0.U,
            "DMA descriptor TMA requires 128B-aligned descriptor in v0")
        }
      }
      (0 until 5).foreach { x =>
        tensor_dim_step_reg(x) := 0.U
        tensor_dim_pos_reg(x) := 0.U
        tensor_base_global_pos_reg(x) := box_offset_elems_setup(x)
        tensor_global_pos_reg(x) := box_offset_elems_setup(x)
        tensor_dim_offset_reg(x) := 0.U
        tensor_shared_offset_reg(x) := 0.U
        tensor_dim_stride_bytes_reg(x) := tensor_dim_stride_bytes_setup(x)
        tensor_shared_stride_bytes_reg(x) := tensor_shared_stride_bytes_setup(x)
      }
      tensor_row_base_reg := tvars.BoxAddress
      tensor_shared_row_base_reg := reg_save.in3(0)
      tensor_dim0_offset_bytes_reg := box_dim0_offset_bytes_setup
      tensor_row_span_bytes_reg := Mux(
        tvars.boxDim(0) === 0.U,
        0.U,
        (tvars.boxDim(0) - 1.U) * tensor_dim_stride_bytes_setup(0) + tvars.datawidth
      )
      tensor_copy_size_reg := Tensorcopysize
      tensor_swizzle_row_low_reg := 0.U
      reg_save.address := alignToL2Line(tvars.BoxAddress)
    }
    is(s_save) {
      when(io.to_tempmem_inst.fire) {
        inst_mem_index_reg := io.inst_mem_index
      }
    }
    is(s_l2cache_tag) {
      when(io.to_tempmem_tag.fire) {
        tag_mem_index_reg := io.tag_mem_index
      }
    }
    is(s_l2cache) {
      when(io.to_l2cache.fire) {
        reg_save.address := address_next
        when(tensor_copy_mode) {
          when(!tensor_row_has_next_line) {
            (0 until 5).foreach { x =>
              tensor_dim_step_reg(x) := tensor_dim_step_next(x)
              tensor_dim_pos_reg(x) := tensor_dim_pos_next(x)
              tensor_global_pos_reg(x) := tensor_global_pos_next(x)
              tensor_dim_offset_reg(x) := tensor_dim_offset_next(x)
              tensor_shared_offset_reg(x) := tensor_shared_offset_next(x)
            }
            when(tensor_has_next_row) {
              tensor_row_base_reg := tensor_next_row_base
              tensor_shared_row_base_reg := tensor_next_shared_row_base
              tensor_swizzle_row_low_reg := tensor_swizzle_row_low_reg + 1.U
            }
          }
        }
        // B2 fix: clear tensor iterators on completion to avoid polluting next instruction.
        when(complete_address) {
          (0 until 5).foreach { x =>
            tensor_dim_step_reg(x) := 0.U
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
  from_l2cache_all.cacheline_info.tensor_dim0_start := tagmem_read_entry.tensor_dim0_start
  from_l2cache_all.cacheline_info.box_dim0_start := tagmem_read_entry.box_dim0_start
  from_l2cache_all.cacheline_info.tensor_high_dim_valid := tagmem_read_entry.tensor_high_dim_valid
  from_l2cache_all.cacheline_info.shared_row_base := tagmem_read_entry.shared_row_base
  from_l2cache_all.cacheline_info.dim0_stride_bytes := tagmem_read_entry.dim0_stride_bytes
  from_l2cache_all.cacheline_info.swizzle_row_low := tagmem_read_entry.swizzle_row_low
  (0 until 5).foreach { x =>
    from_l2cache_all.cacheline_info.tensor_dim_step(x) := tagmem_read_entry.tensor_dim_step(x)
  }

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

  // Mask for slicing L2 cacheline into shared-mem groups
  val mask_l2cache = RegInit(VecInit(Seq.fill(numgroupl2cache)(false.B)))
  val mask_index = PriorityEncoder(mask_l2cache)
  val mask_l2cache_next = Wire(Vec(numgroupl2cache, Bool()))
  (0 until numgroupl2cache).foreach(x =>
    mask_l2cache_next(x) := mask_l2cache(x) && !(x.U >= mask_index && x.U < mask_index + numgroupshared.U)
  )
  val mask_shared = Wire(Vec(numgroupshared, Bool()))
  (0 until numgroupshared).foreach(x =>
    when(x.U + mask_index < numgroupl2cache.U) { mask_shared(x) := mask_l2cache(x.U + mask_index) }
    .otherwise { mask_shared(x) := false.B }
  )

  val output_inst = Reg(new vExeDataDMA)
  val output_data = Reg(new DmaL2CachelineInfo)
  val output_data_entry = Reg(UInt(log2Ceil(max_l2cacheline).W))
  val output_data_4byte = Wire(Vec(numgroupl2cache, UInt((dma_aligned_bulk * BitsOfByte).W)))
  (0 until numgroupl2cache).foreach(x =>
    output_data_4byte(x) := output_data.base.d_data.asUInt((x + 1) * dma_aligned_bulk * BitsOfByte - 1, x * dma_aligned_bulk * BitsOfByte)
  )

  val tag_wire = output_data.cacheline_info.tag

  // ---- OOB fill vectors for tensor mode ----
  import DataType._
  val d_data_bits = output_data.base.d_data.asUInt
  val tmp_1 = Wire(Vec(l2cacheline, UInt((1 * BitsOfByte).W)))
  val tmp_2 = Wire(Vec(l2cacheline / 2, UInt((2 * BitsOfByte).W)))
  val tmp_4 = Wire(Vec(l2cacheline / 4, UInt((4 * BitsOfByte).W)))
  val tmp_8 = Wire(Vec(l2cacheline / 8, UInt((8 * BitsOfByte).W)))
  tmp_1.zipWithIndex.foreach { case (x, i) => x := d_data_bits(i * BitsOfByte + 1 * BitsOfByte - 1, i * BitsOfByte) }
  tmp_2.zipWithIndex.foreach { case (x, i) => x := d_data_bits(i * 2 * BitsOfByte + 2 * BitsOfByte - 1, i * 2 * BitsOfByte) }
  tmp_4.zipWithIndex.foreach { case (x, i) => x := d_data_bits(i * 4 * BitsOfByte + 4 * BitsOfByte - 1, i * 4 * BitsOfByte) }
  tmp_8.zipWithIndex.foreach { case (x, i) => x := d_data_bits(i * 8 * BitsOfByte + 8 * BitsOfByte - 1, i * 8 * BitsOfByte) }

  // OOB fill: keep the requested box row selected, but fill bytes that fall
  // past the real dim0 tensor row instead of leaking data from following rows.
  val t_box_dim0_start = output_data.cacheline_info.box_dim0_start
  val t_tensor_dim0_start = output_data.cacheline_info.tensor_dim0_start
  val t_tensor_high_dim_valid = output_data.cacheline_info.tensor_high_dim_valid
  val t_boxDim0       = output_inst.tensorvars.boxDim(0)
  val t_globalDim0    = output_inst.tensorvars.globalDim(0)
  val t_elementStride0 = output_inst.tensorvars.elementStrides(0)
  val t_datawidth     = output_inst.tensorvars.datawidth
  val t_dataType      = output_inst.tensorvars.dataType
  val t_oobfill       = output_inst.tensorvars.oobfill
  val t_dim0_stride_bytes = t_elementStride0 * t_datawidth
  val t_dim0_row_span_bytes = Mux(
    t_boxDim0 === 0.U,
    0.U,
    (t_boxDim0 - 1.U) * t_dim0_stride_bytes + t_datawidth,
  )
  val t_dim0_end      = t_box_dim0_start + t_dim0_row_span_bytes
  val t_tensor_dim0_end = t_tensor_dim0_start + t_globalDim0 * t_datawidth

  def tensorDim0ByteValid(byteAddr: UInt): Bool = {
    val rel = byteAddr - t_box_dim0_start
    val in_row_span = (byteAddr >= t_box_dim0_start) && (byteAddr < t_dim0_end)
    val in_tensor_row = (byteAddr >= t_tensor_dim0_start) && (byteAddr < t_tensor_dim0_end)
    val is_selected = Mux(
      t_elementStride0 <= 1.U,
      true.B,
      (rel % t_dim0_stride_bytes) < t_datawidth,
    )
    in_row_span && in_tensor_row && t_tensor_high_dim_valid && is_selected
  }

  switch(t_datawidth) {
    is(1.U) {
      (0 until l2cacheline).foreach { x =>
        val in_bounds = tensorDim0ByteValid(tag_wire + x.U)
        when(!in_bounds) { tmp_1(x) := 0.U }
      }
    }
    is(2.U) {
      (0 until l2cacheline by 2).foreach { x =>
        val in_bounds = tensorDim0ByteValid(tag_wire + x.U)
        when(!in_bounds) {
          val fill_val = Mux(t_dataType === UINT16, 0.U((2 * BitsOfByte).W),
            Mux(t_oobfill.asBool, Fill(2 * BitsOfByte, 1.U), 0.U((2 * BitsOfByte).W)))
          tmp_2(x / 2) := fill_val
        }
      }
    }
    is(4.U) {
      (0 until l2cacheline by 4).foreach { x =>
        val in_bounds = tensorDim0ByteValid(tag_wire + x.U)
        when(!in_bounds) {
          val is_int = (t_dataType === UINT32) || (t_dataType === INT32)
          val fill_val = Mux(is_int, 0.U((4 * BitsOfByte).W),
            Mux(t_oobfill.asBool, Fill(4 * BitsOfByte, 1.U), 0.U((4 * BitsOfByte).W)))
          tmp_4(x / 4) := fill_val
        }
      }
    }
    is(8.U) {
      (0 until l2cacheline by 8).foreach { x =>
        val in_bounds = tensorDim0ByteValid(tag_wire + x.U)
        when(!in_bounds) {
          val is_int = (t_dataType === UINT64) || (t_dataType === INT64)
          val fill_val = Mux(is_int, 0.U((8 * BitsOfByte).W),
            Mux(t_oobfill.asBool, Fill(8 * BitsOfByte, 1.U), 0.U((8 * BitsOfByte).W)))
          tmp_8(x / 8) := fill_val
        }
      }
    }
  }


  val s_idle :: s_getdata :: s_shared :: s_shared1 :: s_reset :: Nil = Enum(5)
  val state = RegInit(s_idle)

  io.from_l2cache.ready := !used_cache.andR && (state === s_idle || state === s_getdata)
  io.from_shared.ready := !io.from_addr.fire && state =/= s_reset
  io.from_addr.ready := state === s_idle && !used_inst.andR
  io.from_addr_tag.ready := !used_tag.andR && !io.from_l2cache.fire

  // Design invariant assertions: entry allocation must never fire when slots are full
  when(!reset.asBool) {
    assert(!(io.from_addr.fire && used_inst.andR),
      "DMA Temp_mem: inst entry allocated when all slots full")
    assert(!(io.from_addr_tag.fire && used_tag.andR),
      "DMA Temp_mem: tag entry allocated when all tag slots full")
    assert(!(io.from_l2cache.fire && used_cache.andR),
      "DMA Temp_mem: data entry allocated when all cache slots full")
  }
  io.to_shared.valid := state === s_shared
  io.inst_complete.valid := state === s_reset
  io.inst_mem_index := Mux(io.from_addr.fire, valid_inst_entry, 0.U)
  io.tag_mem_index := Mux(io.from_addr_tag.fire, valid_tag_entry, 0.U)

  // finish_cnt update on inst arrival
  when(io.from_addr.fire) {
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
      is(3.U) {
        finish_cnt(valid_inst_entry) := io.from_addr.bits.copysize >> log2Ceil(dma_aligned_bulk)
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
        datamem.write(valid_data_entry, from_l2cache_all)
        entry_index_reg(valid_data_entry) := current_inst_entry_index
      }
      when(io.from_addr_tag.fire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
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
      when(output_inst.funct === 3.U) {
        // Tensor mode: mask based on dim0 gather selection within the current row.
        val box_start = output_data.cacheline_info.box_dim0_start
        val datawidth = output_inst.tensorvars.datawidth
        val elementStride0 = output_inst.tensorvars.elementStrides(0)
        val dim0_stride_bytes = elementStride0 * datawidth
        val dim0_row_span_bytes = Mux(
          output_inst.tensorvars.boxDim(0) === 0.U,
          0.U,
          (output_inst.tensorvars.boxDim(0) - 1.U) * dim0_stride_bytes + datawidth,
        )
        val box_end = box_start + dim0_row_span_bytes
        (0 until numgroupl2cache).foreach { x =>
          val addr_start = tag_wire + (x.U * dma_aligned_bulk.U)
          val rel = addr_start - box_start
          val in_row_span = (addr_start >= box_start) && (addr_start < box_end)
          val is_selected = Mux(
            elementStride0 <= 1.U,
            true.B,
            (rel % dim0_stride_bytes) < datawidth,
          )
          mask_l2cache(x) := in_row_span && is_selected
        }
        // Apply OOB-filled data back
        switch(output_inst.tensorvars.datawidth) {
          is(1.U) { output_data.base.d_data := tmp_1.asTypeOf(output_data.base.d_data) }
          is(2.U) { output_data.base.d_data := tmp_2.asTypeOf(output_data.base.d_data) }
          is(4.U) { output_data.base.d_data := tmp_4.asTypeOf(output_data.base.d_data) }
          is(8.U) { output_data.base.d_data := tmp_8.asTypeOf(output_data.base.d_data) }
        }
      }.otherwise {
        // Linear copy: mark valid groups by [src, src+copysize)
        (0 until numgroupl2cache).foreach { x =>
          val addr_start = tag_wire + (x.U * dma_aligned_bulk.U)
          val condition = (addr_start >= output_inst.src) && (addr_start < (output_inst.src + output_inst.copysize))
          mask_l2cache(x) := condition
        }
      }
      when(io.from_l2cache.fire) {
        used_cache := used_cache.bitSet(valid_data_entry, true.B)
        used_tag := used_tag.bitSet(io.from_l2cache.bits.d_source(l1cache_sourceBits - 1, l1cache_sourceBits - log2Ceil(max_dma_tag)), false.B)
        datamem.write(valid_data_entry, from_l2cache_all)
        entry_index_reg(valid_data_entry) := current_inst_entry_index
      }
      when(io.from_addr_tag.fire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
      }
    }
    is(s_shared) {
      when(io.to_shared.fire) {
        mask_l2cache := mask_l2cache_next
      }
      when(io.from_addr_tag.fire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
      }
    }
    is(s_shared1) {
      when(mask_l2cache.asUInt === 0.U) {
        used_cache := used_cache.bitSet(output_data_entry, false.B)
        mask_l2cache := VecInit(Seq.fill(numgroupl2cache)(false.B))
        current_inst_entry_index_reg := 0.U
      }
      when(io.from_addr_tag.fire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
      }
    }
    is(s_reset) {
      used_inst := used_inst.bitSet(complete_inst_entry, false.B)
      finish_cnt(PriorityEncoder(complete)) := 1.U
      when(io.from_addr_tag.fire) {
        used_tag := used_tag.bitSet(valid_tag_entry, true.B)
        tagmem.write(valid_tag_entry, io.from_addr_tag.bits)
      }
    }
  }

  // Output to shared
  io.to_shared.bits.entry_index := current_inst_entry_index_reg
  io.to_shared.bits.mask := mask_shared
  (0 until numgroupshared).foreach(x =>
    when(x.U + mask_index < numgroupl2cache.U) {
      io.to_shared.bits.data(x) := output_data_4byte(x.U + mask_index)
    }.otherwise {
      io.to_shared.bits.data(x) := 0.U
    }
  )
  io.to_shared.bits.instinfo := output_inst
  io.to_shared.bits.cacheline_info.tag := tag_wire + mask_index * dma_aligned_bulk.U
  io.to_shared.bits.cacheline_info.tensor_dim0_start := output_data.cacheline_info.tensor_dim0_start
  io.to_shared.bits.cacheline_info.tensor_dim_step := output_data.cacheline_info.tensor_dim_step
  io.to_shared.bits.cacheline_info.box_dim0_start := output_data.cacheline_info.box_dim0_start
  io.to_shared.bits.cacheline_info.tensor_high_dim_valid := output_data.cacheline_info.tensor_high_dim_valid
  io.to_shared.bits.cacheline_info.shared_row_base := output_data.cacheline_info.shared_row_base
  io.to_shared.bits.cacheline_info.dim0_stride_bytes := output_data.cacheline_info.dim0_stride_bytes
  io.to_shared.bits.cacheline_info.swizzle_row_low := output_data.cacheline_info.swizzle_row_low
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

  val s_idle :: s_send :: Nil = Enum(2)
  val state = RegInit(s_idle)
  val reg_save = Reg(new DmaTempOutput)

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

  // Address calculation for each group
  val addr = Wire(Vec(numgroupshared, UInt(xLen.W)))
  // Default: linear address calculation
  (0 until numgroupshared).foreach { x =>
    addr(x) := reg_save.cacheline_info.tag - reg_save.instinfo.src + reg_save.instinfo.dst + (x.U * dma_aligned_bulk.U)
  }
  when(reg_save.instinfo.funct === 3.U) {
    val datawidth = reg_save.instinfo.tensorvars.datawidth
    val supportsPackedDim0Gather =
      (datawidth === dma_aligned_bulk.U) && (reg_save.instinfo.tensorvars.elementStrides(0) > 1.U)
    (0 until numgroupshared).foreach { x =>
      val srcDim0OffsetBytes = reg_save.cacheline_info.tag - reg_save.cacheline_info.box_dim0_start +
        x.U * dma_aligned_bulk.U
      val selectedBeforeInGroup = if (x == 0) 0.U else PopCount(reg_save.mask.take(x))
      val packedDim0OffsetBytes = Mux(
        supportsPackedDim0Gather,
        selectedBeforeInGroup << log2Ceil(dma_aligned_bulk),
        srcDim0OffsetBytes,
      )
      val logicalAddr = reg_save.cacheline_info.shared_row_base + packedDim0OffsetBytes
      addr(x) := swizzleSharedAddr(
        logicalAddr,
        reg_save.instinfo.dst,
        reg_save.instinfo.tensorvars.swizzleMode,
        reg_save.cacheline_info.swizzle_row_low
      )
    }
  }

  val current_tag = Wire(UInt(dcache_TagBits.W))
  val setIdx = Wire(UInt(dcache_SetIdxBits.W))
  val first_valid_addr = addr(PriorityEncoder(reg_save.mask.asUInt))
  current_tag := Mux(reg_save.mask.asUInt.orR, first_valid_addr(xLen - 1, xLen - dcache_TagBits), 0.U)
  setIdx := Mux(reg_save.mask.asUInt.orR, first_valid_addr(xLen - 1 - dcache_TagBits, xLen - dcache_TagBits - dcache_SetIdxBits), 0.U)

  val blockOffset = Wire(Vec(numgroupshared, UInt(dcache_BlockOffsetBits.W)))
  (0 until numgroupshared).foreach(x => blockOffset(x) := addr(x)(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits))

  val current_mask = Wire(Vec(numgroupshared, Bool()))
  (0 until numgroupshared).foreach(x =>
    current_mask(x) := reg_save.mask(x) &&
      (addr(x)(xLen - 1, xLen - dcache_TagBits) === current_tag) &&
      (addr(x)(xLen - 1 - dcache_TagBits, xLen - dcache_TagBits - dcache_SetIdxBits) === setIdx)
  )
  val mask_next = Wire(Vec(numgroupshared, Bool()))
  (0 until numgroupshared).foreach(x =>
    mask_next(x) := reg_save.mask(x) && !current_mask(x)
  )

  io.from_temp.ready := state === s_idle
  io.shared_req.valid := state === s_send
  io.shared_req.bits.instrId := reg_save.entry_index
  io.shared_req.bits.isWrite := true.B
  io.shared_req.bits.setIdx := setIdx
  (0 until numgroupshared).foreach(x => {
    io.shared_req.bits.perLaneAddr(x).blockOffset := blockOffset(x)
    io.shared_req.bits.perLaneAddr(x).wordOffset1H := Fill(BytesOfWord, 1.U)
    io.shared_req.bits.perLaneAddr(x).activeMask := current_mask(x)
    io.shared_req.bits.data(x) := reg_save.data(x)
  })

  switch(state) {
    is(s_idle) {
      when(io.from_temp.fire) {
        reg_save := io.from_temp.bits
        state := s_send
      }
    }
    is(s_send) {
      when(io.shared_req.fire) {
        when(mask_next.asUInt === 0.U) {
          state := s_idle
        }.otherwise {
          reg_save.mask := mask_next
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
    val shared_rsp = Flipped(DecoupledIO(new DCacheCoreRsp_np))
    val dma_cache_req = DecoupledIO(new DCacheMemReq_p)
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
    val fence_end_dma = DecoupledIO(UInt(depth_warp.W))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
  })

  // Input FIFO
  val InputFIFO = Module(new Queue(new vExeData, entries = 1, pipe = true))
  InputFIFO.io.enq <> io.dma_req

  // Address calculator -> L2 cache
  val addrCalc_l2cache = Module(new AddrCalc_l2cache)
  addrCalc_l2cache.io.from_fifo <> InputFIFO.io.deq
  io.dma_cache_req <> addrCalc_l2cache.io.to_l2cache

  // TLB passthrough
  io.to_l2TLB <> addrCalc_l2cache.io.to_l2TLB
  addrCalc_l2cache.io.from_l2TLB <> io.from_l2TLB

  // Temporary memory
  val tempmem = Module(new Temp_mem)
  tempmem.io.from_addr <> addrCalc_l2cache.io.to_tempmem_inst
  tempmem.io.from_addr_tag <> addrCalc_l2cache.io.to_tempmem_tag
  addrCalc_l2cache.io.inst_mem_index := tempmem.io.inst_mem_index
  addrCalc_l2cache.io.tag_mem_index := tempmem.io.tag_mem_index
  val dma_rsp_is_meta = io.dma_cache_rsp.bits.d_source(0)
  addrCalc_l2cache.io.from_l2cache_meta.valid := io.dma_cache_rsp.valid && dma_rsp_is_meta
  addrCalc_l2cache.io.from_l2cache_meta.bits := io.dma_cache_rsp.bits
  tempmem.io.from_l2cache.valid := io.dma_cache_rsp.valid && !dma_rsp_is_meta
  tempmem.io.from_l2cache.bits := io.dma_cache_rsp.bits
  io.dma_cache_rsp.ready := Mux(dma_rsp_is_meta,
    addrCalc_l2cache.io.from_l2cache_meta.ready,
    tempmem.io.from_l2cache.ready)
  tempmem.io.from_shared <> io.shared_rsp
  val dmaCompleteArb = Module(new Arbiter(UInt(depth_warp.W), 2))
  dmaCompleteArb.io.in(0) <> tempmem.io.inst_complete
  dmaCompleteArb.io.in(1) <> addrCalc_l2cache.io.meta_complete
  io.fence_end_dma <> dmaCompleteArb.io.out

  // Address calculator -> shared memory
  val addrCalc_shared = Module(new Addrcalc_shared)
  addrCalc_shared.io.from_temp <> tempmem.io.to_shared
  io.shared_req <> addrCalc_shared.io.shared_req

  // ---- Debug printf for DMA E2E RTL verification ----
  when(io.dma_req.fire) {
    printf(p"[DMA] req fire: funct=${io.dma_req.bits.ctrl.funct} in1=0x${Hexadecimal(io.dma_req.bits.in1(0))} in2=0x${Hexadecimal(io.dma_req.bits.in2(0))} in3=0x${Hexadecimal(io.dma_req.bits.in3(0))} wid=${io.dma_req.bits.ctrl.wid}\n")
  }
  when(io.dma_cache_req.fire) {
    printf(p"[DMA] L2 req fire: addr=0x${Hexadecimal(io.dma_cache_req.bits.a_addr.get)} source=${io.dma_cache_req.bits.a_source}\n")
  }
  when(io.dma_cache_rsp.fire) {
    printf(p"[DMA] L2 rsp fire: source=${io.dma_cache_rsp.bits.d_source}\n")
  }
  when(io.shared_req.fire) {
    printf(p"[DMA] shared_req fire: setIdx=${io.shared_req.bits.setIdx} isWrite=${io.shared_req.bits.isWrite}\n")
  }
  when(io.fence_end_dma.fire) {
    printf(p"[DMA] fence_end_dma fire: wid=${io.fence_end_dma.bits}\n")
  }

}
