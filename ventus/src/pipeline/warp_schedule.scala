/*
 * Copyright (c) 2021-2022 International Innovation Center of Tsinghua University, Shanghai
 * Ventus is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details. */
package pipeline

import chisel3._
import chisel3.util._
import top.parameters._
import gvm._

class warp_scheduler extends Module{
  val io = IO(new Bundle{
    val pc_reset = Input(Bool())
    val warpReq=Flipped(Decoupled(new warpReqData)) //new warp
    val warpRsp=Decoupled(new warpRspData) //endprg
    val wg_id_lookup=Output(UInt(depth_warp.W)) //lookup CTA
    val wg_id_tag=Input(UInt(TAG_WIDTH.W))  //barrier related
    val pc_req=Decoupled(new ICachePipeReq_np) //should flush icache
    val pc_rsp=Flipped(Valid(new ICachePipeRsp_np)) //icache miss state
    val branch = Flipped(DecoupledIO(new BranchCtrl)) //branch, flush pipeline
    val warp_control=Flipped(DecoupledIO(new warpSchedulerExeData)) //engprg and barrier
    val dma_issue = Flipped(ValidIO(UInt(depth_warp.W)))
    val dma_complete = Flipped(ValidIO(new DmaCompletion))
    val dma_issue_allow = Output(Vec(num_warp, Bool()))
    val dma_issue_group = Output(Vec(num_warp, UInt(log2Ceil(dma_group_entries).W)))
    val issued_warp=Flipped(Valid(UInt(depth_warp.W))) //not use
    val scoreboard_busy=Input(UInt(num_warp.W)) //scoreboard race
    val exe_busy=Input(UInt(num_warp.W)) //exe race
    //val pc_icache_ready=Input(Vec(num_warp,Bool()))
    val pc_ibuffer_ready=Input(Vec(num_warp,UInt(depth_ibuffer.W))) //ibuffer ready
    val asid =  if(MMU_ENABLED) Some(Output(UInt(KNL_ASID_WIDTH.W))) else None // 2ibuffer
    val warp_ready=Output(UInt(num_warp.W)) //to issue
    val barrier_busy = Output(UInt(num_warp.W))
    val flush=(ValidIO(UInt(depth_warp.W)))
    val flushCache=(ValidIO(UInt(depth_warp.W)))
    val CTA2csr=ValidIO(new warpReqData) //redirect warpreq
    val dma_fence_wait_dbg = Output(UInt(num_warp.W))
    val dma_inflight_dbg = Output(Vec(num_warp, UInt(log2Ceil(max_dma_inst + 1).W)))
    //val ldst = Input(new warp_schedule_ldst_io()) // assume finish l2cache request
    //val switch = Input(Bool()) // assume coming from LDST unit (or other unit)
    val flushDCache = Decoupled(Bool())
    // val inquire_csr_wid = Output(UInt(depth_warp.W))
    // val inquire_csr_addr = Output(UInt(12.W))
    // val inquire_csr_data = Input(UInt(xLen.W))
  })

  val warp_end=io.warp_control.fire&io.warp_control.bits.ctrl.simt_stack_op
  val warp_end_id=io.warp_control.bits.ctrl.wid
  val is_dma_fence = io.warp_control.bits.ctrl.dma && io.warp_control.bits.ctrl.funct === 6.U
  val warp_ctrl_is_dma_fence = io.warp_control.fire && is_dma_fence
  val warp_ctrl_is_barrier = io.warp_control.fire && !io.warp_control.bits.ctrl.simt_stack_op && !is_dma_fence
  val current_warp=RegInit(0.U(depth_warp.W))
  val next_warp=WireInit(current_warp)
  io.branch.ready:= !io.flushCache.valid
  io.warp_control.ready:= !io.branch.fire & !io.flushCache.valid

  io.warpReq.ready:=true.B
  io.warpRsp.valid:=warp_end // always ready.
  io.warpRsp.bits.wid:=warp_end_id

  io.CTA2csr.bits:=io.warpReq.bits
  io.CTA2csr.valid:=io.warpReq.valid

  val new_warpid = io.warpReq.bits.wid

  if(MMU_ENABLED){
    val asidReg = Reg(Vec(num_warp,UInt(KNL_ASID_WIDTH.W)))
    when(io.warpReq.fire){
      asidReg(new_warpid) := io.warpReq.bits.CTAdata.dispatch2cu_knl_asid.getOrElse(0.U).asUInt
    }
    io.asid.get := asidReg(io.pc_rsp.bits.warpid)
    io.pc_req.bits.asid.get := asidReg(next_warp)
  }


  io.flush.valid:=(io.branch.fire&io.branch.bits.jump) | warp_end//(暂定barrier不flush)
  io.flush.bits:=Mux((io.branch.fire&io.branch.bits.jump),io.branch.bits.wid,warp_end_id)
  io.flushCache.valid:=io.pc_rsp.valid&io.pc_rsp.bits.status(0)
  io.flushCache.bits:=io.pc_rsp.bits.warpid

  val pcControl=VecInit(Seq.fill(num_warp)(Module(new PCcontrol()).io))
  //val pcReplay=VecInit(pcControl.map(x=>RegEnable(x.PC_next,(x.PC_src===2.U)&(!x.PC_replay))))
  //val warp_memory_idle=Reg(Vec(num_warp,Bool()))
  //val warp_barrier_array=RegInit(0.U(num_warp.W))
  //val block_warp_waiting=RegInit(VecInit(Seq.fill(num_block)(0.U(num_warp.W)))) // if meet barrier, switch and set 1. all 1 -> all 0.
  val warp_init_addr=(VecInit(Seq.fill(num_warp)(0.U(32.W))))//,172.U,176.U) //初值怎么传进去，这是个问题？建议走CSR，并且是vec version的
  pcControl.foreach{
    x=>{
      x.New_PC:=io.branch.bits.new_pc
      x.PC_replay:=true.B
      x.PC_src:=0.U
      x.mask_i:=0.U
    }
  }
  val pc_ready=Wire(Vec(num_warp,Bool()))


  current_warp:=next_warp
  pcControl(next_warp).PC_replay:= (!io.pc_req.ready)|(!pc_ready(next_warp))
  pcControl(next_warp).PC_src:=2.U
  io.pc_req.bits.addr := pcControl(next_warp).PC_next
  io.pc_req.bits.warpid := next_warp
  io.pc_req.bits.mask := pcControl(next_warp).mask_o


  io.wg_id_lookup:=Mux(!io.warp_control.bits.ctrl.simt_stack_op,warp_end_id,io.warpRsp.bits.wid) //barrier的时候没有warp_end，只是叫这个名字

  val warp_bar_cur=RegInit(VecInit(Seq.fill(num_block)(0.U(num_warp_in_a_block.W))))
  val warp_bar_exp=RegInit(VecInit(Seq.fill(num_block)(0.U(num_warp_in_a_block.W))))
  val warp_endprg_cnt = RegInit(VecInit(Seq.fill(num_block)(0.U(num_warp_in_a_block.W))))
  val warp_wg_valid = RegInit(VecInit(Seq.fill(num_block)(false.B)))
  val warp_endprg_mask_0 = WireInit(VecInit(Seq.fill(num_block)(false.B)))
  //val warp_bar_cur_next=warp_bar_cur
  //val warp_bar_exp_next=warp_bar_exp
  val WF_ID_WIDTH = log2Ceil(num_warp_in_a_block)
  val warp_bar_lock=WireInit(VecInit(Seq.fill(num_block)(false.B))) //equals to "active block"
  val new_wg_id=io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch(TAG_WIDTH-1, WF_ID_WIDTH)
  val new_wf_id=io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch(WF_ID_WIDTH-1,0)
  val new_wg_wf_count=io.warpReq.bits.CTAdata.dispatch2cu_wg_wf_count
  val end_wg_id=io.wg_id_tag(TAG_WIDTH-1, WF_ID_WIDTH)
  val end_wf_id=io.wg_id_tag(WF_ID_WIDTH-1, 0)
  val warp_bar_data=RegInit(0.U(num_warp.W))  // 0 means not locked by barrier
  val warp_bar_belong=RegInit(VecInit(Seq.fill(num_block)(0.U(num_warp.W))))


  when(io.warpReq.fire){
    warp_bar_belong(new_wg_id):=warp_bar_belong(new_wg_id) | (1.U<<io.warpReq.bits.wid).asUInt  //显示warp中有哪些属于wg
//    warp_bar_exp(new_wg_id):= warp_bar_exp(new_wg_id) | (1.U<<io.warpReq.bits.wid).asUInt
    when(!warp_bar_lock(new_wg_id)) {
      warp_bar_cur(new_wg_id) := 0.U
      warp_bar_exp(new_wg_id) := (1.U << new_wg_wf_count).asUInt - 1.U  // init to 1 for all future wfs in wg
    }
  }
  when(io.warpRsp.fire){
//    warp_bar_exp(end_wg_id):=warp_bar_exp(end_wg_id) & (~(1.U<<io.warpRsp.bits.wid)).asUInt
    warp_bar_belong(end_wg_id):=warp_bar_belong(end_wg_id) & (~(1.U<<io.warpRsp.bits.wid)).asUInt
  }
  warp_bar_lock:=warp_bar_belong.map(x=>x.orR)
  when(warp_ctrl_is_barrier){ // means traditional barrier
    warp_bar_cur(end_wg_id):=warp_bar_cur(end_wg_id) | (1.U<<end_wf_id).asUInt
    warp_bar_data:=warp_bar_data | (1.U<<io.warp_control.bits.ctrl.wid).asUInt
    when((warp_bar_cur(end_wg_id) | (1.U<<end_wf_id).asUInt) === warp_bar_exp(end_wg_id)){
      warp_bar_cur(end_wg_id):=0.U
      warp_bar_data:=warp_bar_data & (~warp_bar_belong(end_wg_id)).asUInt
      if(GVM_ENABLED) {
        val bar_fire_cond = warp_ctrl_is_barrier &&
                    ((warp_bar_cur(end_wg_id) | (1.U<<end_wf_id).asUInt) === warp_bar_exp(end_wg_id))
        val gvm_bar_done = Module(new GvmDutBarrierDone)
        gvm_bar_done.io.clock := clock
        gvm_bar_done.io.reset := reset.asBool
        gvm_bar_done.io.bar_done_fire := bar_fire_cond
        gvm_bar_done.io.sm_id := io.warp_control.bits.ctrl.spike_info.get.sm_id.pad(32)
        gvm_bar_done.io.wg_slot_id := end_wg_id
        gvm_bar_done.io.pc := io.warp_control.bits.ctrl.spike_info.get.pc.pad(32)
        gvm_bar_done.io.inst := io.warp_control.bits.ctrl.spike_info.get.inst.pad(32)
        gvm_bar_done.io.dispatch_id := io.warp_control.bits.ctrl.spike_info.get.dispatch_id.get
      }
    }
  }
  // collect endprg in one wg and issue flush request
  when(io.warpReq.fire){
    warp_endprg_cnt(new_wg_id):=warp_endprg_cnt(new_wg_id) | (1.U<<io.warpReq.bits.wid).asUInt
    warp_wg_valid(new_wg_id):=true.B
  }
  when(io.warpRsp.fire){
    warp_endprg_cnt(end_wg_id) := warp_endprg_cnt(end_wg_id) & (~(1.U<<io.warpRsp.bits.wid)).asUInt
  }
  for(i<-0 until num_block){
    warp_endprg_mask_0(i) := (warp_endprg_cnt(i).orR === false.B) && warp_wg_valid(i)
  }
  val need_flush = warp_endprg_mask_0.asUInt.orR
  val flush_entry = OHToUInt(warp_endprg_mask_0.asUInt)
  when(warp_endprg_mask_0(flush_entry) && io.flushDCache.ready){
    warp_wg_valid(flush_entry) := false.B
  }
  io.flushDCache.valid := need_flush
  io.flushDCache.bits := need_flush


  val warp_active=RegInit(0.U(num_warp.W))
  require((dma_group_entries & (dma_group_entries - 1)) == 0,
    "DMA group wait ring expects a power-of-two group count")
  val dmaInflightWidth = log2Ceil(max_dma_inst + 1)
  val dmaGroupWidth = log2Ceil(dma_group_entries)
  val dmaGroupKeepWidth = log2Ceil(dma_group_entries + 1)
  val maxInflightPerWarp = max_dma_inst.U(dmaInflightWidth.W)
  val dma_inflight_cnt = RegInit(VecInit(Seq.fill(num_warp)(0.U(dmaInflightWidth.W))))
  val dma_fence_wait = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val dma_wait_target = RegInit(VecInit(Seq.fill(num_warp)(0.U(dmaInflightWidth.W))))
  // A per-warp group ring tracks all DMA operations. DMA instructions begin
  // execution immediately and are assigned to the current open group; commit
  // closes that group and advances the issue pointer, while wait_group keeps
  // only the newest committed groups outstanding.
  val dma_group_issue_ptr = RegInit(VecInit(Seq.fill(num_warp)(0.U(dmaGroupWidth.W))))
  val dma_group_open = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val dma_group_wait = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val dma_group_wait_keep = RegInit(VecInit(Seq.fill(num_warp)(0.U(dmaGroupKeepWidth.W))))
  val dma_group_committed = RegInit(VecInit(Seq.fill(num_warp)(
    VecInit(Seq.fill(dma_group_entries)(false.B))
  )))
  val dma_group_count = RegInit(VecInit(Seq.fill(num_warp)(
    VecInit(Seq.fill(dma_group_entries)(0.U(dmaInflightWidth.W)))
  )))
  val dma_issue_allow = Wire(Vec(num_warp, Bool()))
  val dma_fence_zimm = io.warp_control.bits.in1(4, 0)
  val dma_fence_wait_count = dma_fence_zimm.pad(dmaInflightWidth)(dmaInflightWidth - 1, 0)
  val dma_fence_group_commit = dma_fence_zimm === 16.U
  val dma_fence_group_wait = dma_fence_zimm(4) && dma_fence_zimm(3)
  val dma_fence_legacy_wait = !dma_fence_group_commit && !dma_fence_group_wait
  val dma_fence_group_keep = dma_fence_zimm(2, 0).pad(dmaGroupKeepWidth)(dmaGroupKeepWidth - 1, 0)

  def groupBehind(ptr: UInt, distance: Int): UInt =
    (ptr - distance.U)(dmaGroupWidth - 1, 0)

  def groupIsRecent(ptr: UInt, group: UInt, keep: UInt): Bool =
    (1 to dma_group_entries).map { d =>
      keep >= d.U && group === groupBehind(ptr, d)
    }.reduce(_ || _)

  for (i <- 0 until num_warp) {
    val issue_hit = io.dma_issue.valid && io.dma_issue.bits === i.U
    val complete_hit = io.dma_complete.valid && io.dma_complete.bits.wid === i.U
    val can_inc = dma_inflight_cnt(i) =/= maxInflightPerWarp
    val group_issue_slot_free = !dma_group_committed(i)(dma_group_issue_ptr(i))
    val allow_issue = (can_inc || complete_hit) && group_issue_slot_free
    val inc_en = issue_hit && allow_issue
    val dec_en = complete_hit && dma_inflight_cnt(i) =/= 0.U
    val cnt_after_io = dma_inflight_cnt(i) + inc_en.asUInt - dec_en.asUInt
    val fence_issue_here = warp_ctrl_is_dma_fence && io.warp_control.bits.ctrl.wid === i.U
    val fence_legacy_here = fence_issue_here && dma_fence_legacy_wait
    val fence_group_commit_here = fence_issue_here && dma_fence_group_commit
    val fence_group_wait_here = fence_issue_here && dma_fence_group_wait
    val fence_wait_all = dma_fence_wait_count === 0.U || dma_fence_wait_count >= cnt_after_io
    val fence_wait_count = Mux(fence_wait_all, cnt_after_io, dma_fence_wait_count)
    val fence_target = cnt_after_io - fence_wait_count
    val fence_done = cnt_after_io <= dma_wait_target(i)
    val group_count_next = Wire(Vec(dma_group_entries, UInt(dmaInflightWidth.W)))
    val group_committed_next = Wire(Vec(dma_group_entries, Bool()))
    val commit_group_has_work = dma_group_open(i) || dma_group_count(i)(dma_group_issue_ptr(i)) =/= 0.U
    val commit_group_advances = fence_group_commit_here && commit_group_has_work
    val group_ptr_next = Mux(
      commit_group_advances,
      (dma_group_issue_ptr(i) + 1.U)(dmaGroupWidth - 1, 0),
      dma_group_issue_ptr(i)
    )

    for (g <- 0 until dma_group_entries) {
      val group_inc = inc_en && dma_group_issue_ptr(i) === g.U
      val group_dec = complete_hit && io.dma_complete.bits.group === g.U && dma_group_count(i)(g) =/= 0.U
      group_count_next(g) := dma_group_count(i)(g) + group_inc.asUInt - group_dec.asUInt
      group_committed_next(g) := dma_group_committed(i)(g)
      when(group_count_next(g) === 0.U) {
        group_committed_next(g) := false.B
      }
      when(fence_group_commit_here && commit_group_has_work &&
           dma_group_issue_ptr(i) === g.U && group_count_next(g) =/= 0.U) {
        group_committed_next(g) := true.B
      }
    }

    val group_wait_blocked_saved = VecInit((0 until dma_group_entries).map { g =>
      group_committed_next(g) && group_count_next(g) =/= 0.U &&
        !groupIsRecent(dma_group_issue_ptr(i), g.U, dma_group_wait_keep(i))
    }).asUInt.orR
    val group_wait_blocked_new = VecInit((0 until dma_group_entries).map { g =>
      group_committed_next(g) && group_count_next(g) =/= 0.U &&
        !groupIsRecent(dma_group_issue_ptr(i), g.U, dma_fence_group_keep)
    }).asUInt.orR
    val group_wait_done_saved = dma_group_wait_keep(i) >= dma_group_entries.U || !group_wait_blocked_saved
    val group_wait_done_new = dma_fence_group_keep >= dma_group_entries.U || !group_wait_blocked_new
    val group_open_after_commit = Mux(commit_group_advances, false.B, dma_group_open(i) || inc_en)
    val group_open_next = group_open_after_commit && group_count_next(group_ptr_next) =/= 0.U
    val clear_group_state = io.pc_reset || (io.warpReq.fire && io.warpReq.bits.wid === i.U)

    dma_issue_allow(i) := allow_issue

    when(!reset.asBool) {
      assert(!(issue_hit && !allow_issue), s"DMA inflight overflow on warp $i")
      assert(!(complete_hit && dma_inflight_cnt(i) === 0.U && !issue_hit), s"DMA inflight underflow on warp $i")
      assert(!(complete_hit && dma_group_count(i)(io.dma_complete.bits.group) === 0.U &&
        !(inc_en && dma_group_issue_ptr(i) === io.dma_complete.bits.group)),
        s"DMA group counter underflow on warp $i")
      when(io.warpReq.fire && io.warpReq.bits.wid === i.U) {
        assert(dma_inflight_cnt(i) === 0.U, s"Warp reuse before DMA inflight drains on warp $i")
        assert(!dma_fence_wait(i), s"Warp reuse while DMA fence wait is still set on warp $i")
        assert(!dma_group_wait(i), s"Warp reuse while DMA group wait is still set on warp $i")
      }
    }

    dma_inflight_cnt(i) := Mux(io.pc_reset, 0.U, cnt_after_io)
    dma_wait_target(i) := Mux(
      io.pc_reset,
      0.U,
      Mux(
        fence_legacy_here,
        fence_target,
        Mux(fence_done, 0.U, dma_wait_target(i))
      )
    )
    dma_fence_wait(i) := Mux(
      io.pc_reset,
      false.B,
      Mux(
        fence_legacy_here,
        fence_target =/= cnt_after_io,
        Mux(fence_done, false.B, dma_fence_wait(i))
      )
    )
    dma_group_issue_ptr(i) := Mux(clear_group_state, 0.U, group_ptr_next)
    dma_group_open(i) := Mux(clear_group_state, false.B, group_open_next)
    dma_group_wait_keep(i) := Mux(
      clear_group_state,
      0.U,
      Mux(fence_group_wait_here, dma_fence_group_keep, dma_group_wait_keep(i))
    )
    dma_group_wait(i) := Mux(
      clear_group_state,
      false.B,
      Mux(
        fence_group_wait_here,
        !group_wait_done_new,
        Mux(group_wait_done_saved, false.B, dma_group_wait(i))
      )
    )
    for (g <- 0 until dma_group_entries) {
      dma_group_count(i)(g) := Mux(clear_group_state, 0.U, group_count_next(g))
      dma_group_committed(i)(g) := Mux(clear_group_state, false.B, group_committed_next(g))
    }
  }

  io.dma_issue_allow := dma_issue_allow
  io.dma_issue_group := dma_group_issue_ptr



  warp_active:=(warp_active | ((1.U<<io.warpReq.bits.wid).asUInt&Fill(num_warp,io.warpReq.fire))) & (~( Fill(num_warp,warp_end)&(1.U<<warp_end_id).asUInt )).asUInt
  val dma_fence_wait_bits = Cat(dma_fence_wait.reverse)
  val dma_group_wait_bits = Cat(dma_group_wait.reverse)
  val dma_wait_bits = dma_fence_wait_bits | dma_group_wait_bits
  val warp_ready=(~(warp_bar_data | io.scoreboard_busy | io.exe_busy | (~warp_active).asUInt | dma_wait_bits)).asUInt
  io.warp_ready:=warp_ready
  io.barrier_busy := warp_bar_data
  io.dma_fence_wait_dbg := dma_wait_bits
  io.dma_inflight_dbg := dma_inflight_cnt
  for (i<- num_warp-1 to 0 by -1){
    pc_ready(i):= io.pc_ibuffer_ready(i) & warp_active(i) 
    when(pc_ready(i)){next_warp:=i.asUInt}
  }
  io.pc_req.valid:=pc_ready(next_warp)
  //lock one warp to execute
  //next_warp:=0.U
  if(SINGLE_INST) next_warp:=0.U



  when(io.pc_rsp.valid&io.pc_rsp.bits.status(0)){//miss acknowledgement
    pcControl(io.pc_rsp.bits.warpid).PC_replay:=false.B
    pcControl(io.pc_rsp.bits.warpid).PC_src:=3.U
    pcControl(io.pc_rsp.bits.warpid).New_PC:=io.pc_rsp.bits.addr//pcReplay(io.pc_rsp.bits.warpid)
    pcControl(io.pc_rsp.bits.warpid).mask_i:=io.pc_rsp.bits.mask
  }

  when(io.branch.fire&io.branch.bits.jump){
    pcControl(io.branch.bits.wid).PC_replay:=false.B
    pcControl(io.branch.bits.wid).PC_src:=1.U
    pcControl(io.branch.bits.wid).New_PC:=io.branch.bits.new_pc
    //PC:=io.branch.bits.new_pc
    when(io.branch.bits.wid===next_warp){
    io.pc_req.valid:=false.B}
  }


  when(io.warpReq.fire){
    pcControl(io.warpReq.bits.wid).PC_replay:=false.B
    pcControl(io.warpReq.bits.wid).PC_src:=1.U
    pcControl(io.warpReq.bits.wid).New_PC:=io.warpReq.bits.CTAdata.dispatch2cu_start_pc_dispatch
  }


  when(io.pc_reset){
    pcControl.zipWithIndex.foreach{case(x,b)=>{x.PC_src:=1.U;x.New_PC:=warp_init_addr(b);x.PC_replay:=false.B} }
    io.pc_req.valid:=false.B
  }
}
