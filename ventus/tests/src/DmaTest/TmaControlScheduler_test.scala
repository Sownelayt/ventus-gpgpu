package DmaTest

import L1Cache.MyConfig
import chisel3._
import chiseltest._
import config.config.Parameters
import org.scalatest.freespec.AnyFreeSpec
import pipeline.{TmaV2Spec, warp_scheduler}
import top.parameters.num_warp

class TmaControlScheduler_test extends AnyFreeSpec with ChiselScalatestTester {
  implicit val p: Parameters = (new MyConfig).toInstance

  private def bit(value: BigInt, idx: Int): BigInt = (value >> idx) & 1

  private def init(dut: warp_scheduler): Unit = {
    dut.io.pc_reset.poke(false.B)
    dut.io.warpReq.valid.poke(false.B)
    dut.io.warpReq.bits.wid.poke(0.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wg_wf_count.poke(0.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wf_size_dispatch.poke(0.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch.poke(0.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_start_pc_dispatch.poke(0.U)
    dut.io.warpRsp.ready.poke(true.B)
    dut.io.wg_id_tag.poke(0.U)
    dut.io.pc_req.ready.poke(true.B)
    dut.io.pc_rsp.valid.poke(false.B)
    dut.io.pc_rsp.bits.warpid.poke(0.U)
    dut.io.pc_rsp.bits.addr.poke(0.U)
    dut.io.pc_rsp.bits.data.poke(0.U)
    dut.io.pc_rsp.bits.mask.poke(0.U)
    dut.io.pc_rsp.bits.status.poke(0.U)
    dut.io.branch.valid.poke(false.B)
    dut.io.branch.bits.wid.poke(0.U)
    dut.io.branch.bits.jump.poke(false.B)
    dut.io.branch.bits.new_pc.poke(0.U)
    dut.io.warp_control.valid.poke(false.B)
    dut.io.warp_control.bits.ctrl.wid.poke(0.U)
    dut.io.warp_control.bits.ctrl.simt_stack_op.poke(false.B)
    dut.io.warp_control.bits.ctrl.barrier.poke(false.B)
    dut.io.warp_control.bits.ctrl.dma.poke(false.B)
    dut.io.warp_control.bits.ctrl.inst.poke(0.U)
    dut.io.warp_control.bits.ctrl.funct.poke(0.U)
    dut.io.warp_control.bits.in1.poke(0.U)
    dut.io.warp_control.bits.in2.poke(0.U)
    dut.io.dma_group_cmd.ready.poke(true.B)
    dut.io.dma_group_wait.poke(0.U)
    dut.io.dma_group_inflight.foreach(_.poke(0.U))
    dut.io.dma_sync_cmd.ready.poke(true.B)
    dut.io.dma_sync_wait.poke(0.U)
    dut.io.dma_mbarrier_owner_busy.poke(0.U)
    dut.io.lsu_fence_end.poke(((BigInt(1) << num_warp) - 1).U)
    dut.io.issued_warp.valid.poke(false.B)
    dut.io.issued_warp.bits.poke(0.U)
    dut.io.scoreboard_busy.poke(0.U)
    dut.io.exe_busy.poke(0.U)
    dut.io.pc_ibuffer_ready.foreach(_.poke(1.U))
    dut.io.flushDCache.ready.poke(true.B)
    dut.clock.step(2)
  }

  private def activate(dut: warp_scheduler, wid: Int): Unit = {
    dut.io.warpReq.valid.poke(true.B)
    dut.io.warpReq.bits.wid.poke(wid.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wg_wf_count.poke(1.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wf_size_dispatch.poke(1.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch.poke(0.U)
    dut.io.warpReq.bits.CTAdata.dispatch2cu_start_pc_dispatch.poke("h80000000".U)
    dut.clock.step()
    dut.io.warpReq.valid.poke(false.B)
  }

  private def issueControl(dut: warp_scheduler, wid: Int, funct: Int,
                           subop: Int, in1: Int = 0, in2: Int = 0): Unit = {
    dut.io.warp_control.valid.poke(true.B)
    dut.io.warp_control.bits.ctrl.wid.poke(wid.U)
    dut.io.warp_control.bits.ctrl.dma.poke(true.B)
    dut.io.warp_control.bits.ctrl.barrier.poke(true.B)
    dut.io.warp_control.bits.ctrl.inst.poke((BigInt(subop) << 25).U)
    dut.io.warp_control.bits.ctrl.funct.poke(funct.U)
    dut.io.warp_control.bits.in1.poke(in1.U)
    dut.io.warp_control.bits.in2.poke(in2.U)
    while (!dut.io.warp_control.ready.peekBoolean()) dut.clock.step()
    dut.clock.step()
    dut.io.warp_control.valid.poke(false.B)
  }

  private def groupControl(dut: warp_scheduler, wid: Int, zimm: Int): Unit =
    issueControl(dut, wid, TmaV2Spec.FunctS2GGroup, 0, in1 = zimm)

  "group controls are forwarded to the independent TMA tracker" in {
    test(new warp_scheduler) { dut =>
      init(dut); activate(dut, 0)
      dut.io.dma_group_cmd.ready.poke(false.B)
      dut.io.warp_control.valid.poke(true.B)
      dut.io.warp_control.bits.ctrl.wid.poke(0.U)
      dut.io.warp_control.bits.ctrl.dma.poke(true.B)
      dut.io.warp_control.bits.ctrl.funct.poke(TmaV2Spec.FunctS2GGroup.U)
      dut.io.warp_control.bits.in1.poke(TmaV2Spec.S2GGroupWaitBase.U)
      assert(dut.io.dma_group_cmd.valid.peekBoolean())
      assert(!dut.io.warp_control.ready.peekBoolean())
      dut.io.dma_group_cmd.bits.wid.expect(0.U)
      dut.io.dma_group_cmd.bits.zimm.expect(TmaV2Spec.S2GGroupWaitBase.U)
    }
  }

  "proxy fence blocks only its warp until that warp LSU ShiftBoard drains" in {
    test(new warp_scheduler) { dut =>
      init(dut); activate(dut, 0); activate(dut, 1)
      val allButWarp0 = ((BigInt(1) << num_warp) - 1) & ~BigInt(1)
      dut.io.lsu_fence_end.poke(allButWarp0.U)
      issueControl(dut, 0, TmaV2Spec.FunctMbarrierProxy,
        TmaV2Spec.FenceProxyAsyncShared)
      assert(bit(dut.io.warp_ready.peekInt(), 0) == 0)
      assert(bit(dut.io.warp_ready.peekInt(), 1) == 1)
      dut.io.lsu_fence_end.poke(((BigInt(1) << num_warp) - 1).U)
      dut.clock.step()
      assert(bit(dut.io.warp_ready.peekInt(), 0) == 1)
    }
  }

  "mbarrier controls forward both scalar operands without entering DMA data path" in {
    test(new warp_scheduler) { dut =>
      init(dut); activate(dut, 0)
      dut.io.warp_control.valid.poke(true.B)
      dut.io.warp_control.bits.ctrl.wid.poke(0.U)
      dut.io.warp_control.bits.ctrl.dma.poke(true.B)
      dut.io.warp_control.bits.ctrl.barrier.poke(true.B)
      dut.io.warp_control.bits.ctrl.inst.poke(
        (BigInt(TmaV2Spec.MbarrierArriveExpectTx) << 25).U)
      dut.io.warp_control.bits.ctrl.funct.poke(TmaV2Spec.FunctMbarrierProxy.U)
      dut.io.warp_control.bits.in1.poke("h70000040".U)
      dut.io.warp_control.bits.in2.poke(128.U)
      dut.io.wg_id_tag.poke((3 << 3).U)
      assert(dut.io.dma_sync_cmd.valid.peekBoolean())
      assert(dut.io.dma_sync_cmd.bits.owner.peekInt() == 3)
      assert(dut.io.dma_sync_cmd.bits.address.peekInt() == BigInt("70000040", 16))
      assert(dut.io.dma_sync_cmd.bits.value.peekInt() == 128)
      dut.clock.step()
    }
  }

  "the last warp cannot release its WG owner while mbarrier state is busy" in {
    test(new warp_scheduler) { dut =>
      init(dut)
      activate(dut, 0)
      dut.io.wg_id_tag.poke((2 << 3).U)
      dut.io.dma_mbarrier_owner_busy.poke((1 << 2).U)
      dut.io.warp_control.valid.poke(true.B)
      dut.io.warp_control.bits.ctrl.wid.poke(0.U)
      dut.io.warp_control.bits.ctrl.simt_stack_op.poke(true.B)
      assert(!dut.io.warp_control.ready.peekBoolean())
      assert(!dut.io.warpRsp.valid.peekBoolean())
      assert(!dut.io.dma_mbarrier_release.valid.peekBoolean())

      dut.io.dma_mbarrier_owner_busy.poke(0.U)
      assert(dut.io.warp_control.ready.peekBoolean())
      assert(dut.io.warpRsp.valid.peekBoolean())
      assert(dut.io.dma_mbarrier_release.valid.peekBoolean())
      assert(dut.io.dma_mbarrier_release.bits.peekInt() == 2)
      dut.clock.step()
      dut.io.warp_control.valid.poke(false.B)
    }
  }
}
