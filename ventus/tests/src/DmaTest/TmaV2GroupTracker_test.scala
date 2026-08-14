package DmaTest

import chisel3._
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import pipeline.{TmaV2S2GGroupTracker, TmaV2Spec}
import top.parameters.dma_group_entries

class TmaV2GroupTracker_test extends AnyFreeSpec with ChiselScalatestTester {
  private def init(dut: TmaV2S2GGroupTracker): Unit = {
    dut.io.control.valid.poke(false.B)
    dut.io.control.bits.wid.poke(0.U)
    dut.io.control.bits.zimm.poke(0.U)
    dut.io.issue.valid.poke(false.B)
    dut.io.issue.bits.wid.poke(0.U)
    dut.io.issue.bits.isS2G.poke(false.B)
    dut.io.completion.valid.poke(false.B)
    dut.io.completion.bits.wid.poke(0.U)
    dut.io.completion.bits.group.poke(0.U)
    dut.io.completion.bits.is_s2g.poke(false.B)
    dut.io.completion.bits.barrierValid.poke(false.B)
    dut.io.completion.bits.barrierId.poke(0.U)
    dut.io.completion.bits.barrierGeneration.poke(0.U)
    dut.io.completion.bits.transactionBytes.poke(0.U)
    dut.io.warpReset.valid.poke(false.B)
    dut.io.warpReset.bits.poke(0.U)
    dut.io.clearAll.poke(false.B)
    dut.clock.step(2)
  }

  private def issue(dut: TmaV2S2GGroupTracker, wid: Int = 0): Int = {
    val group = dut.io.issueGroup(wid).peekInt().toInt
    assert(dut.io.issueAllow(wid).peekBoolean())
    dut.io.issue.valid.poke(true.B)
    dut.io.issue.bits.wid.poke(wid.U)
    dut.io.issue.bits.isS2G.poke(true.B)
    dut.clock.step()
    dut.io.issue.valid.poke(false.B)
    group
  }

  private def control(dut: TmaV2S2GGroupTracker, zimm: Int, wid: Int = 0): Unit = {
    dut.io.control.valid.poke(true.B)
    dut.io.control.bits.wid.poke(wid.U)
    dut.io.control.bits.zimm.poke(zimm.U)
    dut.clock.step()
    dut.io.control.valid.poke(false.B)
  }

  private def complete(dut: TmaV2S2GGroupTracker, group: Int, wid: Int = 0): Unit = {
    dut.io.completion.valid.poke(true.B)
    dut.io.completion.bits.wid.poke(wid.U)
    dut.io.completion.bits.group.poke(group.U)
    dut.io.completion.bits.is_s2g.poke(true.B)
    dut.clock.step()
    dut.io.completion.valid.poke(false.B)
  }

  "commit/wait and keep-N use only final S2G completions" in {
    test(new TmaV2S2GGroupTracker) { dut =>
      init(dut)
      val old = issue(dut)
      control(dut, TmaV2Spec.S2GGroupCommit)
      val recent = issue(dut)
      control(dut, TmaV2Spec.S2GGroupCommit)
      control(dut, TmaV2Spec.S2GGroupWaitBase + 1)
      assert(dut.io.waitMask.peekInt().testBit(0))

      dut.io.completion.valid.poke(true.B)
      dut.io.completion.bits.is_s2g.poke(false.B)
      dut.clock.step()
      dut.io.completion.valid.poke(false.B)
      assert(dut.io.waitMask.peekInt().testBit(0))

      complete(dut, old)
      assert(!dut.io.waitMask.peekInt().testBit(0))
      assert(dut.io.inflight(0).peekInt() == 1)
      complete(dut, recent)
    }
  }

  "empty commit preserves the slot and wrapped slots wait for completion" in {
    test(new TmaV2S2GGroupTracker) { dut =>
      init(dut)
      val initial = dut.io.issueGroup(0).peekInt().toInt
      control(dut, TmaV2Spec.S2GGroupCommit)
      assert(dut.io.issueGroup(0).peekInt() == initial)
      val groups = (0 until dma_group_entries).map { _ =>
        val group = issue(dut)
        control(dut, TmaV2Spec.S2GGroupCommit)
        group
      }
      assert(!dut.io.issueAllow(0).peekBoolean())
      complete(dut, groups.head)
      assert(dut.io.issueAllow(0).peekBoolean())
      groups.tail.foreach(group => complete(dut, group))
    }
  }

  "invalid operations report status without creating a wait" in {
    test(new TmaV2S2GGroupTracker) { dut =>
      init(dut)
      dut.io.control.valid.poke(true.B)
      dut.io.control.bits.zimm.poke(0.U)
      assert(dut.io.status.valid.peekBoolean())
      dut.io.status.bits.code.expect(TmaV2Spec.StatusInvalidGroupOperation.U)
      dut.clock.step()
      dut.io.control.valid.poke(false.B)
      dut.io.waitMask.expect(0.U)
    }
  }

  "out-of-order group completion and warp reset preserve ring accounting" in {
    test(new TmaV2S2GGroupTracker) { dut =>
      init(dut)
      val oldest = issue(dut)
      control(dut, TmaV2Spec.S2GGroupCommit)
      val newest = issue(dut)
      control(dut, TmaV2Spec.S2GGroupCommit)
      control(dut, TmaV2Spec.S2GGroupWaitBase)
      assert(dut.io.waitMask.peekInt().testBit(0))

      complete(dut, newest)
      assert(dut.io.waitMask.peekInt().testBit(0))
      dut.io.inflight(0).expect(1.U)
      complete(dut, oldest)
      assert(!dut.io.waitMask.peekInt().testBit(0))

      val beforeReset = issue(dut)
      control(dut, TmaV2Spec.S2GGroupCommit)
      complete(dut, beforeReset)
      dut.io.warpReset.valid.poke(true.B)
      dut.io.warpReset.bits.poke(0.U)
      dut.clock.step()
      dut.io.warpReset.valid.poke(false.B)
      dut.io.inflight(0).expect(0.U)
      dut.io.issueGroup(0).expect(0.U)
      assert(dut.io.issueAllow(0).peekBoolean())
    }
  }
}
