package DmaTest

import chisel3._
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import pipeline.{TmaV2MbarrierController, TmaV2Spec}

class TmaV2Completion_test extends AnyFreeSpec with ChiselScalatestTester {
  private def init(dut: TmaV2MbarrierController): Unit = {
    dut.io.syncCommand.valid.poke(false.B)
    dut.io.syncCommand.bits.wid.poke(0.U)
    dut.io.syncCommand.bits.owner.poke(0.U)
    dut.io.syncCommand.bits.op.poke(0.U)
    dut.io.syncCommand.bits.address.poke(0.U)
    dut.io.syncCommand.bits.value.poke(0.U)
    dut.io.reserveRequest.valid.poke(false.B)
    dut.io.reserveRequest.bits.wid.poke(0.U)
    dut.io.reserveRequest.bits.bytes.poke(0.U)
    dut.io.reserveResponse.ready.poke(true.B)
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
    dut.io.wgRelease.valid.poke(false.B)
    dut.io.wgRelease.bits.poke(0.U)
    dut.io.sharedRequest.ready.poke(true.B)
    dut.io.sharedResponse.valid.poke(false.B)
    dut.io.sharedResponse.bits.instrId.poke(0.U)
    dut.io.sharedResponse.bits.isWrite.poke(true.B)
    dut.io.sharedResponse.bits.isMBarrier.poke(true.B)
    dut.io.sharedResponse.bits.data.foreach(_.poke(0.U))
    dut.io.sharedResponse.bits.activeMask.foreach(_.poke(false.B))
    dut.clock.step(2)
  }

  private def command(dut: TmaV2MbarrierController, op: Int, address: BigInt,
                      value: BigInt, wid: Int = 0, owner: Int = 0): Boolean = {
    dut.io.syncCommand.valid.poke(true.B)
    dut.io.syncCommand.bits.wid.poke(wid.U)
    dut.io.syncCommand.bits.owner.poke(owner.U)
    dut.io.syncCommand.bits.op.poke(op.U)
    dut.io.syncCommand.bits.address.poke(address.U)
    dut.io.syncCommand.bits.value.poke(value.U)
    while (!dut.io.syncCommand.ready.peekBoolean()) dut.clock.step()
    val failed = dut.io.status.valid.peekBoolean()
    dut.clock.step()
    dut.io.syncCommand.valid.poke(false.B)
    failed
  }

  private def drainWrites(dut: TmaV2MbarrierController): Unit = {
    for (_ <- 0 until 2) {
      var guard = 0
      while (!dut.io.sharedRequest.valid.peekBoolean() && guard < 20) {
        dut.clock.step(); guard += 1
      }
      assert(dut.io.sharedRequest.valid.peekBoolean())
      dut.clock.step()
      dut.io.sharedResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.sharedResponse.valid.poke(false.B)
    }
  }

  private def reserve(dut: TmaV2MbarrierController, bytes: Int,
                      wid: Int = 0): (Boolean, Boolean, Int, Int) = {
    dut.io.reserveRequest.valid.poke(true.B)
    dut.io.reserveRequest.bits.wid.poke(wid.U)
    dut.io.reserveRequest.bits.bytes.poke(bytes.U)
    while (!dut.io.reserveRequest.ready.peekBoolean()) dut.clock.step()
    dut.clock.step()
    dut.io.reserveRequest.valid.poke(false.B)
    var guard = 0
    while (!dut.io.reserveResponse.valid.peekBoolean() && guard < 10) {
      dut.clock.step(); guard += 1
    }
    assert(dut.io.reserveResponse.valid.peekBoolean())
    val result = (dut.io.reserveResponse.bits.accepted.peekBoolean(),
      dut.io.reserveResponse.bits.barrierValid.peekBoolean(),
      dut.io.reserveResponse.bits.barrierId.peekInt().toInt,
      dut.io.reserveResponse.bits.generation.peekInt().toInt)
    dut.clock.step()
    result
  }

  private def complete(dut: TmaV2MbarrierController, id: Int, generation: Int,
                       bytes: Int, wid: Int = 0): Boolean = {
    dut.io.completion.valid.poke(true.B)
    dut.io.completion.bits.wid.poke(wid.U)
    dut.io.completion.bits.is_s2g.poke(false.B)
    dut.io.completion.bits.barrierValid.poke(true.B)
    dut.io.completion.bits.barrierId.poke(id.U)
    dut.io.completion.bits.barrierGeneration.poke(generation.U)
    dut.io.completion.bits.transactionBytes.poke(bytes.U)
    val failed = dut.io.status.valid.peekBoolean()
    dut.clock.step()
    dut.io.completion.valid.poke(false.B)
    failed
  }

  "one expectation persistently binds multiple unordered G2S operations" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      val address = BigInt("70000040", 16)
      command(dut, TmaV2Spec.MbarrierInit, address, 1)
      drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 128)
      drainWrites(dut)
      val first = reserve(dut, 32)
      val second = reserve(dut, 96)
      assert(first._1 && first._2 && second._1 && second._2)
      assert(first._3 == second._3 && first._4 == second._4)
      val unbound = reserve(dut, 16)
      assert(unbound._1 && !unbound._2)

      complete(dut, second._3, second._4, 96)
      complete(dut, first._3, first._4, 32)
      drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierWait, address, 0)
      dut.io.waitMask.expect(0.U)
    }
  }

  "over-reservation is rejected before the binding is consumed" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      val address = BigInt("70000080", 16)
      command(dut, TmaV2Spec.MbarrierInit, address, 1); drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 64); drainWrites(dut)
      val rejected = reserve(dut, 80)
      assert(!rejected._1)
      val accepted = reserve(dut, 64)
      assert(accepted._1 && accepted._2)
    }
  }

  "a completed phase must be observed before the next arrival" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      val address = BigInt("700000c0", 16)
      command(dut, TmaV2Spec.MbarrierInit, address, 1); drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 32); drainWrites(dut)
      val handle = reserve(dut, 32)
      complete(dut, handle._3, handle._4, 32); drainWrites(dut)

      assert(command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 32))
      command(dut, TmaV2Spec.MbarrierWait, address, 0)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 32)
      val next = reserve(dut, 32)
      assert(next._1 && next._2 && next._4 == ((handle._4 + 1) & 0xff))

      assert(complete(dut, handle._3, handle._4, 32))
    }
  }

  "each WG owns four independent authoritative entries" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      for (slot <- 0 until TmaV2Spec.MbarrierEntriesPerWg) {
        command(dut, TmaV2Spec.MbarrierInit, BigInt("70000100", 16) + slot * 8, 1,
          owner = 0)
        drainWrites(dut)
      }
      assert(command(dut, TmaV2Spec.MbarrierInit,
        BigInt("70000140", 16), 1, owner = 0))
      command(dut, TmaV2Spec.MbarrierInit, BigInt("70000100", 16), 1, owner = 1)
      drainWrites(dut)
    }
  }

  "two warps share one generation and delayed wait releases after mirror writes" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      val address = BigInt("70000180", 16)
      command(dut, TmaV2Spec.MbarrierInit, address, 2); drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 32,
        wid = 0, owner = 0); drainWrites(dut)
      command(dut, TmaV2Spec.MbarrierArriveExpectTx, address, 32,
        wid = 1, owner = 0); drainWrites(dut)
      val warp0 = reserve(dut, 32, wid = 0)
      val warp1 = reserve(dut, 32, wid = 1)
      assert(warp0._3 == warp1._3 && warp0._4 == warp1._4)

      command(dut, TmaV2Spec.MbarrierWait, address, 0, wid = 0, owner = 0)
      assert(dut.io.waitMask.peekInt().testBit(0))
      complete(dut, warp1._3, warp1._4, 32, wid = 1)
      assert(dut.io.waitMask.peekInt().testBit(0))
      complete(dut, warp0._3, warp0._4, 32, wid = 0)
      assert(dut.io.waitMask.peekInt().testBit(0))
      drainWrites(dut)
      dut.clock.step()
      assert(!dut.io.waitMask.peekInt().testBit(0))
    }
  }

  "quiescent WG release frees its fixed partition" in {
    test(new TmaV2MbarrierController) { dut =>
      init(dut)
      val base = BigInt("70000200", 16)
      for (slot <- 0 until TmaV2Spec.MbarrierEntriesPerWg) {
        command(dut, TmaV2Spec.MbarrierInit, base + slot * 8, 1, owner = 0)
        drainWrites(dut)
      }
      dut.io.wgRelease.valid.poke(true.B)
      dut.io.wgRelease.bits.poke(0.U)
      dut.clock.step()
      dut.io.wgRelease.valid.poke(false.B)

      assert(!command(dut, TmaV2Spec.MbarrierInit, base + 0x80, 1, owner = 0))
      drainWrites(dut)
    }
  }
}
