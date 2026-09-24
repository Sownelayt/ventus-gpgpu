package DmaTest

import chisel3._
import chiseltest._
import chiseltest.internal.CachingAnnotation
import org.scalatest.freespec.AnyFreeSpec
import pipeline._

import scala.collection.mutable
import scala.util.Random

class TmaV2CommandQueue_test extends AnyFreeSpec with ChiselScalatestTester {
  private def initialize(dut: TmaV2CommandQueue): Unit = {
    dut.io.enq.valid.poke(false.B)
    dut.io.deq.ready.poke(false.B)
    dut.io.kill.valid.poke(false.B)
    dut.io.kill.bits.asid.poke(0.U)
    dut.io.killActive.poke(false.B)
    dut.io.killAsid.poke(0.U)
    dut.io.cancelled.ready.poke(true.B)
    dut.clock.step(2)
  }

  private def drive(dut: TmaV2CommandQueue, id: Int, asid: Int,
                    s2g: Boolean = false): Unit = {
    dut.io.enq.bits.funct.poke(
      (if (s2g) TmaV2Spec.FunctBulkS2G else TmaV2Spec.FunctBulkG2S).U)
    dut.io.enq.bits.wid.poke((id & 7).U)
    dut.io.enq.bits.asid.poke(asid.U)
    dut.io.enq.bits.group.poke((id & 3).U)
    dut.io.enq.bits.in1.poke(id.U)
    dut.io.enq.bits.in2.foreach(_.poke(0.U))
    dut.io.enq.bits.in3.poke(0.U)
    dut.io.enq.bits.copyDirection.poke(
      (if (s2g) TmaV2Spec.DirectionS2G else TmaV2Spec.DirectionG2S).U)
    dut.io.enq.bits.opBits.poke(0.U)
  }

  for (depth <- Seq(1, 2, 8, 16, 32)) {
    s"depth $depth preserves order and refills on a full dequeue" in {
      test(new TmaV2CommandQueue(depth)).withAnnotations(
        Seq(CachingAnnotation)) { dut =>
        initialize(dut)
        for (id <- 0 until depth) {
          drive(dut, id, asid = id & 1)
          dut.io.enq.valid.poke(true.B)
          dut.io.enq.ready.expect(true.B)
          dut.clock.step()
        }
        drive(dut, depth, asid = 0)
        dut.io.enq.ready.expect(false.B)

        // A full queue may dequeue its oldest item and refill the tail on the
        // same edge without changing occupancy.
        dut.io.deq.ready.poke(true.B)
        dut.io.enq.ready.expect(true.B)
        dut.io.deq.valid.expect(true.B)
        dut.io.deq.bits.in1.expect(0.U)
        dut.clock.step()
        dut.io.enq.valid.poke(false.B)

        for (expected <- 1 to depth) {
          dut.io.deq.valid.expect(true.B)
          dut.io.deq.bits.in1.expect(expected.U)
          dut.clock.step()
        }
        dut.io.deq.valid.expect(false.B)
        dut.io.occupancy.expect(0.U)
      }
    }
  }

  "kill marks resident ASID entries and leaves other commands ordered" in {
    test(new TmaV2CommandQueue(8)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      for (id <- 0 until 8) {
        drive(dut, id, asid = id & 1, s2g = (id & 1) == 1)
        dut.io.enq.valid.poke(true.B)
        dut.clock.step()
      }
      dut.io.enq.valid.poke(false.B)
      dut.io.kill.bits.asid.poke(1.U)
      dut.io.kill.valid.poke(true.B)
      dut.io.killActive.poke(true.B)
      dut.io.killAsid.poke(1.U)
      dut.clock.step()
      dut.io.kill.valid.poke(false.B)
      dut.io.deq.ready.poke(true.B)

      var kept = Vector.empty[Int]
      var cancelled = Vector.empty[Int]
      var cycles = 0
      while ((kept.size < 4 || cancelled.size < 4) && cycles < 24) {
        if (dut.io.deq.valid.peekBoolean())
          kept :+= dut.io.deq.bits.in1.peekInt().toInt
        if (dut.io.cancelled.valid.peekBoolean())
          cancelled :+= dut.io.cancelled.bits.wid.peekInt().toInt
        dut.clock.step()
        cycles += 1
      }
      assert(kept == Vector(0, 2, 4, 6))
      assert(cancelled == Vector(1, 3, 5, 7))
      dut.io.matchingPending.expect(false.B)

      // A command arriving while its ASID is killed is consumed directly by
      // the cancellation completion and never occupies the FIFO.
      drive(dut, 7, asid = 1, s2g = true)
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.ready.expect(true.B)
      dut.io.cancelled.valid.expect(true.B)
      dut.clock.step()
      dut.io.enq.valid.poke(false.B)
      dut.io.occupancy.expect(0.U)
    }
  }

  "same-cycle matching kill suppresses a ready depth-one dequeue" in {
    test(new TmaV2CommandQueue(1)).withAnnotations(
      Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      drive(dut, id = 5, asid = 3, s2g = true)
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.ready.expect(true.B)
      dut.clock.step()
      dut.io.enq.valid.poke(false.B)

      // Reproduce the original escape condition exactly: Prepare is ready
      // while the kill pulse matches the unmarked resident head.
      dut.io.deq.ready.poke(true.B)
      dut.io.kill.bits.asid.poke(3.U)
      dut.io.kill.valid.poke(true.B)
      dut.io.killActive.poke(true.B)
      dut.io.killAsid.poke(3.U)
      dut.io.deq.valid.expect(false.B)
      dut.io.matchingPending.expect(true.B)
      dut.clock.step()
      dut.io.kill.valid.poke(false.B)

      dut.io.deq.valid.expect(false.B)
      dut.io.cancelled.valid.expect(true.B)
      dut.io.cancelled.bits.wid.expect(5.U)
      dut.io.cancelled.bits.group.expect(1.U)
      dut.io.cancelled.bits.is_s2g.expect(true.B)
      dut.io.cancelled.bits.transactionBytes.expect(0.U)
      dut.clock.step()

      dut.io.cancelled.valid.expect(false.B)
      dut.io.occupancy.expect(0.U)
      dut.io.matchingPending.expect(false.B)
    }
  }

  "cancel payload remains stable when a resident kill overtakes an incoming kill" in {
    test(new TmaV2CommandQueue(2)).withAnnotations(
      Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      drive(dut, id = 1, asid = 4)
      dut.io.enq.valid.poke(true.B)
      dut.clock.step()

      // The incoming killed command is selected before the resident head's
      // synchronous cancellation mark becomes visible.  Hold the external
      // completion sink stalled across that priority change.
      drive(dut, id = 6, asid = 4, s2g = true)
      dut.io.cancelled.ready.poke(false.B)
      dut.io.kill.bits.asid.poke(4.U)
      dut.io.kill.valid.poke(true.B)
      dut.io.killActive.poke(true.B)
      dut.io.killAsid.poke(4.U)
      dut.io.enq.ready.expect(true.B)
      dut.io.deq.ready.poke(true.B)
      dut.io.deq.valid.expect(false.B)
      dut.clock.step()
      dut.io.kill.valid.poke(false.B)
      dut.io.enq.valid.poke(false.B)

      for (_ <- 0 until 4) {
        dut.io.cancelled.valid.expect(true.B)
        dut.io.cancelled.bits.wid.expect(6.U)
        dut.io.cancelled.bits.group.expect(2.U)
        dut.io.cancelled.bits.is_s2g.expect(true.B)
        dut.io.cancelled.bits.transactionBytes.expect(0.U)
        dut.io.deq.valid.expect(false.B)
        dut.clock.step()
      }

      // A pipelined refill transfers the resident completion into the skid
      // on the same edge that releases the held incoming completion.
      dut.io.cancelled.ready.poke(true.B)
      dut.io.cancelled.bits.wid.expect(6.U)
      dut.clock.step()
      dut.io.cancelled.valid.expect(true.B)
      dut.io.cancelled.bits.wid.expect(1.U)
      dut.io.cancelled.bits.group.expect(1.U)
      dut.io.cancelled.bits.is_s2g.expect(false.B)
      dut.clock.step()

      dut.io.cancelled.valid.expect(false.B)
      dut.io.occupancy.expect(0.U)
      dut.io.matchingPending.expect(false.B)
    }
  }

  "full wrapped FIFO handles kill enqueue and sustained cancellation backpressure" in {
    test(new TmaV2CommandQueue(4)).withAnnotations(
      Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      for ((id, asid) <- Seq((0, 0), (1, 1), (2, 9), (3, 2))) {
        drive(dut, id, asid)
        dut.io.enq.valid.poke(true.B)
        dut.clock.step()
      }
      dut.io.enq.valid.poke(false.B)
      dut.io.deq.ready.poke(true.B)
      for (expected <- Seq(0, 1)) {
        dut.io.deq.valid.expect(true.B)
        dut.io.deq.bits.in1.expect(expected.U)
        dut.clock.step()
      }
      dut.io.deq.ready.poke(false.B)

      // Refill both wrapped tail slots so the resident queue is full again.
      for ((id, asid) <- Seq((4, 9), (5, 3))) {
        drive(dut, id, asid)
        dut.io.enq.valid.poke(true.B)
        dut.io.enq.ready.expect(true.B)
        dut.clock.step()
      }

      // A matching incoming command bypasses the full resident FIFO into the
      // skid while the same pulse marks both wrapped resident matches.
      drive(dut, id = 6, asid = 9, s2g = true)
      dut.io.cancelled.ready.poke(false.B)
      dut.io.kill.bits.asid.poke(9.U)
      dut.io.kill.valid.poke(true.B)
      dut.io.killActive.poke(true.B)
      dut.io.killAsid.poke(9.U)
      dut.io.enq.ready.expect(true.B)
      dut.io.deq.ready.poke(true.B)
      dut.io.deq.valid.expect(false.B)
      dut.clock.step()
      dut.io.kill.valid.poke(false.B)
      dut.io.enq.valid.poke(false.B)

      for (_ <- 0 until 6) {
        dut.io.cancelled.valid.expect(true.B)
        dut.io.cancelled.bits.wid.expect(6.U)
        dut.io.cancelled.bits.group.expect(2.U)
        dut.io.cancelled.bits.is_s2g.expect(true.B)
        dut.io.deq.valid.expect(false.B)
        dut.clock.step()
      }

      dut.io.cancelled.ready.poke(true.B)
      var cancelled = Vector.empty[Int]
      var kept = Vector.empty[Int]
      var cycles = 0
      while ((cancelled.size < 3 || kept.size < 2) && cycles < 20) {
        if (dut.io.cancelled.valid.peekBoolean())
          cancelled :+= dut.io.cancelled.bits.wid.peekInt().toInt
        if (dut.io.deq.valid.peekBoolean())
          kept :+= dut.io.deq.bits.in1.peekInt().toInt
        dut.clock.step()
        cycles += 1
      }
      assert(cancelled == Vector(6, 2, 4))
      assert(kept == Vector(3, 5))
      dut.io.occupancy.expect(0.U)
      dut.io.matchingPending.expect(false.B)
    }
  }

  "depth 16 survives randomized producer and consumer backpressure" in {
    test(new TmaV2CommandQueue(16)).withAnnotations(Seq(CachingAnnotation)) {
      dut =>
        initialize(dut)
        val random = new Random(0x544d4132L)
        val model = mutable.Queue.empty[Int]
        var offered: Option[Int] = None
        var nextId = 0
        var retired = 0

        for (_ <- 0 until 600) {
          if (offered.isEmpty && nextId < 200 && random.nextBoolean()) {
            offered = Some(nextId)
            nextId += 1
          }
          offered.foreach(id => drive(dut, id, asid = id & 3))
          dut.io.enq.valid.poke(offered.nonEmpty.B)
          dut.io.deq.ready.poke(random.nextBoolean().B)

          val enqFire = offered.nonEmpty && dut.io.enq.ready.peekBoolean()
          val deqFire = dut.io.deq.valid.peekBoolean() &&
            dut.io.deq.ready.peekBoolean()
          if (dut.io.deq.valid.peekBoolean()) {
            assert(model.nonEmpty)
            dut.io.deq.bits.in1.expect(model.front.U)
          }
          val accepted = if (enqFire) offered else None
          dut.clock.step()
          if (deqFire) {
            model.dequeue()
            retired += 1
          }
          accepted.foreach(model.enqueue(_))
          if (enqFire) offered = None
          dut.io.occupancy.expect(model.size.U)
        }

        dut.io.enq.valid.poke(false.B)
        dut.io.deq.ready.poke(true.B)
        while (model.nonEmpty) {
          dut.io.deq.valid.expect(true.B)
          dut.io.deq.bits.in1.expect(model.front.U)
          model.dequeue()
          retired += 1
          dut.clock.step()
        }
        assert(retired == nextId - offered.size)
        dut.io.occupancy.expect(0.U)
    }
  }
}
