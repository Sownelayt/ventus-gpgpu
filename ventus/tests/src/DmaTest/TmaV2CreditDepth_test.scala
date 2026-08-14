package DmaTest

import chisel3._
import chiseltest._
import chiseltest.internal.CachingAnnotation
import org.scalatest.freespec.AnyFreeSpec
import pipeline._

import scala.collection.mutable

/** Diagnose the G2S latency/credit throughput limit without changing the RTL.
  *
  * The production WindowEngine is elaborated once with enough internal
  * capacity.  A synthetic cache then accepts at most N outstanding 128-byte
  * reads, returns them after a fixed latency, and otherwise sustains one
  * request and one response per cycle.  Sweeping N isolates the latency-credit
  * limit without rebuilding the large engine for every point.
  */
class TmaV2CreditDepth_test
    extends AnyFreeSpec
    with ChiselScalatestTester {

  private val lineCount = 256
  private val cacheLatency = 38
  private val creditLimits = Seq(16, 24, 32, 40, 48)

  "G2S fixed-latency effective-credit diagnostic" in {
    test(new TmaV2WindowEngine(
      windowEntries = 48,
      requestEntries = 48,
      sharedEntries = 8,
      writeAckEntries = 8,
      mmuEnabled = false)).withAnnotations(
      Seq(CachingAnnotation, IcarusBackendAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(true.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)

      for ((creditLimit, runIndex) <- creditLimits.zipWithIndex) {
        dut.io.command.bits.wid.poke(0.U)
        dut.io.command.bits.copyDirection.poke(
          TmaV2Spec.DirectionG2S.U)
        dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
        dut.io.command.bits.oobFill.poke(false.B)
        dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
        dut.io.command.bits.asid.poke(0.U)
        dut.io.command.bits.group.poke(0.U)
        dut.io.command.bits.barrierValid.poke(false.B)
        dut.io.command.bits.barrierId.poke(0.U)
        dut.io.command.bits.barrierGeneration.poke(0.U)
        dut.io.command.bits.transactionBytes.poke(
          (lineCount * 128).U)
        dut.io.command.valid.poke(true.B)
        while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.command.valid.poke(false.B)

        val cacheResponses = mutable.Queue.empty[(Int, BigInt)]
        val sharedResponses = mutable.Queue.empty[(Int, BigInt)]
        val cacheIssueCycles = mutable.ArrayBuffer.empty[Int]
        var nextWindow = 0
        var cycle = 0
        var firstWindowCycle = -1
        var completionCycle = -1

        while (completionCycle < 0 && cycle < 4000) {
          if (nextWindow < lineCount) {
            val globalLine =
              0x100000 + runIndex * 0x10000 + nextWindow * 128
            val sharedLine = 0x2000 + nextWindow * 128
            dut.io.window.bits.sharedBase.poke(sharedLine.U)
            dut.io.window.bits.last.poke(
              (nextWindow == lineCount - 1).B)
            for (lane <- 0 until 8) {
              dut.io.window.bits.lanes(lane).valid.poke(true.B)
              dut.io.window.bits.lanes(lane).globalAddress.poke(
                (globalLine + lane * 16).U)
              dut.io.window.bits.lanes(lane).globalBytes.poke(16.U)
              dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
              dut.io.window.bits.lanes(lane).sharedBytes.poke(16.U)
            }
            dut.io.window.valid.poke(true.B)
          } else {
            dut.io.window.valid.poke(false.B)
          }

          val cacheDue = cacheResponses.headOption.filter(_._1 <= cycle)
          dut.io.cacheResponse.valid.poke(cacheDue.nonEmpty.B)
          cacheDue.foreach { case (_, source) =>
            dut.io.cacheResponse.bits.source.poke(source.U)
            dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
          }

          val sharedDue =
            sharedResponses.headOption.filter(_._1 <= cycle)
          dut.io.sharedResponse.valid.poke(sharedDue.nonEmpty.B)
          sharedDue.foreach { case (_, source) =>
            dut.io.sharedResponse.bits.source.poke(source.U)
            dut.io.sharedResponse.bits.data.poke(0.U)
            dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
          }

          val cacheResponseFire =
            cacheDue.nonEmpty && dut.io.cacheResponse.ready.peekBoolean()
          // Permit same-cycle credit recycle when the oldest response fires.
          val creditAvailable =
            cacheResponses.size < creditLimit || cacheResponseFire
          dut.io.cacheRequest.ready.poke(creditAvailable.B)

          val windowFire =
            nextWindow < lineCount && dut.io.window.ready.peekBoolean()
          val cacheRequestFire =
            creditAvailable && dut.io.cacheRequest.valid.peekBoolean()
          val sharedRequestFire =
            dut.io.sharedRequest.valid.peekBoolean()
          val sharedResponseFire =
            sharedDue.nonEmpty && dut.io.sharedResponse.ready.peekBoolean()

          val newCacheResponse =
            if (cacheRequestFire) {
              cacheIssueCycles += cycle
              Some((
                cycle + cacheLatency,
                dut.io.cacheRequest.bits.source.peekInt()))
            } else None
          val newSharedResponse =
            if (sharedRequestFire)
              Some((
                cycle + 1,
                dut.io.sharedRequest.bits.source.peekInt()))
            else None

          if (windowFire && firstWindowCycle < 0)
            firstWindowCycle = cycle
          if (dut.io.completion.valid.peekBoolean())
            completionCycle = cycle

          dut.clock.step()
          if (windowFire) nextWindow += 1
          if (cacheResponseFire) cacheResponses.dequeue()
          if (sharedResponseFire) sharedResponses.dequeue()
          newCacheResponse.foreach(cacheResponses.enqueue(_))
          newSharedResponse.foreach(sharedResponses.enqueue(_))
          cycle += 1
        }

        assert(completionCycle >= 0, "credit diagnostic did not complete")
        assert(nextWindow == lineCount)
        assert(cacheIssueCycles.size == lineCount)
        val issueSpan =
          cacheIssueCycles.last - cacheIssueCycles.head
        val issueCyclesPerLine =
          issueSpan.toDouble / (lineCount - 1).toDouble
        val commandCycles = completionCycle - firstWindowCycle
        println(
          f"CREDIT_DIAG,credits=$creditLimit,latency=$cacheLatency," +
            f"lines=$lineCount,issue_span=$issueSpan," +
            f"issue_cycles_per_line=$issueCyclesPerLine%.6f," +
            f"command_cycles=$commandCycles")
        dut.clock.step(2)
      }
    }
  }
}
