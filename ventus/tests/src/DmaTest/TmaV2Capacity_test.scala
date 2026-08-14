package DmaTest

import chisel3._
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import pipeline._

import scala.collection.mutable

class TmaV2Capacity_test extends AnyFreeSpec with ChiselScalatestTester {
  private val entries = 256

  private def initialize(dut: TmaV2WindowEngine): Unit = {
    dut.io.command.valid.poke(false.B)
    dut.io.seal.valid.poke(false.B)
    dut.io.window.valid.poke(false.B)
    dut.io.tlbRequest.ready.poke(true.B)
    dut.io.tlbResponse.valid.poke(false.B)
    dut.io.sharedRequest.ready.poke(true.B)
    dut.io.sharedResponse.valid.poke(false.B)
    dut.io.cacheRequest.ready.poke(true.B)
    dut.io.cacheResponse.valid.poke(false.B)
    dut.io.completion.ready.poke(true.B)
    dut.clock.step(2)
  }

  private def command(dut: TmaV2WindowEngine, direction: Int): Unit = {
    dut.io.command.bits.wid.poke(0.U)
    dut.io.command.bits.copyDirection.poke(direction.U)
    dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
    dut.io.command.bits.oobFill.poke(false.B)
    dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
    dut.io.command.bits.asid.poke(0.U)
    dut.io.command.bits.group.poke(0.U)
    dut.io.command.bits.barrierValid.poke(false.B)
    dut.io.command.bits.barrierId.poke(0.U)
    dut.io.command.bits.barrierGeneration.poke(0.U)
    dut.io.command.bits.transactionBytes.poke((entries * 128).U)
    dut.io.command.valid.poke(true.B)
    while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
    dut.clock.step()
    dut.io.command.valid.poke(false.B)
  }

  private def driveWindow(
      dut: TmaV2WindowEngine, index: Int): Unit = {
    val globalLine = 0x100000 + index * 128
    val sharedLine = 0x2000 + index * 128
    dut.io.window.bits.sharedBase.poke(sharedLine.U)
    dut.io.window.bits.last.poke((index == entries - 1).B)
    for (lane <- 0 until 8) {
      dut.io.window.bits.lanes(lane).valid.poke(true.B)
      dut.io.window.bits.lanes(lane).globalAddress.poke(
        (globalLine + lane * 16).U)
      dut.io.window.bits.lanes(lane).globalBytes.poke(16.U)
      dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
      dut.io.window.bits.lanes(lane).sharedBytes.poke(16.U)
    }
    dut.io.window.valid.poke(true.B)
  }

  "256 line and ack tags preserve source 255 and retire out of order" in {
    test(new TmaV2WindowEngine(
      windowEntries = 4,
      requestEntries = entries,
      sharedEntries = 4,
      writeAckEntries = entries,
      mmuEnabled = false)).withAnnotations(Seq(IcarusBackendAnnotation)) { dut =>
      initialize(dut)

      command(dut, TmaV2Spec.DirectionG2S)
      val readSources = mutable.ArrayBuffer.empty[Int]
      var nextWindow = 0
      var cycles = 0
      while (readSources.size < entries && cycles < 4000) {
        if (nextWindow < entries)
          driveWindow(dut, nextWindow)
        else
          dut.io.window.valid.poke(false.B)
        val windowFire = nextWindow < entries &&
          dut.io.window.ready.peekBoolean()
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          assert(!dut.io.cacheRequest.bits.write.peekBoolean())
          readSources += dut.io.cacheRequest.bits.source.peekInt().toInt
        }
        dut.clock.step()
        if (windowFire) nextWindow += 1
        cycles += 1
      }
      dut.io.window.valid.poke(false.B)
      assert(readSources.toSet == (0 until entries).toSet)
      dut.io.activeRequests.expect(entries.U)

      val readResponses = mutable.Queue(readSources.reverse.toList: _*)
      val sharedResponses = mutable.Queue.empty[BigInt]
      var readComplete = false
      while (!readComplete && cycles < 12000) {
        dut.io.cacheResponse.valid.poke(readResponses.nonEmpty.B)
        if (readResponses.nonEmpty) {
          dut.io.cacheResponse.bits.source.poke(readResponses.front.U)
          dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        }
        dut.io.sharedResponse.valid.poke(sharedResponses.nonEmpty.B)
        if (sharedResponses.nonEmpty) {
          dut.io.sharedResponse.bits.source.poke(sharedResponses.front.U)
          dut.io.sharedResponse.bits.data.poke(0.U)
          dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        }
        val cacheFire = readResponses.nonEmpty &&
          dut.io.cacheResponse.ready.peekBoolean()
        val sharedFire = sharedResponses.nonEmpty &&
          dut.io.sharedResponse.ready.peekBoolean()
        val newShared = if (dut.io.sharedRequest.valid.peekBoolean())
          Some(dut.io.sharedRequest.bits.source.peekInt()) else None
        readComplete = dut.io.completion.valid.peekBoolean()
        dut.clock.step()
        if (cacheFire) readResponses.dequeue()
        if (sharedFire) sharedResponses.dequeue()
        newShared.foreach(sharedResponses.enqueue(_))
        cycles += 1
      }
      assert(readComplete)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.clock.step(2)

      command(dut, TmaV2Spec.DirectionS2G)
      val writeSources = mutable.ArrayBuffer.empty[Int]
      val readSharedResponses = mutable.Queue.empty[BigInt]
      nextWindow = 0
      cycles = 0
      while (writeSources.size < entries && cycles < 8000) {
        if (nextWindow < entries)
          driveWindow(dut, nextWindow)
        else
          dut.io.window.valid.poke(false.B)
        dut.io.sharedResponse.valid.poke(readSharedResponses.nonEmpty.B)
        if (readSharedResponses.nonEmpty) {
          dut.io.sharedResponse.bits.source.poke(readSharedResponses.front.U)
          dut.io.sharedResponse.bits.data.poke(0.U)
          dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        }
        val windowFire = nextWindow < entries &&
          dut.io.window.ready.peekBoolean()
        val sharedFire = readSharedResponses.nonEmpty &&
          dut.io.sharedResponse.ready.peekBoolean()
        val newShared = if (dut.io.sharedRequest.valid.peekBoolean())
          Some(dut.io.sharedRequest.bits.source.peekInt()) else None
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          assert(dut.io.cacheRequest.bits.write.peekBoolean())
          writeSources += dut.io.cacheRequest.bits.source.peekInt().toInt
        }
        dut.clock.step()
        if (windowFire) nextWindow += 1
        if (sharedFire) readSharedResponses.dequeue()
        newShared.foreach(readSharedResponses.enqueue(_))
        cycles += 1
      }
      dut.io.window.valid.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      assert(writeSources.toSet == (0 until entries).toSet)
      dut.io.activeWriteAcks.expect(entries.U)

      val acks = mutable.Queue(writeSources.reverse.toList: _*)
      var writeComplete = false
      while (!writeComplete && cycles < 12000) {
        dut.io.cacheResponse.valid.poke(acks.nonEmpty.B)
        if (acks.nonEmpty) {
          dut.io.cacheResponse.bits.source.poke(acks.front.U)
          dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        }
        val ackFire = acks.nonEmpty &&
          dut.io.cacheResponse.ready.peekBoolean()
        writeComplete = dut.io.completion.valid.peekBoolean()
        dut.clock.step()
        if (ackFire) acks.dequeue()
        cycles += 1
      }
      assert(writeComplete)
    }
  }
}
