package DmaTest

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.internal.CachingAnnotation
import org.scalatest.freespec.AnyFreeSpec
import pipeline._

import scala.collection.mutable
import scala.util.Random

/** Test-only accumulator for the production DescriptorService event pulses. */
class TmaV2DescriptorPerfHarness(entries: Int = 4) extends Module {
  val io = IO(new Bundle {
    val invalidateAll = Input(Bool())
    val request = Flipped(Vec(2, Decoupled(new TmaV2DescriptorRequest)))
    val response = Vec(2, Decoupled(new TmaV2DescriptorResponse))
    val memoryRequest = Decoupled(new TmaV2DescriptorMemoryRequest)
    val memoryResponse = Flipped(Decoupled(new TmaV2DescriptorMemoryResponse))
    val hitCount = Output(UInt(32.W))
    val missCount = Output(UInt(32.W))
    val compileCount = Output(UInt(32.W))
    val coalesceCount = Output(UInt(32.W))
    val invalidateKillCount = Output(UInt(32.W))
    val compileCycleCount = Output(UInt(32.W))
    val demandHitCount = Output(UInt(32.W))
    val prefetchHitCount = Output(UInt(32.W))
    val demandMissCount = Output(UInt(32.W))
    val prefetchMissCount = Output(UInt(32.W))
    val evictionCount = Output(UInt(32.W))
  })
  val service = Module(new TmaV2DescriptorService(entries))
  service.io.invalidateAll := io.invalidateAll
  service.io.request <> io.request
  io.response <> service.io.response
  io.memoryRequest <> service.io.memoryRequest
  service.io.memoryResponse <> io.memoryResponse

  def counter(event: Bool): UInt = {
    val value = RegInit(0.U(32.W))
    when(event) { value := value + 1.U }
    value
  }
  io.demandHitCount := counter(service.io.events.demandHit)
  io.prefetchHitCount := counter(service.io.events.prefetchHit)
  io.demandMissCount := counter(service.io.events.demandMiss)
  io.prefetchMissCount := counter(service.io.events.prefetchMiss)
  io.compileCount := counter(service.io.events.compile)
  io.coalesceCount := counter(
    service.io.events.demandCoalesce || service.io.events.prefetchCoalesce)
  io.invalidateKillCount := counter(service.io.events.invalidateKill)
  io.compileCycleCount := counter(service.io.events.compileCycle)
  io.evictionCount := counter(service.io.events.eviction)
  io.hitCount := io.demandHitCount + io.prefetchHitCount
  io.missCount := io.demandMissCount + io.prefetchMissCount
}

class TmaV2Backend_test extends AnyFreeSpec with ChiselScalatestTester {
  "compiled descriptor store coalesces, serves busy hits, caches errors and replaces by PLRU" in {
    test(new TmaV2DescriptorPerfHarness(entries = 4))
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(0.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.clock.step(2)

      def issue(
          client: Int,
          address: Int,
          wantResponse: Boolean): Unit = {
        val request = dut.io.request(client)
        request.bits.address.poke(address.U)
        request.bits.asid.poke(3.U)
        request.bits.wantResponse.poke(wantResponse.B)
        request.bits.invalidate.poke(false.B)
        request.valid.poke(true.B)
        var waited = 0
        while (!request.ready.peekBoolean() && waited < 32) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 32,
          f"descriptor request 0x$address%x did not make progress")
        dut.clock.step()
        request.valid.poke(false.B)
      }

      def refill(base: Int, legal: Boolean = true): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        dut.io.memoryResponse.bits.words(0).poke(
          (if (legal) TmaV2Spec.DescriptorMagic else BigInt(0)).U)
        dut.io.memoryResponse.bits.words(1).poke(
          (TmaV2Spec.DTypeU8 | (1 << 5)).U)
        dut.io.memoryResponse.bits.words(2).poke(base.U)
        dut.io.memoryResponse.bits.words(4).poke(16.U)
        dut.io.memoryResponse.bits.words(17).poke(16.U)
        dut.io.memoryResponse.bits.words(22).poke(1.U)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
      }

      def waitCompiled(): Unit = {
        var waited = 0
        while (dut.io.compileCount.peekInt() <=
            dut.io.missCount.peekInt() - 1 && waited < 16) {
          dut.clock.step()
          waited += 1
        }
        while (dut.io.memoryRequest.valid.peekBoolean() && waited < 24) {
          dut.clock.step()
          waited += 1
        }
        dut.clock.step(12)
      }

      def consume(client: Int, address: Int, legal: Boolean): Unit = {
        var waited = 0
        while (!dut.io.response(client).valid.peekBoolean() && waited < 24) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 24)
        dut.io.response(client).bits.compiled.status.expect(
          (if (legal) TmaV2Status.Ok else TmaV2Status.BadMagic))
        dut.io.response(client).ready.poke(true.B)
        dut.clock.step()
        dut.io.response(client).ready.poke(false.B)
      }

      // Slot A becomes a known VALID hit.
      issue(1, 0x1000, wantResponse = false)
      refill(0x10000)
      waitCompiled()

      // Compile B.  A demand for B coalesces while a demand for A is still
      // served as a hit during the same compile.
      issue(1, 0x2000, wantResponse = false)
      refill(0x20000)
      issue(0, 0x1000, wantResponse = true)
      consume(0, 0x1000, legal = true)
      issue(0, 0x2000, wantResponse = true)
      consume(0, 0x2000, legal = true)
      dut.io.coalesceCount.expect(1.U)

      // Static failures occupy a VALID compiled slot and therefore hit
      // without another descriptor-memory request.
      issue(0, 0x3000, wantResponse = true)
      refill(0x30000, legal = false)
      consume(0, 0x3000, legal = false)
      val missesBeforeIllegalHit = dut.io.missCount.peekInt()
      issue(0, 0x3000, wantResponse = true)
      dut.io.memoryRequest.valid.expect(false.B)
      consume(0, 0x3000, legal = false)
      dut.io.missCount.expect(missesBeforeIllegalHit.U)

      // Fill D and then E.  The access history makes A the PLRU victim, so
      // probing A must miss again.
      for ((address, base) <- Seq(
          (0x4000, 0x40000), (0x5000, 0x50000))) {
        issue(1, address, wantResponse = false)
        refill(base)
        waitCompiled()
      }
      val missesBeforeReplacementProbe = dut.io.missCount.peekInt()
      issue(1, 0x1000, wantResponse = false)
      dut.io.missCount.expect((missesBeforeReplacementProbe + 1).U)
    }
  }

  "four-entry compiled descriptor service supports prefetch, hit reuse and both clients" in {
    test(new TmaV2DescriptorPerfHarness)
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(0.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.clock.step(2)

      def issue(client: Int, address: Int, wantResponse: Boolean,
                invalidate: Boolean = false): Unit = {
        dut.io.request(client).bits.address.poke(address.U)
        dut.io.request(client).bits.asid.poke(0.U)
        dut.io.request(client).bits.wantResponse.poke(wantResponse.B)
        dut.io.request(client).bits.invalidate.poke(invalidate.B)
        dut.io.request(client).valid.poke(true.B)
        var waited = 0
        while (!dut.io.request(client).ready.peekBoolean() && waited < 20) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 20)
        dut.clock.step()
        dut.io.request(client).valid.poke(false.B)
      }

      def memoryResponse(seed: Int): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        dut.io.memoryResponse.bits.words(0).poke(
          TmaV2Spec.DescriptorMagic.U)
        dut.io.memoryResponse.bits.words(1).poke(
          (TmaV2Spec.DTypeU8 | (1 << 5)).U)
        dut.io.memoryResponse.bits.words(2).poke(seed.U)
        dut.io.memoryResponse.bits.words(4).poke(16.U)
        dut.io.memoryResponse.bits.words(17).poke(16.U)
        dut.io.memoryResponse.bits.words(22).poke(1.U)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
      }

      def waitResponse(client: Int): Unit = {
        var waited = 0
        while (!dut.io.response(client).valid.peekBoolean() && waited < 20) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 20)
      }

      issue(client = 1, address = 0x1000, wantResponse = false)
      memoryResponse(seed = 0x5000)
      dut.clock.step(12)
      dut.io.missCount.expect(1.U)
      dut.io.response(0).valid.expect(false.B)

      issue(client = 0, address = 0x1000, wantResponse = true)
      dut.io.memoryRequest.valid.expect(false.B)
      waitResponse(0)
      dut.io.response(0).valid.expect(true.B)
      dut.io.response(0).bits.compiled.globalBase.expect(0x5000.U)
      dut.io.response(0).bits.compiled.logicalBytes.expect(16.U)
      dut.io.response(0).bits.compiled.status.expect(TmaV2Status.Ok)
      dut.io.response(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.response(0).ready.poke(false.B)
      dut.io.hitCount.expect(1.U)

      issue(client = 0, address = 0x2000, wantResponse = true)
      memoryResponse(seed = 0x6000)
      waitResponse(0)
      dut.io.response(0).valid.expect(true.B)
      dut.io.response(0).bits.compiled.boxDims(0).expect(16.U)
      dut.io.response(0).bits.compiled.logicalBytes.expect(16.U)
      dut.io.response(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.missCount.expect(2.U)

      dut.io.invalidateAll.poke(true.B)
      dut.clock.step()
      dut.io.invalidateAll.poke(false.B)
      dut.io.response(0).ready.poke(false.B)
      issue(client = 0, address = 0x1000, wantResponse = true)
      dut.io.missCount.expect(3.U)
      memoryResponse(seed = 0x7000)
      waitResponse(0)
      dut.io.response(0).valid.expect(true.B)
      dut.io.response(0).bits.compiled.globalBase.expect(0x7000.U)
    }
  }

  "four prefetched TensorMaps remain demand hits without refill or recompile" in {
    test(new TmaV2DescriptorPerfHarness(entries = 4))
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(5.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }
      dut.clock.step(2)

      def request(client: Int, address: Int, demand: Boolean): Unit = {
        val port = dut.io.request(client)
        port.bits.address.poke(address.U)
        port.bits.wantResponse.poke(demand.B)
        port.valid.poke(true.B)
        var waited = 0
        while (!port.ready.peekBoolean() && waited < 24) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 24)
        dut.clock.step()
        port.valid.poke(false.B)
      }

      def refill(base: Int): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        dut.io.memoryResponse.bits.words(0).poke(
          TmaV2Spec.DescriptorMagic.U)
        dut.io.memoryResponse.bits.words(1).poke(
          (TmaV2Spec.DTypeU8 | (1 << 5)).U)
        dut.io.memoryResponse.bits.words(2).poke(base.U)
        dut.io.memoryResponse.bits.words(4).poke(16.U)
        dut.io.memoryResponse.bits.words(17).poke(16.U)
        dut.io.memoryResponse.bits.words(22).poke(1.U)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
        dut.clock.step(12)
      }

      val maps = (0 until 4).map(index =>
        (0x1000 + index * 0x1000, 0x10000 + index * 0x10000))
      for ((address, base) <- maps) {
        request(client = 1, address = address, demand = false)
        refill(base)
      }
      dut.io.prefetchMissCount.expect(4.U)
      dut.io.compileCount.expect(4.U)

      for ((address, base) <- maps) {
        request(client = 0, address = address, demand = true)
        dut.io.memoryRequest.valid.expect(false.B)
        var waited = 0
        while (!dut.io.response(0).valid.peekBoolean() && waited < 12) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 12)
        dut.io.response(0).bits.compiled.globalBase.expect(base.U)
        dut.io.response(0).ready.poke(true.B)
        dut.clock.step()
        dut.io.response(0).ready.poke(false.B)
      }
      dut.io.demandHitCount.expect(4.U)
      dut.io.missCount.expect(4.U)
      dut.io.compileCount.expect(4.U)
      dut.io.evictionCount.expect(0.U)
    }
  }

  "simultaneous demand and prefetch of one TensorMap allocate and compile once" in {
    test(new TmaV2DescriptorPerfHarness(entries = 4))
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.response.foreach(_.ready.poke(false.B))
      for (client <- 0 until 2) {
        dut.io.request(client).bits.address.poke(0x1800.U)
        dut.io.request(client).bits.asid.poke(2.U)
        dut.io.request(client).bits.wantResponse.poke((client == 0).B)
        dut.io.request(client).bits.invalidate.poke(false.B)
        dut.io.request(client).valid.poke(false.B)
      }
      dut.clock.step(2)
      dut.io.request.foreach(_.valid.poke(true.B))

      dut.io.request(0).ready.expect(true.B)
      dut.io.request(1).ready.expect(false.B)
      dut.io.memoryRequest.valid.expect(true.B)
      dut.clock.step()
      dut.io.request(0).valid.poke(false.B)
      dut.io.request(1).ready.expect(true.B)
      dut.clock.step()
      dut.io.request(1).valid.poke(false.B)

      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.memoryResponse.bits.words(0).poke(
        TmaV2Spec.DescriptorMagic.U)
      dut.io.memoryResponse.bits.words(1).poke(
        (TmaV2Spec.DTypeU8 | (1 << 5)).U)
      dut.io.memoryResponse.bits.words(2).poke(0x9000.U)
      dut.io.memoryResponse.bits.words(4).poke(16.U)
      dut.io.memoryResponse.bits.words(17).poke(16.U)
      dut.io.memoryResponse.bits.words(22).poke(1.U)
      dut.io.memoryResponse.valid.poke(true.B)
      while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.memoryResponse.valid.poke(false.B)

      var waited = 0
      while (!dut.io.response(0).valid.peekBoolean() && waited < 24) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 24)
      dut.io.response(0).bits.compiled.status.expect(TmaV2Status.Ok)
      dut.io.response(0).bits.compiled.globalBase.expect(0x9000.U)
      dut.io.demandMissCount.expect(1.U)
      dut.io.prefetchMissCount.expect(0.U)
      dut.io.compileCount.expect(1.U)
      dut.io.coalesceCount.expect(1.U)
      dut.io.missCount.expect(1.U)
      dut.io.response(0).ready.poke(true.B)
      dut.clock.step()
    }
  }

  "pending demand pins its compiled entry while a control miss replaces its peer" in {
    test(new TmaV2DescriptorPerfHarness(entries = 2))
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(1.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }

      def issue(client: Int, address: Int, demand: Boolean): Unit = {
        val request = dut.io.request(client)
        request.bits.address.poke(address.U)
        request.bits.wantResponse.poke(demand.B)
        request.valid.poke(true.B)
        var waited = 0
        while (!request.ready.peekBoolean() && waited < 32) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 32)
        dut.clock.step()
        request.valid.poke(false.B)
      }

      def refill(base: Int): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        dut.io.memoryResponse.bits.words(0).poke(
          TmaV2Spec.DescriptorMagic.U)
        dut.io.memoryResponse.bits.words(1).poke(
          (TmaV2Spec.DTypeU8 | (1 << 5)).U)
        dut.io.memoryResponse.bits.words(2).poke(base.U)
        dut.io.memoryResponse.bits.words(4).poke(16.U)
        dut.io.memoryResponse.bits.words(17).poke(16.U)
        dut.io.memoryResponse.bits.words(22).poke(1.U)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
        dut.clock.step(12)
      }

      issue(1, 0x1000, demand = false)
      refill(0x10000)
      issue(1, 0x2000, demand = false)
      refill(0x20000)

      // Keep A's compiled response backpressured, then touch B so tree-PLRU
      // nominates A. C must replace B because A is pinned by the demand.
      issue(0, 0x1000, demand = true)
      dut.io.response(0).valid.expect(true.B)
      issue(1, 0x2000, demand = false)
      issue(1, 0x3000, demand = false)
      refill(0x30000)

      dut.io.response(0).valid.expect(true.B)
      dut.io.response(0).bits.compiled.globalBase.expect(0x10000.U)
      dut.io.response(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.response(0).ready.poke(false.B)

      val misses = dut.io.missCount.peekInt()
      issue(0, 0x1000, demand = true)
      dut.io.memoryRequest.valid.expect(false.B)
      dut.io.response(0).bits.compiled.globalBase.expect(0x10000.U)
      dut.io.response(0).ready.poke(true.B)
      dut.clock.step()
      dut.io.response(0).ready.poke(false.B)
      dut.io.missCount.expect(misses.U)
    }
  }

  "two-entry compiled store preserves MRU TensorMap and evicts the PLRU peer" in {
    test(new TmaV2DescriptorPerfHarness(entries = 2))
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(0.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }
      dut.clock.step(2)

      def issue(client: Int, address: Int, demand: Boolean): Unit = {
        val request = dut.io.request(client)
        request.bits.address.poke(address.U)
        request.bits.asid.poke(0.U)
        request.bits.wantResponse.poke(demand.B)
        request.bits.invalidate.poke(false.B)
        request.valid.poke(true.B)
        var waited = 0
        while (!request.ready.peekBoolean() && waited < 24) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 24)
        dut.clock.step()
        request.valid.poke(false.B)
      }

      def refill(base: Int): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        dut.io.memoryResponse.bits.words(0).poke(
          TmaV2Spec.DescriptorMagic.U)
        dut.io.memoryResponse.bits.words(1).poke(
          (TmaV2Spec.DTypeU8 | (1 << 5)).U)
        dut.io.memoryResponse.bits.words(2).poke(base.U)
        dut.io.memoryResponse.bits.words(4).poke(16.U)
        dut.io.memoryResponse.bits.words(17).poke(16.U)
        dut.io.memoryResponse.bits.words(22).poke(1.U)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
        dut.clock.step(12)
      }

      def consume(address: Int): Unit = {
        var waited = 0
        while (!dut.io.response(0).valid.peekBoolean() && waited < 24) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 24)
        dut.io.response(0).ready.poke(true.B)
        dut.clock.step()
        dut.io.response(0).ready.poke(false.B)
      }

      issue(1, 0x1000, demand = false)
      refill(0x10000)
      issue(1, 0x2000, demand = false)
      refill(0x20000)
      issue(0, 0x1000, demand = true)
      consume(0x1000)
      issue(1, 0x3000, demand = false)
      refill(0x30000)
      dut.io.evictionCount.expect(1.U)

      val missesBeforeA = dut.io.missCount.peekInt()
      issue(0, 0x1000, demand = true)
      dut.io.memoryRequest.valid.expect(false.B)
      consume(0x1000)
      dut.io.missCount.expect(missesBeforeA.U)

      val missesBeforeB = dut.io.missCount.peekInt()
      issue(0, 0x2000, demand = true)
      dut.io.missCount.expect((missesBeforeB + 1).U)
      refill(0x22000)
      consume(0x2000)
    }
  }

  "descriptor request arbiter serves simultaneous clients without starvation" in {
    test(new TmaV2DescriptorPerfHarness)
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.response.foreach(_.ready.poke(true.B))
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      for (client <- 0 until 2) {
        dut.io.request(client).valid.poke(true.B)
        dut.io.request(client).bits.address.poke((0x3000 + client * 0x1000).U)
        dut.io.request(client).bits.asid.poke(client.U)
        dut.io.request(client).bits.wantResponse.poke(false.B)
        dut.io.request(client).bits.invalidate.poke(false.B)
      }
      val served = mutable.Set.empty[Int]
      while (served.size < 2) {
        val ready = (0 until 2).filter(dut.io.request(_).ready.peekBoolean())
        assert(ready.size == 1)
        val client = ready.head
        served += client
        dut.clock.step()
        dut.io.request(client).valid.poke(false.B)
        dut.io.memoryResponse.valid.poke(true.B)
        while (!dut.io.memoryResponse.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
        dut.clock.step(12)
      }
      assert(served == Set(0, 1))
      dut.io.missCount.expect(2.U)
    }
  }

  "descriptor invalidation is address and ASID scoped and wins after an in-flight refill" in {
    test(new TmaV2DescriptorPerfHarness)
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.invalidateAll.poke(false.B)
      dut.io.memoryRequest.ready.poke(true.B)
      dut.io.memoryResponse.valid.poke(false.B)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      dut.io.response.foreach(_.ready.poke(false.B))
      dut.io.request.foreach { request =>
        request.valid.poke(false.B)
        request.bits.address.poke(0.U)
        request.bits.asid.poke(0.U)
        request.bits.wantResponse.poke(false.B)
        request.bits.invalidate.poke(false.B)
      }
      dut.clock.step(2)

      def issue(address: Int, asid: Int = 0, invalidate: Boolean = false,
                wantResponse: Boolean = false, client: Int = 0): Unit = {
        val request = dut.io.request(client)
        request.bits.address.poke(address.U)
        request.bits.asid.poke(asid.U)
        request.bits.wantResponse.poke(wantResponse.B)
        request.bits.invalidate.poke(invalidate.B)
        request.valid.poke(true.B)
        var cycles = 0
        while (!request.ready.peekBoolean() && cycles < 30) {
          dut.clock.step()
          cycles += 1
        }
        assert(cycles < 30, s"descriptor request 0x${address.toHexString} stalled")
        dut.clock.step()
        request.valid.poke(false.B)
      }

      def refill(seed: Int): Unit = {
        dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
        for (word <- 0 until 29) {
          dut.io.memoryResponse.bits.words(word).poke((seed + word).U)
        }
        for (word <- 27 until 32) {
          dut.io.memoryResponse.bits.words(word).poke(0.U)
        }
        dut.io.memoryResponse.valid.poke(true.B)
        var refillWait = 0
        while (!dut.io.memoryResponse.ready.peekBoolean() &&
            refillWait < 30) {
          dut.clock.step()
          refillWait += 1
        }
        assert(refillWait < 30)
        dut.clock.step()
        dut.io.memoryResponse.valid.poke(false.B)
        dut.clock.step(12)
      }

      def consume(client: Int, address: Int): Unit = {
        dut.io.response(client).valid.expect(true.B)
        dut.io.response(client).ready.poke(true.B)
        dut.clock.step()
        dut.io.response(client).ready.poke(false.B)
      }

      issue(0x1000)
      refill(0x100)
      issue(0x2000)
      refill(0x200)

      issue(0x1000, invalidate = true, wantResponse = true, client = 1)
      consume(1, 0x1000)
      val missesBeforeOtherHit = dut.io.missCount.peekInt()
      issue(0x2000, wantResponse = true)
      dut.io.memoryRequest.valid.expect(false.B)
      consume(0, 0x2000)
      dut.io.missCount.expect(missesBeforeOtherHit.U)

      issue(0x1000)
      refill(0x300)
      issue(0x1000, asid = 1)
      refill(0x400)
      issue(0x1000, invalidate = true, wantResponse = true)
      consume(0, 0x1000)
      val missesBeforeAsidHit = dut.io.missCount.peekInt()
      issue(0x1000, asid = 1, wantResponse = true)
      consume(0, 0x1000)
      dut.io.missCount.expect(missesBeforeAsidHit.U)

      // A targeted invalidate kills an in-flight compile immediately.  The
      // old refill is consumed but may not commit; the next lookup misses.
      issue(0x3000)
      val invalidate = dut.io.request(1)
      invalidate.bits.address.poke(0x3000.U)
      invalidate.bits.asid.poke(0.U)
      invalidate.bits.wantResponse.poke(true.B)
      invalidate.bits.invalidate.poke(true.B)
      invalidate.valid.poke(true.B)
      val racingLookup = dut.io.request(0)
      racingLookup.bits.address.poke(0x3000.U)
      racingLookup.bits.asid.poke(0.U)
      racingLookup.bits.wantResponse.poke(false.B)
      racingLookup.bits.invalidate.poke(false.B)
      racingLookup.valid.poke(true.B)
      var waitInvalidate = 0
      while (!invalidate.ready.peekBoolean() && waitInvalidate < 20) {
        dut.clock.step()
        waitInvalidate += 1
      }
      assert(waitInvalidate < 20)
      racingLookup.ready.expect(false.B)
      dut.clock.step()
      invalidate.valid.poke(false.B)
      racingLookup.valid.poke(false.B)
      refill(0x500)
      val missesBeforeReload = dut.io.missCount.peekInt()
      consume(1, 0x3000)
      racingLookup.valid.poke(true.B)
      var waitReload = 0
      while (!racingLookup.ready.peekBoolean() && waitReload < 20) {
        dut.clock.step()
        waitReload += 1
      }
      assert(waitReload < 20)
      dut.clock.step()
      racingLookup.valid.poke(false.B)
      dut.io.missCount.expect((missesBeforeReload + 1).U)
      dut.io.invalidateKillCount.expect(1.U)
      refill(0x600)

      // Global invalidation also wins when asserted on the exact cycle that
      // the compiler result becomes valid.  A waiting demand is retried
      // internally and may only observe the freshly refilled descriptor.
      issue(0x4000, wantResponse = true)
      dut.io.memoryResponse.bits.words.foreach(_.poke(0.U))
      for (word <- 0 until 29) {
        dut.io.memoryResponse.bits.words(word).poke((0x700 + word).U)
      }
      for (word <- 27 until 32) {
        dut.io.memoryResponse.bits.words(word).poke(0.U)
      }
      dut.io.memoryResponse.valid.poke(true.B)
      dut.io.memoryResponse.ready.expect(true.B)
      dut.clock.step()
      dut.io.memoryResponse.valid.poke(false.B)
      dut.clock.step(9)
      dut.io.invalidateAll.poke(true.B)
      dut.clock.step()
      dut.io.invalidateAll.poke(false.B)
      refill(0x800)
      var waitKilledResponse = 0
      while (!dut.io.response(0).valid.peekBoolean() &&
          waitKilledResponse < 20) {
        dut.clock.step()
        waitKilledResponse += 1
      }
      assert(waitKilledResponse < 20)
      dut.io.response(0).bits.compiled.status.expect(
        TmaV2Status.BadMagic)
      consume(0, 0x4000)
      val missesBeforeGlobalReload = dut.io.missCount.peekInt()
      issue(0x4000, wantResponse = true)
      consume(0, 0x4000)
      dut.io.missCount.expect(missesBeforeGlobalReload.U)
      dut.io.invalidateKillCount.expect(2.U)
    }
  }

  "window engine bypasses TLB and merges replayed shared response beats" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 8,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
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
      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(3.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)
      dut.io.window.bits.last.poke(true.B)
      for (lane <- 0 until 8) {
        val active = lane == 0
        dut.io.window.bits.lanes(lane).valid.poke(active.B)
        dut.io.window.bits.lanes(lane).globalAddress
          .poke((0x1000 + lane * 16).U)
        dut.io.window.bits.lanes(lane).globalBytes
          .poke((if (active) 16 else 0).U)
        dut.io.window.bits.sharedBase.poke(0x2000.U)
        dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
        dut.io.window.bits.lanes(lane).sharedBytes
          .poke((if (active) 16 else 0).U)
      }
      dut.io.window.valid.poke(true.B)
      while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.window.valid.poke(false.B)

      var cycles = 0
      while (!dut.io.cacheRequest.valid.peekBoolean() && cycles < 40) {
        dut.io.tlbRequest.valid.expect(false.B)
        dut.clock.step()
        cycles += 1
      }
      assert(cycles < 40)
      dut.io.cacheRequest.bits.write.expect(false.B)
      dut.io.cacheRequest.bits.physicalAddress.expect(0x1000.U)
      val cacheSource = dut.io.cacheRequest.bits.source.peekInt()
      dut.clock.step()

      dut.io.cacheResponse.bits.source.poke(cacheSource.U)
      dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
      dut.io.cacheResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.cacheResponse.valid.poke(false.B)

      while (!dut.io.sharedRequest.valid.peekBoolean()) dut.clock.step()
      dut.io.sharedRequest.bits.write.expect(true.B)
      val sharedSource = dut.io.sharedRequest.bits.source.peekInt()
      // The 128-byte PayloadSlot is released as soon as shared accepts
      // the complete write; only the compact acknowledgement tag remains.
      dut.io.windowRetired.expect(true.B)
      dut.clock.step()
      dut.io.sharedResponse.bits.source.poke(sharedSource.U)
      dut.io.sharedResponse.bits.data.poke(0.U)
      dut.io.sharedResponse.bits.wordMask.poke(0x3.U)
      dut.io.sharedResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.windowRetired.expect(false.B)
      dut.io.sharedResponse.bits.wordMask.poke(0xc.U)
      dut.clock.step()
      dut.io.windowRetired.expect(false.B)
      dut.io.sharedResponse.valid.poke(false.B)

      while (!dut.io.completion.valid.peekBoolean()) dut.clock.step()
      dut.io.tlbRequest.valid.expect(false.B)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
    }
  }

  "B6 G2S uses one word-aligned tail without a generic byte permuter" in {
    test(new TmaV2WindowEngine(
      windowEntries = 4, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 4,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
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

      dut.io.command.bits.wid.poke(0.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeB6.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(96.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      dut.io.window.bits.sharedBase.poke(0x2000.U)
      dut.io.window.bits.last.poke(true.B)
      for (lane <- 0 until 8) {
        dut.io.window.bits.lanes(lane).valid.poke(true.B)
        dut.io.window.bits.lanes(lane).globalAddress
          .poke((0x1060 + lane * 12).U)
        dut.io.window.bits.lanes(lane).globalBytes.poke(12.U)
        dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
        dut.io.window.bits.lanes(lane).sharedBytes.poke(12.U)
      }
      dut.io.window.valid.poke(true.B)
      while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.window.valid.poke(false.B)

      val readSources = scala.collection.mutable.Map.empty[BigInt, BigInt]
      var requestCycles = 0
      while (readSources.size < 2 && requestCycles < 40) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          readSources += dut.io.cacheRequest.bits.physicalAddress.peekInt() ->
            dut.io.cacheRequest.bits.source.peekInt()
        }
        dut.clock.step()
        requestCycles += 1
      }
      assert(readSources.keySet == Set(BigInt(0x1000), BigInt(0x1080)))

      def lineAtom(lineBase: Int, atom: Int): BigInt =
        (0 until 16).map { byte =>
          BigInt((lineBase + atom * 16 + byte) & 0xff) << (byte * 8)
        }.reduce(_ | _)

      def laneBytes(start: Int, count: Int, destination: Int = 0): BigInt =
        (0 until count).map { byte =>
          BigInt((start + byte) & 0xff) << ((destination + byte) * 8)
        }.foldLeft(BigInt(0))(_ | _)

      def returnLine(lineBase: Int): Unit = {
        dut.io.cacheResponse.bits.source.poke(readSources(lineBase).U)
        for (atom <- 0 until 8) {
          dut.io.cacheResponse.bits.data(atom).poke(lineAtom(lineBase, atom).U)
        }
        dut.io.cacheResponse.valid.poke(true.B)
        dut.io.cacheResponse.ready.expect(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)
        var waitShared = 0
        while (!dut.io.sharedRequest.valid.peekBoolean() && waitShared < 20) {
          dut.clock.step()
          waitShared += 1
        }
        assert(waitShared < 20)
      }

      returnLine(0x1000)
      val firstMask = dut.io.sharedRequest.bits.mask.peekInt()
      assert((firstMask & 0xffff) == 0x0fff)
      assert(((firstMask >> 16) & 0xffff) == 0x0fff)
      assert(((firstMask >> 32) & 0xffff) == 0x00ff)
      assert((firstMask >> 48) == 0)
      val firstData = dut.io.sharedRequest.bits.data.peekInt()
      assert((firstData & ((BigInt(1) << 128) - 1)) ==
        laneBytes(0x60, 12))
      assert(((firstData >> 128) & ((BigInt(1) << 128) - 1)) ==
        laneBytes(0x6c, 12))
      assert(((firstData >> 256) & ((BigInt(1) << 128) - 1)) ==
        laneBytes(0x78, 8))
      val firstSharedSource = dut.io.sharedRequest.bits.source.peekInt()
      dut.clock.step()
      dut.io.sharedResponse.bits.source.poke(firstSharedSource.U)
      dut.io.sharedResponse.bits.data.poke(0.U)
      dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
      dut.io.sharedResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.sharedResponse.valid.poke(false.B)

      returnLine(0x1080)
      val secondMask = dut.io.sharedRequest.bits.mask.peekInt()
      assert(((secondMask >> 32) & 0xffff) == 0x0f00)
      for (lane <- 3 until 8) {
        assert(((secondMask >> (lane * 16)) & 0xffff) == 0x0fff)
      }
      val secondData = dut.io.sharedRequest.bits.data.peekInt()
      assert(((secondData >> 256) & ((BigInt(1) << 128) - 1)) ==
        laneBytes(0x80, 4, destination = 8))
      for (lane <- 3 until 8) {
        assert(((secondData >> (lane * 128)) &
          ((BigInt(1) << 128) - 1)) ==
          laneBytes(0x84 + (lane - 3) * 12, 12))
      }
      val secondSharedSource = dut.io.sharedRequest.bits.source.peekInt()
      dut.clock.step()
      dut.io.sharedResponse.bits.source.poke(secondSharedSource.U)
      dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
      dut.io.sharedResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.sharedResponse.valid.poke(false.B)

      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() && completionCycles < 30) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 30)
    }
  }

  "S2G final shared response beat preserves data from earlier replays" in {
    test(new TmaV2WindowEngine(
      windowEntries = 4, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 4,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
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

      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionS2G.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU32.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(128.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      dut.io.window.bits.sharedBase.poke(0x2000.U)
      dut.io.window.bits.last.poke(true.B)
      for (lane <- 0 until 8) {
        dut.io.window.bits.lanes(lane).valid.poke(true.B)
        dut.io.window.bits.lanes(lane).globalAddress
          .poke((0x1000 + lane * 16).U)
        dut.io.window.bits.lanes(lane).globalBytes.poke(16.U)
        dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
        dut.io.window.bits.lanes(lane).sharedBytes.poke(16.U)
      }
      dut.io.window.valid.poke(true.B)
      while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.window.valid.poke(false.B)

      while (!dut.io.sharedRequest.valid.peekBoolean()) dut.clock.step()
      dut.io.sharedRequest.bits.write.expect(false.B)
      val sharedSource = dut.io.sharedRequest.bits.source.peekInt()
      dut.clock.step()

      val words = (0 until 32).map(word => BigInt(0x1000 + word))
      val responseData = words.zipWithIndex.map { case (word, index) =>
        word << (index * 32)
      }.reduce(_ | _)
      for (beat <- 0 until 4) {
        val beatMask = (0 until 8).map(index =>
          BigInt(1) << (beat + index * 4)).reduce(_ | _)
        dut.io.sharedResponse.bits.source.poke(sharedSource.U)
        dut.io.sharedResponse.bits.data.poke(responseData.U)
        dut.io.sharedResponse.bits.wordMask.poke(beatMask.U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.io.sharedResponse.ready.expect(true.B)
        dut.clock.step()
      }
      dut.io.sharedResponse.valid.poke(false.B)

      var waitCache = 0
      while (!dut.io.cacheRequest.valid.peekBoolean() && waitCache < 40) {
        dut.clock.step()
        waitCache += 1
      }
      assert(waitCache < 40)
      dut.io.cacheRequest.bits.write.expect(true.B)
      dut.io.cacheRequest.bits.physicalAddress.expect(0x1000.U)
      dut.io.cacheRequest.bits.mask.expect(
        ((BigInt(1) << 128) - 1).U)
      for (atom <- 0 until 8) {
        val expectedAtom = (0 until 4).map { word =>
          words(atom * 4 + word) << (word * 32)
        }.reduce(_ | _)
        dut.io.cacheRequest.bits.data(atom).expect(expectedAtom.U)
      }
    }
  }

  "PayloadSlot, LineContext and shared-tag credits recycle in the completion cycle" in {
    test(new TmaV2WindowEngine(
      windowEntries = 3, requestEntries = 2, sharedEntries = 2,
      writeAckEntries = 2,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(1.U)
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
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      dut.io.command.ready.expect(true.B)
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      def driveWindow(global: Int, shared: Int, last: Boolean): Unit = {
        dut.io.window.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val active = lane == 0
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((global + lane * 16).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
      }

      def issueWindow(global: Int, shared: Int): Unit = {
        driveWindow(global, shared, last = false)
        dut.io.window.valid.poke(true.B)
        while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }

      issueWindow(0x1000, 0x2000)
      issueWindow(0x1080, 0x2080)
      issueWindow(0x1100, 0x2100)
      dut.io.activeWindows.expect(3.U)

      // Fill both legal request entries while a third slot is waiting.
      val issuedReads = scala.collection.mutable.ArrayBuffer.empty[
        (BigInt, BigInt)]
      dut.io.cacheRequest.ready.poke(true.B)
      while (issuedReads.size < 2) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          issuedReads += ((
            dut.io.cacheRequest.bits.source.peekInt(),
            dut.io.cacheRequest.bits.physicalAddress.peekInt()))
        }
        dut.clock.step()
      }
      dut.io.cacheRequest.ready.poke(false.B)
      assert(issuedReads.map(_._2).toSet ==
        Set(BigInt(0x1000), BigInt(0x1080)))
      dut.io.activeRequests.expect(2.U)

      // A full request table returns one line while the third slot needs a
      // line. The returning entry is freed and reallocated on the same edge.
      dut.io.cacheResponse.bits.source.poke(issuedReads.head._1.U)
      dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
      dut.io.cacheResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.activeRequests.expect(2.U)
      // The recycled entry is first allocated as NeedCache and is visible as
      // the following external request without an intervening free cycle.
      while (!dut.io.cacheRequest.valid.peekBoolean()) dut.clock.step()
      dut.io.cacheRequest.bits.physicalAddress.expect(0x1100.U)
      val recycledCacheSource = dut.io.cacheRequest.bits.source.peekInt()
      assert(recycledCacheSource == issuedReads.head._1)
      dut.io.cacheRequest.ready.poke(true.B)
      dut.clock.step()
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.activeRequests.expect(2.U)

      for (source <- Seq(issuedReads(1)._1, recycledCacheSource)) {
        dut.io.cacheResponse.bits.source.poke(source.U)
        dut.io.cacheResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)
      }

      while (!dut.io.sharedRequest.valid.peekBoolean()) dut.clock.step()
      driveWindow(0x1180, 0x2180, last = true)
      dut.io.window.valid.poke(true.B)
      dut.io.sharedRequest.ready.poke(true.B)
      // No slot was free before this cycle. Accepting the shared write must
      // expose its retiring slot directly to the Planner input.
      dut.io.window.ready.expect(true.B)
      dut.io.windowRetired.expect(true.B)
      val firstSharedSource = dut.io.sharedRequest.bits.source.peekInt()
      dut.clock.step()
      dut.io.window.valid.poke(false.B)
      // The accepted request is represented by activeShared rather than
      // activeWindows after ownership moves into the registered egress.
      dut.io.activeWindows.expect(2.U)

      // Consume a second shared write without returning the first ack, filling
      // both shared tags and leaving another transformed slot waiting.
      while (!dut.io.sharedRequest.valid.peekBoolean()) dut.clock.step()
      val secondSharedSource = dut.io.sharedRequest.bits.source.peekInt()
      assert(secondSharedSource != firstSharedSource)
      dut.clock.step()
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.activeShared.expect(2.U)

      // The full shared-tag table receives the first final ack while a new
      // write is waiting. The same tag is captured by the registered egress
      // on this edge, then becomes visible at the external port next cycle.
      dut.clock.step(4)
      dut.io.sharedResponse.bits.source.poke(firstSharedSource.U)
      dut.io.sharedResponse.bits.data.poke(0.U)
      dut.io.sharedResponse.bits.wordMask.poke(0xf.U)
      dut.io.sharedResponse.valid.poke(true.B)
      dut.io.sharedRequest.ready.poke(true.B)
      dut.clock.step()
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.sharedRequest.valid.expect(true.B)
      dut.io.sharedRequest.bits.source.expect(firstSharedSource.U)
      dut.clock.step()
      dut.io.sharedRequest.ready.poke(false.B)
    }
  }

  "recent-line follower routes one G2S response to two overlapping consumers" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 8,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      def issueWindow(global: Int, shared: Int, last: Boolean): Unit = {
        dut.io.window.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val active = lane == 0
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((global + lane * 16).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
        dut.io.window.valid.poke(true.B)
        while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }

      // Both windows consume different bytes from the same global 128B line.
      issueWindow(0x1000, 0x2000, last = false)
      issueWindow(0x1040, 0x2080, last = true)
      dut.clock.step(4)
      dut.io.activeRequests.expect(1.U)

      dut.io.cacheRequest.ready.poke(true.B)
      var cacheRequests = 0
      var cacheSource = BigInt(0)
      for (_ <- 0 until 8) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          dut.io.cacheRequest.bits.write.expect(false.B)
          dut.io.cacheRequest.bits.physicalAddress.expect(0x1000.U)
          cacheSource = dut.io.cacheRequest.bits.source.peekInt()
          cacheRequests += 1
        }
        dut.clock.step()
      }
      assert(cacheRequests == 1,
        s"overlapping windows emitted $cacheRequests cache requests")
      dut.io.cacheRequest.ready.poke(false.B)

      dut.io.cacheResponse.bits.source.poke(cacheSource.U)
      dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
      dut.io.cacheResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.cacheResponse.valid.poke(false.B)

      val sharedSources = mutable.ArrayBuffer.empty[BigInt]
      dut.io.sharedRequest.ready.poke(true.B)
      var sharedCycles = 0
      while (sharedSources.size < 2 && sharedCycles < 80) {
        if (dut.io.sharedRequest.valid.peekBoolean()) {
          dut.io.sharedRequest.bits.write.expect(true.B)
          sharedSources += dut.io.sharedRequest.bits.source.peekInt()
        }
        dut.clock.step()
        sharedCycles += 1
      }
      assert(sharedSources.size == 2)
      dut.io.sharedRequest.ready.poke(false.B)

      sharedSources.reverse.foreach { source =>
        dut.io.sharedResponse.bits.source.poke(source.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)
      }
      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 80) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 80)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeRequests.expect(0.U)
    }
  }

  "window engine retires two overlapping multi-line G2S windows" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 8,
      writeAckEntries = 8,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      def issueWindow(tags: Seq[Int], shared: Int, last: Boolean): Unit = {
        dut.io.window.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val tagIndex = math.min((lane * tags.size) / 8, tags.size - 1)
          val tag = tags(tagIndex)
          val firstLaneForTag =
            (0 until lane).count { earlier =>
              math.min((earlier * tags.size) / 8, tags.size - 1) == tagIndex
            }
          val offset = firstLaneForTag * 16
          dut.io.window.bits.lanes(lane).valid.poke(true.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((tag + offset).U)
          dut.io.window.bits.lanes(lane).globalBytes.poke(16.U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes.poke(16.U)
        }
        dut.io.window.valid.poke(true.B)
        while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }

      // Two lines feed both windows. This mirrors the 192B interleave16
      // shape: 5 canonical line consumers collapse to 3 cache waves, while
      // each window must still retire independently after two broadcasts.
      issueWindow(Seq(0x1000, 0x1080, 0x1100), 0x2000, last = false)
      issueWindow(Seq(0x1080, 0x1100), 0x2080, last = true)
      dut.clock.step(4)
      dut.io.activeRequests.expect(3.U)
      dut.io.activeWindows.expect(5.U)

      val cacheSources = mutable.LinkedHashMap.empty[BigInt, BigInt]
      dut.io.cacheRequest.ready.poke(true.B)
      var requestCycles = 0
      while (cacheSources.size < 3 && requestCycles < 80) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          dut.io.cacheRequest.bits.write.expect(false.B)
          cacheSources +=
            dut.io.cacheRequest.bits.physicalAddress.peekInt() ->
              dut.io.cacheRequest.bits.source.peekInt()
        }
        dut.clock.step()
        requestCycles += 1
      }
      assert(cacheSources.keySet == Set(
        BigInt(0x1000), BigInt(0x1080), BigInt(0x1100)))
      dut.io.cacheRequest.ready.poke(false.B)

      cacheSources.values.toSeq.reverse.foreach { source =>
        dut.io.cacheResponse.bits.source.poke(source.U)
        dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        dut.io.cacheResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)
      }
      // One returned payload already owns a shared tag/egress entry, leaving
      // four resources in the window-owned portion of the engine.
      dut.io.activeWindows.expect(4.U)

      val sharedSources = mutable.ArrayBuffer.empty[BigInt]
      dut.io.sharedRequest.ready.poke(true.B)
      var sharedCycles = 0
      // One partial shared write retires per line consumer; complete 128B
      // windows are not retained until every line has returned.
      // Three unique lines feed five consumer routes in this case.
      while (sharedSources.size < 5 && sharedCycles < 100) {
        if (dut.io.sharedRequest.valid.peekBoolean()) {
          dut.io.sharedRequest.bits.write.expect(true.B)
          sharedSources += dut.io.sharedRequest.bits.source.peekInt()
        }
        dut.clock.step()
        sharedCycles += 1
      }
      assert(sharedSources.size == 5)
      dut.io.sharedRequest.ready.poke(false.B)

      sharedSources.reverse.foreach { source =>
        dut.io.sharedResponse.bits.source.poke(source.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)
      }
      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 100) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 100)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeRequests.expect(0.U)
    }
  }

  "window engine fills and drains payload, line and shared credits under backpressure" in {
    test(new TmaV2WindowEngine(
      windowEntries = 6, requestEntries = 8, sharedEntries = 4,
      writeAckEntries = 8,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(false.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      for (window <- 0 until 8) {
        val global = 0x1000 + window * 0x80
        val shared = 0x2000 + window * 0x80
        dut.io.window.bits.last.poke((window == 7).B)
        for (lane <- 0 until 8) {
          val active = lane == 0
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((global + lane * 16).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
        dut.io.window.valid.poke(true.B)
        while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }
      dut.clock.step(6)
      dut.io.activeWindows.expect(8.U)
      dut.io.robFull.expect(false.B)
      dut.io.activeRequests.expect(8.U)
      dut.io.requestFull.expect(true.B)
      dut.io.tlbRequest.valid.expect(false.B)

      def collectCache(count: Int): Seq[BigInt] = {
        val sources = mutable.ArrayBuffer.empty[BigInt]
        dut.io.cacheRequest.ready.poke(true.B)
        var cycles = 0
        while (sources.size < count && cycles < 80) {
          if (dut.io.cacheRequest.valid.peekBoolean())
            sources += dut.io.cacheRequest.bits.source.peekInt()
          dut.clock.step()
          cycles += 1
        }
        dut.io.cacheRequest.ready.poke(false.B)
        assert(sources.size == count)
        sources.toSeq
      }

      def respondCache(sources: Seq[BigInt]): Unit = {
        sources.reverse.foreach { source =>
          dut.io.cacheResponse.bits.source.poke(source.U)
          dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
          dut.io.cacheResponse.valid.poke(true.B)
          dut.clock.step()
          dut.io.cacheResponse.valid.poke(false.B)
        }
      }

      def collectShared(count: Int): Seq[BigInt] = {
        val sources = mutable.ArrayBuffer.empty[BigInt]
        dut.io.sharedRequest.ready.poke(true.B)
        var cycles = 0
        while (sources.size < count && cycles < 80) {
          if (dut.io.sharedRequest.valid.peekBoolean()) {
            dut.io.sharedRequest.bits.write.expect(true.B)
            sources += dut.io.sharedRequest.bits.source.peekInt()
          }
          dut.clock.step()
          cycles += 1
        }
        dut.io.sharedRequest.ready.poke(false.B)
        assert(sources.size == count)
        sources.toSeq
      }

      def respondShared(sources: Seq[BigInt]): Unit = {
        sources.reverse.foreach { source =>
          dut.io.sharedResponse.bits.source.poke(source.U)
          dut.io.sharedResponse.bits.data.poke(0.U)
          dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
          dut.io.sharedResponse.valid.poke(true.B)
          dut.clock.step()
          dut.io.sharedResponse.valid.poke(false.B)
        }
      }

      val allCache = collectCache(8)
      respondCache(allCache.take(6))
      dut.clock.step(12)
      // The engine-local one-entry egress owns one fully formatted request
      // while the external shared port is backpressured.  Its PayloadSlot is
      // already recyclable, but its shared tag remains live until the
      // response arrives; check that exact ownership transfer rather than
      // treating the egress register as an extra hidden payload copy.
      dut.io.robFull.expect(false.B)
      dut.io.sharedRequest.valid.expect(true.B)
      dut.io.activeShared.expect(1.U)
      dut.io.sharedFull.expect(false.B)
      val firstShared = collectShared(4)
      dut.io.activeShared.expect(4.U)
      dut.io.sharedFull.expect(true.B)
      respondShared(firstShared)

      respondCache(allCache.drop(6))
      dut.clock.step(12)
      dut.io.sharedRequest.valid.expect(true.B)
      dut.io.activeShared.expect(1.U)
      dut.io.sharedFull.expect(false.B)
      val secondShared = collectShared(4)
      dut.io.activeShared.expect(4.U)
      dut.io.sharedFull.expect(true.B)
      respondShared(secondShared)

      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 80) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 80)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeRequests.expect(0.U)
      dut.io.activeShared.expect(0.U)
    }
  }

  "S2G write-ack scoreboard sustains more than four accepted writes" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = TmaV2Spec.DefaultWriteAckEntries,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(2.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionS2G.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      for (window <- 0 until 8) {
        val global = 0x3000 + window * 0x100
        val shared = 0x5000 + window * 0x80
        dut.io.window.bits.last.poke((window == 7).B)
        for (lane <- 0 until 8) {
          val active = lane < 2
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((global + lane * 128).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
        dut.io.window.valid.poke(true.B)
        while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }

      // Complete all eight full-window shared reads while deliberately
      // withholding cache write acceptance and every write acknowledgement.
      for (sharedRead <- 0 until 8) {
        dut.io.sharedRequest.ready.poke(true.B)
        var waited = 0
        while (!dut.io.sharedRequest.valid.peekBoolean() && waited < 80) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 80,
          s"shared read $sharedRead stalled: windows=" +
            s"${dut.io.activeWindows.peekInt()} shared=" +
            s"${dut.io.activeShared.peekInt()}")
        dut.io.sharedRequest.bits.write.expect(false.B)
        val source = dut.io.sharedRequest.bits.source.peekInt()
        dut.clock.step()
        // Do not accidentally accept the next request while returning this
        // response; every accepted source must receive exactly one response.
        dut.io.sharedRequest.ready.poke(false.B)
        dut.io.sharedResponse.bits.source.poke(source.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)
      }
      dut.io.sharedRequest.ready.poke(false.B)

      // The active S2G direction reuses source IDs 0..31 directly as the ack
      // scoreboard. Sixteen writes can become independently outstanding.
      val writeSources = mutable.ArrayBuffer.empty[BigInt]
      dut.io.cacheRequest.ready.poke(true.B)
      var writeCycles = 0
      while (writeSources.size < 16 && writeCycles < 240) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          dut.io.cacheRequest.bits.write.expect(true.B)
          writeSources += dut.io.cacheRequest.bits.source.peekInt()
        }
        dut.clock.step()
        writeCycles += 1
      }
      assert(writeSources.size == 16)
      assert(writeSources.distinct.size == 16)
      assert(writeSources.min == 0)
      assert(writeSources.max == 15)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.activeRequests.expect(0.U)
      dut.io.activeWriteAcks.expect(16.U)
      dut.io.requestFull.expect(false.B)

      writeSources.reverse.foreach { source =>
        dut.io.cacheResponse.bits.source.poke(source.U)
        dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        dut.io.cacheResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)
      }
      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 80) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 80)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeRequests.expect(0.U)
      dut.io.activeWriteAcks.expect(0.U)
    }
  }

  "S2G keeps ownership correct when the last window drains with an older ack" in {
    test(new TmaV2WindowEngine(
      windowEntries = 4, requestEntries = 2, sharedEntries = 2,
      writeAckEntries = 4,
      mmuEnabled = false)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(true.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(false.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)
      dut.io.command.bits.wid.poke(1.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionS2G.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(0.U)
      dut.io.command.bits.group.poke(2.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(0.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      def prepareWindow(index: Int, last: Boolean): Unit = {
        val global = 0x7000 + index * 0x80
        val shared = 0x9000 + index * 0x80
        dut.io.window.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val active = lane == 0
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((global + lane * 16).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(shared.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
      }

      def finishSharedRead(): Unit = {
        dut.io.sharedRequest.ready.poke(true.B)
        var waited = 0
        while (!dut.io.sharedRequest.valid.peekBoolean() && waited < 80) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 80)
        dut.io.sharedRequest.bits.write.expect(false.B)
        val source = dut.io.sharedRequest.bits.source.peekInt()
        dut.clock.step()
        dut.io.sharedRequest.ready.poke(false.B)
        dut.io.sharedResponse.bits.source.poke(source.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)
      }

      def acceptWrite(): BigInt = {
        dut.io.cacheRequest.ready.poke(true.B)
        var waited = 0
        while (!dut.io.cacheRequest.valid.peekBoolean() && waited < 80) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 80)
        dut.io.cacheRequest.bits.write.expect(true.B)
        val source = dut.io.cacheRequest.bits.source.peekInt()
        dut.clock.step()
        dut.io.cacheRequest.ready.poke(false.B)
        source
      }

      prepareWindow(0, last = false)
      dut.io.window.valid.poke(true.B)
      while (!dut.io.window.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.window.valid.poke(false.B)
      finishSharedRead()
      val firstAck = acceptWrite()
      dut.io.activeWriteAcks.expect(1.U)

      // Accepting the final input window while the older write ack returns
      // exercises simultaneous commandOutstanding increment and
      // commandWriteAcks decrement.  The command must not retire early.
      prepareWindow(1, last = true)
      dut.io.window.valid.poke(true.B)
      dut.io.cacheResponse.bits.source.poke(firstAck.U)
      dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
      dut.io.cacheResponse.valid.poke(true.B)
      dut.io.window.ready.expect(true.B)
      dut.io.cacheResponse.ready.expect(true.B)
      dut.clock.step()
      dut.io.window.valid.poke(false.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.valid.expect(false.B)
      dut.io.activeWriteAcks.expect(0.U)
      dut.io.activeWindows.expect(1.U)

      finishSharedRead()
      val finalAck = acceptWrite()
      dut.io.completion.valid.expect(false.B)
      dut.io.cacheResponse.bits.source.poke(finalAck.U)
      dut.io.cacheResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.cacheResponse.valid.poke(false.B)
      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 20) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 20)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeWriteAcks.expect(0.U)
    }
  }

  "window engine coalesces command-local translations and reuses a cached page" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 8,
      mmuEnabled = true)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.command.valid.poke(false.B)
      dut.io.seal.valid.poke(false.B)
      dut.io.window.valid.poke(false.B)
      dut.io.tlbRequest.ready.poke(false.B)
      dut.io.tlbResponse.valid.poke(false.B)
      dut.io.sharedRequest.ready.poke(true.B)
      dut.io.sharedResponse.valid.poke(false.B)
      dut.io.cacheRequest.ready.poke(true.B)
      dut.io.cacheResponse.valid.poke(false.B)
      dut.io.completion.ready.poke(true.B)
      dut.clock.step(2)

      def issueCommand(id: Int, asid: Int): Unit = {
        dut.io.command.bits.wid.poke(1.U)
        dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
        dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
        dut.io.command.bits.oobFill.poke(false.B)
        dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
        dut.io.command.bits.asid.poke(asid.U)
        dut.io.command.bits.group.poke(0.U)
        dut.io.command.bits.barrierValid.poke(false.B)
        dut.io.command.bits.barrierId.poke(0.U)
        dut.io.command.bits.barrierGeneration.poke(0.U)
        dut.io.command.bits.transactionBytes.poke(0.U)
        dut.io.command.valid.poke(true.B)
        var waited = 0
        while (!dut.io.command.ready.peekBoolean() && waited < 32) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 32,
          s"command 0x${id.toHexString} did not acquire the single slot")
        dut.clock.step()
        dut.io.command.valid.poke(false.B)
      }

      def issueWindow(id: Int, address: Int, last: Boolean): Unit = {
        dut.io.window.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val active = lane == 0
          dut.io.window.bits.lanes(lane).valid.poke(active.B)
          dut.io.window.bits.lanes(lane).globalAddress
            .poke((address + lane * 16).U)
          dut.io.window.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.window.bits.sharedBase.poke(0x2000.U)
          dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.window.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
        dut.io.window.valid.poke(true.B)
        var waited = 0
        while (!dut.io.window.ready.peekBoolean() && waited < 32) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 32,
          s"window for command 0x${id.toHexString} did not issue")
        dut.clock.step()
        dut.io.window.valid.poke(false.B)
      }

      issueCommand(0x21, 5)
      issueWindow(0x21, 0x1000, last = false)
      issueWindow(0x21, 0x1080, last = false)

      dut.io.tlbRequest.ready.poke(true.B)
      var tlbCycles = 0
      while (!dut.io.tlbRequest.valid.peekBoolean() && tlbCycles < 30) {
        dut.clock.step()
        tlbCycles += 1
      }
      assert(tlbCycles < 30)
      val tlbSource = dut.io.tlbRequest.bits.source.peekInt()
      dut.io.tlbRequest.bits.virtualAddress.expect(0x1000.U)
      dut.io.tlbRequest.bits.asid.expect(5.U)
      dut.clock.step()
      dut.io.tlbRequest.ready.poke(false.B)

      var coalesced = dut.io.translationCoalesce.peekBoolean()
      for (_ <- 0 until 5) {
        dut.io.tlbRequest.valid.expect(false.B)
        coalesced ||= dut.io.translationCoalesce.peekBoolean()
        dut.clock.step()
      }
      assert(coalesced, "same command+ASID+VPN misses did not coalesce")

      dut.io.cacheRequest.ready.poke(false.B)
      dut.io.tlbResponse.bits.source.poke(tlbSource.U)
      dut.io.tlbResponse.bits.physicalAddress.poke(0x9000.U)
      dut.io.tlbResponse.valid.poke(true.B)
      dut.clock.step()
      dut.io.tlbResponse.valid.poke(false.B)

      issueWindow(0x21, 0x1100, last = true)

      dut.io.cacheRequest.ready.poke(true.B)
      val cacheSources = mutable.ArrayBuffer.empty[BigInt]
      val physicalLines = mutable.Set.empty[BigInt]
      var translationHit = false
      var cacheCycles = 0
      while (cacheSources.size < 3 && cacheCycles < 80) {
        if (dut.io.cacheRequest.valid.peekBoolean()) {
          cacheSources += dut.io.cacheRequest.bits.source.peekInt()
          physicalLines +=
            dut.io.cacheRequest.bits.physicalAddress.peekInt()
        }
        translationHit ||= dut.io.translationHit.peekBoolean()
        dut.io.tlbRequest.valid.expect(false.B)
        dut.clock.step()
        cacheCycles += 1
      }
      assert(cacheSources.size == 3)
      assert(physicalLines == Set(BigInt(0x9000), BigInt(0x9080),
        BigInt(0x9100)))
      assert(translationHit, "third same-page window did not hit translation cache")

      // Hold the shared sink while returning cache lines so the decoupled
      // G2S path cannot retire requests before the collector observes them.
      dut.io.sharedRequest.ready.poke(false.B)
      cacheSources.reverse.foreach { source =>
        dut.io.cacheResponse.bits.source.poke(source.U)
        dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        dut.io.cacheResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)
      }

      val sharedSources = mutable.ArrayBuffer.empty[BigInt]
      dut.io.sharedRequest.ready.poke(true.B)
      var sharedCycles = 0
      while (sharedSources.size < 3 && sharedCycles < 80) {
        if (dut.io.sharedRequest.valid.peekBoolean()) {
          dut.io.sharedRequest.bits.write.expect(true.B)
          sharedSources += dut.io.sharedRequest.bits.source.peekInt()
        }
        dut.clock.step()
        sharedCycles += 1
      }
      assert(sharedSources.size == 3)
      sharedSources.reverse.foreach { source =>
        dut.io.sharedResponse.bits.source.poke(source.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)
      }

      var completionCycles = 0
      while (!dut.io.completion.valid.peekBoolean() &&
          completionCycles < 50) {
        dut.clock.step()
        completionCycles += 1
      }
      assert(completionCycles < 50)
      dut.clock.step()
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)

      // A VPN match must not cross either a command lifetime or an ASID.
      // The engine is deliberately single-active, so exercise the two ASIDs as
      // consecutive commands instead of occupying two command slots.
      val secondPhaseAsids = mutable.ArrayBuffer.empty[BigInt]
      for ((id, asid, physical) <- Seq(
          (0x31, 1, 0xa000), (0x32, 2, 0xb000))) {
        dut.io.cacheRequest.ready.poke(false.B)
        dut.io.sharedRequest.ready.poke(false.B)
        issueCommand(id, asid)
        issueWindow(id, 0x3000, last = true)
        dut.io.tlbRequest.ready.poke(true.B)
        var waited = 0
        while (!dut.io.tlbRequest.valid.peekBoolean() && waited < 30) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 30)
        val tlbSource = dut.io.tlbRequest.bits.source.peekInt()
        dut.io.tlbRequest.bits.asid.expect(asid.U)
        secondPhaseAsids += dut.io.tlbRequest.bits.asid.peekInt()
        dut.clock.step()
        dut.io.tlbRequest.ready.poke(false.B)
        dut.io.tlbResponse.bits.source.poke(tlbSource.U)
        dut.io.tlbResponse.bits.physicalAddress
          .poke(physical.U)
        dut.io.tlbResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.tlbResponse.valid.poke(false.B)

        var cacheCycles = 0
        while (!dut.io.cacheRequest.valid.peekBoolean() &&
            cacheCycles < 50) {
          dut.clock.step()
          cacheCycles += 1
        }
        assert(cacheCycles < 50)
        val cacheSource = dut.io.cacheRequest.bits.source.peekInt()
        dut.io.cacheRequest.bits.physicalAddress.expect(physical.U)
        dut.io.cacheRequest.ready.poke(true.B)
        dut.clock.step()
        dut.io.cacheRequest.ready.poke(false.B)
        dut.io.cacheResponse.bits.source.poke(cacheSource.U)
        dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        dut.io.cacheResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.cacheResponse.valid.poke(false.B)

        var sharedCycles = 0
        while (!dut.io.sharedRequest.valid.peekBoolean() &&
            sharedCycles < 50) {
          dut.clock.step()
          sharedCycles += 1
        }
        assert(sharedCycles < 50)
        val sharedSource = dut.io.sharedRequest.bits.source.peekInt()
        dut.io.sharedRequest.ready.poke(true.B)
        dut.clock.step()
        dut.io.sharedRequest.ready.poke(false.B)
        dut.io.sharedResponse.bits.source.poke(sharedSource.U)
        dut.io.sharedResponse.bits.data.poke(0.U)
        dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        dut.io.sharedResponse.valid.poke(true.B)
        dut.clock.step()
        dut.io.sharedResponse.valid.poke(false.B)

        var completionCycles = 0
        while (!dut.io.completion.valid.peekBoolean() &&
            completionCycles < 50) {
          dut.clock.step()
          completionCycles += 1
        }
        assert(completionCycles < 50)
        dut.clock.step()
      }
      assert(secondPhaseAsids.toSet == Set(BigInt(1), BigInt(2)))
      dut.io.activeCommands.expect(0.U)
    }
  }

  "16KB dense command performs exactly four external page translations" in {
    test(new TmaV2WindowEngine(
      windowEntries = 8, requestEntries = 4, sharedEntries = 4,
      writeAckEntries = 8,
      mmuEnabled = true)).withAnnotations(Seq(CachingAnnotation)) { dut =>
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
      dut.io.command.bits.wid.poke(1.U)
      dut.io.command.bits.copyDirection.poke(TmaV2Spec.DirectionG2S.U)
      dut.io.command.bits.dtype.poke(TmaV2Spec.DTypeU8.U)
      dut.io.command.bits.oobFill.poke(false.B)
      dut.io.command.bits.reduceMode.poke(TmaV2Spec.ReduceCopy.U)
      dut.io.command.bits.asid.poke(9.U)
      dut.io.command.bits.group.poke(0.U)
      dut.io.command.bits.barrierValid.poke(false.B)
      dut.io.command.bits.barrierId.poke(0.U)
      dut.io.command.bits.barrierGeneration.poke(0.U)
      dut.io.command.bits.transactionBytes.poke(16384.U)
      dut.io.command.valid.poke(true.B)
      while (!dut.io.command.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.command.valid.poke(false.B)

      val tlbResponses = mutable.Queue.empty[(BigInt, BigInt)]
      val cacheResponses = mutable.Queue.empty[BigInt]
      val sharedResponses = mutable.Queue.empty[BigInt]
      val translatedVpns = mutable.ArrayBuffer.empty[BigInt]
      var nextWindow = 0
      var completionSeen = false
      var cycles = 0

      while (!completionSeen && cycles < 10000) {
        if (nextWindow < 128) {
          val globalLine = 0x10000 + nextWindow * 128
          val sharedLine = 0x2000 + nextWindow * 128
          dut.io.window.bits.last.poke((nextWindow == 127).B)
          for (lane <- 0 until 8) {
            dut.io.window.bits.lanes(lane).valid.poke(true.B)
            dut.io.window.bits.lanes(lane).globalAddress
              .poke((globalLine + lane * 16).U)
            dut.io.window.bits.lanes(lane).globalBytes.poke(16.U)
            dut.io.window.bits.sharedBase.poke(sharedLine.U)
            dut.io.window.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
            dut.io.window.bits.lanes(lane).sharedBytes.poke(16.U)
          }
          dut.io.window.valid.poke(true.B)
        } else {
          dut.io.window.valid.poke(false.B)
        }

        dut.io.tlbResponse.valid.poke(tlbResponses.nonEmpty.B)
        if (tlbResponses.nonEmpty) {
          dut.io.tlbResponse.bits.source.poke(tlbResponses.front._1.U)
          dut.io.tlbResponse.bits.physicalAddress
            .poke(tlbResponses.front._2.U)
        }
        dut.io.cacheResponse.valid.poke(cacheResponses.nonEmpty.B)
        if (cacheResponses.nonEmpty) {
          dut.io.cacheResponse.bits.source.poke(cacheResponses.front.U)
          dut.io.cacheResponse.bits.data.foreach(_.poke(0.U))
        }
        dut.io.sharedResponse.valid.poke(sharedResponses.nonEmpty.B)
        if (sharedResponses.nonEmpty) {
          dut.io.sharedResponse.bits.source.poke(sharedResponses.front.U)
          dut.io.sharedResponse.bits.data.poke(0.U)
          dut.io.sharedResponse.bits.wordMask.poke("hffffffff".U)
        }

        val windowFire = nextWindow < 128 &&
          dut.io.window.ready.peekBoolean()
        val tlbRequestFire = dut.io.tlbRequest.valid.peekBoolean()
        val cacheRequestFire = dut.io.cacheRequest.valid.peekBoolean()
        val sharedRequestFire = dut.io.sharedRequest.valid.peekBoolean()
        val tlbResponseFire = tlbResponses.nonEmpty &&
          dut.io.tlbResponse.ready.peekBoolean()
        val cacheResponseFire = cacheResponses.nonEmpty &&
          dut.io.cacheResponse.ready.peekBoolean()
        val sharedResponseFire = sharedResponses.nonEmpty &&
          dut.io.sharedResponse.ready.peekBoolean()
        completionSeen = dut.io.completion.valid.peekBoolean()

        val newTlbResponse = if (tlbRequestFire) {
          val virtual = dut.io.tlbRequest.bits.virtualAddress.peekInt()
          translatedVpns += (virtual >> 12)
          Some((dut.io.tlbRequest.bits.source.peekInt(),
            virtual + 0x40000000L))
        } else None
        val newCacheResponse = if (cacheRequestFire)
          Some(dut.io.cacheRequest.bits.source.peekInt()) else None
        val newSharedResponse = if (sharedRequestFire) {
          dut.io.sharedRequest.bits.write.expect(true.B)
          Some(dut.io.sharedRequest.bits.source.peekInt())
        } else None

        dut.clock.step()
        if (windowFire) nextWindow += 1
        if (tlbResponseFire) tlbResponses.dequeue()
        if (cacheResponseFire) cacheResponses.dequeue()
        if (sharedResponseFire) sharedResponses.dequeue()
        newTlbResponse.foreach(tlbResponses.enqueue(_))
        newCacheResponse.foreach(cacheResponses.enqueue(_))
        newSharedResponse.foreach(sharedResponses.enqueue(_))
        cycles += 1
      }

      assert(cycles < 10000, "16KB command did not complete")
      assert(nextWindow == 128)
      assert(translatedVpns == Seq(0x10, 0x11, 0x12, 0x13),
        s"expected one miss per page, saw VPNs $translatedVpns")
      dut.io.activeCommands.expect(0.U)
      dut.io.activeWindows.expect(0.U)
      dut.io.activeRequests.expect(0.U)
      dut.io.activeShared.expect(0.U)
    }
  }

}
