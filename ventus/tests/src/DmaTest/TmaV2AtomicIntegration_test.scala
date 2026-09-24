package DmaTest

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.internal.CachingAnnotation
import config.config.Parameters
import L1Cache.AtomicUnit.AtomicUnit
import L1Cache.MyConfig
import L2cache._
import org.scalatest.freespec.AnyFreeSpec
import pipeline._
import top.parameters._

/**
  * Closed TMA -> AtomicUnit -> line-memory integration harness.
  *
  * The normal SM/cluster source adapter is reproduced at the boundary.  The
  * memory endpoint deliberately permits a Put acknowledgement to be held so
  * the test can prove that TMA completion follows the final AMO acknowledgement.
  */
class TmaV2AtomicIntegrationHarness(implicit p: Parameters) extends Module {
  private val params = InclusiveCacheParameters_lite(
    CacheParameters(2, l2cache_NSets, l2cache_NWays, num_l2cache,
      blockBytes = l2cache_BlockWords << 2,
      beatBytes = l2cache_BlockWords << 2),
    InclusiveCacheMicroParameters(l2cache_writeBytes, l2cache_memCycles,
      l2cache_portFactor, num_warp, num_sm, num_sm_in_cluster, num_cluster,
      dcache_MshrEntry, dcache_NSets, atuns_NInfWriteEntry),
    control = false, mmu = MMU_ENABLED)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val commandReady = Output(Bool())
    val reduceMode = Input(UInt(3.W))
    val reduceType = Input(UInt(2.W))
    val loadMemory = Input(Bool())
    val initialWords = Input(Vec(dcache_BlockWords, UInt(32.W)))
    val sharedWords = Input(Vec(num_thread, UInt(32.W)))
    val allowPutAck = Input(Bool())
    val finalWords = Output(Vec(dcache_BlockWords, UInt(32.W)))
    val completion = Valid(new DmaCompletion)
    val status = Valid(new DmaStatusUpdate)
    val amoRequests = Output(UInt(8.W))
    val putAcks = Output(UInt(8.W))
    val putPending = Output(Bool())
  })

  val core = Module(new TmaV2DmaCore(
    windowEntries = 8, requestEntries = 4,
    sharedEntries = 4, writeAckEntries = 8))
  val atomic = Module(new AtomicUnit(params))

  core.io.dma_req.valid := io.start
  core.io.dma_req.bits := 0.U.asTypeOf(new vExeData)
  core.io.dma_req.bits.ctrl.dma := true.B
  core.io.dma_req.bits.ctrl.funct := TmaV2Spec.FunctBulkS2G.U
  core.io.dma_req.bits.ctrl.wid := 1.U
  core.io.dma_req.bits.ctrl.dma_group := 2.U
  core.io.dma_req.bits.ctrl.inst :=
    Cat(io.reduceMode, io.reduceType, 0.U(27.W))
  core.io.dma_req.bits.ctrl.asid.foreach(_ := 0.U)
  core.io.dma_req.bits.in1(0) := "h2000".U
  core.io.dma_req.bits.in2(0) := 16.U
  core.io.dma_req.bits.in3(0) := "h1000".U
  core.io.dma_req.bits.mask.foreach(_ := true.B)
  io.commandReady := core.io.dma_req.ready

  core.io.txReserve.ready := true.B
  core.io.txReserveResponse.valid := false.B
  core.io.txReserveResponse.bits :=
    0.U.asTypeOf(new DmaTxReserveResponse)
  core.io.killReq.valid := false.B
  core.io.killReq.bits.asid := 0.U
  core.io.perfEnable := false.B
  core.io.perfReset := false.B

  // One-cycle TLB loopback preserves the real request/response ownership
  // boundary while using an identity virtual-to-physical mapping.
  val tlbPending = RegInit(false.B)
  val tlbAddress = Reg(UInt(32.W))
  core.io.to_l2TLB.ready := !tlbPending
  core.io.from_l2TLB.valid := tlbPending
  core.io.from_l2TLB.bits.paddr := tlbAddress
  when(core.io.to_l2TLB.fire) {
    tlbPending := true.B
    tlbAddress := core.io.to_l2TLB.bits.vaddr
  }
  when(core.io.from_l2TLB.fire) { tlbPending := false.B }

  // Return the selected shared words one cycle after the accepted S2G read.
  val sharedPending = RegInit(false.B)
  val sharedSource = Reg(UInt(core.io.shared_req.bits.instrId.getWidth.W))
  val sharedMask = Reg(Vec(num_thread, Bool()))
  core.io.shared_req.ready := !sharedPending
  when(core.io.shared_req.fire) {
    assert(!core.io.shared_req.bits.isWrite)
    sharedPending := true.B
    sharedSource := core.io.shared_req.bits.instrId
    for (lane <- 0 until num_thread) {
      sharedMask(lane) :=
        core.io.shared_req.bits.perLaneAddr(lane).activeMask
    }
  }
  core.io.shared_rsp.valid := sharedPending
  core.io.shared_rsp.bits.instrId := sharedSource
  core.io.shared_rsp.bits.isWrite := false.B
  core.io.shared_rsp.bits.isMBarrier := false.B
  for (lane <- 0 until num_thread) {
    core.io.shared_rsp.bits.data(lane) := io.sharedWords(lane)
    core.io.shared_rsp.bits.activeMask(lane) := sharedMask(lane)
  }
  when(core.io.shared_rsp.fire) { sharedPending := false.B }

  // Reproduce the SM-to-cluster bundle conversion used by GPGPU_top.
  val atomicRequestIngress = Module(new Queue(
    new TLBundleA_lite(params), 1, pipe = false, flow = false))
  atomicRequestIngress.io.enq.valid := core.io.dma_cache_req.valid
  core.io.dma_cache_req.ready := atomicRequestIngress.io.enq.ready
  atomicRequestIngress.io.enq.bits.opcode :=
    core.io.dma_cache_req.bits.a_opcode
  atomicRequestIngress.io.enq.bits.param := core.io.dma_cache_req.bits.a_param
  atomicRequestIngress.io.enq.bits.size := 0.U
  atomicRequestIngress.io.enq.bits.source :=
    core.io.dma_cache_req.bits.a_source
  atomicRequestIngress.io.enq.bits.address :=
    core.io.dma_cache_req.bits.a_addr.get
  atomicRequestIngress.io.enq.bits.mask :=
    core.io.dma_cache_req.bits.a_mask.asUInt
  atomicRequestIngress.io.enq.bits.data :=
    core.io.dma_cache_req.bits.a_data.asUInt
  atomicRequestIngress.io.enq.bits.spike_info.foreach(_ :=
    0.U.asTypeOf(new top.cache_spike_info(mmu.SV32)))
  atomic.io.L12ATUmemReq <> atomicRequestIngress.io.deq

  core.io.dma_cache_rsp.valid := atomic.io.ATU2L1memRsp.valid
  atomic.io.ATU2L1memRsp.ready := core.io.dma_cache_rsp.ready
  core.io.dma_cache_rsp.bits.d_opcode :=
    atomic.io.ATU2L1memRsp.bits.opcode
  core.io.dma_cache_rsp.bits.d_param :=
    atomic.io.ATU2L1memRsp.bits.param
  core.io.dma_cache_rsp.bits.d_source :=
    atomic.io.ATU2L1memRsp.bits.source
  core.io.dma_cache_rsp.bits.d_addr :=
    atomic.io.ATU2L1memRsp.bits.address
  core.io.dma_cache_rsp.bits.d_data :=
    atomic.io.ATU2L1memRsp.bits.data.asTypeOf(
      Vec(dcache_BlockWords, UInt(32.W)))

  // A single cache-line memory is sufficient for the final-value checks;
  // crossing-line and 4 KiB address generation remain covered at DmaCore.
  val memory = RegInit(0.U((dcache_BlockWords * 32).W))
  when(io.loadMemory) { memory := io.initialWords.asUInt }
  io.finalWords := memory.asTypeOf(Vec(dcache_BlockWords, UInt(32.W)))

  val memoryResponsePending = RegInit(false.B)
  val memoryResponseIsPut = RegInit(false.B)
  val memoryResponseSource =
    Reg(UInt(atomic.io.ATU2L2memReq.bits.source.getWidth.W))
  val memoryResponseAddress = Reg(UInt(32.W))
  val memoryResponseData = Reg(UInt((dcache_BlockWords * 32).W))
  atomic.io.ATU2L2memReq.ready := !memoryResponsePending
  when(atomic.io.ATU2L2memReq.fire) {
    val request = atomic.io.ATU2L2memReq.bits
    val isPut = request.opcode === 0.U || request.opcode === 1.U
    memoryResponsePending := true.B
    memoryResponseIsPut := isPut
    memoryResponseSource := request.source
    memoryResponseAddress := request.address
    memoryResponseData := memory
    when(isPut) {
      val oldBytes = memory.asTypeOf(Vec(dcache_BlockWords * 4, UInt(8.W)))
      val newBytes = request.data.asTypeOf(
        Vec(dcache_BlockWords * 4, UInt(8.W)))
      val merged = Wire(Vec(dcache_BlockWords * 4, UInt(8.W)))
      for (byte <- 0 until dcache_BlockWords * 4) {
        merged(byte) := Mux(request.mask(byte), newBytes(byte), oldBytes(byte))
      }
      memory := merged.asUInt
    }
  }

  atomic.io.L22ATUmemRsp.valid := memoryResponsePending &&
    (!memoryResponseIsPut || io.allowPutAck)
  atomic.io.L22ATUmemRsp.bits.opcode :=
    Mux(memoryResponseIsPut, 0.U, 1.U)
  atomic.io.L22ATUmemRsp.bits.param := 0.U
  atomic.io.L22ATUmemRsp.bits.size := 0.U
  atomic.io.L22ATUmemRsp.bits.source := memoryResponseSource
  atomic.io.L22ATUmemRsp.bits.address := memoryResponseAddress
  atomic.io.L22ATUmemRsp.bits.data := memoryResponseData
  when(atomic.io.L22ATUmemRsp.fire) { memoryResponsePending := false.B }

  val amoRequests = RegInit(0.U(8.W))
  val putAcks = RegInit(0.U(8.W))
  when(core.io.dma_cache_req.fire) { amoRequests := amoRequests + 1.U }
  when(atomic.io.L22ATUmemRsp.fire && memoryResponseIsPut) {
    putAcks := putAcks + 1.U
  }
  when(io.loadMemory) {
    amoRequests := 0.U
    putAcks := 0.U
  }
  io.amoRequests := amoRequests
  io.putAcks := putAcks
  io.putPending := memoryResponsePending && memoryResponseIsPut

  core.io.tma_completion.ready := true.B
  io.completion.valid := core.io.tma_completion.valid
  io.completion.bits := core.io.tma_completion.bits
  io.status := core.io.status

  when(!reset.asBool) {
    assert(!io.completion.valid || putAcks === 4.U,
      "bulk reduce completion must follow the fourth AMO Put acknowledgement")
  }
}

class TmaV2AtomicIntegration_test
    extends AnyFreeSpec with ChiselScalatestTester {
  implicit val p: Parameters = (new MyConfig).toInstance

  "TMA bulk reduce commits every operation through AtomicUnit" in {
    test(new TmaV2AtomicIntegrationHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.start.poke(false.B)
      dut.io.loadMemory.poke(false.B)
      dut.io.allowPutAck.poke(true.B)
      dut.io.reduceMode.poke(0.U)
      dut.io.reduceType.poke(0.U)
      dut.io.initialWords.foreach(_.poke(0.U))
      dut.io.sharedWords.foreach(_.poke(0.U))
      dut.clock.step(2)

      val cases = Seq(
        (TmaV2Spec.ReduceAdd, TmaV2Spec.BulkReduceTypeU32,
          BigInt("ffffffff", 16), BigInt(1), BigInt(0)),
        (TmaV2Spec.ReduceMin, TmaV2Spec.BulkReduceTypeS32,
          BigInt("80000000", 16), BigInt("7fffffff", 16),
          BigInt("80000000", 16)),
        (TmaV2Spec.ReduceMin, TmaV2Spec.BulkReduceTypeU32,
          BigInt("80000000", 16), BigInt("7fffffff", 16),
          BigInt("7fffffff", 16)),
        (TmaV2Spec.ReduceMax, TmaV2Spec.BulkReduceTypeS32,
          BigInt("80000000", 16), BigInt("7fffffff", 16),
          BigInt("7fffffff", 16)),
        (TmaV2Spec.ReduceMax, TmaV2Spec.BulkReduceTypeU32,
          BigInt("80000000", 16), BigInt("7fffffff", 16),
          BigInt("80000000", 16)),
        (TmaV2Spec.ReduceAnd, TmaV2Spec.BulkReduceTypeB32,
          BigInt("33333333", 16), BigInt("0ff00ff0", 16),
          BigInt("03300330", 16)),
        (TmaV2Spec.ReduceOr, TmaV2Spec.BulkReduceTypeB32,
          BigInt("30303030", 16), BigInt("0ff00ff0", 16),
          BigInt("3ff03ff0", 16)),
        (TmaV2Spec.ReduceXor, TmaV2Spec.BulkReduceTypeB32,
          BigInt("33333333", 16), BigInt("0ff00ff0", 16),
          BigInt("3cc33cc3", 16)))

      for (((mode, elementType, oldValue, operand, expected), index) <-
          cases.zipWithIndex) {
        for (word <- 0 until dcache_BlockWords) {
          dut.io.initialWords(word).poke(oldValue.U)
          dut.io.sharedWords(word).poke(
            (if (word < 4) operand else BigInt(0)).U)
        }
        dut.io.reduceMode.poke(mode.U)
        dut.io.reduceType.poke(elementType.U)
        dut.io.loadMemory.poke(true.B)
        dut.clock.step()
        dut.io.loadMemory.poke(false.B)

        if (index == 0) dut.io.allowPutAck.poke(false.B)
        dut.io.commandReady.expect(true.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        if (index == 0) {
          var waitPut = 0
          while (!dut.io.putPending.peekBoolean() && waitPut < 160) {
            dut.io.completion.valid.expect(false.B)
            dut.clock.step()
            waitPut += 1
          }
          assert(waitPut < 160)
          dut.clock.step(4)
          dut.io.completion.valid.expect(false.B)
          dut.io.putAcks.expect(0.U)
          dut.io.allowPutAck.poke(true.B)
        }

        var completed = false
        var cycles = 0
        while (!completed && cycles < 500) {
          completed = dut.io.completion.valid.peekBoolean()
          if (!completed) dut.clock.step()
          cycles += 1
        }
        assert(completed, s"reduce mode=$mode did not complete")
        dut.io.amoRequests.expect(4.U)
        dut.io.putAcks.expect(4.U)
        for (word <- 0 until 4) dut.io.finalWords(word).expect(expected.U)
        dut.io.status.valid.expect(false.B)
        dut.clock.step()
      }
    }
  }
}
