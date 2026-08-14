package top
import chisel3._
import chisel3.util._
import parameters._
import L1Cache.MyConfig
import L1Cache.ICache.{InstructionCache, ICacheMemReq_p, ICacheMemRsp, ICacheBundle}
import pipeline.{CTAreqData, CTArspData, CTA2warp, pipe}
import pipeline.{ICachePipeReq_np, ICachePipeRsp_np, DCacheCoreReq_np, DCacheCoreRsp_np, ShareMemCoreReq_np}
import pipeline.{InstClassPerfCounters, PipelinePerfCounters, TmaPerfCounters}
import L1Cache.ShareMem.SharedMemory
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public, Instantiate}
import config.config.Parameters
import gvm._
import mmu.L1TlbAutoReflect

class SMIO_icache(SV: Option[mmu.SVParam] = None)(implicit p: Parameters) extends ICacheBundle {
  val req = DecoupledIO(new ICacheMemReq_p(SV.getOrElse(mmu.SV32)))
  val rsp = Flipped(DecoupledIO(new ICacheMemRsp()))
}

class NoCacheDcacheRspRoute extends Bundle {
  val isDma = Bool()
  val dmaSource = UInt(l1cache_sourceBits.W)
  val dmaAddr = UInt(xLen.W)
  val dmaRspOpcode = UInt(3.W)
}

@instantiable
class SM_wrapper_nocache() extends Module {
  val param = (new MyConfig).toInstance
  @public val sm_id = IO(Input(UInt(8.W)))
  @public val io = IO(new Bundle{
    val CTAreq = Flipped(Decoupled(new CTAreqData))
    val CTArsp = Decoupled(new CTArspData)
    val icache = new SMIO_icache()(param)
    val dcache_req = DecoupledIO(new DCacheCoreReq_np)
    val dcache_rsp = Flipped(DecoupledIO(new DCacheCoreRsp_np))
    val perfEnable = Input(Bool())
    val perfReset = Input(Bool())
    val pipeline_perf = if(PMU_PIPELINE) Some(Output(new PipelinePerfCounters)) else None
    val inst_class_perf = if(PMU_INST_CLASS) Some(Output(new InstClassPerfCounters)) else None
    val tma_perf = if(PMU_TMA) Some(Output(new TmaPerfCounters)) else None
    val icache_invalidate = Input(Bool())
  })

  val cta2warp = Module(new CTA2warp)
  cta2warp.io.CTAreq :<>= io.CTAreq
  io.CTArsp :<>= cta2warp.io.CTArsp

  val pipe = Module(new pipe())
  pipe.sm_id := sm_id
  pipe.io.perfEnable := io.perfEnable
  pipe.io.perfReset := io.perfReset
  io.pipeline_perf.foreach(_ := pipe.io.perf_pipeline.getOrElse(0.U.asTypeOf(new PipelinePerfCounters)))
  io.inst_class_perf.foreach(_ := pipe.io.perf_inst_class.getOrElse(0.U.asTypeOf(new InstClassPerfCounters)))
  io.tma_perf.foreach(_ := pipe.io.perf_tma.getOrElse(0.U.asTypeOf(new TmaPerfCounters)))

  val cnt = Counter(10)
  when(cnt.value < 5.U) { cnt.inc() }
  pipe.io.pc_reset := (cnt.value < 5.U) // Reset the pipe for the first 5 cycles

  pipe.io.warpReq :<>= cta2warp.io.warpReq
  cta2warp.io.warpRsp :<>= pipe.io.warpRsp
  cta2warp.io.wg_id_lookup := pipe.io.wg_id_lookup
  pipe.io.wg_id_tag:=cta2warp.io.wg_id_tag

  val sharedmem = Module(new SharedMemory()(param))
  val sharedReqArb = Module(new Arbiter(new ShareMemCoreReq_np, 2))
  sharedReqArb.io.in(0) <> pipe.io.shared_req
  sharedReqArb.io.in(1) <> pipe.io.dma_shared_req
  sharedmem.io.coreReq.bits.data := sharedReqArb.io.out.bits.data
  sharedmem.io.coreReq.bits.instrId := sharedReqArb.io.out.bits.instrId
  sharedmem.io.coreReq.bits.isWrite := sharedReqArb.io.out.bits.isWrite
  sharedmem.io.coreReq.bits.isMBarrier := sharedReqArb.io.out.bits.isMBarrier
  sharedmem.io.coreReq.bits.setIdx := sharedReqArb.io.out.bits.setIdx
  sharedmem.io.coreReq.bits.perLaneAddr := sharedReqArb.io.out.bits.perLaneAddr
  sharedmem.io.coreReq.bits.sourceTag := sharedReqArb.io.chosen === 1.U
  sharedmem.io.coreReq.valid := sharedReqArb.io.out.valid
  sharedReqArb.io.out.ready := sharedmem.io.coreReq.ready

  val sharedRspFromDma = sharedmem.io.coreRsp.bits.sourceTag
  pipe.io.shared_rsp.valid := sharedmem.io.coreRsp.valid && !sharedRspFromDma
  pipe.io.shared_rsp.bits.data := sharedmem.io.coreRsp.bits.data
  pipe.io.shared_rsp.bits.instrId := sharedmem.io.coreRsp.bits.instrId
  pipe.io.shared_rsp.bits.activeMask := sharedmem.io.coreRsp.bits.activeMask
  pipe.io.dma_shared_rsp.valid := sharedmem.io.coreRsp.valid && sharedRspFromDma
  pipe.io.dma_shared_rsp.bits.data := sharedmem.io.coreRsp.bits.data
  pipe.io.dma_shared_rsp.bits.instrId := sharedmem.io.coreRsp.bits.instrId
  pipe.io.dma_shared_rsp.bits.activeMask := sharedmem.io.coreRsp.bits.activeMask
  pipe.io.dma_shared_rsp.bits.isWrite := sharedmem.io.coreRsp.bits.isWrite
  pipe.io.dma_shared_rsp.bits.isMBarrier := sharedmem.io.coreRsp.bits.isMBarrier
  sharedmem.io.coreRsp.ready := Mux(sharedRspFromDma, pipe.io.dma_shared_rsp.ready, pipe.io.shared_rsp.ready)

  val icache = Module(new InstructionCache()(param))
  icache.io.invalidate := io.icache_invalidate
  // **** icache coreReq ****
  pipe.io.icache_req.ready:=icache.io.coreReq.ready
  icache.io.coreReq.valid:=pipe.io.icache_req.valid
  icache.io.coreReq.bits.addr:=pipe.io.icache_req.bits.addr
  icache.io.coreReq.bits.warpid:=pipe.io.icache_req.bits.warpid
  icache.io.coreReq.bits.mask:=pipe.io.icache_req.bits.mask
  if(MMU_ENABLED){ icache.io.coreReq.bits.asid.get := pipe.io.icache_req.bits.asid.get }
  icache.io.coreReq.bits.spike_info.foreach( _ := DontCare )
  // **** icache coreRsp ****
  pipe.io.icache_rsp.valid:=icache.io.coreRsp.valid
  pipe.io.icache_rsp.bits.warpid:=icache.io.coreRsp.bits.warpid
  pipe.io.icache_rsp.bits.data:=icache.io.coreRsp.bits.data
  pipe.io.icache_rsp.bits.addr:=icache.io.coreRsp.bits.addr
  pipe.io.icache_rsp.bits.status:=icache.io.coreRsp.bits.status
  pipe.io.icache_rsp.bits.mask:=icache.io.coreRsp.bits.mask
  icache.io.coreRsp.ready:=pipe.io.icache_rsp.ready
  icache.io.externalFlushPipe.bits.warpid :=pipe.io.externalFlushPipe.bits
  icache.io.externalFlushPipe.valid :=pipe.io.externalFlushPipe.valid

  io.icache.req :<>= icache.io.memReq
  icache.io.memRsp :<>= io.icache.rsp
  val dmaDcacheReq = Wire(DecoupledIO(new DCacheCoreReq_np))
  val dmaLineAddr = pipe.io.dma_cache_req.bits.a_addr.get
  val dmaBlockByteBits = log2Ceil(dcache_BlockWords * BytesOfWord)
  val dmaTlOpcode = pipe.io.dma_cache_req.bits.a_opcode
  val dmaTlParam = pipe.io.dma_cache_req.bits.a_param
  val dmaIsAtomic = dmaTlOpcode === 2.U || dmaTlOpcode === 3.U
  val dmaCoreAtomicParam = Mux(dmaTlOpcode === 2.U,
    MuxLookup(dmaTlParam, 0.U)(Seq(
      0.U -> 4.U, // signed min
      1.U -> 5.U, // signed max
      2.U -> 6.U, // unsigned min
      3.U -> 7.U, // unsigned max
      4.U -> 0.U  // add
    )),
    MuxLookup(dmaTlParam, 1.U)(Seq(
      0.U -> 1.U,  // xor
      1.U -> 2.U,  // or
      2.U -> 3.U,  // and
      3.U -> 15.U  // swap
    )))
  dmaDcacheReq.valid := pipe.io.dma_cache_req.valid
  pipe.io.dma_cache_req.ready := dmaDcacheReq.ready
  dmaDcacheReq.bits.instrId := 0.U
  dmaDcacheReq.bits.tag := dmaLineAddr(xLen - 1, dmaBlockByteBits + dcache_SetIdxBits)
  dmaDcacheReq.bits.setIdx := dmaLineAddr(dmaBlockByteBits + dcache_SetIdxBits - 1, dmaBlockByteBits)
  dmaDcacheReq.bits.asid.foreach(_ := 0.U)
  dmaDcacheReq.bits.opcode := Mux(
    dmaTlOpcode === 4.U,
    0.U,
    Mux(dmaTlOpcode === 0.U || dmaTlOpcode === 1.U, 1.U,
      Mux(dmaIsAtomic, 2.U, 3.U))
  )
  dmaDcacheReq.bits.param := Mux(dmaIsAtomic,
    dmaCoreAtomicParam, dmaTlParam.pad(4))
  dmaDcacheReq.bits.spike_info.foreach(_ := DontCare)
  (0 until num_thread).foreach { i =>
    // TileLink Get does not use a_mask.  The nocache endpoint must fetch the
    // complete cache line; the TMA engine applies its own payload byte mask
    // after the response returns.
    dmaDcacheReq.bits.perLaneAddr(i).activeMask :=
      dmaTlOpcode === 4.U || pipe.io.dma_cache_req.bits.a_mask(i).orR
    dmaDcacheReq.bits.perLaneAddr(i).blockOffset := i.U
    dmaDcacheReq.bits.perLaneAddr(i).wordOffset1H := pipe.io.dma_cache_req.bits.a_mask(i)
    dmaDcacheReq.bits.data(i) := pipe.io.dma_cache_req.bits.a_data(i)
  }

  val dcacheReqArb = Module(new Arbiter(new DCacheCoreReq_np, 2))
  dcacheReqArb.io.in(0) <> pipe.io.dcache_req
  dcacheReqArb.io.in(1) <> dmaDcacheReq

  val dcacheRspRouteQ = Module(new Queue(new NoCacheDcacheRspRoute, entries = 16))
  val dcacheReqWillRsp = dcacheReqArb.io.out.bits.opcode =/= 3.U
  val dcacheRouteReady = !dcacheReqWillRsp || dcacheRspRouteQ.io.enq.ready
  io.dcache_req.valid := dcacheReqArb.io.out.valid && dcacheRouteReady
  io.dcache_req.bits := dcacheReqArb.io.out.bits
  dcacheReqArb.io.out.ready := io.dcache_req.ready && dcacheRouteReady
  dcacheRspRouteQ.io.enq.valid := io.dcache_req.fire && dcacheReqWillRsp
  dcacheRspRouteQ.io.enq.bits.isDma := dcacheReqArb.io.chosen === 1.U
  dcacheRspRouteQ.io.enq.bits.dmaSource := pipe.io.dma_cache_req.bits.a_source
  dcacheRspRouteQ.io.enq.bits.dmaAddr := dmaLineAddr
  dcacheRspRouteQ.io.enq.bits.dmaRspOpcode := Mux(
    pipe.io.dma_cache_req.bits.a_opcode === 4.U,
    1.U,
    Mux(pipe.io.dma_cache_req.bits.a_opcode === 0.U || pipe.io.dma_cache_req.bits.a_opcode === 1.U, 0.U, 2.U)
  )

  val dcacheRspRouteValid = dcacheRspRouteQ.io.deq.valid
  val dcacheRspFromDma = dcacheRspRouteQ.io.deq.bits.isDma
  pipe.io.dcache_rsp.valid := io.dcache_rsp.valid && dcacheRspRouteValid && !dcacheRspFromDma
  pipe.io.dcache_rsp.bits := io.dcache_rsp.bits
  pipe.io.dma_cache_rsp.valid := io.dcache_rsp.valid && dcacheRspRouteValid && dcacheRspFromDma
  pipe.io.dma_cache_rsp.bits.d_opcode := dcacheRspRouteQ.io.deq.bits.dmaRspOpcode
  pipe.io.dma_cache_rsp.bits.d_param := 0.U
  pipe.io.dma_cache_rsp.bits.d_source := dcacheRspRouteQ.io.deq.bits.dmaSource
  pipe.io.dma_cache_rsp.bits.d_addr := dcacheRspRouteQ.io.deq.bits.dmaAddr
  pipe.io.dma_cache_rsp.bits.d_data := io.dcache_rsp.bits.data
  io.dcache_rsp.ready := dcacheRspRouteValid && Mux(dcacheRspFromDma, pipe.io.dma_cache_rsp.ready, pipe.io.dcache_rsp.ready)
  dcacheRspRouteQ.io.deq.ready := io.dcache_rsp.fire

  if(GVM_ENABLED){
    val WF_ID_WIDTH = log2Ceil(num_warp_in_a_block)
    val gvm_cta2warp = Module(new GvmDutCta2Warp)
    gvm_cta2warp.io.clock := clock
    gvm_cta2warp.io.reset := reset.asBool
    gvm_cta2warp.io.warp_req_fire := cta2warp.io.warpReq.fire
    gvm_cta2warp.io.software_wg_id := cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_wg_id.pad(32)
    gvm_cta2warp.io.software_warp_id :=
      cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch(WF_ID_WIDTH - 1, 0).pad(32)
    gvm_cta2warp.io.sm_id := sm_id
    gvm_cta2warp.io.hardware_warp_id := cta2warp.io.warpReq.bits.wid.pad(32)
    gvm_cta2warp.io.sgpr_base := cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_sgpr_base_dispatch.pad(32)
    gvm_cta2warp.io.vgpr_base := cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_vgpr_base_dispatch.pad(32)
    gvm_cta2warp.io.wg_slot_id_in_warp_sche :=
      cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_wf_tag_dispatch(TAG_WIDTH - 1, WF_ID_WIDTH)
    gvm_cta2warp.io.lds_base := Cat(
      LDS_BASE.U(32.W)(31, LDS_ID_WIDTH + 1),
      cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_lds_base_dispatch
    )
    gvm_cta2warp.io.rtl_num_thread := cta2warp.io.warpReq.bits.CTAdata.dispatch2cu_wf_size_dispatch.pad(32)
  }

  // tma_completion: consumed inside pipe by warp scheduler; sink for observability.
  pipe.io.tma_completion.ready := true.B
  // DMA TLB: identity-mapping bypass (nocache variant has no L1 TLB)
  val dma_tlb_state = RegInit(false.B) // false=idle, true=reply
  val dma_tlb_paddr = Reg(UInt(mmu.SV32.paLen.W))
  when(!dma_tlb_state && pipe.io.dma_tlb_req.valid) {
    dma_tlb_state := true.B
    dma_tlb_paddr := pipe.io.dma_tlb_req.bits.vaddr
  }
  when(dma_tlb_state && pipe.io.dma_tlb_rsp.ready) {
    dma_tlb_state := false.B
  }
  pipe.io.dma_tlb_req.ready      := !dma_tlb_state
  pipe.io.dma_tlb_rsp.valid      := dma_tlb_state
  pipe.io.dma_tlb_rsp.bits.paddr := dma_tlb_paddr
}

class GPGPU_top_nocache() extends Module {
  val num_sm = parameters.num_sm
  val param = (new MyConfig).toInstance
  val io = IO(new Bundle {
    val host_req=Flipped(DecoupledIO(new host2CTA_data))
    val host_rsp=DecoupledIO(new CTA2host_data)
    val icache = Vec(num_sm, new SMIO_icache()(param))
    val dcache_req = Vec(num_sm, DecoupledIO(new DCacheCoreReq_np))
    val dcache_rsp = Vec(num_sm, Flipped(DecoupledIO(new DCacheCoreRsp_np)))
    val perfDump = Input(Bool())
    val perfDumpSummary = Input(Bool())
    val icache_invalidate = Input(Bool())
  })
  val cta = Module(new CTAinterface)
  val sm_wrapper_inst = Seq.tabulate(num_sm) { i => Instantiate(new SM_wrapper_nocache()) }
  sm_wrapper_inst.zipWithIndex.foreach { case (sm, i) => sm.sm_id := i.U }
  val sm_wrapper = VecInit.tabulate(num_sm) { i => sm_wrapper_inst(i).io }

  cta.io.host2CTA :<>= io.host_req
  io.host_rsp :<>= cta.io.CTA2host

  def sumPipelinePerfCounter(select: PipelinePerfCounters => UInt): UInt = {
    if (PMU_PIPELINE) sm_wrapper.map(sm => select(sm.pipeline_perf.get)).reduce(_ + _) else 0.U(64.W)
  }
  def sumInstClassPerfCounter(select: InstClassPerfCounters => UInt): UInt = {
    if (PMU_INST_CLASS) sm_wrapper.map(sm => select(sm.inst_class_perf.get)).reduce(_ + _) else 0.U(64.W)
  }
  def sumTmaPerfCounter(select: TmaPerfCounters => UInt): UInt = {
    if (PMU_TMA) sm_wrapper.map(sm => select(sm.tma_perf.get)).reduce(_ + _) else 0.U(64.W)
  }
  def maxTmaPerfCounter(select: TmaPerfCounters => UInt): UInt = {
    if (PMU_TMA)
      sm_wrapper.map(sm => select(sm.tma_perf.get)).reduce(
        (a, b) => Mux(a > b, a, b))
    else 0.U(64.W)
  }
  def zeroTmaPerf: TmaPerfCounters = 0.U.asTypeOf(new TmaPerfCounters)
  def addTmaPerf(a: TmaPerfCounters, b: TmaPerfCounters): TmaPerfCounters = {
    val sum = Wire(new TmaPerfCounters)
    sum.instIssued := a.instIssued + b.instIssued
    sum.lineIssued := a.lineIssued + b.lineIssued
    sum.putFull := a.putFull + b.putFull
    sum.putPart := a.putPart + b.putPart
    sum.bytesWritten := a.bytesWritten + b.bytesWritten
    sum.sharedReadReq := a.sharedReadReq + b.sharedReadReq
    sum.sharedReadRsp := a.sharedReadRsp + b.sharedReadRsp
    sum.tlbReq := a.tlbReq + b.tlbReq
    sum.lineFullStallCycles := a.lineFullStallCycles + b.lineFullStallCycles
    sum.ackTagFullStallCycles := a.ackTagFullStallCycles + b.ackTagFullStallCycles
    sum.commandSlotFullStallCycles := a.commandSlotFullStallCycles + b.commandSlotFullStallCycles
    sum.pendingFullStallCycles := a.pendingFullStallCycles + b.pendingFullStallCycles
    sum.tlbStallCycles := a.tlbStallCycles + b.tlbStallCycles
    sum.sharedStallCycles := a.sharedStallCycles + b.sharedStallCycles
    sum.cacheReqStallCycles := a.cacheReqStallCycles + b.cacheReqStallCycles
    sum.g2sLineCount := a.g2sLineCount + b.g2sLineCount
    sum.s2gLineCount := a.s2gLineCount + b.s2gLineCount
    sum.g2sCompletionCount := a.g2sCompletionCount + b.g2sCompletionCount
    sum.s2gCompletionCount := a.s2gCompletionCount + b.s2gCompletionCount
    sum.g2sCacheResponseCount := a.g2sCacheResponseCount + b.g2sCacheResponseCount
    sum.s2gCacheResponseCount := a.s2gCacheResponseCount + b.s2gCacheResponseCount
    sum.g2sCacheLatencySum := a.g2sCacheLatencySum + b.g2sCacheLatencySum
    sum.s2gCacheLatencySum := a.s2gCacheLatencySum + b.s2gCacheLatencySum
    sum.activeCommandCycles := a.activeCommandCycles + b.activeCommandCycles
    sum.activeLineCycles := a.activeLineCycles + b.activeLineCycles
    sum.sharedActiveLineCycles :=
      a.sharedActiveLineCycles + b.sharedActiveLineCycles
    sum.requestActiveLineCycles :=
      a.requestActiveLineCycles + b.requestActiveLineCycles
    sum.windowIssued := a.windowIssued + b.windowIssued
    sum.windowRetired := a.windowRetired + b.windowRetired
    sum.uniqueLineWaves := a.uniqueLineWaves + b.uniqueLineWaves
    sum.translationHits := a.translationHits + b.translationHits
    sum.translationMisses := a.translationMisses + b.translationMisses
    sum.translationCoalesces := a.translationCoalesces + b.translationCoalesces
    sum.windowRobFullCycles := a.windowRobFullCycles + b.windowRobFullCycles
    sum.globalRequestFullCycles :=
      a.globalRequestFullCycles + b.globalRequestFullCycles
    sum.sharedQueueFullCycles :=
      a.sharedQueueFullCycles + b.sharedQueueFullCycles
    sum.cacheToSharedLatencyCount :=
      a.cacheToSharedLatencyCount + b.cacheToSharedLatencyCount
    sum.cacheToSharedLatencySum :=
      a.cacheToSharedLatencySum + b.cacheToSharedLatencySum
    sum.descriptorDemandHits :=
      a.descriptorDemandHits + b.descriptorDemandHits
    sum.descriptorDemandMisses :=
      a.descriptorDemandMisses + b.descriptorDemandMisses
    sum.descriptorPrefetchHits :=
      a.descriptorPrefetchHits + b.descriptorPrefetchHits
    sum.descriptorPrefetchMisses :=
      a.descriptorPrefetchMisses + b.descriptorPrefetchMisses
    sum.descriptorEvictions :=
      a.descriptorEvictions + b.descriptorEvictions
    sum.descriptorCompiles := a.descriptorCompiles + b.descriptorCompiles
    sum.descriptorCoalesces :=
      a.descriptorCoalesces + b.descriptorCoalesces
    sum.descriptorInvalidateKills :=
      a.descriptorInvalidateKills + b.descriptorInvalidateKills
    sum.descriptorCompileCycles :=
      a.descriptorCompileCycles + b.descriptorCompileCycles
    sum.bindCycles := a.bindCycles + b.bindCycles
    sum.plannerProduced := a.plannerProduced + b.plannerProduced
    sum.plannerFire := a.plannerFire + b.plannerFire
    sum.plannerStallCycles :=
      a.plannerStallCycles + b.plannerStallCycles
    sum.maxActiveWindows := Mux(
      a.maxActiveWindows > b.maxActiveWindows,
      a.maxActiveWindows, b.maxActiveWindows)
    sum.maxActiveRequests := Mux(
      a.maxActiveRequests > b.maxActiveRequests,
      a.maxActiveRequests, b.maxActiveRequests)
    sum.maxActiveShared := Mux(
      a.maxActiveShared > b.maxActiveShared,
      a.maxActiveShared, b.maxActiveShared)
    sum.maxActiveWriteAcks :=
      Mux(a.maxActiveWriteAcks > b.maxActiveWriteAcks,
        a.maxActiveWriteAcks, b.maxActiveWriteAcks)
    sum.longestWindowFireRun :=
      Mux(a.longestWindowFireRun > b.longestWindowFireRun,
        a.longestWindowFireRun, b.longestWindowFireRun)
    sum
  }

  val pmuActiveCycles = sumPipelinePerfCounter(_.activeCycles)
  val pmuTotalScalarIssued = sumPipelinePerfCounter(_.totalScalarIssued)
  val pmuTotalVectorIssued = sumPipelinePerfCounter(_.totalVectorIssued)
  val pmuExecHazardX = sumPipelinePerfCounter(_.execStructuralHazardCyclesX)
  val pmuExecHazardV = sumPipelinePerfCounter(_.execStructuralHazardCyclesV)
  val pmuDataDepStall = sumPipelinePerfCounter(_.dataDepStallCycles)
  val pmuBarrierStall = sumPipelinePerfCounter(_.barrierStallCycles)
  val pmuCtrlFlushCnt = sumPipelinePerfCounter(_.controlHazardFlushCount)
  val pmuFrontendStall = sumPipelinePerfCounter(_.frontendStallCycles)
  val pmuLsuBackpressure = sumPipelinePerfCounter(_.lsuBackpressureCycles)
  val pmuIbufferFullCycles = sumPipelinePerfCounter(_.ibufferFullCycles)
  val pmuTmaWaitStall = sumPipelinePerfCounter(_.tmaWaitStallCycles)
  val pmuComputeIssued = sumInstClassPerfCounter(_.computeIssued)
  val pmuMemIssued = sumInstClassPerfCounter(_.memIssued)
  val pmuCtrlIssued = sumInstClassPerfCounter(_.ctrlIssued)
  val pmuTotalIssued = pmuTotalScalarIssued + pmuTotalVectorIssued
  val pmuTma = Wire(new TmaPerfCounters)
  pmuTma.instIssued := sumTmaPerfCounter(_.instIssued)
  pmuTma.lineIssued := sumTmaPerfCounter(_.lineIssued)
  pmuTma.putFull := sumTmaPerfCounter(_.putFull)
  pmuTma.putPart := sumTmaPerfCounter(_.putPart)
  pmuTma.bytesWritten := sumTmaPerfCounter(_.bytesWritten)
  pmuTma.sharedReadReq := sumTmaPerfCounter(_.sharedReadReq)
  pmuTma.sharedReadRsp := sumTmaPerfCounter(_.sharedReadRsp)
  pmuTma.tlbReq := sumTmaPerfCounter(_.tlbReq)
  pmuTma.lineFullStallCycles := sumTmaPerfCounter(_.lineFullStallCycles)
  pmuTma.ackTagFullStallCycles := sumTmaPerfCounter(_.ackTagFullStallCycles)
  pmuTma.commandSlotFullStallCycles := sumTmaPerfCounter(_.commandSlotFullStallCycles)
  pmuTma.pendingFullStallCycles := sumTmaPerfCounter(_.pendingFullStallCycles)
  pmuTma.tlbStallCycles := sumTmaPerfCounter(_.tlbStallCycles)
  pmuTma.sharedStallCycles := sumTmaPerfCounter(_.sharedStallCycles)
  pmuTma.cacheReqStallCycles := sumTmaPerfCounter(_.cacheReqStallCycles)
  pmuTma.g2sLineCount := sumTmaPerfCounter(_.g2sLineCount)
  pmuTma.s2gLineCount := sumTmaPerfCounter(_.s2gLineCount)
  pmuTma.g2sCompletionCount := sumTmaPerfCounter(_.g2sCompletionCount)
  pmuTma.s2gCompletionCount := sumTmaPerfCounter(_.s2gCompletionCount)
  pmuTma.g2sCacheResponseCount := sumTmaPerfCounter(_.g2sCacheResponseCount)
  pmuTma.s2gCacheResponseCount := sumTmaPerfCounter(_.s2gCacheResponseCount)
  pmuTma.g2sCacheLatencySum := sumTmaPerfCounter(_.g2sCacheLatencySum)
  pmuTma.s2gCacheLatencySum := sumTmaPerfCounter(_.s2gCacheLatencySum)
  pmuTma.activeCommandCycles := sumTmaPerfCounter(_.activeCommandCycles)
  pmuTma.activeLineCycles := sumTmaPerfCounter(_.activeLineCycles)
  pmuTma.sharedActiveLineCycles :=
    sumTmaPerfCounter(_.sharedActiveLineCycles)
  pmuTma.requestActiveLineCycles :=
    sumTmaPerfCounter(_.requestActiveLineCycles)
  pmuTma.windowIssued := sumTmaPerfCounter(_.windowIssued)
  pmuTma.windowRetired := sumTmaPerfCounter(_.windowRetired)
  pmuTma.uniqueLineWaves := sumTmaPerfCounter(_.uniqueLineWaves)
  pmuTma.translationHits := sumTmaPerfCounter(_.translationHits)
  pmuTma.translationMisses := sumTmaPerfCounter(_.translationMisses)
  pmuTma.translationCoalesces := sumTmaPerfCounter(_.translationCoalesces)
  pmuTma.windowRobFullCycles := sumTmaPerfCounter(_.windowRobFullCycles)
  pmuTma.globalRequestFullCycles :=
    sumTmaPerfCounter(_.globalRequestFullCycles)
  pmuTma.sharedQueueFullCycles :=
    sumTmaPerfCounter(_.sharedQueueFullCycles)
  pmuTma.cacheToSharedLatencyCount :=
    sumTmaPerfCounter(_.cacheToSharedLatencyCount)
  pmuTma.cacheToSharedLatencySum :=
    sumTmaPerfCounter(_.cacheToSharedLatencySum)
  pmuTma.descriptorDemandHits :=
    sumTmaPerfCounter(_.descriptorDemandHits)
  pmuTma.descriptorDemandMisses :=
    sumTmaPerfCounter(_.descriptorDemandMisses)
  pmuTma.descriptorPrefetchHits :=
    sumTmaPerfCounter(_.descriptorPrefetchHits)
  pmuTma.descriptorPrefetchMisses :=
    sumTmaPerfCounter(_.descriptorPrefetchMisses)
  pmuTma.descriptorEvictions :=
    sumTmaPerfCounter(_.descriptorEvictions)
  pmuTma.descriptorCompiles := sumTmaPerfCounter(_.descriptorCompiles)
  pmuTma.descriptorCoalesces := sumTmaPerfCounter(_.descriptorCoalesces)
  pmuTma.descriptorInvalidateKills :=
    sumTmaPerfCounter(_.descriptorInvalidateKills)
  pmuTma.descriptorCompileCycles :=
    sumTmaPerfCounter(_.descriptorCompileCycles)
  pmuTma.bindCycles := sumTmaPerfCounter(_.bindCycles)
  pmuTma.plannerProduced := sumTmaPerfCounter(_.plannerProduced)
  pmuTma.plannerFire := sumTmaPerfCounter(_.plannerFire)
  pmuTma.plannerStallCycles := sumTmaPerfCounter(_.plannerStallCycles)
  pmuTma.maxActiveWindows := maxTmaPerfCounter(_.maxActiveWindows)
  pmuTma.maxActiveRequests := maxTmaPerfCounter(_.maxActiveRequests)
  pmuTma.maxActiveShared := maxTmaPerfCounter(_.maxActiveShared)
  pmuTma.maxActiveWriteAcks := maxTmaPerfCounter(_.maxActiveWriteAcks)
  pmuTma.longestWindowFireRun :=
    maxTmaPerfCounter(_.longestWindowFireRun)

  val perfWindowStarted = RegInit(false.B)
  val perfWindowPrinted = RegInit(false.B)
  val programId = RegInit(0.U(32.W))
  val totalProgramWindows = RegInit(0.U(32.W))
  val totalActiveCycles = RegInit(0.U(64.W))
  val totalScalarIssued = RegInit(0.U(64.W))
  val totalVectorIssued = RegInit(0.U(64.W))
  val totalExecHazardX = RegInit(0.U(64.W))
  val totalExecHazardV = RegInit(0.U(64.W))
  val totalDataDepStall = RegInit(0.U(64.W))
  val totalBarrierStall = RegInit(0.U(64.W))
  val totalCtrlFlushCnt = RegInit(0.U(64.W))
  val totalFrontendStall = RegInit(0.U(64.W))
  val totalLsuBackpressure = RegInit(0.U(64.W))
  val totalIbufferFullCycles = RegInit(0.U(64.W))
  val totalTmaWaitStall = RegInit(0.U(64.W))
  val totalComputeIssued = RegInit(0.U(64.W))
  val totalMemIssued = RegInit(0.U(64.W))
  val totalCtrlIssued = RegInit(0.U(64.W))
  val totalTma = RegInit(0.U.asTypeOf(new TmaPerfCounters))
  val perfStartPulse = io.host_req.fire && !perfWindowStarted
  val perfDumpPulse = io.perfDump && perfWindowStarted && !perfWindowPrinted
  when(perfStartPulse){
    perfWindowStarted := true.B
    perfWindowPrinted := false.B
  }.elsewhen(perfDumpPulse){
    perfWindowStarted := false.B
    perfWindowPrinted := true.B
    programId := programId + 1.U
    totalProgramWindows := totalProgramWindows + 1.U
    totalActiveCycles := totalActiveCycles + pmuActiveCycles
    totalScalarIssued := totalScalarIssued + pmuTotalScalarIssued
    totalVectorIssued := totalVectorIssued + pmuTotalVectorIssued
    totalExecHazardX := totalExecHazardX + pmuExecHazardX
    totalExecHazardV := totalExecHazardV + pmuExecHazardV
    totalDataDepStall := totalDataDepStall + pmuDataDepStall
    totalBarrierStall := totalBarrierStall + pmuBarrierStall
    totalCtrlFlushCnt := totalCtrlFlushCnt + pmuCtrlFlushCnt
    totalFrontendStall := totalFrontendStall + pmuFrontendStall
    totalLsuBackpressure := totalLsuBackpressure + pmuLsuBackpressure
    totalIbufferFullCycles := totalIbufferFullCycles + pmuIbufferFullCycles
    totalTmaWaitStall := totalTmaWaitStall + pmuTmaWaitStall
    totalComputeIssued := totalComputeIssued + pmuComputeIssued
    totalMemIssued := totalMemIssued + pmuMemIssued
    totalCtrlIssued := totalCtrlIssued + pmuCtrlIssued
    totalTma := addTmaPerf(totalTma, pmuTma)
  }

  for (i <- 0 until num_sm) {
    sm_wrapper(i).CTAreq :<>= cta.io.CTA2warp(i)
    cta.io.warp2CTA(i) :<>= sm_wrapper(i).CTArsp

    sm_wrapper(i).perfEnable := perfWindowStarted || perfStartPulse
    sm_wrapper(i).perfReset := perfStartPulse
    io.icache(i).req :<>= sm_wrapper(i).icache.req
    sm_wrapper(i).icache.rsp :<>= io.icache(i).rsp
    sm_wrapper(i).icache_invalidate := io.icache_invalidate

    io.dcache_req(i) :<>= sm_wrapper(i).dcache_req
    sm_wrapper(i).dcache_rsp :<>= io.dcache_rsp(i)
  }

  def includeCurrentWindow(total: UInt, current: UInt): UInt = {
    total + Mux(perfDumpPulse, current, 0.U(total.getWidth.W))
  }

  val summaryProgramWindows = includeCurrentWindow(totalProgramWindows, 1.U(32.W))
  val summaryActiveCycles = includeCurrentWindow(totalActiveCycles, pmuActiveCycles)
  val summaryScalarIssued = includeCurrentWindow(totalScalarIssued, pmuTotalScalarIssued)
  val summaryVectorIssued = includeCurrentWindow(totalVectorIssued, pmuTotalVectorIssued)
  val summaryExecHazardX = includeCurrentWindow(totalExecHazardX, pmuExecHazardX)
  val summaryExecHazardV = includeCurrentWindow(totalExecHazardV, pmuExecHazardV)
  val summaryDataDepStall = includeCurrentWindow(totalDataDepStall, pmuDataDepStall)
  val summaryBarrierStall = includeCurrentWindow(totalBarrierStall, pmuBarrierStall)
  val summaryCtrlFlushCnt = includeCurrentWindow(totalCtrlFlushCnt, pmuCtrlFlushCnt)
  val summaryFrontendStall = includeCurrentWindow(totalFrontendStall, pmuFrontendStall)
  val summaryLsuBackpressure = includeCurrentWindow(totalLsuBackpressure, pmuLsuBackpressure)
  val summaryIbufferFullCycles = includeCurrentWindow(totalIbufferFullCycles, pmuIbufferFullCycles)
  val summaryTmaWaitStall = includeCurrentWindow(totalTmaWaitStall, pmuTmaWaitStall)
  val summaryComputeIssued = includeCurrentWindow(totalComputeIssued, pmuComputeIssued)
  val summaryMemIssued = includeCurrentWindow(totalMemIssued, pmuMemIssued)
  val summaryCtrlIssued = includeCurrentWindow(totalCtrlIssued, pmuCtrlIssued)
  val summaryTma = addTmaPerf(totalTma, Mux(perfDumpPulse, pmuTma, zeroTmaPerf))
  val summaryTotalIssued = summaryScalarIssued + summaryVectorIssued
  val summaryTotalClassIssued = summaryComputeIssued + summaryMemIssued + summaryCtrlIssued

  when(perfDumpPulse){
    printf(p"\n[PROGRAM ${programId}] [PMU] first-kernel-start -> last-kernel-end summary (nocache)\n")
    if (PMU_PIPELINE) {
      printf(p"[PROGRAM ${programId}] [INST+CYCLE] active cycles       : ${pmuActiveCycles}\n")
      printf(p"[PROGRAM ${programId}] [INST+CYCLE] scalar issued       : ${pmuTotalScalarIssued}\n")
      printf(p"[PROGRAM ${programId}] [INST+CYCLE] vector issued       : ${pmuTotalVectorIssued}\n")
      printf(p"[PROGRAM ${programId}] [INST+CYCLE] total issued        : ${pmuTotalIssued}\n")
      printf(p"[PROGRAM ${programId}] [STALL] exec hazard X          : ${pmuExecHazardX}\n")
      printf(p"[PROGRAM ${programId}] [STALL] exec hazard V          : ${pmuExecHazardV}\n")
      printf(p"[PROGRAM ${programId}] [STALL] data dependency        : ${pmuDataDepStall}\n")
      printf(p"[PROGRAM ${programId}] [STALL] barrier stall cycles   : ${pmuBarrierStall}\n")
      printf(p"[PROGRAM ${programId}] [STALL] control flush events   : ${pmuCtrlFlushCnt}\n")
      printf(p"[PROGRAM ${programId}] [STALL] frontend stall cycles  : ${pmuFrontendStall}\n")
      printf(p"[PROGRAM ${programId}] [STALL] lsu backpressure cyc   : ${pmuLsuBackpressure}\n")
      printf(p"[PROGRAM ${programId}] [STALL] ibuffer full cycles    : ${pmuIbufferFullCycles}\n")
      printf(p"[PROGRAM ${programId}] [STALL] tma wait  : ${pmuTmaWaitStall}\n")
    }
    if (PMU_INST_CLASS) {
      printf(p"[PROGRAM ${programId}] [INST CLASS] compute issued    : ${pmuComputeIssued}\n")
      printf(p"[PROGRAM ${programId}] [INST CLASS] mem issued        : ${pmuMemIssued}\n")
      printf(p"[PROGRAM ${programId}] [INST CLASS] ctrl issued       : ${pmuCtrlIssued}\n")
      printf(p"[PROGRAM ${programId}] [INST CLASS] total class issued: ${pmuComputeIssued + pmuMemIssued + pmuCtrlIssued}\n")
    }
    if (PMU_TMA) {
      printf(p"[PROGRAM ${programId}] [TMA PERF] inst issued        : ${pmuTma.instIssued}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] line issued        : ${pmuTma.lineIssued}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] PutFull/PutPart    : ${pmuTma.putFull}/${pmuTma.putPart}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] bytes written      : ${pmuTma.bytesWritten}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] shared req/rsp      : ${pmuTma.sharedReadReq}/${pmuTma.sharedReadRsp}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] tlb req            : ${pmuTma.tlbReq}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] G2S/S2G lines       : ${pmuTma.g2sLineCount}/${pmuTma.s2gLineCount}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] G2S/S2G completions : ${pmuTma.g2sCompletionCount}/${pmuTma.s2gCompletionCount}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] G2S cache responses : ${pmuTma.g2sCacheResponseCount}\n")
      printf(p"[PROGRAM ${programId}] [TMA PERF] S2G cache responses : ${pmuTma.s2gCacheResponseCount}\n")
      if (PMU_TMA_DETAIL) {
        printf(p"[PROGRAM ${programId}] [TMA PERF] line full stall   : ${pmuTma.lineFullStallCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] ack full stall    : ${pmuTma.ackTagFullStallCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] command/pending stall: ${pmuTma.commandSlotFullStallCycles}/${pmuTma.pendingFullStallCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] TLB/shared/cache stall: ${pmuTma.tlbStallCycles}/${pmuTma.sharedStallCycles}/${pmuTma.cacheReqStallCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] G2S/S2G cache latency: ${pmuTma.g2sCacheLatencySum}/${pmuTma.s2gCacheLatencySum}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] active command/line cycles: ${pmuTma.activeCommandCycles}/${pmuTma.activeLineCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] active shared/request line cycles: ${pmuTma.sharedActiveLineCycles}/${pmuTma.requestActiveLineCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA WINDOW] issued/retired/line waves: ${pmuTma.windowIssued}/${pmuTma.windowRetired}/${pmuTma.uniqueLineWaves}\n")
        printf(p"[PROGRAM ${programId}] [TMA WINDOW] translation hit/miss/coalesce: ${pmuTma.translationHits}/${pmuTma.translationMisses}/${pmuTma.translationCoalesces}\n")
        printf(p"[PROGRAM ${programId}] [TMA WINDOW] full ROB/request/shared: ${pmuTma.windowRobFullCycles}/${pmuTma.globalRequestFullCycles}/${pmuTma.sharedQueueFullCycles}; cache-to-shared count/sum: ${pmuTma.cacheToSharedLatencyCount}/${pmuTma.cacheToSharedLatencySum}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] demand hit/miss; prefetch hit/miss: ${pmuTma.descriptorDemandHits}/${pmuTma.descriptorDemandMisses}; ${pmuTma.descriptorPrefetchHits}/${pmuTma.descriptorPrefetchMisses}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] compile/bind cycles: ${pmuTma.descriptorCompileCycles}/${pmuTma.bindCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] planner produced/fire/stall: ${pmuTma.plannerProduced}/${pmuTma.plannerFire}/${pmuTma.plannerStallCycles}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] max window/request/shared/ack: ${pmuTma.maxActiveWindows}/${pmuTma.maxActiveRequests}/${pmuTma.maxActiveShared}/${pmuTma.maxActiveWriteAcks}\n")
        printf(p"[PROGRAM ${programId}] [TMA PERF] longest window fire run: ${pmuTma.longestWindowFireRun}\n")
      }
    }
  }
  when((perfDumpPulse || io.perfDumpSummary) && summaryProgramWindows =/= 0.U){
    printf(p"\n[TESTCASE TOTAL] [PMU] accumulated summary across ${summaryProgramWindows} program windows (nocache)\n")
    if (PMU_PIPELINE) {
      printf(p"[TESTCASE TOTAL] [INST+CYCLE] active cycles       : ${summaryActiveCycles}\n")
      printf(p"[TESTCASE TOTAL] [INST+CYCLE] scalar issued       : ${summaryScalarIssued}\n")
      printf(p"[TESTCASE TOTAL] [INST+CYCLE] vector issued       : ${summaryVectorIssued}\n")
      printf(p"[TESTCASE TOTAL] [INST+CYCLE] total issued        : ${summaryTotalIssued}\n")
      printf(p"[TESTCASE TOTAL] [STALL] exec hazard X          : ${summaryExecHazardX}\n")
      printf(p"[TESTCASE TOTAL] [STALL] exec hazard V          : ${summaryExecHazardV}\n")
      printf(p"[TESTCASE TOTAL] [STALL] data dependency        : ${summaryDataDepStall}\n")
      printf(p"[TESTCASE TOTAL] [STALL] barrier stall cycles   : ${summaryBarrierStall}\n")
      printf(p"[TESTCASE TOTAL] [STALL] control flush events   : ${summaryCtrlFlushCnt}\n")
      printf(p"[TESTCASE TOTAL] [STALL] frontend stall cycles  : ${summaryFrontendStall}\n")
      printf(p"[TESTCASE TOTAL] [STALL] lsu backpressure cyc   : ${summaryLsuBackpressure}\n")
      printf(p"[TESTCASE TOTAL] [STALL] ibuffer full cycles    : ${summaryIbufferFullCycles}\n")
      printf(p"[TESTCASE TOTAL] [STALL] tma wait  : ${summaryTmaWaitStall}\n")
    }
    if (PMU_INST_CLASS) {
      printf(p"[TESTCASE TOTAL] [INST CLASS] compute issued    : ${summaryComputeIssued}\n")
      printf(p"[TESTCASE TOTAL] [INST CLASS] mem issued        : ${summaryMemIssued}\n")
      printf(p"[TESTCASE TOTAL] [INST CLASS] ctrl issued       : ${summaryCtrlIssued}\n")
      printf(p"[TESTCASE TOTAL] [INST CLASS] total class issued: ${summaryTotalClassIssued}\n")
    }
    if (PMU_TMA) {
      printf(p"[TESTCASE TOTAL] [TMA PERF] inst issued        : ${summaryTma.instIssued}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] line issued        : ${summaryTma.lineIssued}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] PutFull/PutPart    : ${summaryTma.putFull}/${summaryTma.putPart}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] bytes written      : ${summaryTma.bytesWritten}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] shared req/rsp      : ${summaryTma.sharedReadReq}/${summaryTma.sharedReadRsp}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] tlb req            : ${summaryTma.tlbReq}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] G2S/S2G lines       : ${summaryTma.g2sLineCount}/${summaryTma.s2gLineCount}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] G2S/S2G completions : ${summaryTma.g2sCompletionCount}/${summaryTma.s2gCompletionCount}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] G2S cache responses : ${summaryTma.g2sCacheResponseCount}\n")
      printf(p"[TESTCASE TOTAL] [TMA PERF] S2G cache responses : ${summaryTma.s2gCacheResponseCount}\n")
      if (PMU_TMA_DETAIL) {
        printf(p"[TESTCASE TOTAL] [TMA PERF] line full stall   : ${summaryTma.lineFullStallCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] ack full stall    : ${summaryTma.ackTagFullStallCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] command/pending stall: ${summaryTma.commandSlotFullStallCycles}/${summaryTma.pendingFullStallCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] TLB/shared/cache stall: ${summaryTma.tlbStallCycles}/${summaryTma.sharedStallCycles}/${summaryTma.cacheReqStallCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] G2S/S2G cache latency: ${summaryTma.g2sCacheLatencySum}/${summaryTma.s2gCacheLatencySum}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] active command/line cycles: ${summaryTma.activeCommandCycles}/${summaryTma.activeLineCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] active shared/request line cycles: ${summaryTma.sharedActiveLineCycles}/${summaryTma.requestActiveLineCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA WINDOW] issued/retired/line waves: ${summaryTma.windowIssued}/${summaryTma.windowRetired}/${summaryTma.uniqueLineWaves}\n")
        printf(p"[TESTCASE TOTAL] [TMA WINDOW] translation hit/miss/coalesce: ${summaryTma.translationHits}/${summaryTma.translationMisses}/${summaryTma.translationCoalesces}\n")
        printf(p"[TESTCASE TOTAL] [TMA WINDOW] full ROB/request/shared: ${summaryTma.windowRobFullCycles}/${summaryTma.globalRequestFullCycles}/${summaryTma.sharedQueueFullCycles}; cache-to-shared count/sum: ${summaryTma.cacheToSharedLatencyCount}/${summaryTma.cacheToSharedLatencySum}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] demand hit/miss; prefetch hit/miss: ${summaryTma.descriptorDemandHits}/${summaryTma.descriptorDemandMisses}; ${summaryTma.descriptorPrefetchHits}/${summaryTma.descriptorPrefetchMisses}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] compile/bind cycles: ${summaryTma.descriptorCompileCycles}/${summaryTma.bindCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] planner produced/fire/stall: ${summaryTma.plannerProduced}/${summaryTma.plannerFire}/${summaryTma.plannerStallCycles}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] max window/request/shared/ack: ${summaryTma.maxActiveWindows}/${summaryTma.maxActiveRequests}/${summaryTma.maxActiveShared}/${summaryTma.maxActiveWriteAcks}\n")
        printf(p"[TESTCASE TOTAL] [TMA PERF] longest window fire run: ${summaryTma.longestWindowFireRun}\n")
      }
    }
  }

  import top.ParametersToJson
  ParametersToJson.saveToJson("sim-verilator-nocache/parameters.json")
}
