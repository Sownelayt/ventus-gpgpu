package pipeline

import chisel3._
import chisel3.util._
import mmu.SV32
import top.parameters._

class TmaS2GPayloadMeta extends Bundle {
  val globalAddress = Vec(8, UInt(32.W))
  val globalBytes = Vec(8, UInt(5.W))
  val sharedBase = UInt(32.W)
  val sharedDelta = Vec(8, UInt(3.W))
}

/**
  * S2G metadata after the shared request has consumed base/delta addressing.
  *
  * The shared-address lifetime ends when the shared request fires, so the
  * post-read view reuses the same PayloadSlot metadata bank for the first
  * global-line route seed.
  */
class TmaS2GPostSharedMeta extends Bundle {
  val globalAddress = Vec(8, UInt(32.W))
  val globalBytes = Vec(8, UInt(5.W))
  val firstClear = UInt(32.W)
  val firstTag = UInt(25.W)
  val firstElement = UInt(5.W)
}

class TmaG2SLineRoute extends Bundle {
  val sharedBase = UInt(32.W)
  // Legal Tensor atoms begin on a 32-bit boundary. Ordinary formats begin
  // on a 16B boundary; the only other phases are the fixed 8B B4P64 phase
  // and the fixed 12B B6 packing phase.
  val cacheWord = Vec(8, UInt(5.W))
  val destinationWord = Vec(8, UInt(2.W))
  val bytes = Vec(8, UInt(5.W))
  val sharedAtomDelta = Vec(8, UInt(3.W))
}

/** Lane-aligned payload metadata after a G2S cache response is captured. */
class TmaG2SReadyMeta extends Bundle {
  val sharedBase = UInt(32.W)
  val mask = Vec(8, UInt(16.W))
  val sharedAtomDelta = Vec(8, UInt(3.W))
}

/** Completion metadata queued after all transfer resources retire. */
class TmaCompletionPayload extends Bundle {
  val wid = UInt(3.W)
  val copyDirection = UInt(1.W)
  val group = UInt(log2Ceil(TmaV2Spec.S2GGroupsPerWarp).W)
  val barrierValid = Bool()
  val barrierId = UInt(log2Ceil(TmaV2Spec.MbarrierEntries).W)
  val barrierGeneration = UInt(8.W)
  val transactionBytes = UInt(32.W)
}

/**
  * Single-command TMA data engine.
  *
  * requestEntries are narrow LineContexts.  windowEntries are the only
  * entries which own a complete 128B payload.  A G2S cache response moves
  * directly from its LineContext into one (or, for a paired recent-line
  * follower, two) PayloadSlots and releases the line credit immediately.
  * S2G allocates a PayloadSlot before shared read.  Its first narrow line
  * route is precomputed into metadata bits whose shared-address lifetime ends
  * when the request fires; later lines are created after data capture.
  */
class TmaV2WindowEngine(
    windowEntries: Int = TmaV2Spec.DefaultWindowEntries,
    requestEntries: Int = TmaV2Spec.DefaultGlobalRequestEntries,
    sharedEntries: Int = TmaV2Spec.DefaultSharedReadyEntries,
    writeAckEntries: Int = TmaV2Spec.DefaultWriteAckEntries,
    mmuEnabled: Boolean = MMU_ENABLED) extends Module {
  require(windowEntries >= 2)
  require(requestEntries >= 2 && requestEntries <= 256,
    "TMA requestEntries must be in 2..256")
  require(sharedEntries >= 2)
  require(isPow2(sharedEntries))
  require(writeAckEntries >= 2 && writeAckEntries <= 256,
    "TMA writeAckEntries must be in 2..256")

  private val payloadEntries = windowEntries
  private val lineEntries = requestEntries
  private val payloadWidth = log2Ceil(payloadEntries)
  private val lineWidth = log2Ceil(lineEntries)
  private val sharedWidth = log2Ceil(sharedEntries)
  private val ackWidth = log2Ceil(writeAckEntries)
  private val cacheSourceEntries =
    TmaV2Spec.cacheSourceEntries(requestEntries, writeAckEntries)
  private val sharedSetIdxLo =
    dcache_BlockOffsetBits + dcache_WordOffsetBits
  private val sharedSetIdxHi =
    sharedSetIdxLo + log2Ceil(sharedmem_depth) - 1
  private val lineCountWidth = log2Ceil(lineEntries + 1)
  private val activeWidth =
    log2Ceil(lineEntries + payloadEntries + sharedEntries + 1)
  private val lineGroups = (lineEntries + 3) / 4
  private val lineGroupWidth = math.max(1, log2Ceil(lineGroups))
  private val recentEntries = 4

  val io = IO(new Bundle {
    val command = Flipped(Decoupled(new TmaV2EngineCommand))
    val seal = Flipped(Decoupled(Bool()))
    val window = Flipped(Decoupled(new TmaV2Window))
    val tlbRequest = Decoupled(new TmaV2TlbRequest(lineEntries))
    val tlbResponse =
      Flipped(Decoupled(new TmaV2TlbResponse(lineEntries)))
    val sharedRequest =
      Decoupled(new TmaV2SharedRequest(sharedEntries))
    // Set only after this exact queue head has been presented at the final
    // external boundary and stalled.  Such a beat must survive command kill.
    val sharedRequestHeld = Input(Bool())
    val sharedResponse =
      Flipped(Decoupled(new TmaV2SharedResponse(sharedEntries)))
    val cacheRequest =
      Decoupled(new TmaV2CacheRequest(cacheSourceEntries))
    val cacheRequestHeld = Input(Bool())
    val cacheResponse =
      Flipped(Decoupled(new TmaV2CacheResponse(cacheSourceEntries)))
    val completion = Decoupled(new TmaV2EngineCompletion)
    val kill = Flipped(Valid(new TmaV2KillRequest))
    val killAsid = Input(UInt(SV32.asidLen.W))
    val killPending = Output(Bool())

    val activeCommands = Output(UInt(1.W))
    val activeWindows = Output(UInt(activeWidth.W))
    val activeRequests = Output(UInt(log2Ceil(lineEntries + 1).W))
    val activeShared = Output(UInt(log2Ceil(sharedEntries + 1).W))
    val activeWriteAcks =
      Output(UInt(log2Ceil(writeAckEntries + 1).W))
    val windowIssued = Output(Bool())
    val windowRetired = Output(Bool())
    val uniqueLineWave = Output(Bool())
    val translationHit = Output(Bool())
    val translationMiss = Output(Bool())
    val translationCoalesce = Output(Bool())
    val robFull = Output(Bool())
    val requestFull = Output(Bool())
    val sharedFull = Output(Bool())
    val writeAckFull = Output(Bool())
    val cacheToSharedLatencyValid = Output(Bool())
    val cacheToSharedLatency = Output(UInt(32.W))
  })

  // ------------------------------------------------------------------------
  // One immutable active command
  // ------------------------------------------------------------------------
  val commandValid = RegInit(false.B)
  val commandSealed = RegInit(false.B)
  val commandWid = Reg(UInt(3.W))
  val commandDirection = Reg(UInt(1.W))
  val commandDtype = Reg(UInt(5.W))
  val commandOobFill = Reg(Bool())
  val commandReduceMode = Reg(UInt(3.W))
  val commandAsid = Reg(UInt(SV32.asidLen.W))
  val commandGroup =
    Reg(UInt(log2Ceil(TmaV2Spec.S2GGroupsPerWarp).W))
  val commandBarrierValid = Reg(Bool())
  val commandBarrierId =
    Reg(UInt(log2Ceil(TmaV2Spec.MbarrierEntries).W))
  val commandBarrierGeneration = Reg(UInt(8.W))
  val commandTransactionBytes = Reg(UInt(32.W))
  val commandKilled = RegInit(false.B)
  // Keep the small-transfer issue credit at 16 to control L2 tail latency;
  // larger commands retain
  // all 40 LineContexts and therefore the intended long-transfer overlap.
  val g2sOutstanding = RegInit(0.U(lineCountWidth.W))
  val g2sReadRetire = WireDefault(false.B)

  io.command.ready := !commandValid
  when(io.command.fire) {
    commandValid := true.B
    commandSealed := false.B
    commandWid := io.command.bits.wid
    commandDirection := io.command.bits.copyDirection
    commandDtype := io.command.bits.dtype
    commandOobFill := io.command.bits.oobFill
    commandReduceMode := io.command.bits.reduceMode
    commandAsid := io.command.bits.asid
    commandGroup := io.command.bits.group
    commandBarrierValid := io.command.bits.barrierValid
    commandBarrierId := io.command.bits.barrierId
    commandBarrierGeneration := io.command.bits.barrierGeneration
    commandTransactionBytes := io.command.bits.transactionBytes
    commandKilled := false.B
  }
  val killMatch = io.kill.valid && commandValid &&
    commandAsid === io.kill.bits.asid
  val killMode = commandKilled || killMatch
  io.killPending := commandValid && commandAsid === io.killAsid
  val dtypeBits = TmaV2DescriptorDerived.dtypeBits(commandDtype)
  val paddedFp4 = commandDtype === TmaV2Spec.DTypeB4x16P64.U
  val paddedFp6 = commandDtype === TmaV2Spec.DTypeB6.U
  val paddedSubByte = paddedFp4 || paddedFp6

  // Compute the NaN byte pattern once.  The selected fill pattern is then
  // broadcast to all eight lanes rather than instantiating eight dtype
  // decoders.
  val nanBytes = Wire(Vec(16, UInt(8.W)))
  for (byte <- 0 until 16) {
    val elementByte = MuxLookup(dtypeBits, 0.U(3.W))(Seq(
      16.U -> (byte & 1).U,
      32.U -> (byte & 3).U,
      64.U -> (byte & 7).U))
    val fp16Nan = Mux(elementByte === 0.U, "h00".U, "h7e".U)
    val bf16Nan = Mux(elementByte === 0.U, "hc0".U, "h7f".U)
    val fp32Nan = MuxLookup(elementByte, 0.U)(Seq(
      0.U -> "h00".U, 1.U -> "h00".U,
      2.U -> "hc0".U, 3.U -> "h7f".U))
    val fp64Nan = MuxLookup(elementByte, 0.U)(Seq(
      6.U -> "hf8".U, 7.U -> "h7f".U))
    nanBytes(byte) := MuxLookup(commandDtype, 0.U)(Seq(
      TmaV2Spec.DTypeFp16.U -> fp16Nan,
      TmaV2Spec.DTypeBf16.U -> bf16Nan,
      TmaV2Spec.DTypeFp32.U -> fp32Nan,
      TmaV2Spec.DTypeFp32Ftz.U -> fp32Nan,
      TmaV2Spec.DTypeTf32.U -> fp32Nan,
      TmaV2Spec.DTypeTf32Ftz.U -> fp32Nan,
      TmaV2Spec.DTypeFp64.U -> fp64Nan))
  }
  val commandFillPattern =
    Mux(commandOobFill, nanBytes.asUInt, 0.U(128.W))

  def bytePrefixMask(bytes: UInt): UInt = {
    MuxLookup(bytes, 0.U(16.W))(
      (1 to 16).map(count =>
        count.U -> ((BigInt(1) << count) - 1).U(16.W)))
  }

  def cacheLineTag(address: UInt): UInt = address(31, 7)

  def endingCacheLineTag(address: UInt, bytes: UInt): UInt =
    ((address + Mux(bytes.orR, bytes - 1.U, 0.U)) >> 7)(24, 0)

  // ------------------------------------------------------------------------
  // 40 narrow LineContexts
  //
  // G2S route layout is deliberately kept below 192 bits:
  //   shared base + 8 * {cache offset, destination offset, bytes, atom delta}
  // The common address/state/follower fields keep a production entry below
  // 256 sequential bits.
  // ------------------------------------------------------------------------
  object LineState {
    val Free = 0.U(3.W)
    val NeedTranslate = 1.U(3.W)
    val WaitTranslate = 2.U(3.W)
    val NeedCache = 3.U(3.W)
    val WaitCache = 4.U(3.W)
    val Follower = 5.U(3.W)
    // Allocation reserves the source ID immediately; the wide route and tag
    // are committed from a registered write intent on the following edge.
    val RouteWrite = 6.U(3.W)
  }
  val lineState =
    RegInit(VecInit(Seq.fill(lineEntries)(LineState.Free)))
  // Low seven bits are architecturally zero for canonical cache lines.
  // Address storage is phase-tagged by lineState. Before translation this is
  // the 25-bit VA line tag. Once translation completes, the high 20 bits are
  // overwritten by the PPN while VA[11:7] remains in the low five bits. G2S
  // routes already carry cache offsets and therefore do not need the VPN.
  // S2G keeps its virtual comparison tag in otherwise-unused route bits.
  val lineAddressTag = Reg(Vec(lineEntries, UInt(25.W)))
  // G2S consumes the complete compact route. S2G aliases its low bits with
  // {reduce element, payload owner}; both physical views cannot be live
  // because the unique ActiveCommand fixes direction until all state drains.
  private val g2sLineRouteWidth = (new TmaG2SLineRoute).getWidth
  private val s2gLineRouteWidth = 25 + 5 + payloadWidth
  private val lineRouteWidth = math.max(
    g2sLineRouteWidth, s2gLineRouteWidth)
  val lineRoute = Reg(Vec(lineEntries, UInt(lineRouteWidth.W)))
  def g2sLineRoute(index: UInt): TmaG2SLineRoute =
    lineRoute(index).asTypeOf(new TmaG2SLineRoute)
  def linePayload(index: UInt): UInt =
    lineRoute(index)(payloadWidth - 1, 0)
  def lineReduceElement(index: UInt): UInt =
    lineRoute(index)(payloadWidth + 4, payloadWidth)
  def s2gVirtualTag(index: UInt): UInt =
    lineRoute(index)(lineRouteWidth - 1, lineRouteWidth - 25)

  // One allocation can enter and one previous allocation can commit every
  // cycle.  Registering both the write index and the 192-bit route removes
  // the free-selector/decomposer cone from all 40 * 192 LineContext D pins
  // without reducing the one-line/cycle steady-state rate.
  val lineWriteValid = RegInit(false.B)
  val lineWriteTarget = Reg(UInt(lineWidth.W))
  val lineWriteTag = Reg(UInt(25.W))
  val lineWriteRoute = Reg(UInt(lineRouteWidth.W))
  val lineWriteNextState = Reg(UInt(3.W))
  when(lineWriteValid) {
    lineAddressTag(lineWriteTarget) := lineWriteTag
    lineRoute(lineWriteTarget) := lineWriteRoute
    lineState(lineWriteTarget) := lineWriteNextState
    lineWriteValid := false.B
  }

  // At most one cache response and one cache write can retire line contexts
  // in a cycle.  Their IDs feed the hierarchical free selector so a new
  // canonical line can reuse the credit without an empty cycle.
  val recycleLine0Valid = WireDefault(false.B)
  val recycleLine0 = WireDefault(0.U(lineWidth.W))
  val recycleLine1Valid = WireDefault(false.B)
  val recycleLine1 = WireDefault(0.U(lineWidth.W))
  val lineFreeVec = Wire(Vec(lineEntries, Bool()))
  for (line <- 0 until lineEntries) {
    lineFreeVec(line) :=
      lineState(line) === LineState.Free ||
        (recycleLine0Valid && recycleLine0 === line.U) ||
        (recycleLine1Valid && recycleLine1 === line.U)
  }
  val freeLineGroup = Wire(Vec(lineGroups, Bool()))
  val freeLineLocal = Wire(Vec(lineGroups, UInt(2.W)))
  for (group <- 0 until lineGroups) {
    val local = Wire(Vec(4, Bool()))
    for (sub <- 0 until 4) {
      val index = group * 4 + sub
      local(sub) := (if (index < lineEntries) lineFreeVec(index)
                     else false.B)
    }
    freeLineGroup(group) := local.asUInt.orR
    freeLineLocal(group) := PriorityEncoder(local)
  }
  val freeGroup = PriorityEncoder(freeLineGroup)
  val freeLine =
    (freeGroup * 4.U + freeLineLocal(freeGroup))(lineWidth - 1, 0)
  val hasFreeLine = freeLineGroup.asUInt.orR

  // ------------------------------------------------------------------------
  // Six bidirectional PayloadSlots
  // ------------------------------------------------------------------------
  object PayloadState {
    val Free = 0.U(3.W)
    val G2SReady = 1.U(3.W)
    val S2GNeedShared = 2.U(3.W)
    val S2GWaitShared = 3.U(3.W)
    val S2GDecompose = 4.U(3.W)
  }
  val payloadState =
    RegInit(VecInit(Seq.fill(payloadEntries)(PayloadState.Free)))
  val payloadData = Reg(Vec(payloadEntries, UInt(1024.W)))

  // Direction-exclusive physical union. Keeping one array is important: two named
  // Reg(Vec(...)) declarations would synthesize both banks even though a
  // single active command makes their lifetimes mutually exclusive.
  private val g2sReadyMetaWidth = (new TmaG2SReadyMeta).getWidth
  private val s2gPayloadMetaWidth = (new TmaS2GPayloadMeta).getWidth
  private val s2gPostMetaWidth = (new TmaS2GPostSharedMeta).getWidth
  private val payloadMetaWidth = Seq(
    g2sReadyMetaWidth, s2gPayloadMetaWidth, s2gPostMetaWidth).max
  val payloadMeta =
    Reg(Vec(payloadEntries, UInt(payloadMetaWidth.W)))
  def g2sReadyMeta(index: UInt): TmaG2SReadyMeta =
    payloadMeta(index)(g2sReadyMetaWidth - 1, 0)
      .asTypeOf(new TmaG2SReadyMeta)
  def s2gMeta(index: UInt): TmaS2GPayloadMeta =
    payloadMeta(index)(s2gPayloadMetaWidth - 1, 0)
      .asTypeOf(new TmaS2GPayloadMeta)
  def s2gPostMeta(index: UInt): TmaS2GPostSharedMeta =
    payloadMeta(index)(s2gPostMetaWidth - 1, 0)
      .asTypeOf(new TmaS2GPostSharedMeta)

  val payloadPending = Reg(Vec(payloadEntries, UInt(32.W)))
  // This counter is only consumed by S2G, but it must still have a defined
  // value while the unique active command is G2S.  In particular the
  // resource-invariant assertions inspect every physical slot independent of
  // direction; leaving the mutually-exclusive view uninitialized makes an
  // otherwise legal first G2S command depend on simulator/register power-up
  // state.
  val payloadOutstanding =
    RegInit(VecInit(Seq.fill(payloadEntries)(0.U(lineCountWidth.W))))
  // Scala aliases retained for the existing failure diagnostics. They do
  // not elaborate additional state.
  val slotState = payloadState
  val slotOutstanding = payloadOutstanding

  val recyclePayloadValid = WireDefault(false.B)
  val recyclePayload = WireDefault(0.U(payloadWidth.W))
  val s2gPayloadCompletes = WireDefault(false.B)
  val payloadFreeVec = Wire(Vec(payloadEntries, Bool()))
  for (payload <- 0 until payloadEntries) {
    payloadFreeVec(payload) :=
      payloadState(payload) === PayloadState.Free ||
        // Both directions have a real ownership-transfer point before the
        // next window allocation: G2S at shared-request acceptance and S2G
        // at final cache-write acceptance. Neither path depends on
        // io.window.ready, so this same-cycle credit does not form a ready
        // loop and avoids one bubble each time the compact payload pool fills.
        (recyclePayloadValid && recyclePayload === payload.U)
  }
  val freePayload0 = PriorityEncoder(payloadFreeVec)
  val payloadFreeWithout0 = Wire(Vec(payloadEntries, Bool()))
  for (payload <- 0 until payloadEntries) {
    payloadFreeWithout0(payload) :=
      payloadFreeVec(payload) && payload.U =/= freePayload0
  }
  val freePayload1 = PriorityEncoder(payloadFreeWithout0)
  val hasFreePayload = payloadFreeVec.asUInt.orR
  val hasTwoFreePayloads = PopCount(payloadFreeVec) >= 2.U

  // ------------------------------------------------------------------------
  // One compact G2S WindowDecomposer
  // ------------------------------------------------------------------------
  val decompValid = RegInit(false.B)
  val decompGlobalAddress = Reg(Vec(8, UInt(32.W)))
  val decompGlobalBytes = Reg(Vec(8, UInt(5.W)))
  val decompSharedBase = Reg(UInt(32.W))
  val decompSharedDelta = Reg(Vec(8, UInt(3.W)))
  val decompSharedBytes = Reg(Vec(8, UInt(5.W)))
  // Eight primary word-aligned lane segments plus the one possible B6 tail.
  // A legal padded B6 window is exactly 128 elements (eight 16-element
  // lanes), so at most one lane crosses a cache-line boundary.
  val decompPending = Reg(UInt(9.W))
  val decompTailLane = Reg(UInt(3.W))
  val decompFillPending = RegInit(false.B)

  val decompSegmentTags = Wire(Vec(9, UInt(25.W)))
  for (lane <- 0 until 8) {
    decompSegmentTags(lane) := cacheLineTag(decompGlobalAddress(lane))
  }
  decompSegmentTags(8) := endingCacheLineTag(
    decompGlobalAddress(decompTailLane),
    decompGlobalBytes(decompTailLane))
  val selectedDecompSegment = PriorityEncoder(decompPending)
  val selectedDecompTag = decompSegmentTags(selectedDecompSegment)
  val decompSameTag = VecInit((0 until 9).map { segment =>
    decompPending(segment) &&
      decompSegmentTags(segment) === selectedDecompTag
  }).asUInt
  val decompPendingAfter = decompPending & ~decompSameTag
  // Registered output of WindowDecomposer. The 192-bit route terminates the
  // segment/tag/interval cone before the LineContext write mux.
  val routeCandidateValid = RegInit(false.B)
  val routeCandidateTag = Reg(UInt(25.W))
  val routeCandidate = Reg(new TmaG2SLineRoute)

  // Four recent leaders preserve the useful adjacent two-consumer merge
  // without a lineEntries-wide associative owner network.
  val recentValid =
    RegInit(VecInit(Seq.fill(recentEntries)(false.B)))
  val recentTag = Reg(Vec(recentEntries, UInt(25.W)))
  val recentLeader = Reg(Vec(recentEntries, UInt(lineWidth.W)))
  val recentRr = RegInit(0.U(2.W))
  // Once a recent leader acquires its only follower, retain the compact pair
  // in a four-entry directory until the response arrives.  This avoids both
  // a 40-way follower search and any dependence on later recent-tag
  // replacement.
  val pairValid =
    RegInit(VecInit(Seq.fill(recentEntries)(false.B)))
  val pairLeader = Reg(Vec(recentEntries, UInt(lineWidth.W)))
  val pairFollower = Reg(Vec(recentEntries, UInt(lineWidth.W)))
  val pairFreeVec = VecInit(pairValid.map(!_))
  val pairHasFree = pairFreeVec.asUInt.orR
  val pairFree = PriorityEncoder(pairFreeVec)
  val returningRead = WireDefault(false.B)
  val returningReadLine = WireDefault(0.U(lineWidth.W))
  val recentMatchVec = VecInit((0 until recentEntries).map { entry =>
    val alreadyPaired = VecInit((0 until recentEntries).map { pair =>
      pairValid(pair) && pairLeader(pair) === recentLeader(entry)
    }).asUInt.orR
    routeCandidateValid && recentValid(entry) &&
      recentTag(entry) === routeCandidateTag &&
      lineState(recentLeader(entry)) =/= LineState.Free &&
      !alreadyPaired &&
      pairHasFree
  })
  val recentMatch = recentMatchVec.asUInt.orR
  val recentMatchEntry = PriorityEncoder(recentMatchVec)
  val recentMatchLeader = recentLeader(recentMatchEntry)

  val decompEmitValid = decompValid && decompPending.orR
  // A returning leader may feed the registered candidate as a second
  // consumer. Route metadata is already stable, so no decomp address logic
  // lies between the cache response and PayloadSlot capture.
  val directResponseCandidate = routeCandidateValid && recentMatch &&
    returningRead && returningReadLine === recentMatchLeader
  val directResponseFire = WireDefault(false.B)
  val routeCandidateFire = routeCandidateValid && !killMode &&
    !directResponseCandidate && hasFreeLine
  val routeCandidateReady =
    !routeCandidateValid || routeCandidateFire || directResponseFire
  val decompEmitFire = decompEmitValid && routeCandidateReady
  val decompRouteFire = routeCandidateFire || directResponseFire
  // Ownership transfers out of the window decomposer when its last narrow
  // route enters the candidate stage, not when that candidate later acquires
  // a LineContext. This is what permits one single-line window every cycle.
  val decompLineDone = decompEmitFire && !decompPendingAfter.orR &&
    !decompFillPending

  // Fill is produced only after all real line routes have left the
  // decomposer. A returning cache response has priority over this synthetic
  // payload allocation.
  val fillCanAllocate = decompValid && !killMode && !decompPending.orR &&
    (!routeCandidateValid || routeCandidateFire) &&
    decompFillPending && hasFreePayload && !returningRead
  val fillFire = fillCanAllocate
  val decompRelease = decompLineDone || fillFire

  // ------------------------------------------------------------------------
  // Shared tag table (direction comes from ActiveCommand)
  // ------------------------------------------------------------------------
  val sharedValid =
    RegInit(VecInit(Seq.fill(sharedEntries)(false.B)))
  val sharedPayload = Reg(Vec(sharedEntries, UInt(payloadWidth.W)))
  val sharedPendingWords = Reg(Vec(sharedEntries, UInt(32.W)))
  val recycleSharedValid = WireDefault(false.B)
  val recycleShared = WireDefault(0.U(sharedWidth.W))
  val freeSharedVec = VecInit((0 until sharedEntries).map { entry =>
    !sharedValid(entry) ||
      (recycleSharedValid && recycleShared === entry.U)
  })
  val freeShared = PriorityEncoder(freeSharedVec)
  val hasFreeShared = freeSharedVec.asUInt.orR
  val selectedSharedSource = io.sharedResponse.bits.source
  val selectedSharedMask = io.sharedResponse.bits.wordMask
  val selectedSharedRemaining =
    sharedPendingWords(selectedSharedSource) & ~selectedSharedMask

  // ------------------------------------------------------------------------
  // S2G write acknowledgement namespace; source IDs are reused by direction.
  // ------------------------------------------------------------------------
  val writeAckValid =
    RegInit(VecInit(Seq.fill(writeAckEntries)(false.B)))
  val recycleAckValid = WireDefault(false.B)
  val recycleAck = WireDefault(0.U(ackWidth.W))
  val freeAckVec = VecInit((0 until writeAckEntries).map { entry =>
    !writeAckValid(entry) ||
      (recycleAckValid && recycleAck === entry.U)
  })
  val freeAck = PriorityEncoder(freeAckVec)
  val hasFreeAck = freeAckVec.asUInt.orR

  // ------------------------------------------------------------------------
  // Window input
  // ------------------------------------------------------------------------
  val normalizedWindow = io.window.bits
  val normalizedWindowValid = io.window.valid
  val windowMatches = commandValid
  val incomingGlobalBytes = Wire(Vec(8, UInt(5.W)))
  val incomingSharedBytes = Wire(Vec(8, UInt(5.W)))
  val incomingPrimary = Wire(Vec(8, Bool()))
  val incomingB6Tail = Wire(Vec(8, Bool()))
  val incomingMissingMask = Wire(Vec(8, UInt(16.W)))
  val incomingHasFill = Wire(Vec(8, Bool()))
  val incomingSharedDelta = Wire(Vec(8, UInt(3.W)))
  val incomingSharedBase = normalizedWindow.sharedBase
  val incomingReduceWords = Wire(Vec(32, Bool()))
  for (lane <- 0 until 8) {
    val globalBytes = normalizedWindow.lanes(lane).globalBytes
    val sharedBytes = normalizedWindow.lanes(lane).sharedBytes
    val globalMask = bytePrefixMask(globalBytes)
    val sharedMask = bytePrefixMask(sharedBytes)
    incomingGlobalBytes(lane) := globalBytes
    incomingSharedBytes(lane) := sharedBytes
    val addressOffset = normalizedWindow.lanes(lane).globalAddress(6, 0)
    val crossesLine = addressOffset +& globalBytes > 128.U
    incomingPrimary(lane) :=
      normalizedWindow.lanes(lane).valid &&
        globalBytes.orR
    incomingB6Tail(lane) :=
      normalizedWindow.lanes(lane).valid &&
        globalBytes.orR && paddedFp6 && crossesLine
    incomingMissingMask(lane) :=
      Mux(normalizedWindow.lanes(lane).valid,
        sharedMask & ~globalMask, 0.U)
    incomingHasFill(lane) := incomingMissingMask(lane).orR
    incomingSharedDelta(lane) :=
      normalizedWindow.lanes(lane).sharedAtomDelta
    for (word <- 0 until 4) {
      incomingReduceWords(lane * 4 + word) :=
        normalizedWindow.lanes(lane).valid &&
          globalBytes >= (word * 4 + 4).U
    }
  }
  val incomingCopyPending = Cat(
    incomingB6Tail.asUInt.orR, incomingPrimary.asUInt)
  val incomingTailLane = PriorityEncoder(incomingB6Tail)
  val incomingReducePending = incomingReduceWords.asUInt
  val incomingPending = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    incomingCopyPending.pad(32), incomingReducePending)

  val s2gWindowReady = hasFreePayload && !returningRead
  val g2sWindowReady = !decompValid || decompRelease
  val normalizedWindowReady = windowMatches && !killMode && Mux(
    commandDirection === TmaV2Spec.DirectionS2G.U,
    s2gWindowReady, g2sWindowReady)
  io.window.ready := normalizedWindowReady
  val normalizedWindowFire =
    normalizedWindowValid && normalizedWindowReady

  // ------------------------------------------------------------------------
  // G2S decomposer -> LineContext
  // ------------------------------------------------------------------------
  val decompSelectedRoute = Wire(new TmaG2SLineRoute)
  decompSelectedRoute.sharedBase := decompSharedBase
  for (lane <- 0 until 8) {
    val address = decompGlobalAddress(lane)
    val addressOffset = address(6, 0)
    val firstCapacity = 128.U - addressOffset
    val firstTag = cacheLineTag(address)
    val secondTag = endingCacheLineTag(
      address, decompGlobalBytes(lane))
    val firstBytes = Mux(
      decompGlobalBytes(lane) > firstCapacity,
      firstCapacity, decompGlobalBytes(lane))
    val secondBytes = decompGlobalBytes(lane) - firstBytes
    val selectsFirst = firstTag === selectedDecompTag
    val selectsSecond =
      lane.U === decompTailLane && secondTag =/= firstTag &&
        secondTag === selectedDecompTag
    val bytes = Mux(
      selectsFirst, firstBytes,
      Mux(selectsSecond, secondBytes, 0.U))
    decompSelectedRoute.cacheWord(lane) :=
      Mux(selectsFirst, address(6, 2), 0.U)
    decompSelectedRoute.destinationWord(lane) :=
      Mux(selectsSecond, firstBytes(3, 2), 0.U)
    decompSelectedRoute.bytes(lane) := bytes
    decompSelectedRoute.sharedAtomDelta(lane) :=
      decompSharedDelta(lane)
  }

  when(routeCandidateFire) {
    val target = freeLine
    val attach = recentMatch
    val leader = recentMatchLeader
    lineState(target) := LineState.RouteWrite
    lineWriteValid := true.B
    lineWriteTarget := target
    lineWriteTag := routeCandidateTag
    lineWriteRoute := routeCandidate.asUInt
    lineWriteNextState := Mux(
      attach, LineState.Follower,
      (if (mmuEnabled) LineState.NeedTranslate
       else LineState.NeedCache))
    when(attach) {
      pairValid(pairFree) := true.B
      pairLeader(pairFree) := leader
      pairFollower(pairFree) := target
      recentValid(recentMatchEntry) := false.B
    }.otherwise {
      recentValid(recentRr) := true.B
      recentTag(recentRr) := routeCandidateTag
      recentLeader(recentRr) := target
      recentRr := recentRr + 1.U
    }
  }

  when(decompRouteFire && !decompEmitFire) {
    routeCandidateValid := false.B
  }
  when(decompEmitFire) {
    routeCandidateValid := true.B
    routeCandidateTag := selectedDecompTag
    routeCandidate := decompSelectedRoute
    decompPending := decompPendingAfter
  }

  // Synthetic zero/NaN fill occupies a normal payload entry but never a
  // line/TLB/L2 credit.
  when(fillFire) {
    payloadState(freePayload0) := PayloadState.G2SReady
    val fillMeta = Wire(new TmaG2SReadyMeta)
    fillMeta.sharedBase := decompSharedBase
    val fillLanes = Wire(Vec(8, UInt(128.W)))
    for (lane <- 0 until 8) {
      val laneMissing = bytePrefixMask(decompSharedBytes(lane)) &
        ~bytePrefixMask(decompGlobalBytes(lane))
      fillLanes(lane) := commandFillPattern
      fillMeta.mask(lane) := laneMissing
      fillMeta.sharedAtomDelta(lane) := decompSharedDelta(lane)
    }
    payloadData(freePayload0) := fillLanes.asUInt
    payloadMeta(freePayload0) :=
      fillMeta.asUInt.pad(payloadMetaWidth)
    decompFillPending := false.B
  }

  // ------------------------------------------------------------------------
  // S2G PayloadSlot -> LineContext decomposer
  // ------------------------------------------------------------------------
  val residentS2GDecompVec =
    VecInit(payloadState.map(_ === PayloadState.S2GDecompose))
  val residentS2GDecompPayload = PriorityEncoder(residentS2GDecompVec)
  val completeS2GResponse = io.sharedResponse.valid &&
    commandDirection === TmaV2Spec.DirectionS2G.U &&
    sharedValid(selectedSharedSource) && !selectedSharedRemaining.orR
  // A completing response uses the route seed written when its shared request
  // fired.  It bypasses only compact control metadata; response data still
  // terminates in payloadData at this clock edge.
  val s2gDecompPayload = Mux(
    completeS2GResponse,
    sharedPayload(selectedSharedSource),
    residentS2GDecompPayload)
  val selectedS2GPending = payloadPending(s2gDecompPayload)
  val selectedS2GSegment = PriorityEncoder(selectedS2GPending(8, 0))
  val selectedS2GElement = PriorityEncoder(selectedS2GPending)
  val selectedS2GMeta = s2gPostMeta(s2gDecompPayload)
  val selectedS2GTailVec = VecInit((0 until 8).map { lane =>
    val addressOffset = selectedS2GMeta.globalAddress(lane)(6, 0)
    paddedFp6 && selectedS2GMeta.globalBytes(lane).orR &&
      addressOffset +& selectedS2GMeta.globalBytes(lane) > 128.U
  })
  val selectedS2GTailLane = PriorityEncoder(selectedS2GTailVec)
  val selectedS2GLane = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    Mux(selectedS2GSegment === 8.U,
      selectedS2GTailLane, selectedS2GSegment(2, 0)),
    selectedS2GElement >> 2)(2, 0)
  val selectedS2GWord = selectedS2GElement(1, 0)
  val selectedS2GTags = Wire(Vec(9, UInt(25.W)))
  for (lane <- 0 until 8) {
    selectedS2GTags(lane) :=
      cacheLineTag(selectedS2GMeta.globalAddress(lane))
  }
  selectedS2GTags(8) := endingCacheLineTag(
    selectedS2GMeta.globalAddress(selectedS2GTailLane),
    selectedS2GMeta.globalBytes(selectedS2GTailLane))
  val selectedS2GCopyTag =
    selectedS2GTags(selectedS2GSegment)
  val selectedS2GReduceTag = cacheLineTag(
    selectedS2GMeta.globalAddress(selectedS2GLane) +
      (selectedS2GWord << 2))
  val calculatedS2GTag = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    selectedS2GCopyTag, selectedS2GReduceTag)
  val selectedS2GSameTag = VecInit((0 until 9).map { segment =>
    selectedS2GPending(segment) &&
      selectedS2GTags(segment) === selectedS2GCopyTag
  }).asUInt
  val calculatedS2GClear = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    selectedS2GSameTag.pad(32),
    UIntToOH(selectedS2GElement, 32))
  val selectedS2GTag = Mux(
    completeS2GResponse, selectedS2GMeta.firstTag, calculatedS2GTag)
  val selectedS2GRouteElement = Mux(
    completeS2GResponse,
    selectedS2GMeta.firstElement,
    selectedS2GElement)
  val selectedS2GClear = Mux(
    completeS2GResponse,
    selectedS2GMeta.firstClear,
    calculatedS2GClear)
  val s2gDecompValid =
    (completeS2GResponse || residentS2GDecompVec.asUInt.orR) &&
    selectedS2GPending.orR

  val s2gDecompFire = s2gDecompValid && hasFreeLine && !killMode &&
    commandDirection === TmaV2Spec.DirectionS2G.U

  when(s2gDecompFire) {
    payloadPending(s2gDecompPayload) :=
      selectedS2GPending & ~selectedS2GClear
    lineState(freeLine) := LineState.RouteWrite
    lineWriteValid := true.B
    lineWriteTarget := freeLine
    lineWriteTag := selectedS2GTag
    lineWriteNextState :=
      (if (mmuEnabled) LineState.NeedTranslate
       else LineState.NeedCache)
    val routePaddingWidth =
      lineRouteWidth - 25 - 5 - payloadWidth
    lineWriteRoute := Cat(
      selectedS2GTag,
      0.U(routePaddingWidth.W),
      selectedS2GRouteElement,
      s2gDecompPayload)
  }

  // ------------------------------------------------------------------------
  // Single lookup/cycle translation service with one active external miss
  // ------------------------------------------------------------------------
  val translationValid = RegInit(VecInit(Seq.fill(
    TmaV2Spec.TranslationEntriesPerCommand)(false.B)))
  val translationVpn = Reg(Vec(
    TmaV2Spec.TranslationEntriesPerCommand, UInt(20.W)))
  val translationPpn = Reg(Vec(
    TmaV2Spec.TranslationEntriesPerCommand, UInt(20.W)))
  val translationRr = RegInit(0.U(2.W))
  val tlbMissValid = RegInit(false.B)
  val tlbMissSource = Reg(UInt(lineWidth.W))
  val tlbMissVpn = Reg(UInt(20.W))

  when(io.command.fire) {
    translationValid.foreach(_ := false.B)
    translationRr := 0.U
    tlbMissValid := false.B
    recentValid.foreach(_ := false.B)
    pairValid.foreach(_ := false.B)
    routeCandidateValid := false.B
  }

  val translateGroupArb =
    Module(new RRArbiter(UInt(lineGroupWidth.W), lineGroups))
  val translateLocal = Wire(Vec(lineGroups, UInt(2.W)))
  for (group <- 0 until lineGroups) {
    val local = Wire(Vec(4, Bool()))
    for (sub <- 0 until 4) {
      val index = group * 4 + sub
      local(sub) := (if (index < lineEntries)
        lineState(index) === LineState.NeedTranslate else false.B)
    }
    translateLocal(group) := PriorityEncoder(local)
    translateGroupArb.io.in(group).valid := local.asUInt.orR
    translateGroupArb.io.in(group).bits := group.U
  }
  val translateGroup = translateGroupArb.io.out.bits
  val translateLocalSelected =
    if (lineGroups == 1) translateLocal(0)
    else translateLocal(translateGroup)
  val translateLine = (translateGroup * 4.U +
    translateLocalSelected)(lineWidth - 1, 0)
  val translateVpn = lineAddressTag(translateLine)(24, 5)
  val translationHitVec = VecInit((0 until
    TmaV2Spec.TranslationEntriesPerCommand).map { entry =>
    translationValid(entry) &&
      translationVpn(entry) === translateVpn
  })
  val selectedTranslationHit = translationHitVec.asUInt.orR
  val selectedTranslationEntry = PriorityEncoder(translationHitVec)

  io.translationHit := false.B
  io.translationMiss := false.B
  io.translationCoalesce := false.B
  io.tlbRequest.valid := false.B
  io.tlbRequest.bits.source := translateLine
  io.tlbRequest.bits.virtualAddress :=
    Cat(lineAddressTag(translateLine), 0.U(7.W))
  io.tlbRequest.bits.asid := commandAsid
  translateGroupArb.io.out.ready := false.B
  when(translateGroupArb.io.out.valid && !killMode) {
    when(selectedTranslationHit) {
      translateGroupArb.io.out.ready := true.B
      lineAddressTag(translateLine) :=
        Cat(translationPpn(selectedTranslationEntry),
          lineAddressTag(translateLine)(4, 0))
      lineState(translateLine) := LineState.NeedCache
      io.translationHit := true.B
    }.elsewhen(!tlbMissValid) {
      io.tlbRequest.valid := mmuEnabled.B
      translateGroupArb.io.out.ready :=
        !mmuEnabled.B || io.tlbRequest.ready
      when(!mmuEnabled.B || io.tlbRequest.fire) {
        lineState(translateLine) := Mux(
          mmuEnabled.B, LineState.WaitTranslate, LineState.NeedCache)
        when(mmuEnabled.B) {
          tlbMissValid := true.B
          tlbMissSource := translateLine
          tlbMissVpn := translateVpn
          io.translationMiss := true.B
        }
      }
    }.otherwise {
      // Advance the small RR selector without changing the context. It will
      // hit after the outstanding translation fills the cache.
      translateGroupArb.io.out.ready := true.B
      io.translationCoalesce := translateVpn === tlbMissVpn
    }
  }
  if (!mmuEnabled) {
    io.tlbRequest.valid := false.B
  }
  io.tlbResponse.ready := mmuEnabled.B && tlbMissValid
  when(io.tlbResponse.fire) {
    assert(io.tlbResponse.bits.source === tlbMissSource)
    val ppn = io.tlbResponse.bits.physicalAddress(31, 12)
    translationValid(translationRr) := !killMode
    translationVpn(translationRr) := tlbMissVpn
    translationPpn(translationRr) := ppn
    translationRr := translationRr + 1.U
    lineAddressTag(tlbMissSource) :=
      Cat(ppn, lineAddressTag(tlbMissSource)(4, 0))
    lineState(tlbMissSource) := Mux(
      killMode, LineState.Free, LineState.NeedCache)
    tlbMissValid := false.B
  }

  // ------------------------------------------------------------------------
  // One cache request candidate/cycle
  // ------------------------------------------------------------------------
  val cacheGroupArb =
    Module(new RRArbiter(UInt(lineGroupWidth.W), lineGroups))
  val cacheLocal = Wire(Vec(lineGroups, UInt(2.W)))
  for (group <- 0 until lineGroups) {
    val local = Wire(Vec(4, Bool()))
    for (sub <- 0 until 4) {
      val index = group * 4 + sub
      local(sub) := (if (index < lineEntries)
        lineState(index) === LineState.NeedCache else false.B)
    }
    cacheLocal(group) := PriorityEncoder(local)
    cacheGroupArb.io.in(group).valid := local.asUInt.orR
    cacheGroupArb.io.in(group).bits := group.U
  }
  val cacheGroup = cacheGroupArb.io.out.bits
  val cacheLocalSelected =
    if (lineGroups == 1) cacheLocal(0)
    else cacheLocal(cacheGroup)
  val cacheLine = (cacheGroup * 4.U +
    cacheLocalSelected)(lineWidth - 1, 0)
  val cachePayload = linePayload(cacheLine)
  val cachePhysicalAddress =
    Cat(lineAddressTag(cacheLine), 0.U(7.W))

  // Direction-exclusive payload selection. G2S consumes a ready payload at
  // the shared port while S2G consumes the payload owned by the selected
  // LineContext at the cache port. A single active direction means both
  // paths share one physical payload read mux.
  val g2sReadyVec =
    VecInit(payloadState.map(_ === PayloadState.G2SReady))
  val s2gNeedSharedVec =
    VecInit(payloadState.map(_ === PayloadState.S2GNeedShared))
  val g2sSharedPayload = PriorityEncoder(g2sReadyVec)
  val s2gSharedPayload = PriorityEncoder(s2gNeedSharedVec)
  val outgoingG2S = commandDirection ===
    TmaV2Spec.DirectionG2S.U
  val outgoingPayload = Mux(
    outgoingG2S, g2sSharedPayload, s2gSharedPayload)
  val outgoingPayloadValid = Mux(
    outgoingG2S, g2sReadyVec.asUInt.orR,
    s2gNeedSharedVec.asUInt.orR)
  val outgoingG2SMeta = g2sReadyMeta(outgoingPayload)
  val outgoingS2GMeta = s2gMeta(outgoingPayload)
  val outgoingS2GPending = payloadPending(outgoingPayload)
  val outgoingS2GElement = PriorityEncoder(outgoingS2GPending)
  val outgoingS2GSegment = PriorityEncoder(outgoingS2GPending(8, 0))
  val outgoingS2GTailVec = VecInit((0 until 8).map { lane =>
    val addressOffset = outgoingS2GMeta.globalAddress(lane)(6, 0)
    paddedFp6 && outgoingS2GMeta.globalBytes(lane).orR &&
      addressOffset +& outgoingS2GMeta.globalBytes(lane) > 128.U
  })
  val outgoingS2GTailLane = PriorityEncoder(outgoingS2GTailVec)
  val outgoingS2GLane = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    Mux(outgoingS2GSegment === 8.U,
      outgoingS2GTailLane, outgoingS2GSegment(2, 0)),
    outgoingS2GElement >> 2)(2, 0)
  val outgoingS2GWord = outgoingS2GElement(1, 0)
  val outgoingS2GTags = Wire(Vec(9, UInt(25.W)))
  for (lane <- 0 until 8) {
    outgoingS2GTags(lane) :=
      cacheLineTag(outgoingS2GMeta.globalAddress(lane))
  }
  outgoingS2GTags(8) := endingCacheLineTag(
    outgoingS2GMeta.globalAddress(outgoingS2GTailLane),
    outgoingS2GMeta.globalBytes(outgoingS2GTailLane))
  val outgoingS2GCopyTag = outgoingS2GTags(outgoingS2GSegment)
  val outgoingS2GReduceTag = cacheLineTag(
    outgoingS2GMeta.globalAddress(outgoingS2GLane) +
      (outgoingS2GWord << 2))
  val outgoingS2GFirstTag = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    outgoingS2GCopyTag, outgoingS2GReduceTag)
  val outgoingS2GSameTag = VecInit((0 until 9).map { segment =>
    outgoingS2GPending(segment) &&
      outgoingS2GTags(segment) === outgoingS2GCopyTag
  }).asUInt
  val outgoingS2GFirstClear = Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    outgoingS2GSameTag.pad(32),
    UIntToOH(outgoingS2GElement, 32))

  // Every S2G shared response beat must merge against the selected payload.
  // A bank-conflicting 128B read returns several beats, including a final
  // beat whose mask covers only the last subset of words; "final" therefore
  // does not mean that the response overwrites the complete payload. Give
  // all response beats the single payload read port and backpressure the
  // mutually independent cache issue for that cycle.
  val s2gSharedMerge = io.sharedResponse.valid &&
    commandDirection === TmaV2Spec.DirectionS2G.U &&
    sharedValid(selectedSharedSource)

  // Tensor addresses are word aligned by the public format restrictions.
  // Route cache data with small word selectors instead of a generic byte
  // byte shifter. Ordinary formats are 16B aligned, B4P64 has a fixed 8B phase,
  // and B6 alone may contribute one fixed word-aligned tail to the next line.
  val cachePayloadMeta = s2gPostMeta(cachePayload)
  val selectedS2GDataPayload = Mux(
    s2gSharedMerge,
    sharedPayload(selectedSharedSource),
    cachePayload)
  val selectedS2GData = payloadData(selectedS2GDataPayload)
  val s2gPayloadLanes =
    selectedS2GData.asTypeOf(Vec(8, UInt(128.W)))
  val cacheDataParts = Wire(Vec(8, UInt(1024.W)))
  val cacheMaskParts = Wire(Vec(8, UInt(128.W)))
  for (lane <- 0 until 8) {
    val laneAddress = cachePayloadMeta.globalAddress(lane)
    val laneBytes = cachePayloadMeta.globalBytes(lane)
    val laneMask = bytePrefixMask(laneBytes)
    val lineTag = s2gVirtualTag(cacheLine)
    val addressOffset = laneAddress(6, 0)
    val addressWord = laneAddress(6, 2)
    val firstCapacity = 128.U - addressOffset
    val firstBytes = Mux(
      laneBytes > firstCapacity, firstCapacity, laneBytes)
    val tailBytes = laneBytes - firstBytes
    val firstTag = cacheLineTag(laneAddress)
    val secondTag = endingCacheLineTag(laneAddress, laneBytes)
    val firstBelongs = firstTag === lineTag
    val secondBelongs =
      secondTag =/= firstTag && secondTag === lineTag
    val laneWords = s2gPayloadLanes(lane)
      .asTypeOf(Vec(4, UInt(32.W)))
    val laneWordMask = VecInit((0 until 4).map { word =>
      laneMask(word * 4 + 3, word * 4)
    })
    val lineWords = Wire(Vec(32, UInt(32.W)))
    val lineWordMasks = Wire(Vec(32, UInt(4.W)))
    for (word <- 0 until 32) {
      val relative = word.U(6.W) - addressWord
      val primaryActive = firstBelongs &&
        word.U >= addressWord &&
        (relative << 2) < firstBytes
      val tailActive = secondBelongs &&
        (word * 4).U < tailBytes
      val sourceWord = Mux(
        primaryActive,
        relative(1, 0),
        ((firstBytes >> 2) + word.U)(1, 0))
      lineWords(word) := Mux(
        primaryActive || tailActive, laneWords(sourceWord), 0.U)
      lineWordMasks(word) := Mux(
        primaryActive || tailActive, laneWordMask(sourceWord), 0.U)
    }
    cacheDataParts(lane) := lineWords.asUInt
    cacheMaskParts(lane) := lineWordMasks.asUInt
  }
  val gatheredCacheData = cacheDataParts.reduce(_ | _)
  val gatheredCacheMask = cacheMaskParts.reduce(_ | _)
  val reduceElement = lineReduceElement(cacheLine)
  val reduceLane = reduceElement >> 2
  val reduceWord = reduceElement(1, 0)
  val reduceWordIndex =
    (cachePayloadMeta.globalAddress(reduceLane)(6, 2) +
      reduceWord)(4, 0)
  val reduceMask = VecInit((0 until 32).map { word =>
    Mux(reduceWordIndex === word.U, "hf".U(4.W), 0.U(4.W))
  }).asUInt

  val cacheWrite = commandDirection === TmaV2Spec.DirectionS2G.U
  val smallG2sCredit = commandDirection ===
    TmaV2Spec.DirectionG2S.U &&
    commandTransactionBytes <= 4096.U
  val g2sCanIssue = !smallG2sCredit ||
    g2sOutstanding < 16.U
  // This is the sole payload cache-egress register. It ends the LineContext
  // select/word-routing cone before request-format and descriptor arbitration.
  // The one-entry pipe still accepts one request per cycle.
  val cacheIssueQueue = Module(new Queue(
    new TmaV2CacheRequest(cacheSourceEntries),
    1, pipe = true, flow = false))
  val retainCacheRequest = killMode && io.cacheRequestHeld
  io.cacheRequest.valid := cacheIssueQueue.io.deq.valid &&
    (!killMode || retainCacheRequest)
  io.cacheRequest.bits := cacheIssueQueue.io.deq.bits
  cacheIssueQueue.io.deq.ready := Mux(
    killMode && !retainCacheRequest, true.B, io.cacheRequest.ready)
  cacheGroupArb.io.out.ready := !killMode && !s2gSharedMerge &&
    cacheIssueQueue.io.enq.ready &&
    (!cacheWrite || hasFreeAck) && g2sCanIssue
  cacheIssueQueue.io.enq.valid := !killMode && !s2gSharedMerge &&
    cacheGroupArb.io.out.valid &&
    (!cacheWrite || hasFreeAck) && g2sCanIssue
  cacheIssueQueue.io.enq.bits.write := cacheWrite
  cacheIssueQueue.io.enq.bits.reduceMode := commandReduceMode
  cacheIssueQueue.io.enq.bits.signed :=
    commandDtype === TmaV2Spec.DTypeS32.U
  cacheIssueQueue.io.enq.bits.source :=
    Mux(cacheWrite, freeAck, cacheLine)
  cacheIssueQueue.io.enq.bits.physicalAddress := cachePhysicalAddress
  cacheIssueQueue.io.enq.bits.data :=
    gatheredCacheData.asTypeOf(Vec(8, UInt(128.W)))
  cacheIssueQueue.io.enq.bits.mask := Mux(
    commandReduceMode === TmaV2Spec.ReduceCopy.U,
    gatheredCacheMask, reduceMask)

  val cacheCaptureFire = cacheIssueQueue.io.enq.fire
  val s2gWriteFire = cacheCaptureFire && cacheWrite
  when(cacheCaptureFire) {
    when(cacheWrite) {
      writeAckValid(freeAck) := true.B
      lineState(cacheLine) := LineState.Free
    }.otherwise {
      lineState(cacheLine) := LineState.WaitCache
    }
  }
  // ------------------------------------------------------------------------
  // Cache response -> one/two G2S payloads, or S2G ack
  // ------------------------------------------------------------------------
  val responseSource = io.cacheResponse.bits.source
  val responseLine =
    responseSource(lineWidth - 1, 0)
  val responseAck =
    responseSource(ackWidth - 1, 0)
  val responseIsRead = commandDirection ===
    TmaV2Spec.DirectionG2S.U &&
    responseSource < lineEntries.U
  val responseIsAck = commandDirection ===
    TmaV2Spec.DirectionS2G.U &&
    responseSource < writeAckEntries.U
  // Do not feed a truncated, out-of-range source into a non-power-of-two
  // Vec. Only architecturally allocated IDs can select table state.
  val responseLineWaiting = VecInit((0 until lineEntries).map { entry =>
    responseSource === entry.U &&
      lineState(entry) === LineState.WaitCache
  }).asUInt.orR
  val responseAckWaiting = VecInit((0 until writeAckEntries).map { entry =>
    responseSource === entry.U && writeAckValid(entry)
  }).asUInt.orR
  returningRead := io.cacheResponse.valid && responseIsRead &&
    responseLineWaiting
  returningReadLine := responseLine
  val responsePairVec = VecInit((0 until recentEntries).map { entry =>
    pairValid(entry) && pairLeader(entry) === responseLine
  })
  val responsePairValid = responsePairVec.asUInt.orR
  val responsePairEntry = PriorityEncoder(responsePairVec)
  val responseFollower = pairFollower(responsePairEntry)
  val responseFollowerRoutePending =
    responsePairValid && lineWriteValid &&
      lineWriteTarget === responseFollower
  val responseHasFollower =
    returningRead && (responsePairValid || directResponseCandidate)
  val responsePayloadSpace = Mux(
    responseHasFollower, hasTwoFreePayloads, hasFreePayload)
  io.cacheResponse.ready := Mux(
    responseIsRead,
      Mux(killMode, responseLineWaiting,
        responsePayloadSpace && !responseFollowerRoutePending),
    responseIsAck && responseAckWaiting)

  def alignLineToPayload(routeBits: UInt): (UInt, UInt) = {
    val route = routeBits.asTypeOf(new TmaG2SLineRoute)
    val cacheWords = io.cacheResponse.bits.data.asUInt
      .asTypeOf(Vec(32, UInt(32.W)))
    val lanes = Wire(Vec(8, UInt(128.W)))
    val readyMeta = Wire(new TmaG2SReadyMeta)
    readyMeta.sharedBase := route.sharedBase
    for (lane <- 0 until 8) {
      val laneWords = Wire(Vec(4, UInt(32.W)))
      for (word <- 0 until 4) {
        val relative = word.U - route.destinationWord(lane)
        val active = word.U >= route.destinationWord(lane) &&
          (relative << 2) < route.bytes(lane)
        val cacheWord = route.cacheWord(lane) + relative
        laneWords(word) := Mux(active, cacheWords(cacheWord), 0.U)
      }
      val prefix = bytePrefixMask(route.bytes(lane))
      readyMeta.mask(lane) := MuxLookup(
        route.destinationWord(lane), prefix)(Seq(
          1.U -> (prefix << 4)(15, 0),
          2.U -> (prefix << 8)(15, 0),
          3.U -> (prefix << 12)(15, 0)))
      readyMeta.sharedAtomDelta(lane) :=
        route.sharedAtomDelta(lane)
      lanes(lane) := laneWords.asUInt
    }
    (lanes.asUInt, readyMeta.asUInt.pad(payloadMetaWidth))
  }

  val (responseLeaderData, responseLeaderMeta) =
    alignLineToPayload(lineRoute(responseLine))
  val responseSecondRoute = Mux(
    responsePairValid,
    lineRoute(responseFollower), routeCandidate.asUInt)
  val (responseSecondData, responseSecondMeta) =
    alignLineToPayload(responseSecondRoute)

  directResponseFire := io.cacheResponse.fire &&
    responseIsRead && directResponseCandidate
  when(io.cacheResponse.fire && responseIsRead) {
    assert(lineState(responseLine) === LineState.WaitCache)
    lineState(responseLine) := LineState.Free
    when(!killMode) {
      payloadData(freePayload0) := responseLeaderData
      payloadMeta(freePayload0) := responseLeaderMeta
      payloadState(freePayload0) := PayloadState.G2SReady
    }
    when(responsePairValid) {
      val follower = responseFollower
      when(!killMode) {
        payloadData(freePayload1) := responseSecondData
        payloadMeta(freePayload1) := responseSecondMeta
        payloadState(freePayload1) := PayloadState.G2SReady
      }
      lineState(follower) := LineState.Free
      pairValid(responsePairEntry) := false.B
    }.elsewhen(directResponseCandidate && !killMode) {
      payloadData(freePayload1) := responseSecondData
      payloadMeta(freePayload1) := responseSecondMeta
      payloadState(freePayload1) := PayloadState.G2SReady
      recentValid(recentMatchEntry) := false.B
    }
    for (entry <- 0 until recentEntries) {
      when(recentValid(entry) &&
          recentLeader(entry) === responseLine) {
        recentValid(entry) := false.B
      }
    }
  }
  when(io.cacheResponse.fire && responseIsAck) {
    assert(writeAckValid(responseAck))
    writeAckValid(responseAck) := false.B
  }
  // A cache write accepted on the same edge as an old ack reuses that source
  // ID. Allocation must win over the response-side clear.
  when(s2gWriteFire && recycleAckValid && freeAck === recycleAck) {
    writeAckValid(freeAck) := true.B
  }
  val g2sReadIssue = io.cacheRequest.fire && !cacheWrite
  g2sReadRetire := io.cacheResponse.fire && responseIsRead
  when(g2sReadIssue =/= g2sReadRetire) {
    g2sOutstanding := g2sOutstanding +
      g2sReadIssue.asUInt - g2sReadRetire.asUInt
  }
  recycleLine0Valid := s2gWriteFire ||
    (io.cacheResponse.fire && responseIsRead)
  recycleLine0 := Mux(
    io.cacheResponse.fire && responseIsRead,
    responseLine, cacheLine)
  recycleLine1Valid := io.cacheResponse.fire &&
    responseIsRead && responsePairValid
  recycleLine1 := responseFollower
  // Ack ready is exactly the scoreboard hit below, so a valid matching ack
  // is guaranteed to fire. Use that pre-ready predicate for the recyclable
  // credit to keep cache request valid independent of response ready.
  recycleAckValid := io.cacheResponse.valid && responseIsAck &&
    responseAckWaiting
  recycleAck := responseAck

  // ------------------------------------------------------------------------
  // Shared request/response
  // ------------------------------------------------------------------------
  val outgoingG2SLanes = Wire(Vec(8, UInt(128.W)))
  val outgoingSharedMask = Wire(Vec(8, UInt(16.W)))
  val outgoingSharedAtom = Wire(Vec(8, UInt(3.W)))
  val outgoingSharedAtomWide = Wire(Vec(8, UInt(4.W)))
  val outgoingSharedBase = Mux(
    outgoingG2S,
    outgoingG2SMeta.sharedBase,
    outgoingS2GMeta.sharedBase)
  val outgoingSharedBaseAtom = outgoingSharedBase(6, 4)
  val readyG2SLanes = payloadData(outgoingPayload)
    .asTypeOf(Vec(8, UInt(128.W)))
  for (lane <- 0 until 8) {
    val s2gBytes = Mux(paddedSubByte, 16.U,
      outgoingS2GMeta.globalBytes(lane))
    val g2sMask = outgoingG2SMeta.mask(lane)
    val g2sBitMask = Cat((0 until 16).reverse.map(byte =>
      Fill(8, g2sMask(byte))))
    outgoingG2SLanes(lane) :=
      readyG2SLanes(lane) & g2sBitMask
    outgoingSharedMask(lane) := Mux(
      outgoingG2S,
      g2sMask,
      bytePrefixMask(s2gBytes))
    val delta = Mux(
      outgoingG2S,
      outgoingG2SMeta.sharedAtomDelta(lane),
      outgoingS2GMeta.sharedDelta(lane))
    outgoingSharedAtomWide(lane) := outgoingSharedBaseAtom +& delta
    outgoingSharedAtom(lane) := outgoingSharedAtomWide(lane)(2, 0)
  }
  val outgoingPendingWords = VecInit((0 until 32).map { word =>
    val lane = word / 4
    val byte = (word % 4) * 4
    outgoingSharedMask(lane)(byte + 3, byte).orR
  }).asUInt

  // As with cache traffic, this is the sole one-entry shared egress.  Keeping
  // it next to the payload word router terminates that cone before the DmaCore
  // per-lane request formatter without adding another storage stage.
  val sharedIssueQueue = Module(new Queue(
    new TmaV2SharedRequest(sharedEntries),
    1, pipe = true, flow = false))
  val retainSharedRequest = killMode && io.sharedRequestHeld
  io.sharedRequest.valid := sharedIssueQueue.io.deq.valid &&
    (!killMode || retainSharedRequest)
  io.sharedRequest.bits := sharedIssueQueue.io.deq.bits
  sharedIssueQueue.io.deq.ready := Mux(
    killMode && !retainSharedRequest, true.B, io.sharedRequest.ready)
  sharedIssueQueue.io.enq.valid :=
    outgoingPayloadValid && hasFreeShared && !killMode
  sharedIssueQueue.io.enq.bits.write := outgoingG2S
  sharedIssueQueue.io.enq.bits.source := freeShared
  sharedIssueQueue.io.enq.bits.sharedSetIdx :=
    outgoingSharedBase(sharedSetIdxHi, sharedSetIdxLo)
  sharedIssueQueue.io.enq.bits.sharedAtomIndex := outgoingSharedAtom
  sharedIssueQueue.io.enq.bits.data :=
    Mux(outgoingG2S, outgoingG2SLanes.asUInt, 0.U)
  sharedIssueQueue.io.enq.bits.mask := outgoingSharedMask.asUInt

  val sharedCaptureFire = sharedIssueQueue.io.enq.fire
  val g2sSharedFire = sharedCaptureFire && outgoingG2S
  when(sharedCaptureFire) {
    sharedValid(freeShared) := true.B
    sharedPayload(freeShared) := outgoingPayload
    sharedPendingWords(freeShared) := outgoingPendingWords
    when(outgoingG2S) {
      payloadState(outgoingPayload) := PayloadState.Free
    }.otherwise {
      val postMeta = Wire(new TmaS2GPostSharedMeta)
      postMeta.globalAddress := outgoingS2GMeta.globalAddress
      postMeta.globalBytes := outgoingS2GMeta.globalBytes
      postMeta.firstClear := outgoingS2GFirstClear
      postMeta.firstTag := outgoingS2GFirstTag
      postMeta.firstElement := outgoingS2GElement
      payloadMeta(outgoingPayload) := postMeta.asUInt
      payloadState(outgoingPayload) := PayloadState.S2GWaitShared
    }
  }
  val sharedSource = selectedSharedSource
  val sharedMask = selectedSharedMask
  val sharedRemaining = selectedSharedRemaining
  io.sharedResponse.ready := sharedValid(sharedSource)
  recycleSharedValid := io.sharedResponse.fire &&
    !sharedRemaining.orR
  recycleShared := sharedSource

  when(io.sharedResponse.fire) {
    assert(sharedValid(sharedSource))
    sharedPendingWords(sharedSource) := sharedRemaining
    when(!sharedRemaining.orR) {
      sharedValid(sharedSource) := false.B
    }
    when(commandDirection === TmaV2Spec.DirectionS2G.U && !killMode) {
      val payload = sharedPayload(sharedSource)
      val responseMeta = s2gPostMeta(payload)
      // s2gSharedMerge selects this payload onto the sole read port for every
      // beat, including the final partial-mask beat.
      val oldWords =
        selectedS2GData.asTypeOf(Vec(32, UInt(32.W)))
      val responseWords =
        io.sharedResponse.bits.data.asTypeOf(Vec(32, UInt(32.W)))
      val mergedWords = Wire(Vec(32, UInt(32.W)))
      for (word <- 0 until 32) {
        mergedWords(word) := Mux(
          sharedMask(word), responseWords(word), oldWords(word))
      }
      when(!sharedRemaining.orR) {
        val mergedLanes =
          mergedWords.asUInt.asTypeOf(Vec(8, UInt(128.W)))
        val transformed = Wire(Vec(8, UInt(128.W)))
        for (lane <- 0 until 8) {
          val bytes = mergedLanes(lane).asTypeOf(Vec(16, UInt(8.W)))
          val fp6 = VecInit((0 until 16).map(element =>
            bytes(element)(5, 0)))
          transformed(lane) := Mux(
            commandDtype === TmaV2Spec.DTypeB6.U,
            fp6.asUInt.pad(128), mergedLanes(lane))
        }
        payloadData(payload) := transformed.asUInt
        payloadState(payload) := PayloadState.S2GDecompose
      }.otherwise {
        payloadData(payload) := mergedWords.asUInt
      }
    }.elsewhen(commandDirection === TmaV2Spec.DirectionS2G.U &&
        !selectedSharedRemaining.orR) {
      payloadState(sharedPayload(sharedSource)) := PayloadState.Free
    }
  }

  // ------------------------------------------------------------------------
  // State/counter updates and same-cycle recycling
  // ------------------------------------------------------------------------
  for (payload <- 0 until payloadEntries) {
    val allocatingLine = s2gDecompFire &&
      s2gDecompPayload === payload.U
    val completingLine = s2gWriteFire &&
      cachePayload === payload.U
    val nextPending = Mux(allocatingLine,
      payloadPending(payload) & ~selectedS2GClear,
      payloadPending(payload))
    val nextOutstanding = payloadOutstanding(payload) +
      allocatingLine.asUInt - completingLine.asUInt
    when(allocatingLine =/= completingLine) {
      payloadOutstanding(payload) := nextOutstanding
    }
    when(payloadState(payload) === PayloadState.S2GDecompose &&
        nextPending === 0.U && nextOutstanding === 0.U) {
      payloadState(payload) := PayloadState.Free
    }
  }
  s2gPayloadCompletes := s2gWriteFire &&
    payloadPending(cachePayload) === 0.U &&
    payloadOutstanding(cachePayload) === 1.U
  recyclePayloadValid := g2sSharedFire || s2gPayloadCompletes
  recyclePayload := Mux(g2sSharedFire, outgoingPayload, cachePayload)

  when(normalizedWindowFire) {
    when(normalizedWindow.last) { commandSealed := true.B }
    when(commandDirection === TmaV2Spec.DirectionG2S.U) {
      decompValid := true.B
      decompPending := incomingCopyPending
      decompTailLane := incomingTailLane
      decompFillPending := incomingHasFill.asUInt.orR
      decompSharedBase := incomingSharedBase
      for (lane <- 0 until 8) {
        decompGlobalAddress(lane) :=
          normalizedWindow.lanes(lane).globalAddress
        decompGlobalBytes(lane) := incomingGlobalBytes(lane)
        decompSharedDelta(lane) := incomingSharedDelta(lane)
        decompSharedBytes(lane) := incomingSharedBytes(lane)
      }
    }.otherwise {
      val target = freePayload0
      val meta = Wire(new TmaS2GPayloadMeta)
      payloadState(target) := PayloadState.S2GNeedShared
      meta.sharedBase := incomingSharedBase
      payloadPending(target) := incomingPending
      payloadOutstanding(target) := 0.U
      for (lane <- 0 until 8) {
        meta.globalAddress(lane) :=
          normalizedWindow.lanes(lane).globalAddress
        meta.globalBytes(lane) := incomingGlobalBytes(lane)
        meta.sharedDelta(lane) := incomingSharedDelta(lane)
      }
      payloadMeta(target) := meta.asUInt
    }
  }.elsewhen(decompRelease) {
    decompValid := false.B
  }

  // Reallocation wins over a same-cycle release.
  when(normalizedWindowFire &&
      commandDirection === TmaV2Spec.DirectionS2G.U &&
      recyclePayloadValid && freePayload0 === recyclePayload) {
    payloadState(freePayload0) := PayloadState.S2GNeedShared
  }
  when(io.cacheResponse.fire && responseIsRead && !killMode) {
    payloadState(freePayload0) := PayloadState.G2SReady
    when(responseHasFollower) {
      payloadState(freePayload1) := PayloadState.G2SReady
    }
  }
  // Synthetic fill can consume the PayloadSlot released by the outgoing
  // G2S shared request on this same edge.  As with a returning cache line,
  // the new allocation must win over the earlier release assignment.  The
  // payload data/route were already written in fillFire above.
  when(fillFire) {
    payloadState(freePayload0) := PayloadState.G2SReady
  }
  when(sharedCaptureFire && recycleSharedValid &&
      freeShared === recycleShared) {
    sharedValid(freeShared) := true.B
    sharedPayload(freeShared) := outgoingPayload
    sharedPendingWords(freeShared) := outgoingPendingWords
  }
  // A decomposer allocation may select a context retired by the cache
  // interface on this same edge. Restate the new state after all retirement
  // assignments so allocation wins; the compact route was written above.
  when(routeCandidateFire) {
    lineState(freeLine) := LineState.RouteWrite
  }
  when(s2gDecompFire) {
    lineState(freeLine) := LineState.RouteWrite
  }

  // Kill is a command-local flush plus response drain. States which have not
  // crossed an external Decoupled boundary are discarded immediately. TLB,
  // cache, shared and write-ack owners which already fired remain resident
  // until their matching response is consumed.
  when(killMatch) {
    commandKilled := true.B
    commandSealed := true.B
    decompValid := false.B
    decompFillPending := false.B
    routeCandidateValid := false.B
    lineWriteValid := false.B
    recentValid.foreach(_ := false.B)
    pairValid.foreach(_ := false.B)
    translationValid.foreach(_ := false.B)

    for (line <- 0 until lineEntries) {
      when(lineState(line) =/= LineState.WaitTranslate &&
          lineState(line) =/= LineState.WaitCache) {
        lineState(line) := LineState.Free
      }
    }
    for (payload <- 0 until payloadEntries) {
      when(payloadState(payload) =/= PayloadState.S2GWaitShared) {
        payloadState(payload) := PayloadState.Free
        payloadOutstanding(payload) := 0.U
        payloadPending(payload) := 0.U
      }
    }

    // These two one-entry queues own resources at enqueue time. If their
    // request has not fired externally, drop the queue head and undo that
    // ownership here.
    when(cacheIssueQueue.io.deq.valid && !io.cacheRequestHeld) {
      val source = cacheIssueQueue.io.deq.bits.source
      when(cacheIssueQueue.io.deq.bits.write) {
        for (ack <- 0 until writeAckEntries) {
          when(source === ack.U) { writeAckValid(ack) := false.B }
        }
      }.otherwise {
        for (line <- 0 until lineEntries) {
          when(source === line.U) { lineState(line) := LineState.Free }
        }
      }
    }
    when(sharedIssueQueue.io.deq.valid && !io.sharedRequestHeld) {
      val source = sharedIssueQueue.io.deq.bits.source
      for (entry <- 0 until sharedEntries) {
        when(source === entry.U) {
          sharedValid(entry) := false.B
          payloadState(sharedPayload(entry)) := PayloadState.Free
          payloadOutstanding(sharedPayload(entry)) := 0.U
          payloadPending(sharedPayload(entry)) := 0.U
        }
      }
    }
  }

  io.seal.ready := commandValid && !commandSealed
  when(io.seal.fire) { commandSealed := true.B }

  // ------------------------------------------------------------------------
  // Completion
  // ------------------------------------------------------------------------
  val completionQ =
    Module(new Queue(new TmaCompletionPayload, 2))
  val allLinesFree =
    !VecInit(lineState.map(_ =/= LineState.Free)).asUInt.orR
  val allPayloadsFree =
    !VecInit(payloadState.map(_ =/= PayloadState.Free)).asUInt.orR
  val allSharedFree = !sharedValid.asUInt.orR
  val allAcksFree = !writeAckValid.asUInt.orR
  completionQ.io.enq.valid := commandValid && commandSealed &&
    !decompValid && !routeCandidateValid &&
    allLinesFree && allPayloadsFree &&
    allSharedFree && allAcksFree && !tlbMissValid
  completionQ.io.enq.bits.wid := commandWid
  completionQ.io.enq.bits.copyDirection := commandDirection
  completionQ.io.enq.bits.group := commandGroup
  completionQ.io.enq.bits.barrierValid := commandBarrierValid
  completionQ.io.enq.bits.barrierId := commandBarrierId
  completionQ.io.enq.bits.barrierGeneration :=
    commandBarrierGeneration
  completionQ.io.enq.bits.transactionBytes :=
    commandTransactionBytes
  when(completionQ.io.enq.fire) {
    commandValid := false.B
    commandSealed := false.B
    translationValid.foreach(_ := false.B)
    recentValid.foreach(_ := false.B)
    pairValid.foreach(_ := false.B)
    commandKilled := false.B
  }
  io.completion.valid := completionQ.io.deq.valid
  completionQ.io.deq.ready := io.completion.ready
  io.completion.bits.wid := completionQ.io.deq.bits.wid
  io.completion.bits.copyDirection :=
    completionQ.io.deq.bits.copyDirection
  io.completion.bits.group := completionQ.io.deq.bits.group
  io.completion.bits.barrierValid :=
    completionQ.io.deq.bits.barrierValid
  io.completion.bits.barrierId :=
    completionQ.io.deq.bits.barrierId
  io.completion.bits.barrierGeneration :=
    completionQ.io.deq.bits.barrierGeneration
  io.completion.bits.transactionBytes :=
    completionQ.io.deq.bits.transactionBytes

  val activeLines = PopCount(lineState.map(_ =/= LineState.Free))
  val activeCacheLines = PopCount(lineState.map(state =>
    state =/= LineState.Free && state =/= LineState.Follower))
  val activePayloads =
    PopCount(payloadState.map(_ =/= PayloadState.Free))
  io.activeCommands := commandValid.asUInt
  io.activeWindows :=
    (activeLines +& activePayloads +& decompValid.asUInt +
      routeCandidateValid.asUInt).pad(activeWidth)
  io.activeRequests := activeCacheLines
  io.activeShared := PopCount(sharedValid)
  io.activeWriteAcks := PopCount(writeAckValid)
  io.windowIssued := normalizedWindowFire
  // Report G2S retirement at the visible shared interface even though the
  // internal PayloadSlot ownership transfers into the egress register one
  // cycle earlier.  This preserves the external PMU/test contract.
  io.windowRetired :=
    (io.sharedRequest.fire && io.sharedRequest.bits.write) ||
      s2gPayloadCompletes
  io.uniqueLineWave := io.cacheRequest.fire
  io.robFull := !hasFreePayload
  io.requestFull := !hasFreeLine
  io.sharedFull := !hasFreeShared
  io.writeAckFull := !hasFreeAck
  io.cacheToSharedLatencyValid := false.B
  io.cacheToSharedLatency := 0.U
  if (PMU_TMA) {
    // Simulation-only attribution. Both arrays and the timestamp counter are
    // absent from PMU-off release/DC RTL by Scala elaboration.
    val perfCycle = RegInit(0.U(32.W))
    val cacheIssueCycle = Reg(Vec(cacheSourceEntries, UInt(32.W)))
    perfCycle := perfCycle + 1.U
    when(io.cacheRequest.fire) {
      cacheIssueCycle(io.cacheRequest.bits.source) := perfCycle
    }
    when(io.cacheResponse.fire) {
      io.cacheToSharedLatencyValid := true.B
      io.cacheToSharedLatency :=
        perfCycle - cacheIssueCycle(io.cacheResponse.bits.source)
    }
  }

  when(!reset.asBool) {
    when(killMode) {
      assert(!io.tlbRequest.fire,
        "a killed TMA command must not issue a new TLB request")
      assert(!io.cacheRequest.fire || io.cacheRequestHeld,
        "only a pre-kill irrevocable cache request may fire after kill")
      assert(!io.sharedRequest.fire || io.sharedRequestHeld,
        "only a pre-kill irrevocable shared request may fire after kill")
    }
    assert(!normalizedWindowFire ||
      PopCount(normalizedWindow.lanes.map(_.valid)) > 0.U)
    assert(!io.command.fire || !commandValid)
    assert(!io.cacheResponse.valid ||
      (responseIsRead && responseLineWaiting) ||
      (responseIsAck && responseAckWaiting))
    assert(!io.sharedResponse.valid ||
      sharedValid(io.sharedResponse.bits.source))
    assert(!(commandDirection === TmaV2Spec.DirectionG2S.U &&
      writeAckValid.asUInt.orR))
    assert(g2sOutstanding <= lineEntries.U)
    when(normalizedWindowFire) {
      assert(PopCount(incomingB6Tail) <= 1.U,
        "one legal padded B6 window has at most one cache-line tail")
    }
    for (line <- 0 until lineEntries) {
      when(lineState(line) === LineState.Follower) {
        assert(VecInit((0 until recentEntries).map { entry =>
          pairValid(entry) && pairFollower(entry) === line.U
        }).asUInt.orR)
      }
    }
    for (payload <- 0 until payloadEntries) {
      assert(payloadOutstanding(payload) <= lineEntries.U)
    }
    for (lane <- 0 until 8) {
      when(sharedIssueQueue.io.enq.fire && outgoingSharedMask(lane).orR) {
        assert(outgoingSharedBase(3, 0) === 0.U)
        assert(outgoingSharedAtomWide(lane) < 8.U,
          "one TMA shared request must remain inside one 128B set")
      }
      when(normalizedWindowFire && normalizedWindow.lanes(lane).valid) {
        assert(incomingGlobalBytes(lane) <= 16.U)
        assert(incomingSharedBytes(lane) <= 16.U)
        assert(incomingSharedDelta(lane) < 8.U)
        assert(normalizedWindow.lanes(lane).globalAddress(1, 0) === 0.U,
          "all legal Tensor lane addresses are word aligned")
        when(!paddedSubByte) {
          assert(normalizedWindow.lanes(lane).globalAddress(3, 0) === 0.U,
            "ordinary Tensor lanes are 16B aligned")
        }
      }
    }
  }
}
