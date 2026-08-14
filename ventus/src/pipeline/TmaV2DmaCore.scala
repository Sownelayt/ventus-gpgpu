package pipeline

import chisel3._
import chisel3.util._
import config.config.Parameters
import L1Cache.{DCacheMemReq_p, DCacheMemRsp}
import mmu.{L1TlbReq, L1TlbRsp, SV32}
import top.parameters._

class DmaSharedRsp extends Bundle {
  val instrId = UInt(log2Up(lsu_nMshrEntry).W)
  val isWrite = Bool()
  val isMBarrier = Bool()
  val data = Vec(num_thread, UInt(xLen.W))
  val activeMask = Vec(num_thread, Bool())
}

class DmaCoreIO(implicit p: Parameters) extends Bundle {
  val dma_req = Flipped(Decoupled(new vExeData))
  val dma_cache_rsp = Flipped(Decoupled(new DCacheMemRsp))
  val shared_rsp = Flipped(Decoupled(new DmaSharedRsp))
  val dma_cache_req = Decoupled(new DCacheMemReq_p)
  val shared_req = Decoupled(new ShareMemCoreReq_np)
  val tma_completion = Decoupled(new DmaCompletion)
  val txReserve = Decoupled(new DmaTxReserveRequest)
  val txReserveResponse = Flipped(Decoupled(new DmaTxReserveResponse))
  val to_l2TLB = Decoupled(new L1TlbReq(SV32))
  val from_l2TLB = Flipped(Decoupled(new L1TlbRsp(SV32)))
  val status = Valid(new DmaStatusUpdate)
  val perfEnable = Input(Bool())
  val perfReset = Input(Bool())
  val perf_tma = if (PMU_TMA) Some(Output(new TmaPerfCounters)) else None
}

object TmaV2AtomicEncoding {
  def opcode(mode: UInt): UInt = {
    val arithmetic = mode === TmaV2Spec.ReduceAdd.U ||
      mode === TmaV2Spec.ReduceMin.U ||
      mode === TmaV2Spec.ReduceMax.U
    Mux(arithmetic, 2.U, 3.U)
  }

  def param(mode: UInt, signed: Bool): UInt = MuxLookup(mode, 0.U)(Seq(
    TmaV2Spec.ReduceAdd.U -> 4.U,
    TmaV2Spec.ReduceMin.U -> Mux(signed, 0.U, 2.U),
    TmaV2Spec.ReduceMax.U -> Mux(signed, 1.U, 3.U),
    TmaV2Spec.ReduceAnd.U -> 2.U,
    TmaV2Spec.ReduceOr.U -> 1.U,
    TmaV2Spec.ReduceXor.U -> 0.U))
}

class TmaV2CompactRequest extends Bundle {
  val funct = UInt(3.W)
  val wid = UInt(depth_warp.W)
  val asid = UInt(SV32.asidLen.W)
  val group = UInt(log2Ceil(TmaV2Spec.S2GGroupsPerWarp).W)
  val in1 = UInt(32.W)
  val in2 = Vec(TmaV2Spec.RankMax, UInt(32.W))
  val in3 = UInt(32.W)
  val copyDirection = UInt(1.W)
  // Instruction bits 31:27 have one physical copy. Their interpretation is
  // selected by funct at the point of use.
  val opBits = UInt(5.W)
}

/** Fields used by the independent TensorMap control lane. */
class TmaV2ControlRequest extends Bundle {
  val address = UInt(32.W)
  val asid = UInt(SV32.asidLen.W)
  val wid = UInt(depth_warp.W)
  val subop = UInt(5.W)
}

/** Tensor fields that survive binding inside the single lookahead slot. */
class TmaV2PreparedTensor extends Bundle {
  val compiled = new TmaV2CompiledDescriptor
  val originComponents =
    Vec(TmaV2Spec.RankMax, SInt(64.W))
}

/**
  * One active command may own the data plane while exactly one later command
  * performs validation, descriptor lookup, binding and transaction reserve.
  * The lookahead never allocates a window or emits a payload request.
  */
class TmaV2Ingress extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new vExeData))
    val descriptorRequest = Vec(2, Decoupled(new TmaV2DescriptorRequest))
    val descriptorResponse =
      Flipped(Vec(2, Decoupled(new TmaV2DescriptorResponse)))
    val command = Decoupled(new TmaV2EngineCommand)
    val txReserve = Decoupled(new DmaTxReserveRequest)
    val txReserveResponse = Flipped(Decoupled(new DmaTxReserveResponse))
    val window = Decoupled(new TmaV2Window)
    val seal = Decoupled(Bool())
    val engineCompletion = Flipped(Valid(new TmaV2EngineCompletion))
    val directCompletion = Decoupled(new DmaCompletion)
    val status = Valid(new DmaStatusUpdate)
    val commandSlotFullStall = Output(Bool())
    val pendingFullStall = Output(Bool())
    val bindCycle = Output(Bool())
    val plannerProduced = Output(Bool())
    val plannerFire = Output(Bool())
    val plannerStalled = Output(Bool())
  })

  val Seq(sIdle, sValidate, sBulkReserve, sBulkReserveWait, sBulkReady,
    sDescReq, sDescWait, sTensorSetup, sTensorBindWait, sTensorReserve,
    sTensorReserveWait, sTensorReady, sTensorPlanner, sErrorReady,
    sSeal) = Enum(15)
  val Seq(cIdle, cPrefetchReq, cInvalidateReq, cInvalidateWait,
    cComplete) = Enum(5)
  val state = RegInit(sIdle)
  val controlState = RegInit(cIdle)
  val controlSaved = Reg(new TmaV2ControlRequest)
  // Raw coordinates/direction/shared base remain in saved and are reused
  // after binding. Only newly derived fields occupy prepared storage.
  val boundCommand = Reg(new TmaV2PreparedTensor)

  val engineActive = RegInit(false.B)
  val bulkEmitActive = RegInit(false.B)
  val tensorEmitActive = RegInit(false.B)
  // The lookahead must never overwrite the cursor of a stalled active bulk
  // transfer. Keep a compact prepared snapshot and copy it only at promotion.
  val preparedBulkGlobal = Reg(UInt(32.W))
  val preparedBulkShared = Reg(UInt(32.W))
  val preparedBulkWindows = Reg(UInt(26.W))
  val preparedBulkFirstChunk = Reg(UInt(8.W))
  val preparedBulkLastChunk = Reg(UInt(8.W))
  val bulkGlobal = Reg(UInt(32.W))
  val bulkShared = Reg(UInt(32.W))
  val bulkWindowsRemaining = Reg(UInt(26.W))
  val bulkFirst = RegInit(false.B)
  val bulkFirstChunk = Reg(UInt(8.W))
  val bulkLastChunk = Reg(UInt(8.W))

  val reserveAccepted = RegInit(false.B)
  val reserveBarrierValid = RegInit(false.B)
  val reserveBarrierId =
    Reg(UInt(log2Ceil(TmaV2Spec.MbarrierEntries).W))
  val reserveGeneration = Reg(UInt(8.W))
  val preparedTransactionBytes = Reg(UInt(32.W))

  when(io.engineCompletion.valid) {
    assert(engineActive)
    engineActive := false.B
  }

  val incomingTensorMapControl =
    io.in.bits.ctrl.funct === TmaV2Spec.FunctPrefetchTensormap.U
  val compactInput =
    WireDefault(0.U.asTypeOf(new TmaV2CompactRequest))
  compactInput.funct := io.in.bits.ctrl.funct
  compactInput.wid := io.in.bits.ctrl.wid
  compactInput.asid := io.in.bits.ctrl.asid.getOrElse(0.U)
  compactInput.group := io.in.bits.ctrl.dma_group
  compactInput.in1 := io.in.bits.in1(0)
  for (dimension <- 0 until TmaV2Spec.RankMax) {
    compactInput.in2(dimension) := io.in.bits.in2(dimension)
  }
  compactInput.in3 := io.in.bits.in3(0)
  compactInput.copyDirection := Mux(
    io.in.bits.ctrl.funct === TmaV2Spec.FunctBulkS2G.U ||
      io.in.bits.ctrl.funct === TmaV2Spec.FunctTensorS2G.U,
    TmaV2Spec.DirectionS2G.U, TmaV2Spec.DirectionG2S.U)
  compactInput.opBits := io.in.bits.ctrl.inst(31, 27)
  // Control hints remain independent of both the active command and the
  // single data lookahead.
  val controlRequestQ =
    Module(new Queue(new TmaV2ControlRequest, 1, pipe = true))
  controlRequestQ.io.enq.valid :=
    io.in.valid && incomingTensorMapControl
  controlRequestQ.io.enq.bits.address := compactInput.in1
  controlRequestQ.io.enq.bits.asid := compactInput.asid
  controlRequestQ.io.enq.bits.wid := compactInput.wid
  controlRequestQ.io.enq.bits.subop := compactInput.opBits
  // This queue is the sole LookaheadSlot. Its head remains stable through
  // validation, descriptor lookup, binding and reservation; no second raw
  // request register exists. `pipe` permits promotion and refill on one edge.
  val dataRequestQ =
    Module(new Queue(new TmaV2CompactRequest, 1, pipe = true, flow = false))
  dataRequestQ.io.enq.valid :=
    io.in.valid && !incomingTensorMapControl
  dataRequestQ.io.enq.bits := compactInput
  val saved = dataRequestQ.io.deq.bits
  val savedReduceMode = Mux(
    saved.funct === TmaV2Spec.FunctBulkS2G.U ||
      saved.funct === TmaV2Spec.FunctTensorS2G.U,
    saved.opBits(4, 2), TmaV2Spec.ReduceCopy.U)
  val savedBulkReduceType = saved.opBits(1, 0)
  val popDataRequest = WireDefault(false.B)
  dataRequestQ.io.deq.ready := popDataRequest
  io.in.ready := Mux(incomingTensorMapControl,
    controlRequestQ.io.enq.ready,
    dataRequestQ.io.enq.ready)
  io.commandSlotFullStall :=
    io.in.valid && !incomingTensorMapControl &&
      !dataRequestQ.io.enq.ready
  io.pendingFullStall := io.in.valid && incomingTensorMapControl &&
    !controlRequestQ.io.enq.ready

  when(state === sIdle && dataRequestQ.io.deq.valid) {
    reserveAccepted := false.B
    reserveBarrierValid := false.B
    preparedTransactionBytes := 0.U
    state := sValidate
  }

  val dataStatus = Wire(Valid(new DmaStatusUpdate))
  dataStatus.valid := false.B
  dataStatus.bits := 0.U.asTypeOf(new DmaStatusUpdate)
  def report(code: UInt, detail: UInt): Unit = {
    dataStatus.valid := true.B
    dataStatus.bits.wid := saved.wid
    dataStatus.bits.code := code
    dataStatus.bits.detail := detail
  }
  val controlStatusValid = RegInit(false.B)
  val controlStatus = Reg(new DmaStatusUpdate)
  io.status.valid := dataStatus.valid || controlStatusValid
  io.status.bits := Mux(dataStatus.valid, dataStatus.bits, controlStatus)
  when(controlStatusValid && !dataStatus.valid) {
    controlStatusValid := false.B
  }

  val readyCommand =
    state === sBulkReady || state === sTensorReady ||
      state === sErrorReady
  io.command.valid := readyCommand && !engineActive
  io.command.bits.wid := saved.wid
  io.command.bits.copyDirection := saved.copyDirection
  val bulkReduce = state === sBulkReady &&
    savedReduceMode =/= TmaV2Spec.ReduceCopy.U
  io.command.bits.dtype := Mux(state === sTensorReady,
    boundCommand.compiled.dtype,
    Mux(bulkReduce,
      Mux(savedBulkReduceType === TmaV2Spec.BulkReduceTypeS32.U,
        TmaV2Spec.DTypeS32.U, TmaV2Spec.DTypeU32.U),
      TmaV2Spec.DTypeU8.U))
  io.command.bits.oobFill := state === sTensorReady &&
    boundCommand.compiled.oobFill
  io.command.bits.reduceMode := Mux(
    state === sTensorReady || state === sBulkReady,
    savedReduceMode, TmaV2Spec.ReduceCopy.U)
  io.command.bits.asid := saved.asid
  io.command.bits.group := saved.group
  io.command.bits.barrierValid :=
    reserveAccepted && reserveBarrierValid && state =/= sErrorReady
  io.command.bits.barrierId := Mux(io.command.bits.barrierValid,
    reserveBarrierId, 0.U)
  io.command.bits.barrierGeneration := Mux(
    io.command.bits.barrierValid, reserveGeneration, 0.U)
  io.command.bits.transactionBytes := Mux(
    reserveAccepted && reserveBarrierValid && state =/= sErrorReady,
    preparedTransactionBytes, 0.U)

  for (client <- 0 until 2) {
    io.descriptorRequest(client).valid := false.B
    io.descriptorRequest(client).bits :=
      0.U.asTypeOf(new TmaV2DescriptorRequest)
  }
  io.descriptorRequest(0).valid := state === sDescReq
  io.descriptorRequest(0).bits.address := saved.in1
  io.descriptorRequest(0).bits.asid := saved.asid
  io.descriptorRequest(0).bits.wantResponse := true.B
  io.descriptorRequest(0).bits.invalidate := false.B
  io.descriptorRequest(1).valid :=
    controlState === cPrefetchReq || controlState === cInvalidateReq
  io.descriptorRequest(1).bits.address := controlSaved.address
  io.descriptorRequest(1).bits.asid := controlSaved.asid
  io.descriptorRequest(1).bits.wantResponse :=
    controlState === cInvalidateReq
  io.descriptorRequest(1).bits.invalidate :=
    controlState === cInvalidateReq
  io.descriptorResponse(0).ready := state === sDescWait
  io.descriptorResponse(1).ready := controlState === cInvalidateWait

  val binder = Module(new TmaV2CommandBinder)
  val windowPlanner = Module(new TmaV2WindowPlanner)
  io.bindCycle := binder.io.bindCycle
  io.plannerProduced := windowPlanner.io.produced
  io.plannerFire := windowPlanner.io.out.fire
  io.plannerStalled := windowPlanner.io.stalled

  binder.io.in.valid := state === sTensorSetup &&
    boundCommand.compiled.status === TmaV2Status.Ok
  binder.io.in.bits.compiled := boundCommand.compiled
  binder.io.in.bits.request.copyDirection := saved.copyDirection
  binder.io.in.bits.request.reduceMode := savedReduceMode
  binder.io.in.bits.request.sharedBase := saved.in3
  for (dimension <- 0 until TmaV2Spec.RankMax) {
    binder.io.in.bits.request.coordinates(dimension) :=
      saved.in2(dimension).asSInt
  }
  binder.io.out.ready := state === sTensorBindWait

  val plannerCommand =
    WireDefault(0.U.asTypeOf(new TmaV2BoundCommand))
  plannerCommand.compiled := boundCommand.compiled
  plannerCommand.copyDirection := saved.copyDirection
  plannerCommand.sharedBase := saved.in3
  for (dimension <- 0 until TmaV2Spec.RankMax) {
    plannerCommand.coordinates(dimension) :=
      saved.in2(dimension).asSInt
  }
  plannerCommand.originComponents := boundCommand.originComponents
  plannerCommand.reduceMode := savedReduceMode
  windowPlanner.io.in.valid :=
    state === sTensorPlanner && !tensorEmitActive
  windowPlanner.io.in.bits := plannerCommand

  val reserveBytes = Mux(
    state === sBulkReserve || state === sBulkReserveWait,
    saved.in2(0), boundCommand.compiled.logicalBytes)
  io.txReserve.valid :=
    state === sBulkReserve || state === sTensorReserve
  io.txReserve.bits.wid := saved.wid
  io.txReserve.bits.bytes := reserveBytes
  io.txReserveResponse.ready :=
    state === sBulkReserveWait || state === sTensorReserveWait

  // Lookahead precomputes count/edge chunks.  The active recurrence is now a
  // narrow decrement plus a three-way chunk select instead of a 32-bit
  // compare/subtract feedback path on every emitted window.
  val bulkChunk = Mux(
    bulkFirst, bulkFirstChunk,
    Mux(bulkWindowsRemaining === 1.U,
      bulkLastChunk, 128.U(8.W)))
  val bulkChunkWide = bulkChunk.pad(32)
  val bulkWindow = WireDefault(0.U.asTypeOf(new TmaV2Window))
  bulkWindow.sharedBase := bulkShared
  bulkWindow.last := bulkWindowsRemaining === 1.U
  for (lane <- 0 until 8) {
    val laneOffset = (lane * 16).U(32.W)
    val active = laneOffset < bulkChunkWide
    bulkWindow.lanes(lane).valid := active
    bulkWindow.lanes(lane).globalAddress := bulkGlobal + laneOffset
    bulkWindow.lanes(lane).globalBytes := Mux(active, 16.U, 0.U)
    bulkWindow.lanes(lane).sharedAtomDelta := lane.U
    bulkWindow.lanes(lane).sharedBytes := Mux(active, 16.U, 0.U)
  }

  io.window.valid := Mux(bulkEmitActive, true.B,
    tensorEmitActive && windowPlanner.io.out.valid)
  io.window.bits := Mux(bulkEmitActive, bulkWindow,
    windowPlanner.io.out.bits)
  windowPlanner.io.out.ready :=
    tensorEmitActive && !bulkEmitActive && io.window.ready

  val sealQ = Module(new Queue(Bool(), 1, pipe = true, flow = false))
  sealQ.io.enq.valid := state === sSeal
  sealQ.io.enq.bits := true.B
  io.seal <> sealQ.io.deq
  popDataRequest :=
    (state === sBulkReady && io.command.fire) ||
      (state === sTensorPlanner && windowPlanner.io.in.fire) ||
      (state === sSeal && sealQ.io.enq.fire)

  io.directCompletion.valid := controlState === cComplete
  io.directCompletion.bits.wid := controlSaved.wid
  io.directCompletion.bits.group := 0.U
  io.directCompletion.bits.is_s2g := false.B
  io.directCompletion.bits.barrierValid := false.B
  io.directCompletion.bits.barrierId := 0.U
  io.directCompletion.bits.barrierGeneration := 0.U
  io.directCompletion.bits.transactionBytes := 0.U

  controlRequestQ.io.deq.ready := controlState === cIdle
  when(controlRequestQ.io.deq.fire) {
    controlSaved := controlRequestQ.io.deq.bits
    val aligned = controlRequestQ.io.deq.bits.address(6, 0) === 0.U
    val prefetch = controlRequestQ.io.deq.bits.subop ===
      TmaV2Spec.TensorMapPrefetchSubop.U
    val invalidate = controlRequestQ.io.deq.bits.subop ===
      TmaV2Spec.TensorMapInvalidateSubop.U
    controlState := Mux(
      !aligned || (!prefetch && !invalidate), cComplete,
      Mux(prefetch, cPrefetchReq, cInvalidateReq))
    when(!aligned || (!prefetch && !invalidate)) {
      controlStatusValid := true.B
      controlStatus.wid := controlRequestQ.io.deq.bits.wid
      controlStatus.code := Mux(
        !aligned, TmaV2Spec.StatusInvalidDescriptor.U,
        TmaV2Spec.StatusUnsupportedFeature.U)
      controlStatus.detail := Mux(
        !aligned, TmaV2Status.BadAlignment,
        controlRequestQ.io.deq.bits.subop)
    }
  }

  when(state === sValidate) {
    val bulk = saved.funct === TmaV2Spec.FunctBulkG2S.U ||
      saved.funct === TmaV2Spec.FunctBulkS2G.U
    val tensor = saved.funct === TmaV2Spec.FunctTensorG2S.U ||
      saved.funct === TmaV2Spec.FunctTensorS2G.U
    val bulkS2G = saved.funct === TmaV2Spec.FunctBulkS2G.U
    val bulkCopy = savedReduceMode === TmaV2Spec.ReduceCopy.U
    val bulkArithmetic = savedReduceMode === TmaV2Spec.ReduceAdd.U ||
      savedReduceMode === TmaV2Spec.ReduceMin.U ||
      savedReduceMode === TmaV2Spec.ReduceMax.U
    val bulkBitwise = savedReduceMode === TmaV2Spec.ReduceAnd.U ||
      savedReduceMode === TmaV2Spec.ReduceOr.U ||
      savedReduceMode === TmaV2Spec.ReduceXor.U
    val bulkArithmeticType =
      savedBulkReduceType === TmaV2Spec.BulkReduceTypeU32.U ||
        savedBulkReduceType === TmaV2Spec.BulkReduceTypeS32.U
    val bulkReduceEncodingLegal = !bulkS2G ||
      (bulkCopy &&
        savedBulkReduceType === TmaV2Spec.BulkReduceTypeU32.U) ||
      (bulkArithmetic && bulkArithmeticType) ||
      (bulkBitwise &&
        savedBulkReduceType === TmaV2Spec.BulkReduceTypeB32.U)
    val globalAddress = Mux(
      saved.copyDirection === TmaV2Spec.DirectionG2S.U,
      saved.in1, saved.in3)
    val sharedAddress = Mux(
      saved.copyDirection === TmaV2Spec.DirectionG2S.U,
      saved.in3, saved.in1)
    val badBulk = globalAddress(3, 0) =/= 0.U ||
      sharedAddress(3, 0) =/= 0.U ||
      saved.in2(0) === 0.U || saved.in2(0)(3, 0) =/= 0.U ||
      !bulkReduceEncodingLegal
    when(bulk && !badBulk) {
      val sharedPhase = sharedAddress(6, 0)
      val firstCapacity = 128.U(9.W) - sharedPhase
      val firstChunk = Mux(
        saved.in2(0) > firstCapacity,
        firstCapacity, saved.in2(0))
      val span = saved.in2(0) +& sharedPhase
      val windows = (span + 127.U) >> 7
      val tail = span(6, 0)
      val lastChunk = Mux(
        windows === 1.U, saved.in2(0)(7, 0),
        Mux(tail === 0.U, 128.U, tail))
      preparedBulkGlobal := globalAddress
      preparedBulkShared := sharedAddress
      preparedBulkWindows := windows(25, 0)
      preparedBulkFirstChunk := firstChunk(7, 0)
      preparedBulkLastChunk := lastChunk(7, 0)
      preparedTransactionBytes := saved.in2(0)
      state := Mux(
        saved.copyDirection === TmaV2Spec.DirectionG2S.U,
        sBulkReserve, sBulkReady)
    }.elsewhen(tensor && saved.in1(6, 0) === 0.U) {
      state := sDescReq
    }.otherwise {
      state := sErrorReady
      report(TmaV2Spec.StatusUnsupportedFeature.U, saved.funct)
    }
  }

  when(io.txReserve.fire) {
    state := Mux(state === sBulkReserve,
      sBulkReserveWait, sTensorReserveWait)
  }
  when(io.txReserveResponse.fire) {
    assert(io.txReserveResponse.bits.wid === saved.wid)
    reserveAccepted := io.txReserveResponse.bits.accepted
    reserveBarrierValid := io.txReserveResponse.bits.barrierValid
    reserveBarrierId := io.txReserveResponse.bits.barrierId
    reserveGeneration := io.txReserveResponse.bits.generation
    when(!io.txReserveResponse.bits.accepted) {
      state := sErrorReady
    }.otherwise {
      state := Mux(state === sBulkReserveWait,
        sBulkReady, sTensorReady)
    }
  }

  when(io.descriptorRequest(0).fire) { state := sDescWait }
  when(io.descriptorResponse(0).fire) {
    boundCommand.compiled := io.descriptorResponse(0).bits.compiled
    state := sTensorSetup
  }
  when(state === sTensorSetup) {
    when(boundCommand.compiled.status =/= TmaV2Status.Ok) {
      state := sErrorReady
      val unsupported =
        boundCommand.compiled.status === TmaV2Status.UnsupportedDType ||
          boundCommand.compiled.status === TmaV2Status.UnsupportedFeature
      val architecturalStatus = Mux(
        boundCommand.compiled.status === TmaV2Status.AddressOverflow,
        TmaV2Spec.StatusAddressOverflow.U,
        Mux(unsupported, TmaV2Spec.StatusUnsupportedFeature.U,
          TmaV2Spec.StatusInvalidDescriptor.U))
      report(architecturalStatus, boundCommand.compiled.status)
    }.elsewhen(binder.io.in.fire) {
      state := sTensorBindWait
    }
  }
  when(state === sTensorBindWait && binder.io.out.fire) {
    when(!binder.io.out.bits.legal) {
      state := sErrorReady
      val unsupported =
        binder.io.out.bits.status === TmaV2Status.UnsupportedDType ||
          binder.io.out.bits.status === TmaV2Status.UnsupportedFeature
      val architecturalStatus = Mux(
        binder.io.out.bits.status === TmaV2Status.AddressOverflow,
        TmaV2Spec.StatusAddressOverflow.U,
        Mux(unsupported, TmaV2Spec.StatusUnsupportedFeature.U,
          TmaV2Spec.StatusInvalidDescriptor.U))
      report(architecturalStatus, binder.io.out.bits.status)
    }.otherwise {
      boundCommand.compiled := binder.io.out.bits.command.compiled
      boundCommand.originComponents :=
        binder.io.out.bits.command.originComponents
      preparedTransactionBytes :=
        binder.io.out.bits.command.compiled.logicalBytes
      state := Mux(
        saved.copyDirection === TmaV2Spec.DirectionG2S.U,
        sTensorReserve, sTensorReady)
    }
  }

  when(io.command.fire) {
    engineActive := true.B
    when(state === sBulkReady) {
      bulkGlobal := preparedBulkGlobal
      bulkShared := preparedBulkShared
      bulkWindowsRemaining := preparedBulkWindows
      bulkFirst := true.B
      bulkFirstChunk := preparedBulkFirstChunk
      bulkLastChunk := preparedBulkLastChunk
      bulkEmitActive := true.B
      state := sIdle
    }.elsewhen(state === sTensorReady) {
      state := sTensorPlanner
    }.otherwise {
      state := sSeal
    }
  }
  when(windowPlanner.io.in.fire) {
    tensorEmitActive := true.B
    state := sIdle
  }
  when(bulkEmitActive && io.window.fire) {
    bulkGlobal := bulkGlobal + bulkChunkWide
    bulkShared := bulkShared + bulkChunkWide
    bulkWindowsRemaining := bulkWindowsRemaining - 1.U
    bulkFirst := false.B
    when(bulkWindow.last) { bulkEmitActive := false.B }
  }
  when(tensorEmitActive && io.window.fire && !bulkEmitActive &&
      io.window.bits.last) {
    tensorEmitActive := false.B
  }
  when(state === sSeal && sealQ.io.enq.fire) { state := sIdle }

  when(io.descriptorRequest(1).fire) {
    controlState := Mux(
      controlState === cInvalidateReq, cInvalidateWait, cComplete)
  }
  when(io.descriptorResponse(1).fire) { controlState := cComplete }
  when(io.directCompletion.fire) { controlState := cIdle }

  when(!reset.asBool) {
    assert(!(bulkEmitActive && tensorEmitActive))
    assert(!io.window.fire ||
      PopCount(io.window.bits.lanes.map(_.valid)) > 0.U)
    assert(!(windowPlanner.io.in.fire && tensorEmitActive))
    when(state =/= sIdle && engineActive) {
      assert(!io.window.fire || bulkEmitActive || tensorEmitActive,
        "lookahead must not emit payload windows")
    }
  }
}

class TmaV2DmaCore(
    windowEntries: Int = TmaV2Spec.DefaultWindowEntries,
    requestEntries: Int = TmaV2Spec.DefaultGlobalRequestEntries,
    sharedEntries: Int = TmaV2Spec.DefaultSharedReadyEntries,
    writeAckEntries: Int = TmaV2Spec.DefaultWriteAckEntries,
    descriptorEntries: Int = TmaV2Spec.DefaultDescriptorEntries)(
    implicit p: Parameters) extends Module {
  require(requestEntries >= 2 && requestEntries <= 256,
    "TMA requestEntries must be in 2..256")
  require(writeAckEntries >= 2 && writeAckEntries <= 256,
    "TMA writeAckEntries must be in 2..256")
  private val cacheSourceEntries =
    TmaV2Spec.cacheSourceEntries(requestEntries, writeAckEntries)
  val io = IO(new DmaCoreIO)
  require(l1cache_sourceBits >
    log2Ceil(cacheSourceEntries),
    "descriptor and global-request responses need distinct DMA source IDs")
  val ingress = Module(new TmaV2Ingress)
  val subsystem = Module(new TmaV2WindowSubsystem(
    windowEntries = windowEntries,
    requestEntries = requestEntries,
    sharedEntries = sharedEntries,
    writeAckEntries = writeAckEntries,
    descriptorEntries = descriptorEntries))
  // PMU reset must not evict compiled descriptors.  Global invalidation is
  // provided by module reset; architectural address-scoped invalidation
  // travels through descriptor client 1.
  subsystem.io.descriptorInvalidateAll := false.B

  ingress.io.in <> io.dma_req
  for (client <- 0 until 2) {
    subsystem.io.descriptorRequest(client) <> ingress.io.descriptorRequest(client)
    ingress.io.descriptorResponse(client) <> subsystem.io.descriptorResponse(client)
  }
  subsystem.io.command <> ingress.io.command
  io.txReserve <> ingress.io.txReserve
  ingress.io.txReserveResponse <> io.txReserveResponse
  // Terminate both the tensor Planner and bulk arithmetic before the Engine
  // allocators.  The one-entry pipe accepts/refills every cycle, so this is a
  // fill-latency change only; it does not insert bubbles into an II=1 stream.
  val windowIngress = Module(new Queue(
    new TmaV2Window, 1, pipe = true, flow = false))
  windowIngress.io.enq <> ingress.io.window
  subsystem.io.window <> windowIngress.io.deq
  subsystem.io.seal <> ingress.io.seal
  io.status := ingress.io.status

  val tlbArb = Module(new Arbiter(new L1TlbReq(SV32), 2))
  val tlbBusy = RegInit(false.B)
  val tlbOwnerLine = RegInit(false.B)
  val tlbLineSource = Reg(UInt(log2Ceil(requestEntries).W))
  val descriptorPaddr = Reg(UInt(SV32.paLen.W))
  val descriptorNeedCache = RegInit(false.B)
  val descriptorCacheWait = RegInit(false.B)

  tlbArb.io.in(0).valid := (if (MMU_ENABLED)
    subsystem.io.descriptorMemoryRequest.valid &&
      !descriptorNeedCache && !descriptorCacheWait
  else false.B)
  tlbArb.io.in(0).bits.vaddr := subsystem.io.descriptorMemoryRequest.bits.address
  tlbArb.io.in(0).bits.asid := subsystem.io.descriptorMemoryRequest.bits.asid
  subsystem.io.descriptorMemoryRequest.ready := (if (MMU_ENABLED)
    tlbArb.io.in(0).ready && !descriptorNeedCache &&
      !descriptorCacheWait
  else !descriptorNeedCache && !descriptorCacheWait)
  tlbArb.io.in(1).valid := subsystem.io.tlbRequest.valid
  tlbArb.io.in(1).bits.vaddr := subsystem.io.tlbRequest.bits.virtualAddress
  tlbArb.io.in(1).bits.asid := subsystem.io.tlbRequest.bits.asid
  subsystem.io.tlbRequest.ready := tlbArb.io.in(1).ready
  val tlbIssueValid = RegInit(false.B)
  val tlbIssue = Reg(new L1TlbReq(SV32))
  val tlbIssueOwnerLine = Reg(Bool())
  val tlbIssueLineSource =
    Reg(UInt(log2Ceil(requestEntries).W))
  tlbArb.io.out.ready := (if (MMU_ENABLED)
    !tlbIssueValid && !tlbBusy else false.B)
  when(tlbArb.io.out.fire) {
    tlbIssueValid := true.B
    tlbIssue := tlbArb.io.out.bits
    tlbIssueOwnerLine := tlbArb.io.chosen === 1.U
    tlbIssueLineSource := subsystem.io.tlbRequest.bits.source
  }
  io.to_l2TLB.valid := (if (MMU_ENABLED)
    tlbIssueValid && !tlbBusy else false.B)
  io.to_l2TLB.bits := tlbIssue
  when(io.to_l2TLB.fire) {
    tlbIssueValid := false.B
    tlbBusy := true.B
    tlbOwnerLine := tlbIssueOwnerLine
    tlbLineSource := tlbIssueLineSource
  }
  subsystem.io.tlbResponse.valid := (if (MMU_ENABLED)
    io.from_l2TLB.valid && tlbBusy && tlbOwnerLine else false.B)
  subsystem.io.tlbResponse.bits.source := tlbLineSource
  subsystem.io.tlbResponse.bits.physicalAddress := io.from_l2TLB.bits.paddr
  io.from_l2TLB.ready := (if (MMU_ENABLED)
    tlbBusy && Mux(tlbOwnerLine, subsystem.io.tlbResponse.ready,
      !descriptorNeedCache && !descriptorCacheWait)
  else false.B)
  when(io.from_l2TLB.fire) {
    tlbBusy := false.B
    when(!tlbOwnerLine) {
      descriptorPaddr := io.from_l2TLB.bits.paddr
      descriptorNeedCache := true.B
    }
  }
  if (!MMU_ENABLED) {
    when(subsystem.io.descriptorMemoryRequest.fire) {
      descriptorPaddr :=
        subsystem.io.descriptorMemoryRequest.bits.address
      descriptorNeedCache := true.B
    }
  }

  val descriptorSource = Fill(l1cache_sourceBits, 1.U)
  val descriptorCacheRequest = Wire(Decoupled(new DCacheMemReq_p))
  descriptorCacheRequest.valid := descriptorNeedCache
  descriptorCacheRequest.bits := 0.U.asTypeOf(new DCacheMemReq_p)
  descriptorCacheRequest.bits.a_opcode := 4.U
  descriptorCacheRequest.bits.a_source := descriptorSource
  descriptorCacheRequest.bits.a_addr.foreach(_ := descriptorPaddr)
  descriptorCacheRequest.bits.a_mask.foreach(_ := Fill(BytesOfWord, 1.U))
  descriptorCacheRequest.bits.spike_info.foreach(_ := descriptorCacheRequest.bits.defaultSpikeInfo)
  when(descriptorCacheRequest.fire) {
    descriptorNeedCache := false.B
    descriptorCacheWait := true.B
  }

  // The backend has already scalarized tensor reduction to one complete
  // 32-bit word.  Map that word to the existing TL AMO encoding and let the
  // shared L2 AtomicUnit own the global read/modify/write sequence.
  val lineMask = subsystem.io.cacheRequest.bits.mask
  val requestedReduce = subsystem.io.cacheRequest.bits.reduceMode =/=
    TmaV2Spec.ReduceCopy.U
  val requestWordMasks = VecInit((0 until dcache_BlockWords).map { word =>
    lineMask(word * 4 + 3, word * 4)
  })
  val requestWordActive = VecInit(requestWordMasks.map(_.orR))
  val requestWords = Wire(Vec(dcache_BlockWords, UInt(32.W)))
  for (word <- 0 until dcache_BlockWords) {
    val bank = word / 4
    val subword = word % 4
    requestWords(word) :=
      subsystem.io.cacheRequest.bits.data(bank)(subword * 32 + 31, subword * 32)
  }

  val reduceOpcode =
    TmaV2AtomicEncoding.opcode(subsystem.io.cacheRequest.bits.reduceMode)
  val reduceParam = TmaV2AtomicEncoding.param(
    subsystem.io.cacheRequest.bits.reduceMode,
    subsystem.io.cacheRequest.bits.signed)
  val lineCacheRequest = Wire(Decoupled(new DCacheMemReq_p))
  lineCacheRequest.valid := subsystem.io.cacheRequest.valid
  lineCacheRequest.bits := 0.U.asTypeOf(new DCacheMemReq_p)
  lineCacheRequest.bits.a_opcode := Mux(requestedReduce, reduceOpcode,
    Mux(subsystem.io.cacheRequest.bits.write,
      Mux(lineMask.andR, 0.U, 1.U), 4.U))
  lineCacheRequest.bits.a_param := Mux(requestedReduce, reduceParam, 0.U)
  lineCacheRequest.bits.a_source := subsystem.io.cacheRequest.bits.source
  lineCacheRequest.bits.a_addr.foreach(_ :=
    subsystem.io.cacheRequest.bits.physicalAddress)
  for (word <- 0 until dcache_BlockWords) {
    lineCacheRequest.bits.a_data(word) := requestWords(word)
    lineCacheRequest.bits.a_mask(word) := lineMask(word * 4 + 3, word * 4)
  }
  lineCacheRequest.bits.spike_info.foreach(_ := lineCacheRequest.bits.defaultSpikeInfo)
  subsystem.io.cacheRequest.ready := lineCacheRequest.ready

  when(subsystem.io.cacheRequest.fire && requestedReduce) {
    assert(subsystem.io.cacheRequest.bits.write,
      "TMA reduce is only legal for S2G requests")
    assert(PopCount(requestWordActive) === 1.U,
      "TMA atomic reduce accepts exactly one 32-bit element per request")
    for (word <- 0 until dcache_BlockWords) {
      assert(requestWordMasks(word) === 0.U || requestWordMasks(word) === "hf".U,
        s"TMA atomic reduce word $word must be a complete 32-bit element")
    }
  }

  // Payload traffic has normal priority.  After seven payload grants a
  // waiting descriptor refill receives the next grant, which keeps prefetch
  // asynchronous without permitting compiler starvation.
  val descriptorDeferrals = RegInit(0.U(3.W))
  val forceDescriptor = descriptorCacheRequest.valid &&
    (!lineCacheRequest.valid || descriptorDeferrals === 7.U)
  // Payload traffic is already held by the sole one-entry cache egress in
  // WindowEngine.  Arbitrate that registered request directly with the
  // descriptor producer; retaining the former post-arbitration queue would
  // add a second residency stage and reduce the effective 40+6 capacity.
  io.dma_cache_req.valid :=
    descriptorCacheRequest.valid || lineCacheRequest.valid
  io.dma_cache_req.bits := Mux(
    forceDescriptor, descriptorCacheRequest.bits, lineCacheRequest.bits)
  descriptorCacheRequest.ready :=
    io.dma_cache_req.ready && forceDescriptor
  lineCacheRequest.ready :=
    io.dma_cache_req.ready && !forceDescriptor
  when(io.dma_cache_req.fire) {
    when(forceDescriptor) {
      descriptorDeferrals := 0.U
    }.elsewhen(descriptorCacheRequest.valid) {
      descriptorDeferrals := descriptorDeferrals + 1.U
    }
  }

  // Register incoming 128B responses before descriptor decode or PayloadSlot
  // selection.  This is an elastic boundary (simultaneous dequeue/refill is
  // legal), so it removes the external input delay from every wide storage
  // D pin without reducing response throughput.
  val cacheIngress = Module(new Queue(
    new DCacheMemRsp, 1, pipe = true, flow = false))
  cacheIngress.io.enq <> io.dma_cache_rsp
  val cacheResponse = cacheIngress.io.deq
  val responseIsDescriptor =
    cacheResponse.bits.d_source === descriptorSource
  subsystem.io.descriptorMemoryResponse.valid :=
    cacheResponse.valid && responseIsDescriptor
  subsystem.io.descriptorMemoryResponse.bits.words :=
    cacheResponse.bits.d_data
  subsystem.io.cacheResponse.valid := cacheResponse.valid &&
    !responseIsDescriptor
  subsystem.io.cacheResponse.bits.source :=
    cacheResponse.bits.d_source(log2Ceil(cacheSourceEntries) - 1, 0)
  for (bank <- 0 until 8) {
    subsystem.io.cacheResponse.bits.data(bank) := Cat((0 until 4).reverse.map { subword =>
      cacheResponse.bits.d_data(bank * 4 + subword)
    })
  }
  cacheResponse.ready := Mux(responseIsDescriptor,
    subsystem.io.descriptorMemoryResponse.ready, subsystem.io.cacheResponse.ready)
  when(subsystem.io.descriptorMemoryResponse.fire) { descriptorCacheWait := false.B }

  val sharedSource = subsystem.io.sharedRequest.bits.source
  // WindowEngine owns the sole registered shared egress.  Format that stable
  // request directly for the shared-memory interface; a second queue here
  // would only extend PayloadSlot residency.
  io.shared_req.valid := subsystem.io.sharedRequest.valid
  io.shared_req.bits := 0.U.asTypeOf(new ShareMemCoreReq_np)
  io.shared_req.bits.instrId := sharedSource
  io.shared_req.bits.isWrite := subsystem.io.sharedRequest.bits.write
  io.shared_req.bits.isMBarrier := false.B
  io.shared_req.bits.setIdx := subsystem.io.sharedRequest.bits.sharedSetIdx
  for (atom <- 0 until 8; word <- 0 until 4) {
    val lane = atom * 4 + word
    val byteMask = subsystem.io.sharedRequest.bits.mask(atom * 16 + word * 4 + 3,
      atom * 16 + word * 4)
    io.shared_req.bits.perLaneAddr(lane).activeMask :=
      byteMask.orR
    io.shared_req.bits.perLaneAddr(lane).blockOffset :=
      Cat(subsystem.io.sharedRequest.bits.sharedAtomIndex(atom),
        word.U(2.W))
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H :=
      byteMask
    io.shared_req.bits.data(lane) :=
      subsystem.io.sharedRequest.bits.data(lane * 32 + 31, lane * 32)
  }
  subsystem.io.sharedRequest.ready := io.shared_req.ready
  // Shared responses receive the same one-beat elastic input boundary.  The
  // registered data then feeds the compact word-routing/merge datapath.
  val sharedIngress = Module(new Queue(
    new DmaSharedRsp, 1, pipe = true, flow = false))
  sharedIngress.io.enq <> io.shared_rsp
  val sharedResponse = sharedIngress.io.deq
  subsystem.io.sharedResponse.valid := sharedResponse.valid
  subsystem.io.sharedResponse.bits.source := sharedResponse.bits.instrId
  subsystem.io.sharedResponse.bits.data :=
    Cat((0 until 32).reverse.map(sharedResponse.bits.data(_)))
  subsystem.io.sharedResponse.bits.wordMask :=
    sharedResponse.bits.activeMask.asUInt
  sharedResponse.ready := subsystem.io.sharedResponse.ready

  val completionArb = Module(new Arbiter(new DmaCompletion, 2))
  completionArb.io.in(0).valid := subsystem.io.completion.valid
  completionArb.io.in(0).bits.wid := subsystem.io.completion.bits.wid
  completionArb.io.in(0).bits.group := subsystem.io.completion.bits.group
  completionArb.io.in(0).bits.is_s2g :=
    subsystem.io.completion.bits.copyDirection === TmaV2Spec.DirectionS2G.U
  completionArb.io.in(0).bits.barrierValid := subsystem.io.completion.bits.barrierValid
  completionArb.io.in(0).bits.barrierId := subsystem.io.completion.bits.barrierId
  completionArb.io.in(0).bits.barrierGeneration :=
    subsystem.io.completion.bits.barrierGeneration
  completionArb.io.in(0).bits.transactionBytes := subsystem.io.completion.bits.transactionBytes
  subsystem.io.completion.ready := completionArb.io.in(0).ready
  completionArb.io.in(1) <> ingress.io.directCompletion
  val completionEgress = Module(new Queue(
    new DmaCompletion, 1, pipe = true, flow = false))
  completionEgress.io.enq <> completionArb.io.out
  io.tma_completion <> completionEgress.io.deq
  ingress.io.engineCompletion.valid := subsystem.io.completion.fire
  ingress.io.engineCompletion.bits := subsystem.io.completion.bits

  if (PMU_TMA) {
    val perf = RegInit(0.U.asTypeOf(new TmaPerfCounters))
    val perfCycle = RegInit(0.U(32.W))
    val windowFireRun = RegInit(0.U(26.W))
    val longestWindowFireRun = RegInit(0.U(26.W))
    val maxActiveWindows =
      RegInit(0.U(subsystem.io.activeWindows.getWidth.W))
    val maxActiveRequests =
      RegInit(0.U(subsystem.io.activeRequests.getWidth.W))
    val maxActiveShared =
      RegInit(0.U(subsystem.io.activeShared.getWidth.W))
    val maxActiveWriteAcks =
      RegInit(0.U(subsystem.io.activeWriteAcks.getWidth.W))
    val cacheIssueCycle = Reg(Vec(cacheSourceEntries, UInt(32.W)))
    val perfCommandDirection = RegInit(TmaV2Spec.DirectionG2S.U(1.W))
    val cacheResponseSource = subsystem.io.cacheResponse.bits.source
    val cacheResponseWrite =
      perfCommandDirection === TmaV2Spec.DirectionS2G.U
    val cacheResponseLatency = perfCycle - cacheIssueCycle(cacheResponseSource)
    val s2gCacheRequest = subsystem.io.cacheRequest.fire &&
      subsystem.io.cacheRequest.bits.write
    val s2gCopyCacheRequest = s2gCacheRequest &&
      subsystem.io.cacheRequest.bits.reduceMode === TmaV2Spec.ReduceCopy.U
    val s2gCacheResponse = subsystem.io.cacheResponse.fire && cacheResponseWrite
    val g2sCacheResponse = subsystem.io.cacheResponse.fire && !cacheResponseWrite

    when(subsystem.io.cacheRequest.fire) {
      cacheIssueCycle(subsystem.io.cacheRequest.bits.source) := perfCycle
    }
    when(subsystem.io.command.fire) {
      perfCommandDirection := subsystem.io.command.bits.copyDirection
    }

    when(io.perfReset) {
      perf := 0.U.asTypeOf(new TmaPerfCounters)
      perfCycle := 0.U
      windowFireRun := 0.U
      longestWindowFireRun := 0.U
      maxActiveWindows := 0.U
      maxActiveRequests := 0.U
      maxActiveShared := 0.U
      maxActiveWriteAcks := 0.U
    }.elsewhen(io.perfEnable) {
      perfCycle := perfCycle + 1.U
      when(ingress.io.window.fire) {
        val nextRun = windowFireRun + 1.U
        windowFireRun := nextRun
        when(nextRun > longestWindowFireRun) {
          longestWindowFireRun := nextRun
        }
      }.otherwise {
        windowFireRun := 0.U
      }
      perf.instIssued := perf.instIssued + io.dma_req.fire.asUInt
      perf.lineIssued := perf.lineIssued + ingress.io.window.fire.asUInt
      perf.putFull := perf.putFull + (s2gCopyCacheRequest && lineMask.andR).asUInt
      perf.putPart := perf.putPart + (s2gCopyCacheRequest && !lineMask.andR).asUInt
      perf.bytesWritten := perf.bytesWritten + Mux(s2gCacheRequest,
        PopCount(lineMask), 0.U)
      perf.sharedReadReq := perf.sharedReadReq +
        (subsystem.io.sharedRequest.fire && !subsystem.io.sharedRequest.bits.write).asUInt
      perf.sharedReadRsp := perf.sharedReadRsp +
        (sharedResponse.fire && !sharedResponse.bits.isWrite).asUInt
      perf.tlbReq := perf.tlbReq + io.to_l2TLB.fire.asUInt
      perf.g2sLineCount := perf.g2sLineCount +
        (ingress.io.window.fire &&
          perfCommandDirection === TmaV2Spec.DirectionG2S.U).asUInt
      perf.s2gLineCount := perf.s2gLineCount +
        (ingress.io.window.fire &&
          perfCommandDirection === TmaV2Spec.DirectionS2G.U).asUInt
      perf.g2sCompletionCount := perf.g2sCompletionCount +
        (subsystem.io.completion.fire &&
          subsystem.io.completion.bits.copyDirection === TmaV2Spec.DirectionG2S.U).asUInt
      perf.s2gCompletionCount := perf.s2gCompletionCount +
        (subsystem.io.completion.fire &&
          subsystem.io.completion.bits.copyDirection === TmaV2Spec.DirectionS2G.U).asUInt
      perf.g2sCacheResponseCount := perf.g2sCacheResponseCount + g2sCacheResponse.asUInt
      perf.s2gCacheResponseCount := perf.s2gCacheResponseCount + s2gCacheResponse.asUInt
      perf.windowIssued := perf.windowIssued +
        subsystem.io.windowIssued.asUInt
      perf.windowRetired := perf.windowRetired +
        subsystem.io.windowRetired.asUInt
      perf.uniqueLineWaves := perf.uniqueLineWaves +
        subsystem.io.uniqueLineWave.asUInt
      perf.translationHits := perf.translationHits +
        subsystem.io.translationHit.asUInt
      perf.translationMisses := perf.translationMisses +
        subsystem.io.translationMiss.asUInt
      perf.translationCoalesces := perf.translationCoalesces +
        subsystem.io.translationCoalesce.asUInt
      perf.windowRobFullCycles := perf.windowRobFullCycles +
        subsystem.io.robFull.asUInt
      perf.globalRequestFullCycles := perf.globalRequestFullCycles +
        subsystem.io.requestFull.asUInt
      perf.sharedQueueFullCycles := perf.sharedQueueFullCycles +
        subsystem.io.sharedFull.asUInt
      perf.cacheToSharedLatencyCount :=
        perf.cacheToSharedLatencyCount +
          subsystem.io.cacheToSharedLatencyValid.asUInt
      perf.cacheToSharedLatencySum :=
        perf.cacheToSharedLatencySum +
          Mux(subsystem.io.cacheToSharedLatencyValid,
            subsystem.io.cacheToSharedLatency, 0.U)
      perf.descriptorDemandHits :=
        perf.descriptorDemandHits + subsystem.io.descriptorEvents.demandHit.asUInt
      perf.descriptorDemandMisses :=
        perf.descriptorDemandMisses + subsystem.io.descriptorEvents.demandMiss.asUInt
      perf.descriptorPrefetchHits :=
        perf.descriptorPrefetchHits + subsystem.io.descriptorEvents.prefetchHit.asUInt
      perf.descriptorPrefetchMisses :=
        perf.descriptorPrefetchMisses + subsystem.io.descriptorEvents.prefetchMiss.asUInt
      perf.descriptorEvictions :=
        perf.descriptorEvictions + subsystem.io.descriptorEvents.eviction.asUInt
      perf.descriptorCompiles :=
        perf.descriptorCompiles + subsystem.io.descriptorEvents.compile.asUInt
      perf.descriptorCoalesces :=
        perf.descriptorCoalesces +
          (subsystem.io.descriptorEvents.demandCoalesce ||
            subsystem.io.descriptorEvents.prefetchCoalesce).asUInt
      perf.descriptorInvalidateKills :=
        perf.descriptorInvalidateKills +
          subsystem.io.descriptorEvents.invalidateKill.asUInt
      perf.descriptorCompileCycles :=
        perf.descriptorCompileCycles +
          subsystem.io.descriptorEvents.compileCycle.asUInt
      perf.bindCycles := perf.bindCycles + ingress.io.bindCycle.asUInt
      perf.plannerProduced :=
        perf.plannerProduced + ingress.io.plannerProduced.asUInt
      perf.plannerFire :=
        perf.plannerFire + ingress.io.plannerFire.asUInt
      perf.plannerStallCycles :=
        perf.plannerStallCycles + ingress.io.plannerStalled.asUInt
      when(subsystem.io.activeWindows > maxActiveWindows) {
        maxActiveWindows := subsystem.io.activeWindows
      }
      when(subsystem.io.activeRequests > maxActiveRequests) {
        maxActiveRequests := subsystem.io.activeRequests
      }
      when(subsystem.io.activeShared > maxActiveShared) {
        maxActiveShared := subsystem.io.activeShared
      }
      when(subsystem.io.activeWriteAcks > maxActiveWriteAcks) {
        maxActiveWriteAcks := subsystem.io.activeWriteAcks
      }

      if (PMU_TMA_DETAIL) {
        perf.lineFullStallCycles := perf.lineFullStallCycles +
          (ingress.io.window.valid && !ingress.io.window.ready).asUInt
        perf.ackTagFullStallCycles := perf.ackTagFullStallCycles +
          subsystem.io.writeAckFull.asUInt
        perf.commandSlotFullStallCycles := perf.commandSlotFullStallCycles +
          ingress.io.commandSlotFullStall.asUInt
        perf.pendingFullStallCycles := perf.pendingFullStallCycles +
          ingress.io.pendingFullStall.asUInt
        perf.tlbStallCycles := perf.tlbStallCycles +
          (subsystem.io.tlbRequest.valid && !subsystem.io.tlbRequest.ready).asUInt
        perf.sharedStallCycles := perf.sharedStallCycles +
          (subsystem.io.sharedRequest.valid && !subsystem.io.sharedRequest.ready).asUInt
        perf.cacheReqStallCycles := perf.cacheReqStallCycles +
          (lineCacheRequest.valid && !lineCacheRequest.ready).asUInt
        perf.g2sCacheLatencySum := perf.g2sCacheLatencySum +
          Mux(g2sCacheResponse, cacheResponseLatency, 0.U)
        perf.s2gCacheLatencySum := perf.s2gCacheLatencySum +
          Mux(s2gCacheResponse, cacheResponseLatency, 0.U)
        perf.activeCommandCycles := perf.activeCommandCycles + subsystem.io.activeCommands
        perf.activeLineCycles := perf.activeLineCycles +
          subsystem.io.activeWindows
        perf.sharedActiveLineCycles := perf.sharedActiveLineCycles +
          subsystem.io.activeShared
        perf.requestActiveLineCycles := perf.requestActiveLineCycles +
          subsystem.io.activeRequests
      }
    }
    val perfOutput = WireDefault(perf)
    perfOutput.maxActiveWindows := maxActiveWindows
    perfOutput.maxActiveRequests := maxActiveRequests
    perfOutput.maxActiveShared := maxActiveShared
    perfOutput.maxActiveWriteAcks := maxActiveWriteAcks
    perfOutput.longestWindowFireRun := longestWindowFireRun
    io.perf_tma.foreach(_ := perfOutput)
  }

  when(!reset.asBool) {
    assert(!(cacheResponse.valid && responseIsDescriptor &&
      !descriptorCacheWait))
    assert(!(subsystem.io.cacheResponse.fire &&
      subsystem.io.cacheResponse.bits.source >=
        cacheSourceEntries.U))
    assert(!(sharedResponse.fire && sharedResponse.bits.isMBarrier))
    if (!MMU_ENABLED) {
      assert(!io.to_l2TLB.valid,
        "MMU-off TMA must use identity translation without an external TLB request")
    }
  }
}

/** Standalone synthesis boundary for the complete TMA v2 DMA engine. */
class tma(
    windowEntries: Int = TmaV2Spec.DefaultWindowEntries,
    requestEntries: Int = TmaV2Spec.DefaultGlobalRequestEntries,
    sharedEntries: Int = TmaV2Spec.DefaultSharedReadyEntries,
    writeAckEntries: Int = TmaV2Spec.DefaultWriteAckEntries,
    descriptorEntries: Int = TmaV2Spec.DefaultDescriptorEntries)(
    implicit p: Parameters) extends Module {
  val io = IO(new DmaCoreIO)
  val core = Module(new TmaV2DmaCore(
    windowEntries = windowEntries,
    requestEntries = requestEntries,
    sharedEntries = sharedEntries,
    writeAckEntries = writeAckEntries,
    descriptorEntries = descriptorEntries))
  core.io.dma_req <> io.dma_req
  core.io.dma_cache_rsp <> io.dma_cache_rsp
  core.io.shared_rsp <> io.shared_rsp
  io.dma_cache_req <> core.io.dma_cache_req
  io.shared_req <> core.io.shared_req
  io.tma_completion <> core.io.tma_completion
  io.txReserve <> core.io.txReserve
  core.io.txReserveResponse <> io.txReserveResponse
  io.to_l2TLB <> core.io.to_l2TLB
  core.io.from_l2TLB <> io.from_l2TLB
  io.status := core.io.status
  core.io.perfEnable := io.perfEnable
  core.io.perfReset := io.perfReset
  if (PMU_TMA) io.perf_tma.foreach(_ := core.io.perf_tma.get)
}
