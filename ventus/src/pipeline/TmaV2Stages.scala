package pipeline

import chisel3._
import chisel3.util._
import mmu.SV32
import top.parameters._

/** One outstanding TMA-local cancellation domain. */
class TmaV2KillRequest extends Bundle {
  val asid = UInt(SV32.asidLen.W)
}

object TmaV2PreparedKind {
  val Bulk = 0.U(2.W)
  val Tensor = 1.U(2.W)
  val Empty = 2.U(2.W)
}

/** Atomic hand-off from the single Prepare slot to the Execute stage. */
class TmaV2PreparedCommand extends Bundle {
  val engine = new TmaV2EngineCommand
  val kind = UInt(2.W)

  val bulkGlobal = UInt(32.W)
  val bulkShared = UInt(32.W)
  val bulkWindows = UInt(26.W)
  val bulkFirstChunk = UInt(8.W)
  val bulkLastChunk = UInt(8.W)

  val tensor = new TmaV2BoundCommand
}

/**
  * Parameterized raw-command FIFO.
  *
  * Cancellation is deliberately off the normal ready path. A kill pulse
  * marks matching resident entries; cancelled heads are retired through a
  * cleanup-completion port in FIFO order. The physical entry remains
  * occupied until that completion is accepted, so source/group accounting
  * cannot be lost under completion backpressure.
  */
class TmaV2CommandQueue(entries: Int = TmaV2Spec.DefaultCommandQueueEntries)
    extends Module {
  require(entries >= 1 && entries <= 64,
    "TMA command queue entries must be in 1..64")
  private val pointerWidth = math.max(1, log2Ceil(entries))
  private val countWidth = log2Ceil(entries + 1)

  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(new TmaV2CompactRequest))
    val deq = Decoupled(new TmaV2CompactRequest)
    val kill = Flipped(Valid(new TmaV2KillRequest))
    val killActive = Input(Bool())
    val killAsid = Input(UInt(SV32.asidLen.W))
    val cancelled = Decoupled(new DmaCompletion)
    val matchingPending = Output(Bool())
    val occupancy = Output(UInt(countWidth.W))
    val full = Output(Bool())
  })

  val storage = Reg(Vec(entries, new TmaV2CompactRequest))
  val valid = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val cancelled = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val head = RegInit(0.U(pointerWidth.W))
  val tail = RegInit(0.U(pointerWidth.W))
  val count = RegInit(0.U(countWidth.W))

  def advance(pointer: UInt): UInt = {
    if (entries == 1) 0.U
    else Mux(pointer === (entries - 1).U, 0.U, pointer + 1.U)
  }

  val headValid = if (entries == 1) valid(0) else valid(head)
  val headCancelled = headValid &&
    (if (entries == 1) cancelled(0) else cancelled(head))
  val headEntry = if (entries == 1) storage(0) else storage(head)
  val incomingKilled =
    (io.killActive && io.enq.bits.asid === io.killAsid) ||
      (io.kill.valid && io.enq.bits.asid === io.kill.bits.asid)

  val cancelArb = Module(new Arbiter(new DmaCompletion, 2))
  cancelArb.io.in(0).valid := headCancelled
  cancelArb.io.in(0).bits.wid := headEntry.wid
  cancelArb.io.in(0).bits.group := headEntry.group
  cancelArb.io.in(0).bits.is_s2g :=
    headEntry.copyDirection === TmaV2Spec.DirectionS2G.U
  cancelArb.io.in(0).bits.barrierValid := false.B
  cancelArb.io.in(0).bits.barrierId := 0.U
  cancelArb.io.in(0).bits.barrierGeneration := 0.U
  cancelArb.io.in(0).bits.transactionBytes := 0.U

  cancelArb.io.in(1).valid := io.enq.valid && incomingKilled
  cancelArb.io.in(1).bits.wid := io.enq.bits.wid
  cancelArb.io.in(1).bits.group := io.enq.bits.group
  cancelArb.io.in(1).bits.is_s2g :=
    io.enq.bits.copyDirection === TmaV2Spec.DirectionS2G.U
  cancelArb.io.in(1).bits.barrierValid := false.B
  cancelArb.io.in(1).bits.barrierId := 0.U
  cancelArb.io.in(1).bits.barrierGeneration := 0.U
  cancelArb.io.in(1).bits.transactionBytes := 0.U
  // An unlocked Arbiter may change its selected input while its output is
  // stalled if a higher-priority resident cancellation appears.  Capture the
  // selected completion in a flow-through one-entry slot so the externally
  // visible Decoupled payload remains stable without adding latency on the
  // ready path.
  val cancelSkid = Module(new Queue(
    new DmaCompletion, 1, pipe = true, flow = true))
  cancelSkid.io.enq <> cancelArb.io.out
  io.cancelled <> cancelSkid.io.deq

  // A kill mark is registered below.  Suppress a matching head
  // combinationally as well, otherwise a ready Prepare slot can dequeue it
  // on the same edge on which the cancellation mark is written.
  val headKilledNow = io.kill.valid && headValid &&
    headEntry.asid === io.kill.bits.asid
  io.deq.valid := headValid && !headCancelled && !headKilledNow
  io.deq.bits := headEntry
  val normalPop = io.deq.fire
  val cancelledPop = cancelArb.io.in(0).fire
  val pop = normalPop || cancelledPop

  val full = count === entries.U
  val normalEnqReady = !full || pop
  io.enq.ready := Mux(incomingKilled,
    cancelArb.io.in(1).ready, normalEnqReady)
  val store = io.enq.fire && !incomingKilled

  when(io.kill.valid) {
    for (entry <- 0 until entries) {
      when(valid(entry) && storage(entry).asid === io.kill.bits.asid) {
        cancelled(entry) := true.B
      }
    }
  }

  when(pop) {
    if (entries == 1) {
      valid(0) := false.B
      cancelled(0) := false.B
    } else {
      valid(head) := false.B
      cancelled(head) := false.B
    }
    head := advance(head)
  }
  when(store) {
    if (entries == 1) {
      storage(0) := io.enq.bits
      valid(0) := true.B
      cancelled(0) := false.B
    } else {
      storage(tail) := io.enq.bits
      valid(tail) := true.B
      cancelled(tail) := false.B
    }
    tail := advance(tail)
  }
  when(store =/= pop) {
    count := count + store.asUInt - pop.asUInt
  }

  io.matchingPending := VecInit((0 until entries).map { entry =>
    valid(entry) && storage(entry).asid === io.killAsid
  }).asUInt.orR
  io.occupancy := count
  io.full := full

  when(!reset.asBool) {
    assert(count === PopCount(valid),
      "TMA command FIFO valid bits must match occupancy")
    assert(!headCancelled || !io.deq.valid,
      "cancelled TMA commands must not reach Prepare")
    assert(!headKilledNow || !io.deq.valid,
      "a same-cycle matching kill must suppress TMA dequeue")
  }
}

/**
  * Single-active-command execute stage. It owns both iterator variants and
  * converts one atomic PreparedCommand into the Engine command/window stream.
  */
class TmaV2ExecuteStage extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new TmaV2PreparedCommand))
    val command = Decoupled(new TmaV2EngineCommand)
    val window = Decoupled(new TmaV2Window)
    val seal = Decoupled(Bool())
    val engineCompletion = Flipped(Valid(new TmaV2EngineCompletion))
    val kill = Flipped(Valid(new TmaV2KillRequest))
    val killActive = Input(Bool())
    val killAsid = Input(UInt(SV32.asidLen.W))
    val killPending = Output(Bool())
    val active = Output(Bool())
    val activeAsid = Output(UInt(SV32.asidLen.W))
    val plannerProduced = Output(Bool())
    val plannerFire = Output(Bool())
    val plannerStalled = Output(Bool())
    val activeKill = Output(Bool())
    val killDrain = Output(Bool())
  })

  val engineActive = RegInit(false.B)
  val activeAsid = Reg(UInt(SV32.asidLen.W))
  val bulkEmitActive = RegInit(false.B)
  val tensorLaunchPending = RegInit(false.B)
  val tensorEmitActive = RegInit(false.B)
  val tensorDim0StrideActive = RegInit(false.B)
  val tensorSetupActive = RegInit(false.B)
  val tensorSetupLowPhase = RegInit(true.B)
  val tensorSetupDimension = RegInit(1.U(3.W))
  val tensorSetupPending = RegInit(0.U((TmaV2Spec.RankMax - 1).W))
  val tensorSetupLow = Reg(UInt(9.W))
  val sealPending = RegInit(false.B)

  val bulkGlobal = Reg(UInt(32.W))
  val bulkShared = Reg(UInt(32.W))
  val bulkWindowsRemaining = Reg(UInt(26.W))
  val bulkFirst = RegInit(false.B)
  val bulkFirstChunk = Reg(UInt(8.W))
  val bulkLastChunk = Reg(UInt(8.W))
  val tensorCommand = Reg(new TmaV2BoundCommand)

  // Element stride one remains a direct launch into the original planner.
  // Only active outer dimensions with a non-zero strideMinus1 bit enter the
  // command-local setup lane.
  val incomingOuterStride = VecInit(
    (1 until TmaV2Spec.RankMax).map { dimension =>
      dimension.U < io.in.bits.tensor.compiled.rank &&
        !(io.in.bits.tensor.compiled.interleave =/=
            TmaV2Spec.InterleaveNone.U &&
          (dimension + 2).U === io.in.bits.tensor.compiled.rank) &&
        io.in.bits.tensor.compiled.strideMinus1(dimension) =/= 0.U
    }).asUInt

  // Shift/add multiply for the 1..8 traversal factor.  The result is a
  // planner-private stride; the descriptor-store copy remains untouched.
  def multiplyElementStride(value: UInt, stride: UInt): UInt = {
    val x2 = (value << 1)(63, 0)
    val x4 = (value << 2)(63, 0)
    val x8 = (value << 3)(63, 0)
    MuxLookup(stride, value)(Seq(
      2.U -> x2,
      3.U -> (x2 +& value)(63, 0),
      4.U -> x4,
      5.U -> (x4 +& value)(63, 0),
      6.U -> (x4 +& x2)(63, 0),
      7.U -> ((x4 +& x2) +& value)(63, 0),
      8.U -> x8))
  }

  // A 13-step restoring divider is shared by every outer dimension and by
  // the low/high setup micro-ops.  The dividend is first saturated at
  // effectiveBox*stride (at most 2048), so no wide coordinate divider is
  // needed even though global coordinates are 32/33 bits.
  def restoringDivideSmall(numerator: UInt, divisor: UInt): UInt = {
    var remainder = 0.U(5.W)
    var quotient = 0.U(13.W)
    for (bit <- 12 to 0 by -1) {
      val shifted = Cat(remainder(3, 0), numerator(bit))
      val subtract = shifted >= divisor.pad(5)
      remainder = Mux(subtract,
        (shifted - divisor.pad(5))(4, 0), shifted)
      quotient = quotient | (subtract.asUInt << bit)
    }
    quotient
  }

  def clippedCeilDivide(
      distance: UInt, count: UInt, stride: UInt): UInt = {
    val countWide = count.pad(13)
    val strideWide = stride.pad(13)
    val x2 = (countWide << 1)(12, 0)
    val x4 = (countWide << 2)(12, 0)
    val x8 = (countWide << 3)(12, 0)
    val limit = MuxLookup(stride, countWide)(Seq(
      2.U -> x2,
      3.U -> (x2 +& countWide)(12, 0),
      4.U -> x4,
      5.U -> (x4 +& countWide)(12, 0),
      6.U -> (x4 +& x2)(12, 0),
      7.U -> ((x4 +& x2) +& countWide)(12, 0),
      8.U -> x8))
    val bounded = Mux(distance > limit, limit, distance)
    val rounded = bounded +& strideWide - 1.U
    val quotient = restoringDivideSmall(rounded(12, 0), stride)
    Mux(quotient > count, count, quotient(8, 0))
  }

  val activeKillNow = io.kill.valid && engineActive &&
    activeAsid === io.kill.bits.asid
  val incomingKillNow = io.kill.valid && io.in.valid &&
    io.in.bits.engine.asid === io.kill.bits.asid
  io.command.valid := io.in.valid && !engineActive && !incomingKillNow
  io.command.bits := io.in.bits.engine
  io.in.ready := io.command.ready && !engineActive && !incomingKillNow

  when(io.command.fire) {
    engineActive := true.B
    activeAsid := io.in.bits.engine.asid
    when(io.in.bits.kind === TmaV2PreparedKind.Bulk) {
      bulkGlobal := io.in.bits.bulkGlobal
      bulkShared := io.in.bits.bulkShared
      bulkWindowsRemaining := io.in.bits.bulkWindows
      bulkFirst := true.B
      bulkFirstChunk := io.in.bits.bulkFirstChunk
      bulkLastChunk := io.in.bits.bulkLastChunk
      bulkEmitActive := true.B
    }.elsewhen(io.in.bits.kind === TmaV2PreparedKind.Tensor) {
      tensorCommand := io.in.bits.tensor
      when(incomingOuterStride.orR) {
        tensorSetupActive := true.B
        tensorSetupLowPhase := true.B
        tensorSetupPending := incomingOuterStride
        tensorSetupDimension := PriorityEncoder(incomingOuterStride) + 1.U
      }.otherwise {
        tensorLaunchPending := true.B
      }
    }.otherwise {
      sealPending := true.B
    }
  }
  when(io.engineCompletion.valid) {
    assert(engineActive)
    engineActive := false.B
  }

  when(activeKillNow) {
    bulkEmitActive := false.B
    tensorLaunchPending := false.B
    tensorEmitActive := false.B
    tensorDim0StrideActive := false.B
    tensorSetupActive := false.B
    tensorSetupPending := 0.U
    sealPending := false.B
  }

  val setupStride =
    tensorCommand.compiled.strideMinus1(tensorSetupDimension).pad(4) +
      1.U(4.W)
  val setupCount = tensorCommand.compiled.boxDims(tensorSetupDimension)
  val setupCoordinate = tensorCommand.coordinates(tensorSetupDimension)
  val setupCoordinateWide = setupCoordinate.pad(34)
  val setupGlobalWide =
    tensorCommand.compiled.globalDims(tensorSetupDimension).zext.pad(34)
  val setupNegativeDistance =
    (0.S(34.W) - setupCoordinateWide).asUInt
  val setupHighDistance =
    (setupGlobalWide - setupCoordinateWide).asUInt
  // Low and high clipping are separate micro-ops.  Select their dividend
  // before the restoring divider so hardware contains one shared small
  // divider rather than relying on synthesis to merge two expressions.
  val setupDistance = Mux(
    tensorSetupLowPhase, setupNegativeDistance, setupHighDistance)
  val setupDivideValue = clippedCeilDivide(
    setupDistance, setupCount, setupStride)
  val setupClippedValue = Mux(tensorSetupLowPhase,
    Mux(setupCoordinate < 0.S, setupDivideValue, 0.U),
    Mux(setupCoordinateWide >= setupGlobalWide, 0.U, setupDivideValue))

  when(tensorSetupActive && !activeKillNow) {
    when(tensorSetupLowPhase) {
      tensorSetupLow := setupClippedValue
      val rawStride = tensorCommand.compiled.globalStrides(
        (tensorSetupDimension - 1.U)(1, 0))
      tensorCommand.compiled.globalStrides(
        (tensorSetupDimension - 1.U)(1, 0)) :=
        multiplyElementStride(rawStride, setupStride)
      tensorSetupLowPhase := false.B
    }.otherwise {
      val clippedExtent = Mux(
        setupClippedValue > tensorSetupLow,
        setupClippedValue - tensorSetupLow, 0.U)
      tensorCommand.coordinates(tensorSetupDimension) :=
        -tensorSetupLow.zext
      tensorCommand.compiled.globalDims(tensorSetupDimension) :=
        clippedExtent.pad(33)

      val selectedBit =
        (1.U((TmaV2Spec.RankMax - 1).W) <<
          (tensorSetupDimension - 1.U))
      val remaining = tensorSetupPending & ~selectedBit
      tensorSetupPending := remaining
      when(remaining.orR) {
        tensorSetupDimension := PriorityEncoder(remaining) + 1.U
        tensorSetupLowPhase := true.B
      }.otherwise {
        tensorSetupActive := false.B
        tensorLaunchPending := true.B
      }
    }
  }

  val tensorDim0Stride =
    tensorCommand.compiled.strideMinus1(0) =/= 0.U
  val plannerCommand = WireDefault(tensorCommand)
  // The compiler stores the selected dim0 count for accounting.  The slow
  // path must instead let the planner enumerate every raw atom position so
  // the following compactor can select 0, stride, 2*stride, ... and densely
  // renumber the surviving atoms. Other dimensions already use their
  // effective traversal counts; the rank-2 channel slice is canonicalized
  // to one by the compiler because CUDA treats that box field as metadata.
  when(tensorDim0Stride) {
    plannerCommand.compiled.boxDims(0) :=
      tensorCommand.compiled.rawDim0Box
  }

  val planner = Module(new TmaV2WindowPlanner)
  val dim0Compactor = Module(new TmaV2Dim0StrideCompactor)
  planner.io.flush := activeKillNow
  dim0Compactor.io.flush := activeKillNow
  planner.io.in.valid := tensorLaunchPending && !activeKillNow
  planner.io.in.bits := plannerCommand
  dim0Compactor.io.start.valid :=
    planner.io.in.fire && tensorDim0Stride && !activeKillNow
  dim0Compactor.io.start.bits.sharedBase := tensorCommand.sharedBase
  dim0Compactor.io.start.bits.strideMinus1 :=
    tensorCommand.compiled.strideMinus1(0)
  dim0Compactor.io.start.bits.rawPositions :=
    tensorCommand.compiled.rawDim0Box
  dim0Compactor.io.start.bits.interleave :=
    tensorCommand.compiled.interleave
  dim0Compactor.io.start.bits.swizzle :=
    tensorCommand.compiled.swizzle
  when(planner.io.in.fire) {
    tensorLaunchPending := false.B
    tensorEmitActive := true.B
    tensorDim0StrideActive := tensorDim0Stride
  }

  dim0Compactor.io.in.valid := planner.io.out.valid &&
    tensorEmitActive && tensorDim0StrideActive && !activeKillNow
  dim0Compactor.io.in.bits := planner.io.out.bits

  val bulkChunk = Mux(
    bulkFirst, bulkFirstChunk,
    Mux(bulkWindowsRemaining === 1.U, bulkLastChunk, 128.U(8.W)))
  val bulkChunkWide = bulkChunk.pad(32)
  val bulkWindow = WireDefault(0.U.asTypeOf(new TmaV2Window))
  bulkWindow.sharedBase := bulkShared
  bulkWindow.last := bulkWindowsRemaining === 1.U
  for (lane <- 0 until 8) {
    val laneOffset = (lane * 16).U(32.W)
    val laneActive = laneOffset < bulkChunkWide
    bulkWindow.lanes(lane).valid := laneActive
    bulkWindow.lanes(lane).globalAddress := bulkGlobal + laneOffset
    bulkWindow.lanes(lane).globalBytes := Mux(laneActive, 16.U, 0.U)
    bulkWindow.lanes(lane).sharedAtomDelta := lane.U
    bulkWindow.lanes(lane).sharedBytes := Mux(laneActive, 16.U, 0.U)
  }

  val tensorWindowValid = Mux(
    tensorDim0StrideActive,
    dim0Compactor.io.out.valid, planner.io.out.valid)
  val tensorWindow = Mux(
    tensorDim0StrideActive,
    dim0Compactor.io.out.bits, planner.io.out.bits)
  io.window.valid := !activeKillNow && Mux(bulkEmitActive, true.B,
    tensorEmitActive && tensorWindowValid)
  io.window.bits := Mux(bulkEmitActive, bulkWindow, tensorWindow)
  planner.io.out.ready := !activeKillNow && tensorEmitActive &&
    !bulkEmitActive && Mux(
      tensorDim0StrideActive,
      dim0Compactor.io.in.ready, io.window.ready)
  dim0Compactor.io.out.ready := !activeKillNow && tensorEmitActive &&
    tensorDim0StrideActive && !bulkEmitActive && io.window.ready

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
    tensorDim0StrideActive := false.B
  }

  io.seal.valid := sealPending && !activeKillNow
  io.seal.bits := true.B
  when(io.seal.fire) { sealPending := false.B }

  io.active := engineActive
  io.activeAsid := activeAsid
  io.plannerProduced := planner.io.produced
  io.plannerFire := planner.io.out.fire
  io.plannerStalled := planner.io.stalled
  io.killPending := engineActive && activeAsid === io.killAsid
  io.activeKill := activeKillNow
  io.killDrain := io.killActive && io.killPending

  when(!reset.asBool) {
    assert(!(bulkEmitActive && tensorEmitActive))
    assert(!io.window.fire || PopCount(io.window.bits.lanes.map(_.valid)) > 0.U)
    assert(!(planner.io.in.fire && tensorEmitActive))
    assert(!(tensorDim0StrideActive && !dim0Compactor.io.busy &&
      tensorEmitActive && !activeKillNow),
      "active dim0 slow path must retain compactor ownership")
    assert(!(tensorSetupActive && tensorLaunchPending),
      "outer-stride setup and planner launch must be disjoint")
    when(activeKillNow) {
      assert(!io.window.valid,
        "an active kill must suppress Execute window issue immediately")
      assert(!io.seal.valid,
        "an active kill must suppress Execute seal issue immediately")
      assert(!planner.io.in.valid,
        "an active kill must suppress a pending planner launch")
    }
  }
}
