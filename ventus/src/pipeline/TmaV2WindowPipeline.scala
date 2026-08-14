package pipeline

import chisel3._
import chisel3.util._
import mmu.SV32
import top.parameters.MMU_ENABLED

/** Stage-1 token: cursor snapshots only, before any address mapping. */
class TmaV2CursorExpandToken extends Bundle {
  // Coordinates are origin + this compact box index. Carrying five full
  // 32-bit coordinates per lane duplicated command-stable origins and made
  // the CursorExpand queue the largest Planner register bank.
  val laneIndex =
    Vec(8, Vec(TmaV2Spec.RankMax, UInt(9.W)))
  // GlobalMap consumes only the sum, never the five independent dimension
  // components. Cursor state keeps the vector needed for the next window.
  val laneComponentSum = Vec(8, SInt(64.W))
  val laneRow = Vec(8, UInt(3.W))
  val laneValid = Vec(8, Bool())
  // Low seven bits of a logical window base are always zero.
  val logicalWindowIndex = UInt(25.W)
  val last = Bool()
}

/**
  * Elastic cut between mixed-radix cursor carry and component formation.
  *
  * Cursor indices and address components are independent recurrences.  The
  * index recurrence advances when this token is enqueued; the component
  * recurrence advances when it is dequeued into CursorExpand.  A depth-one
  * pipe therefore keeps both recurrences at II=1 without placing the rank
  * carry chain and the 64-bit component network in the same cycle.
  */
class TmaV2CursorCarryToken extends Bundle {
  // Lane eight is the next-window cursor and is not emitted downstream.
  val laneIndex =
    Vec(9, Vec(TmaV2Spec.RankMax, UInt(9.W)))
  // Entry d stores carry into dimension d+1.  This is enough to reconstruct
  // both the incoming carry and quotient for every nonzero dimension.
  val laneCarry =
    Vec(9, Vec(TmaV2Spec.RankMax, UInt(4.W)))
  val cursorRow = UInt(3.W)
  val logicalWindowIndex = UInt(25.W)
}

/**
  * Elastic cut inside the five-rank mixed-radix recurrence.
  *
  * The low-rank producer owns dimensions zero and one.  The following
  * MidCarry stage owns dimensions two and three. Carry values for both
  * completed low dimensions are retained because ComponentMap needs the
  * dimension-zero wrap as well as the carry entering dimension two.
  */
class TmaV2CursorLowCarryToken extends Bundle {
  val laneIndex = Vec(9, Vec(2, UInt(9.W)))
  val laneCarry = Vec(9, Vec(2, UInt(4.W)))
  val cursorRow = UInt(3.W)
  val logicalWindowIndex = UInt(25.W)
}

/**
  * Second elastic cut inside the five-rank mixed-radix recurrence.
  *
  * MidCarry owns dimensions two and three.  It retains the four completed
  * indices/carries required by ComponentMap while letting LowCarry start the
  * following window and HighCarry finish dimension four concurrently.
  */
class TmaV2CursorMidCarryToken extends Bundle {
  val laneIndex = Vec(9, Vec(4, UInt(9.W)))
  val laneCarry = Vec(9, Vec(4, UInt(4.W)))
  val cursorRow = UInt(3.W)
  val logicalWindowIndex = UInt(25.W)
}

class TmaV2GlobalMapLane extends Bundle {
  val valid = Bool()
  val globalAddress = UInt(32.W)
  val globalBytes = UInt(5.W)
}

/** Address/bounds cut between cursor expansion and byte-interval mapping. */
class TmaV2CoordinateMapLane extends Bundle {
  val valid = Bool()
  val coordinate0 = SInt(32.W)
  val outerInBounds = Bool()
  val globalAddress = UInt(32.W)
}

class TmaV2CoordinateMapToken extends Bundle {
  val lanes = Vec(8, new TmaV2CoordinateMapLane)
  val laneRow = Vec(8, UInt(3.W))
  val logicalWindowIndex = UInt(25.W)
  val last = Bool()
}

class TmaV2GlobalMapToken extends Bundle {
  val lanes = Vec(8, new TmaV2GlobalMapLane)
  val laneRow = Vec(8, UInt(3.W))
  val logicalWindowIndex = UInt(25.W)
  val last = Bool()
}

/** Registered planner output; command sideband remains in ActiveCommand. */
class TmaV2WindowLane extends Bundle {
  val valid = Bool()
  val globalAddress = UInt(32.W)
  val globalBytes = UInt(5.W)
  val sharedAtomDelta = UInt(3.W)
  val sharedBytes = UInt(5.W)
}

class TmaV2Window extends Bundle {
  val sharedBase = UInt(32.W)
  val lanes = Vec(8, new TmaV2WindowLane)
  val last = Bool()
}

/**
  * Eight-stage elastic eight-lane 5D planner.
  *
  * LowCarry (rank 0..1), MidCarry (rank 2..3), HighCarry (rank 4),
  * ComponentMap, CoordinateMap, GlobalMap, SharedMap and SlotEncode are
  * separated by one-entry pipe queues. Every queue holds its token under
  * backpressure.  The three rank slices advance their owned index cursors
  * independently when their corresponding stage fires, while the component
  * cursor advances only when the matching token enters ComponentMap.  Once
  * filled, every stage accepts one token per cycle.
  */
class TmaV2WindowPlanner extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new TmaV2BoundCommand))
    val out = Decoupled(new TmaV2Window)
    val produced = Output(Bool())
    val stalled = Output(Bool())
  })

  val commandBusy = RegInit(false.B)
  val producerDone = RegInit(false.B)
  val command = Reg(new TmaV2BoundCommand)
  val cursorIndex = Reg(Vec(TmaV2Spec.RankMax, UInt(9.W)))
  val componentCursor = Reg(Vec(TmaV2Spec.RankMax, SInt(64.W)))
  val cursorRow = RegInit(0.U(3.W))
  val logicalWindowIndex = RegInit(0.U(25.W))

  // Decode command-stable format information once so dtype decoding is not
  // recreated in every lane and planner stage.
  val commandDtypeBits = Reg(UInt(7.W))
  val commandByteShift = Reg(UInt(3.W))
  val commandChannelsLog2 = Reg(UInt(4.W))
  val paddedFp4 = Reg(Bool())
  val paddedFp6 = Reg(Bool())
  val packedFp4 = Reg(Bool())
  val laneElementShift = Reg(UInt(3.W))
  val payloadBytes = Reg(UInt(5.W))
  val atomsPerSwizzleRow = Reg(UInt(4.W))
  val plannedLaneCount = Reg(UInt(4.W))
  val paddedSubByte = paddedFp4 || paddedFp6

  val incomingDtypeBits =
    TmaV2DescriptorDerived.dtypeBits(io.in.bits.compiled.dtype)
  val incomingByteShift =
    TmaV2DescriptorDerived.elementByteShift(incomingDtypeBits)
  val incomingChannelsLog2 =
    TmaV2DescriptorDerived.channelsLog2(
      io.in.bits.compiled.interleave, incomingByteShift)
  val incomingPaddedFp4 =
    io.in.bits.compiled.dtype === TmaV2Spec.DTypeB4x16P64.U
  val incomingPaddedFp6 =
    io.in.bits.compiled.dtype === TmaV2Spec.DTypeB6.U
  val incomingPackedFp4 =
    io.in.bits.compiled.dtype === TmaV2Spec.DTypeB4x16.U
  val incomingPaddedSubByte = incomingPaddedFp4 || incomingPaddedFp6
  val incomingLaneElementShift = MuxLookup(
    incomingDtypeBits, 0.U(3.W))(Seq(
      4.U -> Mux(incomingPaddedFp4, 4.U, 5.U),
      6.U -> 4.U,
      8.U -> 4.U,
      16.U -> 3.U,
      32.U -> 2.U,
      64.U -> 1.U))
  val incomingPayloadBytes = Mux(incomingPaddedFp4, 8.U(5.W),
    Mux(incomingPaddedFp6, 12.U(5.W), 16.U(5.W)))
  // CUDA gives every swizzled logical row a physical pitch equal to the
  // selected span.  When the logical row is shorter than that span, one
  // 128B shared window therefore contains fewer than eight useful atoms.
  // Keep the fixed eight-lane combinational planner and mask the unused
  // lanes instead of enumerating atoms or emitting multiple shared accesses.
  val incomingOrdinarySharedRowBytes =
    MuxLookup(incomingDtypeBits, 0.U(10.W))(Seq(
      4.U -> ((io.in.bits.compiled.boxDims(0) + 1.U) >> 1),
      6.U -> (((io.in.bits.compiled.boxDims(0) << 2) +
        (io.in.bits.compiled.boxDims(0) << 1) + 7.U) >> 3),
      8.U -> io.in.bits.compiled.boxDims(0),
      16.U -> (io.in.bits.compiled.boxDims(0) << 1),
      32.U -> (io.in.bits.compiled.boxDims(0) << 2),
      64.U -> (io.in.bits.compiled.boxDims(0) << 3)))
  val incomingSharedRowBytes = Mux(
    incomingPaddedSubByte, 128.U, incomingOrdinarySharedRowBytes)
  val incomingAtomsPerSwizzleRow = incomingSharedRowBytes >> 4
  val incomingRowsPerSwizzleShift = 3.U - io.in.bits.compiled.swizzle
  val incomingSwizzleLaneCount =
    incomingAtomsPerSwizzleRow << incomingRowsPerSwizzleShift
  val incomingPlannedLaneCount = Mux(
    io.in.bits.compiled.swizzle === TmaV2Spec.SwizzleNone.U,
    8.U(4.W), incomingSwizzleLaneCount(3, 0))

  // Multiply an at-most 64-bit value by a factor in [0, 8] using shifts and
  // a balanced add tree.  This is used only for small mixed-radix carries;
  // it must never infer a general multiplier or divider.
  def multiplySmall(value: UInt, factor: UInt, width: Int): UInt = {
    val wide = value.pad(width)
    // Scala literals use their minimum width (0.U/1.U are one bit).  Widen
    // the selector before MuxLookup so keys 2..8 cannot alias after FIRRTL
    // width inference.
    val normalizedFactor = factor.pad(4)
    val x2 = (wide << 1)(width - 1, 0)
    val x4 = (wide << 2)(width - 1, 0)
    val x8 = (wide << 3)(width - 1, 0)
    MuxLookup(normalizedFactor, 0.U(width.W))(Seq(
      1.U -> wide,
      2.U -> x2,
      3.U -> (x2 +& wide)(width - 1, 0),
      4.U -> x4,
      5.U -> (x4 +& wide)(width - 1, 0),
      6.U -> (x4 +& x2)(width - 1, 0),
      7.U -> ((x4 +& x2) +& wide)(width - 1, 0),
      8.U -> x8))
  }

  // Divide a value by a 1..256 box dimension when the quotient is known to
  // be at most eight.  The comparisons form a thermometer vector, so a
  // balanced popcount is the quotient. This avoids both / or % hardware and
  // a descending priority mux inside every rank step.
  def smallDivRem(total: UInt, divisor: UInt): (UInt, UInt) = {
    // Keep these thresholds explicit so synthesis can balance them and the
    // 256-element divisor remains represented at the required width.
    val d1 = divisor.pad(13)
    val d2 = (d1 << 1)(12, 0)
    val d3 = (d2 +& d1)(12, 0)
    val d4 = (d1 << 2)(12, 0)
    val d5 = (d4 +& d1)(12, 0)
    val d6 = (d4 +& d2)(12, 0)
    val d7 = ((d4 +& d2) +& d1)(12, 0)
    val d8 = (d1 << 3)(12, 0)
    val wideTotal = total.pad(13)
    val ge1 = (wideTotal >= d1).asUInt
    val ge2 = (wideTotal >= d2).asUInt
    val ge3 = (wideTotal >= d3).asUInt
    val ge4 = (wideTotal >= d4).asUInt
    val ge5 = (wideTotal >= d5).asUInt
    val ge6 = (wideTotal >= d6).asUInt
    val ge7 = (wideTotal >= d7).asUInt
    val ge8 = (wideTotal >= d8).asUInt
    val count01 = ge1 +& ge2
    val count23 = ge3 +& ge4
    val count45 = ge5 +& ge6
    val count67 = ge7 +& ge8
    val count03 = count01 +& count23
    val count47 = count45 +& count67
    val quotient = count03 +& count47

    // Reconstruct divisor*quotient from quotient bits using two balanced
    // add levels instead of selecting among eight precomputed products.
    val product1 = Mux(quotient(0), d1, 0.U(13.W))
    val product2 = Mux(quotient(1), d2, 0.U(13.W))
    val product4 = Mux(quotient(2), d4, 0.U(13.W))
    val product8 = Mux(quotient(3), d8, 0.U(13.W))
    val product12 = product1 +& product2
    val product48 = product4 +& product8
    val product = product12 +& product48
    val remainder = wideTotal - product(12, 0)
    (quotient, remainder)
  }

  // An inclusive parallel prefix sum with log2(8) adder depth.  Cursor
  // component deltas used to form an eight-element serial 64-bit chain.
  def prefixSums(values: Seq[UInt]): Seq[UInt] = {
    var stage = values.map(_.pad(67))
    for (distance <- Seq(1, 2, 4)) {
      val previous = stage
      stage = previous.indices.map { index =>
        if (index >= distance)
          (previous(index) +& previous(index - distance))(66, 0)
        else previous(index)
      }
    }
    0.U(67.W) +: stage
  }

  def dim0DeltaAt(index: UInt): UInt = {
    val coordinate = command.coordinates(0) + index.zext
    val inSlice = coordinate.asUInt &
      ((1.U(64.W) << commandChannelsLog2) - 1.U)
    Mux(paddedSubByte, payloadBytes,
      MuxLookup(command.compiled.interleave, 16.U(64.W))(Seq(
        TmaV2Spec.Interleave16.U ->
          command.compiled.interleaveSliceStride,
        TmaV2Spec.Interleave32.U -> Mux(inSlice === 0.U, 16.U,
          command.compiled.interleaveSliceStride - 16.U))))
  }

  // Split the five-rank mixed-radix recurrence 2+2+1.  Each stage owns its
  // slice of cursorIndex and therefore forms a systolic recurrence: LowCarry
  // can start window N+2 while MidCarry works on N+1 and HighCarry finishes N.
  // Merely registering a partial combinational result without splitting the
  // feedback ownership would lengthen the recurrence and reduce throughput.
  val lowLaneIndex = Wire(Vec(9, Vec(2, UInt(9.W))))
  val lowLaneCarry = Wire(Vec(9, Vec(3, UInt(4.W))))

  for (lane <- 0 to 8) {
    lowLaneCarry(lane)(0) := lane.U
    for (dimension <- 0 until 2) {
      val used = dimension.U < command.compiled.rank
      val increment = if (dimension == 0) {
        (lane.U(12.W) << laneElementShift)(11, 0)
      } else {
        lowLaneCarry(lane)(dimension).pad(12)
      }
      val total = cursorIndex(dimension).pad(12) +& increment
      val (quotient, remainder) = smallDivRem(
        total(11, 0), command.compiled.boxDims(dimension).pad(12))
      lowLaneIndex(lane)(dimension) := Mux(
        used, remainder(8, 0), cursorIndex(dimension))
      lowLaneCarry(lane)(dimension + 1) := Mux(
        used, quotient, increment(3, 0))
    }
  }

  val lowCarryStage = Module(new Queue(
    new TmaV2CursorLowCarryToken, 1, pipe = true, flow = false))
  lowCarryStage.io.enq.valid := commandBusy && !producerDone
  lowCarryStage.io.enq.bits :=
    0.U.asTypeOf(new TmaV2CursorLowCarryToken)
  lowCarryStage.io.enq.bits.cursorRow := cursorRow
  lowCarryStage.io.enq.bits.logicalWindowIndex := logicalWindowIndex
  for (lane <- 0 to 8) {
    lowCarryStage.io.enq.bits.laneIndex(lane) := lowLaneIndex(lane)
    for (dimension <- 0 until 2) {
      lowCarryStage.io.enq.bits.laneCarry(lane)(dimension) :=
        lowLaneCarry(lane)(dimension + 1)
    }
  }

  when(lowCarryStage.io.enq.fire) {
    cursorIndex(0) := lowLaneIndex(plannedLaneCount)(0)
    cursorIndex(1) := lowLaneIndex(plannedLaneCount)(1)
    cursorRow :=
      (cursorRow + lowLaneCarry(plannedLaneCount)(1))(2, 0)
    logicalWindowIndex := logicalWindowIndex + 1.U
  }

  val lowCarryToken = lowCarryStage.io.deq.bits
  val midLaneIndex = Wire(Vec(9, Vec(4, UInt(9.W))))
  val midLaneCarry = Wire(Vec(9, Vec(4, UInt(4.W))))
  val midCarryChain = Wire(Vec(9, Vec(3, UInt(4.W))))

  for (lane <- 0 to 8) {
    for (dimension <- 0 until 2) {
      midLaneIndex(lane)(dimension) :=
        lowCarryToken.laneIndex(lane)(dimension)
      midLaneCarry(lane)(dimension) :=
        lowCarryToken.laneCarry(lane)(dimension)
    }
    midCarryChain(lane)(0) := lowCarryToken.laneCarry(lane)(1)
    for (dimension <- 2 until 4) {
      val used = dimension.U < command.compiled.rank
      val midDimension = dimension - 2
      val increment = midCarryChain(lane)(midDimension).pad(12)
      val total = cursorIndex(dimension).pad(12) +& increment
      val (quotient, remainder) = smallDivRem(
        total(11, 0), command.compiled.boxDims(dimension).pad(12))
      midLaneIndex(lane)(dimension) := Mux(
        used, remainder(8, 0), cursorIndex(dimension))
      midCarryChain(lane)(midDimension + 1) := Mux(
        used, quotient, increment(3, 0))
      midLaneCarry(lane)(dimension) :=
        midCarryChain(lane)(midDimension + 1)
    }
  }

  val midCarryStage = Module(new Queue(
    new TmaV2CursorMidCarryToken, 1, pipe = true, flow = false))
  midCarryStage.io.enq.valid := lowCarryStage.io.deq.valid && !producerDone
  lowCarryStage.io.deq.ready := Mux(
    producerDone, true.B, midCarryStage.io.enq.ready)
  midCarryStage.io.enq.bits :=
    0.U.asTypeOf(new TmaV2CursorMidCarryToken)
  midCarryStage.io.enq.bits.cursorRow := lowCarryToken.cursorRow
  midCarryStage.io.enq.bits.logicalWindowIndex :=
    lowCarryToken.logicalWindowIndex
  for (lane <- 0 to 8) {
    midCarryStage.io.enq.bits.laneIndex(lane) := midLaneIndex(lane)
    midCarryStage.io.enq.bits.laneCarry(lane) := midLaneCarry(lane)
  }

  when(midCarryStage.io.enq.fire) {
    cursorIndex(2) := midLaneIndex(plannedLaneCount)(2)
    cursorIndex(3) := midLaneIndex(plannedLaneCount)(3)
  }

  val midCarryToken = midCarryStage.io.deq.bits
  val laneIndex = Wire(Vec(9, Vec(TmaV2Spec.RankMax, UInt(9.W))))
  val laneCarry = Wire(Vec(9, Vec(TmaV2Spec.RankMax, UInt(4.W))))

  for (lane <- 0 to 8) {
    for (dimension <- 0 until 4) {
      laneIndex(lane)(dimension) :=
        midCarryToken.laneIndex(lane)(dimension)
      laneCarry(lane)(dimension) :=
        midCarryToken.laneCarry(lane)(dimension)
    }
    val used = 4.U < command.compiled.rank
    val increment = midCarryToken.laneCarry(lane)(3).pad(12)
    val total = cursorIndex(4).pad(12) +& increment
    val (quotient, remainder) = smallDivRem(
      total(11, 0), command.compiled.boxDims(4).pad(12))
    laneIndex(lane)(4) := Mux(used, remainder(8, 0), cursorIndex(4))
    laneCarry(lane)(4) := Mux(used, quotient, increment(3, 0))
  }

  val carryStage = Module(new Queue(
    new TmaV2CursorCarryToken, 1, pipe = true, flow = false))
  carryStage.io.enq.valid := midCarryStage.io.deq.valid && !producerDone
  // HighCarry discovers the final carry only as it accepts the last token.
  // Pipe queues may simultaneously admit one younger token in each upstream
  // stage; once producerDone is set, drain both speculative tokens without
  // forwarding either into the architectural window stream.
  midCarryStage.io.deq.ready := Mux(
    producerDone, true.B, carryStage.io.enq.ready)
  carryStage.io.enq.bits := 0.U.asTypeOf(new TmaV2CursorCarryToken)
  carryStage.io.enq.bits.cursorRow := midCarryToken.cursorRow
  carryStage.io.enq.bits.logicalWindowIndex :=
    midCarryToken.logicalWindowIndex
  for (lane <- 0 to 8) {
    carryStage.io.enq.bits.laneIndex(lane) := laneIndex(lane)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      carryStage.io.enq.bits.laneCarry(lane)(dimension) :=
        laneCarry(lane)(dimension)
    }
  }

  val nextIndexDone =
    laneCarry(plannedLaneCount)(TmaV2Spec.RankMax - 1) =/= 0.U
  when(carryStage.io.enq.fire) {
    cursorIndex(4) := laneIndex(plannedLaneCount)(4)
    when(nextIndexDone) {
      producerDone := true.B
    }
  }

  val carryToken = carryStage.io.deq.bits
  val baseDeltas = (0 until 8).map { step =>
    val index = carryToken.laneIndex(0)(0) +
      (step.U(12.W) << laneElementShift)(11, 0)
    dim0DeltaAt(index(8, 0))
  }
  val originDeltas = (0 until 8).map { step =>
    val index = (step.U(12.W) << laneElementShift)(11, 0)
    dim0DeltaAt(index(8, 0))
  }
  val baseDeltaPrefix = prefixSums(baseDeltas)
  val originDeltaPrefix = prefixSums(originDeltas)
  val laneComponent =
    Wire(Vec(9, Vec(TmaV2Spec.RankMax, SInt(64.W))))
  val componentLaneRow = Wire(Vec(9, UInt(3.W)))
  val componentLaneDone = Wire(Vec(9, Bool()))

  for (lane <- 0 to 8) {
    for (dimension <- 1 until TmaV2Spec.RankMax) {
      val used = dimension.U < command.compiled.rank
      val incomingCarry =
        carryToken.laneCarry(lane)(dimension - 1)
      val quotient = carryToken.laneCarry(lane)(dimension)
      val remainder = carryToken.laneIndex(lane)(dimension)
      val stride = command.compiled.globalStrides(dimension - 1)
      val noWrapDelta = multiplySmall(stride, incomingCarry, 67)
      val wrappedDelta = multiplySmall(stride, remainder(3, 0), 67)
      val noWrapComponent =
        componentCursor(dimension).asUInt.pad(67) +& noWrapDelta
      val wrappedComponent =
        command.originComponents(dimension).asUInt.pad(67) +&
          wrappedDelta
      laneComponent(lane)(dimension) := Mux(
        !used || incomingCarry === 0.U,
        componentCursor(dimension),
        Mux(quotient === 0.U,
          noWrapComponent(63, 0).asSInt,
          wrappedComponent(63, 0).asSInt))
    }

    val wrappedDim0 = carryToken.laneCarry(lane)(0) =/= 0.U
    val originSteps =
      (carryToken.laneIndex(lane)(0) >> laneElementShift)(3, 0)
    val continued0 = componentCursor(0).asUInt.pad(68) +&
      baseDeltaPrefix(lane).pad(68)
    val wrapped0 = command.originComponents(0).asUInt.pad(68) +&
      VecInit(originDeltaPrefix)(originSteps).pad(68)
    laneComponent(lane)(0) := Mux(
      wrappedDim0, wrapped0(63, 0).asSInt,
      continued0(63, 0).asSInt)
    componentLaneRow(lane) :=
      (carryToken.cursorRow + carryToken.laneCarry(lane)(0))(2, 0)
    componentLaneDone(lane) :=
      carryToken.laneCarry(lane)(TmaV2Spec.RankMax - 1) =/= 0.U
  }

  val cursorStage = Module(new Queue(
    new TmaV2CursorExpandToken, 1, pipe = true, flow = false))
  cursorStage.io.enq.valid := carryStage.io.deq.valid
  carryStage.io.deq.ready := cursorStage.io.enq.ready
  cursorStage.io.enq.bits := 0.U.asTypeOf(new TmaV2CursorExpandToken)
  cursorStage.io.enq.bits.logicalWindowIndex :=
    carryToken.logicalWindowIndex
  cursorStage.io.enq.bits.last :=
    componentLaneDone(plannedLaneCount)
  for (lane <- 0 until 8) {
    cursorStage.io.enq.bits.laneIndex(lane) :=
      carryToken.laneIndex(lane)
    // Keep the five signed components in a balanced adder tree. A Scala
    // `reduce` produced a four-adder serial cone on every lane.
    val component01 =
      laneComponent(lane)(0) +& laneComponent(lane)(1)
    val component23 =
      laneComponent(lane)(2) +& laneComponent(lane)(3)
    val component03 = component01 +& component23
    val componentAll = component03 +& laneComponent(lane)(4)
    cursorStage.io.enq.bits.laneComponentSum(lane) :=
      componentAll.asUInt(63, 0).asSInt
    cursorStage.io.enq.bits.laneRow(lane) := componentLaneRow(lane)
    cursorStage.io.enq.bits.laneValid(lane) :=
      !componentLaneDone(lane) && lane.U < plannedLaneCount
  }

  when(cursorStage.io.enq.fire) {
    componentCursor := laneComponent(plannedLaneCount)
  }

  val cursorToken = cursorStage.io.deq.bits
  // ActiveCommand is immutable until the last output is accepted, so stage
  // tokens carry only cursor data rather than cloning the bound descriptor.
  val mappedCommand = command
  val mappedPaddedFp4 = paddedFp4
  val mappedPaddedFp6 = paddedFp6
  val mappedPaddedSubByte = mappedPaddedFp4 || mappedPaddedFp6
  val mappedDtypeBits = commandDtypeBits
  val mappedPackedFp4 = packedFp4
  val mappedElementByteShift = commandByteShift
  val mappedElementsPerLane =
    (1.U(9.W) << laneElementShift)(8, 0)
  val mappedPayloadBytes = payloadBytes
  val mappedAtomsPerSwizzleRow = atomsPerSwizzleRow

  // Bounds/address and byte-interval derivation were formerly one large
  // eight-lane stage.  Split them at an elastic register so the four outer
  // coordinate comparisons and 64-bit address add never share a cycle with
  // clipping, packed-format rounding and byte-run selection.
  val coordinateStage = Module(new Queue(
    new TmaV2CoordinateMapToken, 1, pipe = true, flow = false))
  coordinateStage.io.enq.valid := cursorStage.io.deq.valid
  cursorStage.io.deq.ready := coordinateStage.io.enq.ready
  coordinateStage.io.enq.bits :=
    0.U.asTypeOf(new TmaV2CoordinateMapToken)
  coordinateStage.io.enq.bits.logicalWindowIndex :=
    cursorToken.logicalWindowIndex
  coordinateStage.io.enq.bits.last := cursorToken.last

  for (lane <- 0 until 8) {
    val laneCoordinate = Wire(Vec(TmaV2Spec.RankMax, SInt(32.W)))
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      val coordinateWide =
        mappedCommand.coordinates(dimension) +
          cursorToken.laneIndex(lane)(dimension).zext
      laneCoordinate(dimension) :=
        coordinateWide.asUInt(31, 0).asSInt
    }
    val laneValid = cursorToken.laneValid(lane)
    val outerInBounds = (1 until TmaV2Spec.RankMax).map { dimension =>
      dimension.U >= mappedCommand.compiled.rank ||
        (laneCoordinate(dimension) >= 0.S &&
          laneCoordinate(dimension).asUInt <
            mappedCommand.compiled.globalDims(dimension))
    }.reduce(_ && _)
    val globalWide = mappedCommand.compiled.globalBase.zext +
      cursorToken.laneComponentSum(lane)
    coordinateStage.io.enq.bits.lanes(lane).valid :=
      cursorToken.laneValid(lane)
    coordinateStage.io.enq.bits.lanes(lane).coordinate0 :=
      laneCoordinate(0)
    coordinateStage.io.enq.bits.lanes(lane).outerInBounds :=
      outerInBounds
    coordinateStage.io.enq.bits.lanes(lane).globalAddress :=
      globalWide.asUInt(31, 0)
    coordinateStage.io.enq.bits.laneRow(lane) :=
      cursorToken.laneRow(lane)
  }

  val coordinateToken = coordinateStage.io.deq.bits
  val globalStage = Module(new Queue(
    new TmaV2GlobalMapToken, 1, pipe = true, flow = false))
  globalStage.io.enq.valid := coordinateStage.io.deq.valid
  coordinateStage.io.deq.ready := globalStage.io.enq.ready
  globalStage.io.enq.bits :=
    0.U.asTypeOf(new TmaV2GlobalMapToken)
  globalStage.io.enq.bits.logicalWindowIndex :=
    coordinateToken.logicalWindowIndex
  globalStage.io.enq.bits.last := coordinateToken.last

  for (lane <- 0 until 8) {
    val laneValid = coordinateToken.lanes(lane).valid
    val outerInBounds =
      coordinateToken.lanes(lane).outerInBounds
    val coordinate0 =
      coordinateToken.lanes(lane).coordinate0
    val completePaddedRow = outerInBounds &&
      coordinate0 >= 0.S &&
      (coordinate0 + 15.S).asUInt <
        mappedCommand.compiled.globalDims(0)

    // A legal command starts on a 16B bounding-box boundary.  Every leading
    // negative-OOB lane is therefore wholly OOB; only the right edge can
    // retain a byte prefix shorter than one lane.
    val dim0 = mappedCommand.compiled.globalDims(0)
    val coordinateNegative = coordinate0.asUInt(31)
    val availableElementsSigned = dim0.zext - coordinate0
    val availableElements = Mux(
      !coordinateNegative && availableElementsSigned > 0.S,
      availableElementsSigned.asUInt, 0.U)
    val validElements = Mux(
      availableElements > mappedElementsPerLane,
      mappedElementsPerLane, availableElements(8, 0))
    val ordinaryBytes =
      (validElements << mappedElementByteShift)(5, 0)

    // Packed FP4 commands start on 32-element boundaries and descriptors
    // require an even dim0, so clipping is byte-granular without a nibble
    // phase or a nonzero lane offset.
    val packedBytes = (validElements + 1.U) >> 1

    val intervalBytes = Mux(
      mappedPaddedSubByte,
      Mux(completePaddedRow, mappedPayloadBytes, 0.U),
      Mux(mappedPackedFp4, packedBytes, ordinaryBytes))
    val validIntervalBytes = Mux(
      laneValid && outerInBounds, intervalBytes, 0.U)

    globalStage.io.enq.bits.lanes(lane).valid := laneValid
    globalStage.io.enq.bits.lanes(lane).globalAddress :=
      coordinateToken.lanes(lane).globalAddress
    globalStage.io.enq.bits.lanes(lane).globalBytes :=
      validIntervalBytes
    globalStage.io.enq.bits.laneRow(lane) :=
      coordinateToken.laneRow(lane)
  }

  val globalToken = globalStage.io.deq.bits
  val laneStage = Module(new Queue(
    new TmaV2Window, 1, pipe = true, flow = false))
  laneStage.io.enq.valid := globalStage.io.deq.valid
  globalStage.io.deq.ready := laneStage.io.enq.ready
  laneStage.io.enq.bits := 0.U.asTypeOf(new TmaV2Window)
  laneStage.io.enq.bits.sharedBase :=
    mappedCommand.sharedBase +
      (globalToken.logicalWindowIndex << 7)
  laneStage.io.enq.bits.last := globalToken.last

  for (lane <- 0 until 8) {
    val chunkMask =
      (1.U(32.W) << mappedCommand.compiled.swizzle) - 1.U
    val phase = (mappedCommand.sharedBase >> 7) & chunkMask
    val rowWithinWindow = MuxLookup(
      mappedAtomsPerSwizzleRow, 0.U(4.W))(
      (1 to 8).map(atoms => atoms.U -> (lane / atoms).U))
    val atomWithinRow = MuxLookup(
      mappedAtomsPerSwizzleRow, 0.U(4.W))(
      (1 to 8).map(atoms => atoms.U -> (lane % atoms).U))
    val rowPhase = (globalToken.laneRow(lane) + phase) & chunkMask
    val swizzled =
      (rowWithinWindow << (4.U + mappedCommand.compiled.swizzle)) +
      ((atomWithinRow ^ rowPhase) << 4)
    val logical = (lane * 16).U
    val physical = Mux(
      mappedCommand.compiled.swizzle === TmaV2Spec.SwizzleNone.U,
      logical, swizzled)
    val laneValid = globalToken.lanes(lane).valid
    laneStage.io.enq.bits.lanes(lane).valid := laneValid
    laneStage.io.enq.bits.lanes(lane).globalAddress :=
      globalToken.lanes(lane).globalAddress
    laneStage.io.enq.bits.lanes(lane).globalBytes :=
      globalToken.lanes(lane).globalBytes
    laneStage.io.enq.bits.lanes(lane).sharedAtomDelta :=
      physical(6, 4)
    // Every supported codec occupies a contiguous prefix of its 16-byte atom,
    // so the lane needs only the prefix length.
    laneStage.io.enq.bits.lanes(lane).sharedBytes := Mux(laneValid,
      Mux(mappedCommand.copyDirection === TmaV2Spec.DirectionG2S.U,
        mappedPayloadBytes,
        Mux(mappedPaddedSubByte, 16.U, mappedPayloadBytes)), 0.U)
  }

  io.out.valid := laneStage.io.deq.valid
  io.out.bits := laneStage.io.deq.bits
  laneStage.io.deq.ready := io.out.ready

  io.in.ready := !commandBusy
  io.produced := cursorStage.io.enq.fire
  io.stalled := laneStage.io.deq.valid && !laneStage.io.deq.ready

  when(io.in.fire) {
    command := io.in.bits
    commandDtypeBits := incomingDtypeBits
    commandByteShift := incomingByteShift
    commandChannelsLog2 := incomingChannelsLog2
    paddedFp4 := incomingPaddedFp4
    paddedFp6 := incomingPaddedFp6
    packedFp4 := incomingPackedFp4
    laneElementShift := incomingLaneElementShift
    payloadBytes := incomingPayloadBytes
    atomsPerSwizzleRow := incomingAtomsPerSwizzleRow(3, 0)
    plannedLaneCount := incomingPlannedLaneCount
    cursorIndex.foreach(_ := 0.U)
    componentCursor := io.in.bits.originComponents
    cursorRow := 0.U
    logicalWindowIndex := 0.U
    commandBusy := true.B
    producerDone := false.B
  }
  when(io.out.fire && io.out.bits.last) {
    commandBusy := false.B
    producerDone := false.B
  }

  when(!reset.asBool && io.out.fire) {
    assert(PopCount(io.out.bits.lanes.map(_.valid)) > 0.U)
    for (lane <- 0 until 8) {
      when(io.out.bits.lanes(lane).valid) {
        assert(io.out.bits.lanes(lane).sharedAtomDelta < 8.U)
      }
    }
    assert(!(packedFp4 && command.compiled.interleave =/=
      TmaV2Spec.InterleaveNone.U),
      "packed FP4 interleave requires a future byte-within-lane codec")
  }
  when(!reset.asBool) {
    when(cursorStage.io.enq.fire) {
      assert(plannedLaneCount > 0.U && plannedLaneCount <= 8.U)
    }
    assert(!(producerDone && carryStage.io.enq.valid),
      "last carry token must stop the cursor-index producer")
    assert(!(producerDone && midCarryStage.io.enq.valid),
      "last high-rank carry must stop the mid-rank producer")
    assert(!(producerDone && lowCarryStage.io.enq.valid),
      "last high-rank carry must stop the low-rank producer")
    assert(!io.in.fire || !commandBusy,
      "a second planner command cannot overlap the active command")
  }
}

/** Descriptor cache plus the full-window copy engine. */
class TmaV2WindowSubsystem(
    windowEntries: Int = TmaV2Spec.DefaultWindowEntries,
    requestEntries: Int = TmaV2Spec.DefaultGlobalRequestEntries,
    sharedEntries: Int = TmaV2Spec.DefaultSharedReadyEntries,
    writeAckEntries: Int = TmaV2Spec.DefaultWriteAckEntries,
    descriptorEntries: Int = TmaV2Spec.DefaultDescriptorEntries,
    mmuEnabled: Boolean = MMU_ENABLED) extends Module {
  require(requestEntries >= 2 && requestEntries <= 256,
    "TMA requestEntries must be in 2..256")
  require(writeAckEntries >= 2 && writeAckEntries <= 256,
    "TMA writeAckEntries must be in 2..256")
  private val cacheSourceEntries =
    TmaV2Spec.cacheSourceEntries(requestEntries, writeAckEntries)
  val io = IO(new Bundle {
    val descriptorInvalidateAll = Input(Bool())
    val descriptorRequest =
      Flipped(Vec(2, Decoupled(new TmaV2DescriptorRequest)))
    val descriptorResponse =
      Vec(2, Decoupled(new TmaV2DescriptorResponse))
    val descriptorMemoryRequest =
      Decoupled(new TmaV2DescriptorMemoryRequest)
    val descriptorMemoryResponse =
      Flipped(Decoupled(new TmaV2DescriptorMemoryResponse))
    val command = Flipped(Decoupled(new TmaV2EngineCommand))
    val seal = Flipped(Decoupled(Bool()))
    val window = Flipped(Decoupled(new TmaV2Window))
    val tlbRequest = Decoupled(new TmaV2TlbRequest(requestEntries))
    val tlbResponse =
      Flipped(Decoupled(new TmaV2TlbResponse(requestEntries)))
    val sharedRequest =
      Decoupled(new TmaV2SharedRequest(sharedEntries))
    val sharedResponse =
      Flipped(Decoupled(new TmaV2SharedResponse(sharedEntries)))
    val cacheRequest =
      Decoupled(new TmaV2CacheRequest(cacheSourceEntries))
    val cacheResponse =
      Flipped(Decoupled(new TmaV2CacheResponse(cacheSourceEntries)))
    val completion = Decoupled(new TmaV2EngineCompletion)
    val activeCommands = Output(UInt(1.W))
    val activeWindows =
      Output(UInt(log2Ceil(
        requestEntries + windowEntries + sharedEntries + 1).W))
    val activeRequests =
      Output(UInt(log2Ceil(requestEntries + 1).W))
    val activeShared =
      Output(UInt(log2Ceil(sharedEntries + 1).W))
    val activeWriteAcks = Output(UInt(log2Ceil(
      writeAckEntries + 1).W))
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
    val descriptorEvents = Output(new TmaV2DescriptorEvents)
  })

  val descriptor = Module(new TmaV2DescriptorService(descriptorEntries))
  val engine = Module(new TmaV2WindowEngine(
    windowEntries, requestEntries, sharedEntries,
    writeAckEntries, mmuEnabled))
  descriptor.io.invalidateAll := io.descriptorInvalidateAll
  for (client <- 0 until 2) {
    descriptor.io.request(client) <> io.descriptorRequest(client)
    io.descriptorResponse(client) <> descriptor.io.response(client)
  }
  io.descriptorMemoryRequest <> descriptor.io.memoryRequest
  descriptor.io.memoryResponse <> io.descriptorMemoryResponse
  engine.io.command <> io.command
  engine.io.seal <> io.seal
  engine.io.window <> io.window
  io.tlbRequest <> engine.io.tlbRequest
  engine.io.tlbResponse <> io.tlbResponse
  io.sharedRequest <> engine.io.sharedRequest
  engine.io.sharedResponse <> io.sharedResponse
  io.cacheRequest <> engine.io.cacheRequest
  engine.io.cacheResponse <> io.cacheResponse
  io.completion <> engine.io.completion
  io.activeCommands := engine.io.activeCommands
  io.activeWindows := engine.io.activeWindows
  io.activeRequests := engine.io.activeRequests
  io.activeShared := engine.io.activeShared
  io.activeWriteAcks := engine.io.activeWriteAcks
  io.windowIssued := engine.io.windowIssued
  io.windowRetired := engine.io.windowRetired
  io.uniqueLineWave := engine.io.uniqueLineWave
  io.translationHit := engine.io.translationHit
  io.translationMiss := engine.io.translationMiss
  io.translationCoalesce := engine.io.translationCoalesce
  io.robFull := engine.io.robFull
  io.requestFull := engine.io.requestFull
  io.sharedFull := engine.io.sharedFull
  io.writeAckFull := engine.io.writeAckFull
  io.cacheToSharedLatencyValid := engine.io.cacheToSharedLatencyValid
  io.cacheToSharedLatency := engine.io.cacheToSharedLatency
  io.descriptorEvents := descriptor.io.events
}
