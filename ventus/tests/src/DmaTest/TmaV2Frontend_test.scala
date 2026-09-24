package DmaTest

import chisel3._
import chisel3.util.Decoupled
import chiseltest._
import chiseltest.internal.CachingAnnotation
import org.scalatest.freespec.AnyFreeSpec
import pipeline._

import scala.collection.mutable
import scala.io.Source
import scala.util.Random

private object TmaV2FrontendTestWiring {
  def connectPayload(
      payload: TmaV2DescriptorPayload,
      words: Vec[UInt]): Unit = {
    payload.lowWords(0) := words(0)
    payload.lowWords(1) := words(1)
    payload.lowWords(2) := words(2)
    payload.globalBaseHigh := words(3)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      payload.lowWords(3 + dimension) := words(4 + dimension)
      payload.boxDims(dimension) := words(17 + dimension)
      payload.elementStrides(dimension) := words(22 + dimension)
      if (dimension < TmaV2Spec.RankMax - 1) {
        payload.lowWords(8 + dimension) := words(9 + dimension * 2)
        payload.globalStrideHigh(dimension) :=
          words(10 + dimension * 2)
      }
    }
    payload.reservedBad :=
      (27 until 32).map(word => words(word) =/= 0.U).reduce(_ || _)
  }
}

class TmaV2CompilerHarness extends Module {
  val io = IO(new Bundle {
    val words = Input(Vec(32, UInt(32.W)))
    val in = Flipped(Decoupled(Bool()))
    val out = Decoupled(new TmaV2CompiledDescriptor)
    val busy = Output(Bool())
  })

  val compiler = Module(new TmaV2DescriptorCompiler)
  TmaV2FrontendTestWiring.connectPayload(
    compiler.io.in.bits, io.words)
  compiler.io.in.valid := io.in.valid
  io.in.ready := compiler.io.in.ready
  io.out <> compiler.io.out
  io.busy := compiler.io.busy
}

class TmaV2BinderHarness extends Module {
  val io = IO(new Bundle {
    val words = Input(Vec(32, UInt(32.W)))
    val request = Input(new TmaV2Request)
    val in = Flipped(Decoupled(Bool()))
    val out = Decoupled(new TmaV2BindResponse)
    val bindCycle = Output(Bool())
  })

  val compiler = Module(new TmaV2DescriptorCompiler)
  val binder = Module(new TmaV2CommandBinder)
  val request = Reg(new TmaV2Request)
  TmaV2FrontendTestWiring.connectPayload(compiler.io.in.bits, io.words)
  compiler.io.in.valid := io.in.valid
  io.in.ready := compiler.io.in.ready
  when(io.in.fire) { request := io.request }
  binder.io.in.valid := compiler.io.out.valid
  compiler.io.out.ready := binder.io.in.ready
  binder.io.in.bits.compiled := compiler.io.out.bits
  binder.io.in.bits.request := request
  io.out <> binder.io.out
  io.bindCycle := binder.io.bindCycle
}

class TmaV2WindowHarness extends Module {
  val io = IO(new Bundle {
    val words = Input(Vec(32, UInt(32.W)))
    val request = Input(new TmaV2Request)
    val in = Flipped(Decoupled(Bool()))
    val out = Decoupled(new TmaV2Window)
    val flush = Input(Bool())
    val produced = Output(Bool())
    val stalled = Output(Bool())
  })

  val compiler = Module(new TmaV2DescriptorCompiler)
  val binder = Module(new TmaV2CommandBinder)
  val planner = Module(new TmaV2WindowPlanner)
  planner.io.flush := io.flush
  val request = Reg(new TmaV2Request)
  TmaV2FrontendTestWiring.connectPayload(compiler.io.in.bits, io.words)
  compiler.io.in.valid := io.in.valid
  io.in.ready := compiler.io.in.ready
  when(io.in.fire) { request := io.request }
  binder.io.in.valid := compiler.io.out.valid
  compiler.io.out.ready := binder.io.in.ready
  binder.io.in.bits.compiled := compiler.io.out.bits
  binder.io.in.bits.request := request
  planner.io.in.valid := binder.io.out.valid && binder.io.out.bits.legal
  planner.io.in.bits := binder.io.out.bits.command
  binder.io.out.ready := Mux(
    binder.io.out.bits.legal, planner.io.in.ready, true.B)
  io.out <> planner.io.out
  io.produced := planner.io.produced
  io.stalled := planner.io.stalled
}

/** Compiler + Binder + Execute harness used to exercise the command-local
  * outer-stride setup without involving cache/TLB plumbing. */
class TmaV2ExecuteWindowHarness extends Module {
  val io = IO(new Bundle {
    val words = Input(Vec(32, UInt(32.W)))
    val request = Input(new TmaV2Request)
    val in = Flipped(Decoupled(Bool()))
    val out = Decoupled(new TmaV2Window)
    val kill = Input(Bool())
    val active = Output(Bool())
    val rawPlannerFire = Output(Bool())
    val plannerProduced = Output(Bool())
    val plannerStalled = Output(Bool())
    val compiledStatus = Output(UInt(TmaV2Status.width.W))
    val bindStatus = Output(UInt(TmaV2Status.width.W))
  })

  val compiler = Module(new TmaV2DescriptorCompiler)
  val binder = Module(new TmaV2CommandBinder)
  val execute = Module(new TmaV2ExecuteStage)
  val request = Reg(new TmaV2Request)

  TmaV2FrontendTestWiring.connectPayload(compiler.io.in.bits, io.words)
  compiler.io.in.valid := io.in.valid
  io.in.ready := compiler.io.in.ready
  when(io.in.fire) { request := io.request }

  binder.io.in.valid := compiler.io.out.valid
  compiler.io.out.ready := binder.io.in.ready
  binder.io.in.bits.compiled := compiler.io.out.bits
  binder.io.in.bits.request := request

  execute.io.in.valid := binder.io.out.valid && binder.io.out.bits.legal
  execute.io.in.bits := 0.U.asTypeOf(new TmaV2PreparedCommand)
  execute.io.in.bits.kind := TmaV2PreparedKind.Tensor
  execute.io.in.bits.engine.wid := 0.U
  execute.io.in.bits.engine.copyDirection := request.copyDirection
  execute.io.in.bits.engine.dtype := binder.io.out.bits.command.compiled.dtype
  execute.io.in.bits.engine.oobFill :=
    binder.io.out.bits.command.compiled.oobFill
  execute.io.in.bits.engine.reduceMode := request.reduceMode
  execute.io.in.bits.engine.asid := 0.U
  execute.io.in.bits.engine.group := 0.U
  execute.io.in.bits.engine.transactionBytes :=
    binder.io.out.bits.command.compiled.logicalBytes
  execute.io.in.bits.tensor := binder.io.out.bits.command
  binder.io.out.ready := Mux(
    binder.io.out.bits.legal, execute.io.in.ready, true.B)

  execute.io.command.ready := true.B
  execute.io.seal.ready := true.B
  io.out <> execute.io.window
  execute.io.kill.valid := io.kill
  execute.io.kill.bits := 0.U.asTypeOf(new TmaV2KillRequest)
  execute.io.killActive := io.kill
  execute.io.killAsid := 0.U
  val completion = RegNext(
    (io.out.fire && io.out.bits.last) || io.kill, false.B)
  execute.io.engineCompletion.valid := completion
  execute.io.engineCompletion.bits :=
    0.U.asTypeOf(new TmaV2EngineCompletion)
  io.active := execute.io.active
  io.rawPlannerFire := execute.io.plannerFire
  io.plannerProduced := execute.io.plannerProduced
  io.plannerStalled := execute.io.plannerStalled
  io.compiledStatus := compiler.io.out.bits.status
  io.bindStatus := binder.io.out.bits.status
}

class TmaV2Frontend_test extends AnyFreeSpec with ChiselScalatestTester {
  private case class Golden(
      data: Set[(BigInt, BigInt)],
      fill: Set[BigInt])

  private def descriptor(
      dtype: Int,
      rank: Int,
      dims: Seq[Int],
      strides: Seq[BigInt],
      box: Seq[Int],
      elementStrides: Seq[Int] = Seq.empty,
      interleave: Int = TmaV2Spec.InterleaveNone,
      swizzle: Int = TmaV2Spec.SwizzleNone,
      oob: Int = TmaV2Spec.OobZero): Array[BigInt] = {
    require(dims.size == rank)
    require(strides.size == rank - 1)
    require(box.size == rank)
    require(elementStrides.isEmpty || elementStrides.size == rank)
    val words = Array.fill[BigInt](32)(0)
    words(0) = TmaV2Spec.DescriptorMagic
    words(1) = dtype | (rank << 5) | (interleave << 8) |
      (swizzle << 10) | (oob << 18)
    words(2) = 0x10000
    for (dimension <- 0 until rank) {
      words(4 + dimension) = dims(dimension)
      words(17 + dimension) = box(dimension)
      words(22 + dimension) =
        if (elementStrides.isEmpty) 1 else elementStrides(dimension)
      if (dimension + 1 < rank) {
        words(9 + dimension * 2) =
          strides(dimension) & 0xffffffffL
        words(10 + dimension * 2) =
          strides(dimension) >> 32
      }
    }
    words
  }

  private def legalDescriptor(
      dtype: Int,
      rank: Int): Array[BigInt] = {
    val rowElements = dtype match {
      case TmaV2Spec.DTypeB4x16P64 | TmaV2Spec.DTypeB6 => 128
      case TmaV2Spec.DTypeB4x16 => 32
      case TmaV2Spec.DTypeU8 => 16
      case TmaV2Spec.DTypeU16 | TmaV2Spec.DTypeFp16 |
          TmaV2Spec.DTypeBf16 => 8
      case TmaV2Spec.DTypeU64 | TmaV2Spec.DTypeS64 |
          TmaV2Spec.DTypeFp64 => 2
      case _ => 4
    }
    val rowBytes = dtype match {
      case TmaV2Spec.DTypeB4x16P64 => 64
      case TmaV2Spec.DTypeB6 => 96
      case _ => 16
    }
    val dimensions = Seq(rowElements) ++ Seq.fill(rank - 1)(2)
    val strides = mutable.ArrayBuffer.empty[BigInt]
    var span = BigInt(rowBytes)
    for (_ <- 1 until rank) {
      val alignment =
        if (dtype == TmaV2Spec.DTypeB4x16P64 ||
            dtype == TmaV2Spec.DTypeB6) 32 else 16
      val stride = (span + alignment - 1) / alignment * alignment
      strides += stride
      span = stride * 2
    }
    descriptor(dtype, rank, dimensions, strides.toSeq, dimensions)
  }

  private def pokeWords(
      port: Vec[UInt],
      words: Array[BigInt]): Unit =
    for (word <- words.indices) port(word).poke(words(word).U)

  private def pokeRequest(
      request: TmaV2Request,
      direction: Int = TmaV2Spec.DirectionG2S,
      coordinates: Seq[Int] = Seq.empty,
      sharedBase: BigInt = 0x2000,
      reduceMode: Int = TmaV2Spec.ReduceCopy): Unit = {
    request.copyDirection.poke(direction.U)
    request.reduceMode.poke(reduceMode.U)
    request.sharedBase.poke(sharedBase.U)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      request.coordinates(dimension)
        .poke(coordinates.lift(dimension).getOrElse(0).S)
    }
  }

  private def compileOne(
      dut: TmaV2CompilerHarness,
      words: Array[BigInt]): Int = {
    pokeWords(dut.io.words, words)
    dut.io.out.ready.poke(true.B)
    dut.io.in.valid.poke(true.B)
    dut.io.in.bits.poke(true.B)
    var acceptCycles = 0
    while (!dut.io.in.ready.peekBoolean() && acceptCycles < 4) {
      dut.clock.step()
      acceptCycles += 1
    }
    assert(acceptCycles < 4)
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
    var compileCycles = 0
    while (!dut.io.out.valid.peekBoolean() && compileCycles < 20) {
      assert(dut.io.busy.peekBoolean())
      dut.clock.step()
      compileCycles += 1
    }
    assert(compileCycles <= 19,
      s"compiler exceeded the 19-cycle pipelined budget: $compileCycles")
    assert(dut.io.out.valid.peekBoolean())
    compileCycles
  }

  private def bindOne(
      dut: TmaV2BinderHarness,
      words: Array[BigInt],
      direction: Int,
      coordinates: Seq[Int],
      sharedBase: BigInt = 0x2000,
      reduceMode: Int = TmaV2Spec.ReduceCopy): Int = {
    pokeWords(dut.io.words, words)
    pokeRequest(dut.io.request, direction, coordinates,
      sharedBase, reduceMode)
    dut.io.out.ready.poke(true.B)
    dut.io.in.valid.poke(true.B)
    dut.io.in.bits.poke(true.B)
    while (!dut.io.in.ready.peekBoolean()) dut.clock.step()
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
    var cycles = 0
    var waitCycles = 0
    while (!dut.io.out.valid.peekBoolean() && waitCycles < 32) {
      if (dut.io.bindCycle.peekBoolean()) cycles += 1
      dut.clock.step()
      waitCycles += 1
    }
    assert(waitCycles < 32)
    assert(cycles <= TmaV2Spec.RankMax)
    cycles
  }

  private def startWindow(
      dut: TmaV2WindowHarness,
      words: Array[BigInt],
      direction: Int,
      coordinates: Seq[Int],
      reduceMode: Int = TmaV2Spec.ReduceCopy): Unit = {
    dut.io.flush.poke(false.B)
    pokeWords(dut.io.words, words)
    pokeRequest(dut.io.request, direction, coordinates,
      reduceMode = reduceMode)
    dut.io.in.valid.poke(true.B)
    dut.io.in.bits.poke(true.B)
    var cycles = 0
    while (!dut.io.in.ready.peekBoolean() && cycles < 32) {
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < 32)
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
  }

  private def golden(caseName: String): Golden = {
    val stream = getClass.getResourceAsStream("/tma_v2_atom_trace.csv")
    require(stream != null, "independent C-model atom trace is missing")
    val source = Source.fromInputStream(stream)
    try {
      val data = mutable.Set.empty[(BigInt, BigInt)]
      val fill = mutable.Set.empty[BigInt]
      source.getLines().drop(1).foreach { line =>
        val column = line.split(",", -1)
        if (column(0) == caseName && column(1) == "ok" &&
            column(2) != "-1") {
          val globalBase = BigInt(column(5))
          val sharedBase = BigInt(column(6))
          val sharedMask = column(8).toInt
          if (column(4) == "1") {
            for (byte <- 0 until 16
                 if (sharedMask & (1 << byte)) != 0)
              fill += sharedBase + byte
          } else {
            val mapping = column(9)
            for (globalByte <- 0 until 16) {
              val sharedByte = Integer.parseInt(
                mapping.slice(globalByte * 2, globalByte * 2 + 2), 16)
              if (sharedByte != 0xff)
                data += ((globalBase + globalByte,
                  sharedBase + sharedByte))
            }
          }
        }
      }
      Golden(data.toSet, fill.toSet)
    } finally source.close()
  }

  private def collectWindows(
      dut: TmaV2WindowHarness,
      seed: Int,
      maxCycles: Int = 10000): (Golden, Int, Int) = {
    val random = new Random(seed)
    val data = mutable.Set.empty[(BigInt, BigInt)]
    val fill = mutable.Set.empty[BigInt]
    var windows = 0
    var cycles = 0
    var done = false
    while (!done && cycles < maxCycles) {
      val ready = random.nextInt(4) != 0
      dut.io.out.ready.poke(ready.B)
      if (dut.io.out.valid.peekBoolean() && ready) {
        for (lane <- 0 until 8
             if dut.io.out.bits.lanes(lane).valid.peekBoolean()) {
          val global = dut.io.out.bits.lanes(lane)
            .globalAddress.peekInt()
          val shared = dut.io.out.bits.sharedBase.peekInt() +
            (dut.io.out.bits.lanes(lane).sharedAtomDelta.peekInt() << 4)
          val globalBytes = dut.io.out.bits.lanes(lane)
            .globalBytes.peekInt().toInt
          val sharedBytes = dut.io.out.bits.lanes(lane)
            .sharedBytes.peekInt().toInt
          for (byte <- 0 until 16) {
            val globalValid = byte < globalBytes
            val sharedValid = byte < sharedBytes
            if (globalValid && sharedValid)
              data += ((global + byte, shared + byte))
            else if (!globalValid && sharedValid)
              fill += shared + byte
          }
        }
        done = dut.io.out.bits.last.peekBoolean()
        windows += 1
      }
      dut.clock.step()
      cycles += 1
    }
    assert(done, s"planner did not complete within $maxCycles cycles")
    (Golden(data.toSet, fill.toSet), windows, cycles)
  }

  private def startExecuteWindow(
      dut: TmaV2ExecuteWindowHarness,
      words: Array[BigInt],
      coordinates: Seq[Int]): Unit = {
    dut.io.kill.poke(false.B)
    pokeWords(dut.io.words, words)
    pokeRequest(dut.io.request, TmaV2Spec.DirectionG2S, coordinates)
    dut.io.out.ready.poke(false.B)
    dut.io.in.valid.poke(true.B)
    dut.io.in.bits.poke(true.B)
    var cycles = 0
    while (!dut.io.in.ready.peekBoolean() && cycles < 32) {
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < 32)
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
  }

  "descriptor compiler covers rank 1-5 and every supported dtype in at most thirteen cycles" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      for {
        rank <- TmaV2Spec.RankMin to TmaV2Spec.RankMax
        dtype <- TmaV2Spec.DTypeU8 to TmaV2Spec.DTypeB6
      } {
        val words = legalDescriptor(dtype, rank)
        val cycles = compileOne(dut, words)
        dut.io.out.bits.status.expect(TmaV2Status.Ok)
        dut.io.out.bits.status.expect(TmaV2Status.Ok)
        dut.io.out.bits.rank.expect(rank.U)
        dut.io.out.bits.dtype.expect(dtype.U)
        assert(cycles <= 11)
        dut.clock.step()
      }
    }
  }

  "descriptor compiler reports static errors and retains required layout products" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val interleave16 = descriptor(
        TmaV2Spec.DTypeFp32, 3, Seq(4, 8, 2),
        Seq(16, 128), Seq(4, 8, 2),
        interleave = TmaV2Spec.Interleave16)
      val interleave16Cycles = compileOne(dut, interleave16)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.globalStrides(0).expect(16.U)
      dut.io.out.bits.globalStrides(1).expect(128.U)
      dut.io.out.bits.boxDims(1).expect(1.U)
      dut.io.out.bits.logicalBytes.expect(128.U)
      dut.io.out.bits.interleaveSliceStride.expect(128.U)
      dut.clock.step()

      val interleave32 = descriptor(
        TmaV2Spec.DTypeFp32, 3, Seq(8, 4, 2),
        Seq(32, 128), Seq(8, 4, 2),
        interleave = TmaV2Spec.Interleave32,
        swizzle = TmaV2Spec.Swizzle32)
      compileOne(dut, interleave32)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.boxDims(1).expect(1.U)
      dut.io.out.bits.logicalBytes.expect(512.U)
      dut.clock.step()

      val unitElementStride = descriptor(
        TmaV2Spec.DTypeU8, 2, Seq(128, 64),
        Seq(128), Seq(128, 64),
        elementStrides = Seq(1, 1))
      compileOne(dut, unitElementStride)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.globalStrides(0).expect(128.U)
      dut.io.out.bits.logicalBytes.expect((128 * 64).U)
      dut.io.out.bits.rawDim0Box.expect(128.U)
      for (dimension <- 0 until 2)
        dut.io.out.bits.strideMinus1(dimension).expect(0.U)
      dut.clock.step()

      for (elementStride <- 2 to 8) {
        val outerElementStride = descriptor(
          TmaV2Spec.DTypeU8, 2, Seq(128, 64),
          Seq(128), Seq(128, 64),
          elementStrides = Seq(1, elementStride))
        compileOne(dut, outerElementStride)
        val effective = (64 + elementStride - 1) / elementStride
        dut.io.out.bits.status.expect(TmaV2Status.Ok)
        dut.io.out.bits.boxDims(0).expect(128.U)
        dut.io.out.bits.boxDims(1).expect(effective.U)
        dut.io.out.bits.strideMinus1(0).expect(0.U)
        dut.io.out.bits.strideMinus1(1).expect((elementStride - 1).U)
        dut.io.out.bits.logicalBytes.expect((128 * effective).U)
        dut.clock.step()
      }

      val interleaveDim0Stride = descriptor(
        TmaV2Spec.DTypeU16, 3, Seq(8, 8, 3),
        Seq(16, 128), Seq(8, 8, 3),
        elementStrides = Seq(2, 1, 1),
        interleave = TmaV2Spec.Interleave16)
      compileOne(dut, interleaveDim0Stride)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.rawDim0Box.expect(8.U)
      dut.io.out.bits.boxDims(0).expect(4.U)
      dut.io.out.bits.logicalBytes.expect((4 * 2 * 8 * 3).U)
      dut.clock.step()

      val ignoredChannelStride = descriptor(
        TmaV2Spec.DTypeU16, 3, Seq(16, 8, 2),
        Seq(256, 2048), Seq(8, 8, 2),
        elementStrides = Seq(1, 7, 1),
        interleave = TmaV2Spec.Interleave16)
      val ignoredChannelCycles = compileOne(dut, ignoredChannelStride)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.boxDims(1).expect(1.U)
      dut.io.out.bits.strideMinus1(1).expect(6.U)
      dut.io.out.bits.logicalBytes.expect((8 * 16 * 2).U)
      assert(ignoredChannelCycles == interleave16Cycles,
        "channel metadata stride must not enter compiler setup")
      dut.clock.step()

      for (badStride <- Seq(0, 9)) {
        val badOuterStride = descriptor(
          TmaV2Spec.DTypeU8, 2, Seq(128, 64),
          Seq(128), Seq(128, 64),
          elementStrides = Seq(1, badStride))
        compileOne(dut, badOuterStride)
        dut.io.out.bits.status.expect(TmaV2Status.BadStride)
        dut.clock.step()
      }

      val nonInterleaveDim0Stride = descriptor(
        TmaV2Spec.DTypeU8, 2, Seq(128, 64),
        Seq(128), Seq(128, 64), elementStrides = Seq(2, 1))
      compileOne(dut, nonInterleaveDim0Stride)
      dut.io.out.bits.status.expect(TmaV2Status.BadStride)
      dut.clock.step()

      val inactiveStride = legalDescriptor(TmaV2Spec.DTypeU8, 1)
      inactiveStride(23) = 1
      compileOne(dut, inactiveStride)
      dut.io.out.bits.status.expect(TmaV2Status.ReservedBits)
      dut.clock.step()

      val priority = nonInterleaveDim0Stride.clone()
      priority(0) = 0
      compileOne(dut, priority)
      dut.io.out.bits.status.expect(TmaV2Status.BadMagic)
      dut.clock.step()

      val packedB4At16B = legalDescriptor(TmaV2Spec.DTypeB4x16, 2)
      packedB4At16B(2) = 0x10010
      packedB4At16B(9) = 16
      compileOne(dut, packedB4At16B)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.clock.step()

      val oddPackedB4 = legalDescriptor(TmaV2Spec.DTypeB4x16, 1)
      oddPackedB4(4) = 65
      compileOne(dut, oddPackedB4)
      dut.io.out.bits.status.expect(TmaV2Status.BadDimension)
      dut.clock.step()

      val misalignedPaddedB6 = legalDescriptor(TmaV2Spec.DTypeB6, 1)
      misalignedPaddedB6(2) = 0x10010
      compileOne(dut, misalignedPaddedB6)
      dut.io.out.bits.status.expect(TmaV2Status.BadAlignment)
      dut.clock.step()

      val badMagic = legalDescriptor(TmaV2Spec.DTypeU8, 1)
      badMagic(0) = 0
      val badRank = legalDescriptor(TmaV2Spec.DTypeU8, 1)
      badRank(1) &= ~(BigInt(7) << 5)
      val invalid = Seq(
        (badMagic, TmaV2Status.BadMagic),
        (badRank, TmaV2Status.BadRank),
        (descriptor(TmaV2Spec.DTypeFp32, 2, Seq(4, 4),
          Seq(16), Seq(4, 4), interleave = TmaV2Spec.Interleave16),
          TmaV2Status.BadLayout),
        (descriptor(TmaV2Spec.DTypeU8, 2, Seq(32, 2),
          Seq(16), Seq(16, 2)), TmaV2Status.BadStride))
      for ((words, status) <- invalid) {
        compileOne(dut, words)
        dut.io.out.bits.status.expect(status)
        dut.clock.step()
      }
    }
  }

  "descriptor compiler exhaustively validates element strides by rank and dimension" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      for (rank <- TmaV2Spec.RankMin to TmaV2Spec.RankMax) {
        for (dimension <- 0 until rank; elementStride <- 0 to 9) {
          val interleavedDim0 = dimension == 0 && rank >= 3
          val dimensions =
            if (interleavedDim0) Seq(8) ++ Seq.fill(rank - 1)(2)
            else Seq(16) ++ Seq.fill(rank - 1)(2)
          val strides =
            if (interleavedDim0)
              Seq[BigInt](16, 32, 64, 128).take(rank - 1)
            else Seq[BigInt](16, 32, 64, 128).take(rank - 1)
          val elementStrides = Seq.tabulate(rank) { index =>
            if (index == dimension) elementStride else 1
          }
          val words = descriptor(
            if (interleavedDim0) TmaV2Spec.DTypeU16 else TmaV2Spec.DTypeU8,
            rank, dimensions, strides, dimensions,
            elementStrides = elementStrides,
            interleave = if (interleavedDim0)
              TmaV2Spec.Interleave16 else TmaV2Spec.InterleaveNone)
          compileOne(dut, words)
          val legal = elementStride >= TmaV2Spec.ElementStrideMin &&
            elementStride <= TmaV2Spec.ElementStrideMax &&
            (dimension != 0 || interleavedDim0 || elementStride == 1)
          dut.io.out.bits.status.expect(
            (if (legal) TmaV2Status.Ok else TmaV2Status.BadStride))
          if (legal) {
            val effective =
              (dimensions(dimension) + elementStride - 1) / elementStride
            dut.io.out.bits.boxDims(dimension).expect(effective.U)
            dut.io.out.bits.strideMinus1(dimension)
              .expect((elementStride - 1).U)
          }
          dut.clock.step()
        }

        for (inactive <- rank until TmaV2Spec.RankMax) {
          val words = legalDescriptor(TmaV2Spec.DTypeU8, rank)
          words(22 + inactive) = 1
          compileOne(dut, words)
          dut.io.out.bits.status.expect(TmaV2Status.ReservedBits)
          dut.clock.step()
        }
      }

      // Error priority remains architectural: an earlier malformed field
      // wins even when the same descriptor also carries a bad traversal
      // stride.
      val priority = legalDescriptor(TmaV2Spec.DTypeU8, 2)
      priority(0) = 0
      priority(23) = 9
      compileOne(dut, priority)
      dut.io.out.bits.status.expect(TmaV2Status.BadMagic)
      dut.clock.step()

      val scaledStrideOverflow = descriptor(
        TmaV2Spec.DTypeU8, 2, Seq(16, 1),
        Seq(BigInt("fffffffffffffff0", 16)), Seq(16, 1),
        elementStrides = Seq(1, 2))
      compileOne(dut, scaledStrideOverflow)
      dut.io.out.bits.status.expect(TmaV2Status.AddressOverflow)
    }
  }

  "command binder computes only dynamic origins and rejects command-local errors" in {
    test(new TmaV2BinderHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val words = descriptor(
        TmaV2Spec.DTypeU32, 3, Seq(32, 8, 2),
        Seq(128, 1024), Seq(16, 4, 2))

      bindOne(dut, words, TmaV2Spec.DirectionG2S, Seq(4, 2, 1))
      dut.io.out.bits.legal.expect(true.B)
      dut.io.out.bits.command.originComponents(0).expect(16.S)
      dut.io.out.bits.command.originComponents(1).expect(256.S)
      dut.io.out.bits.command.originComponents(2).expect(1024.S)
      dut.clock.step()

      val interleaveWords = descriptor(
        TmaV2Spec.DTypeU16, 3, Seq(16, 8, 2),
        Seq(256, 2048), Seq(8, 8, 2),
        interleave = TmaV2Spec.Interleave16)
      bindOne(dut, interleaveWords, TmaV2Spec.DirectionG2S,
        Seq(1, 1, 1))
      dut.io.out.bits.legal.expect(true.B)
      dut.io.out.bits.command.originComponents(0).expect(16.S)
      dut.io.out.bits.command.originComponents(1).expect(256.S)
      dut.io.out.bits.command.originComponents(2).expect(2048.S)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionG2S, Seq(3, 2, 1))
      dut.io.out.bits.status.expect(TmaV2Status.BadAlignment)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionG2S, Seq(-4, 0, 0))
      dut.io.out.bits.legal.expect(true.B)
      dut.io.out.bits.command.originComponents(0).expect((-16).S)
      dut.clock.step()

      val packedB4 = legalDescriptor(TmaV2Spec.DTypeB4x16, 1)
      bindOne(dut, packedB4, TmaV2Spec.DirectionG2S, Seq(32))
      dut.io.out.bits.legal.expect(true.B)
      dut.clock.step()
      bindOne(dut, packedB4, TmaV2Spec.DirectionG2S, Seq(16))
      dut.io.out.bits.status.expect(TmaV2Status.BadAlignment)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionS2G, Seq(-1, 0, 0))
      dut.io.out.bits.status.expect(TmaV2Status.BadCoordinate)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionS2G, Seq(-1, 0, 0),
        reduceMode = TmaV2Spec.ReduceAdd)
      dut.io.out.bits.status.expect(TmaV2Status.BadCoordinate)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionG2S, Seq(0, 0, 0),
        sharedBase = 0x2040)
      dut.io.out.bits.status.expect(TmaV2Status.BadAlignment)
      dut.clock.step()

      bindOne(dut, words, TmaV2Spec.DirectionG2S, Seq(0, 0, 0),
        reduceMode = TmaV2Spec.ReduceAdd)
      dut.io.out.bits.status.expect(TmaV2Status.UnsupportedFeature)
    }
  }

  "window planner covers rank 1-5 and keeps random-backpressure output stable" in {
    for (rank <- TmaV2Spec.RankMin to TmaV2Spec.RankMax) {
      test(new TmaV2WindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        val dimensions = Seq(16) ++ Seq.fill(rank - 1)(2)
        val strides = Seq[BigInt](16, 32, 64, 128).take(rank - 1)
        val words = descriptor(TmaV2Spec.DTypeU8, rank,
          dimensions, strides, dimensions)
        dut.io.out.ready.poke(false.B)
        dut.io.in.valid.poke(false.B)
        startWindow(dut, words, TmaV2Spec.DirectionG2S,
          Seq.fill(rank)(0))
        val observed = collectWindows(dut, 0x3100 + rank)
        assert(observed._1.data.size == dimensions.product)
        assert(observed._1.fill.isEmpty)
        assert(observed._2 == (dimensions.product + 127) / 128)
      }
    }
  }

  "32KiB contiguous command emits exactly 256 consecutive windows after fill" in {
    test(new TmaV2WindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      val words = descriptor(TmaV2Spec.DTypeU8, 2,
        Seq(256, 128), Seq(256), Seq(256, 128))
      dut.io.out.ready.poke(true.B)
      dut.io.in.valid.poke(false.B)
      startWindow(dut, words, TmaV2Spec.DirectionG2S, Seq(0, 0))

      var cycle = 0
      var fires = 0
      var previousFire = -1
      var sawLast = false
      while (!sawLast && cycle < 300) {
        if (dut.io.out.valid.peekBoolean()) {
          if (previousFire >= 0)
            assert(cycle == previousFire + 1,
              s"internal bubble between windows $fires and ${fires + 1}")
          previousFire = cycle
          fires += 1
          sawLast = dut.io.out.bits.last.peekBoolean()
        }
        dut.clock.step()
        cycle += 1
      }
      assert(sawLast,
        s"planner did not assert last: fires=$fires cycles=$cycle")
      assert(fires == 256)
    }
  }

  "planner flush drops every buffered window and permits the next command" in {
    test(new TmaV2WindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.flush.poke(false.B)
      dut.clock.step(2)

      val long = descriptor(TmaV2Spec.DTypeU8, 2,
        Seq(256, 128), Seq(256), Seq(256, 128))
      startWindow(dut, long, TmaV2Spec.DirectionG2S, Seq(0, 0))
      var waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 64) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 64, "long command never reached the stalled planner output")

      dut.io.flush.poke(true.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step()
      dut.io.flush.poke(false.B)
      dut.io.out.ready.poke(true.B)
      for (_ <- 0 until 12) {
        dut.io.out.valid.expect(false.B)
        dut.clock.step()
      }

      val short = descriptor(TmaV2Spec.DTypeU8, 1,
        Seq(128), Seq.empty, Seq(128))
      startWindow(dut, short, TmaV2Spec.DirectionG2S, Seq(0))
      val observed = collectWindows(dut, seed = 0x46534c48)
      assert(observed._1.data.size == 128)
      assert(observed._1.fill.isEmpty)
      assert(observed._2 == 1)
    }
  }

  "compiler retains outer element-stride metadata for Execute setup" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val words = descriptor(TmaV2Spec.DTypeU8, 2, Seq(128, 64),
        Seq(128), Seq(128, 64), elementStrides = Seq(1, 2))
      compileOne(dut, words)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.boxDims(1).expect(32.U)
      dut.io.out.bits.strideMinus1(1).expect(1.U)
      dut.io.out.bits.logicalBytes.expect((128 * 32).U)
    }
  }

  "compiler preserves unit-stride latency and adds one setup cycle per strided dimension" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)

      val dimensions = Seq(16, 16, 16, 16, 16)
      val globalStrides = Seq[BigInt](16, 256, 4096, 65536)
      val boxDims = Seq(16, 8, 8, 8, 8)
      val unit = descriptor(TmaV2Spec.DTypeU8, 5,
        dimensions, globalStrides, boxDims,
        elementStrides = Seq(1, 1, 1, 1, 1))
      val unitCycles = compileOne(dut, unit)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      assert(unitCycles <= 11,
        s"unit-stride compiler latency regressed to $unitCycles cycles")
      dut.clock.step()

      val strided = descriptor(TmaV2Spec.DTypeU8, 5,
        dimensions, globalStrides, boxDims,
        elementStrides = Seq(1, 2, 3, 4, 8))
      val stridedCycles = compileOne(dut, strided)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      assert(stridedCycles == unitCycles + 4,
        s"expected four command-local setup cycles: unit=$unitCycles " +
          s"strided=$stridedCycles")
      for ((effective, dimension) <- Seq(16, 4, 3, 2, 1).zipWithIndex)
        dut.io.out.bits.boxDims(dimension).expect(effective.U)
      for ((minusOne, dimension) <- Seq(0, 1, 2, 3, 7).zipWithIndex)
        dut.io.out.bits.strideMinus1(dimension).expect(minusOne.U)
      dut.io.out.bits.logicalBytes.expect((16 * 4 * 3 * 2).U)
    }
  }

  "outer element stride scales source addresses and keeps shared output dense" in {
    for ((coordinate, expectedBytes) <- Seq(
        (1, Seq(16, 16, 16)),
        (-3, Seq(0, 0, 16)),
        (15, Seq(16, 0, 0)))) {
      test(new TmaV2ExecuteWindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        val words = descriptor(TmaV2Spec.DTypeU8, 2,
          Seq(16, 16), Seq(16), Seq(16, 5),
          elementStrides = Seq(1, 2))
        dut.io.in.valid.poke(false.B)
        startExecuteWindow(dut, words, Seq(0, coordinate))
        dut.io.out.ready.poke(true.B)

        var waited = 0
        while (!dut.io.out.valid.peekBoolean() && waited < 64) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 64)
        dut.io.out.bits.last.expect(true.B)
        dut.io.out.bits.sharedBase.expect(0x2000.U)
        val expectedAddresses = Seq(coordinate, coordinate + 2, coordinate + 4)
          .map(row => BigInt(0x10000 + row * 16))
        for (lane <- 0 until 3) {
          dut.io.out.bits.lanes(lane).valid.expect(true.B)
          dut.io.out.bits.lanes(lane).globalAddress
            .expect((expectedAddresses(lane) & 0xffffffffL).U)
          dut.io.out.bits.lanes(lane).globalBytes
            .expect(expectedBytes(lane).U)
          dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(lane.U)
          dut.io.out.bits.lanes(lane).sharedBytes.expect(16.U)
        }
        for (lane <- 3 until 8)
          dut.io.out.bits.lanes(lane).valid.expect(false.B)
        dut.clock.step()
      }
    }
  }

  "outer element stride eight does not wrap and preserves dense shared packing" in {
    test(new TmaV2ExecuteWindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      val words = descriptor(TmaV2Spec.DTypeU8, 2,
        Seq(16, 32), Seq(16), Seq(16, 17),
        elementStrides = Seq(1, 8))
      dut.io.in.valid.poke(false.B)
      startExecuteWindow(dut, words, Seq(0, 1))
      dut.io.out.ready.poke(true.B)

      var waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 64) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 64)
      dut.io.out.bits.last.expect(true.B)
      dut.io.out.bits.sharedBase.expect(0x2000.U)
      val expectedAddresses = Seq(1, 9, 17)
        .map(row => BigInt(0x10000 + row * 16))
      for (lane <- 0 until 3) {
        dut.io.out.bits.lanes(lane).valid.expect(true.B)
        dut.io.out.bits.lanes(lane).globalAddress
          .expect(expectedAddresses(lane).U)
        dut.io.out.bits.lanes(lane).globalBytes.expect(16.U)
        dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(lane.U)
        dut.io.out.bits.lanes(lane).sharedBytes.expect(16.U)
      }
      for (lane <- 3 until 8)
        dut.io.out.bits.lanes(lane).valid.expect(false.B)
      dut.clock.step()
    }
  }

  "interleave16 dim0 stride selects whole atoms and compacts shared output" in {
    for (elementStride <- 1 to 8) {
      test(new TmaV2ExecuteWindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        val words = descriptor(
          TmaV2Spec.DTypeU16, 3,
          Seq(8, 64, 1), Seq(16, 1024), Seq(8, 8, 1),
          elementStrides = Seq(elementStride, 1, 1),
          interleave = TmaV2Spec.Interleave16)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(false.B)
        startExecuteWindow(dut, words, Seq(0, 0, 0))

        var waited = 0
        while (!dut.io.out.valid.peekBoolean() && waited < 96) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 96,
          s"interleave16 stride $elementStride produced no window")

        // A stalled slow-path output must retain both its atom mask and its
        // dense shared-memory assignment.
        val heldBase = dut.io.out.bits.sharedBase.peekInt()
        val heldLast = dut.io.out.bits.last.peekBoolean()
        val held = (0 until 8).map { lane => (
          dut.io.out.bits.lanes(lane).valid.peekBoolean(),
          dut.io.out.bits.lanes(lane).globalAddress.peekInt(),
          dut.io.out.bits.lanes(lane).sharedAtomDelta.peekInt()) }
        dut.clock.step(3)
        dut.io.out.bits.sharedBase.expect(heldBase.U)
        dut.io.out.bits.last.expect(heldLast.B)
        for (lane <- 0 until 8) {
          dut.io.out.bits.lanes(lane).valid.expect(held(lane)._1.B)
          dut.io.out.bits.lanes(lane).globalAddress.expect(held(lane)._2.U)
          dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(held(lane)._3.U)
        }

        val positions = 0 until 8 by elementStride
        dut.io.out.bits.last.expect(true.B)
        dut.io.out.bits.sharedBase.expect(0x2000.U)
        for (lane <- 0 until 8) {
          val active = lane < positions.size
          dut.io.out.bits.lanes(lane).valid.expect(active.B)
          if (active) {
            dut.io.out.bits.lanes(lane).globalAddress
              .expect((0x10000 + positions(lane) * 16).U)
            dut.io.out.bits.lanes(lane).globalBytes.expect(16.U)
            dut.io.out.bits.lanes(lane).sharedBytes.expect(16.U)
            dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(lane.U)
          }
        }
        dut.io.out.ready.poke(true.B)
        dut.clock.step()
      }
    }
  }

  "interleave32 dim0 stride keeps adjacent atom halves and dense CUDA swizzle" in {
    for (elementStride <- 2 to 8) {
      test(new TmaV2ExecuteWindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        val words = descriptor(
          TmaV2Spec.DTypeU8, 3,
          Seq(32, 64, 1), Seq(32, 2048), Seq(16, 32, 1),
          elementStrides = Seq(elementStride, 1, 1),
          interleave = TmaV2Spec.Interleave32,
          swizzle = TmaV2Spec.Swizzle32)
        dut.io.in.valid.poke(false.B)
        dut.io.out.ready.poke(true.B)
        startExecuteWindow(dut, words, Seq(0, 0, 0))

        val positions = (0 until 16 by elementStride).toVector
        val expectedAtoms = positions.flatMap(position =>
          Seq(position * 2, position * 2 + 1))
        val observedAtoms = mutable.ArrayBuffer.empty[Int]
        var window = 0
        var cycles = 0
        var done = false
        while (!done && cycles < 160) {
          // Exercise a deterministic stall at each architectural window.
          val ready = cycles % 5 != 2
          dut.io.out.ready.poke(ready.B)
          if (dut.io.out.valid.peekBoolean() && ready) {
            dut.io.out.bits.sharedBase
              .expect((0x2000 + window * 128).U)
            val windowAtoms = expectedAtoms.slice(window * 8, window * 8 + 8)
            for (lane <- 0 until 8) {
              val active = lane < windowAtoms.size
              dut.io.out.bits.lanes(lane).valid.expect(active.B)
              if (active) {
                val atom = windowAtoms(lane)
                observedAtoms += atom
                dut.io.out.bits.lanes(lane).globalAddress
                  .expect((0x10000 + atom * 16).U)
                dut.io.out.bits.lanes(lane).globalBytes.expect(16.U)
                dut.io.out.bits.lanes(lane).sharedBytes.expect(16.U)
                // 32B swizzle uses x ^ y within each 128B window.  The
                // shared base is 8KiB aligned, so y is the output-window
                // parity here.
                dut.io.out.bits.lanes(lane).sharedAtomDelta
                  .expect((lane ^ (window & 1)).U)
              }
            }
            done = dut.io.out.bits.last.peekBoolean()
            window += 1
          }
          dut.clock.step()
          cycles += 1
        }
        assert(done,
          s"interleave32 stride $elementStride did not finish")
        assert(observedAtoms.toSeq == expectedAtoms,
          s"interleave32 stride $elementStride split or reordered a pair")
        assert(window == (expectedAtoms.size + 7) / 8)
      }
    }
  }

  "interleave32 trailing empty raw window preserves the final full compacted window" in {
    test(new TmaV2ExecuteWindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      // Each of two outer rows has sixteen raw 32B positions.  Stride eight
      // selects two 32B positions (four lanes) per row, for exactly one dense
      // 128B output window.  The final raw planner window contains only the
      // seven unselected positions following the last row's second pair.
      val words = descriptor(
        TmaV2Spec.DTypeU8, 3,
        Seq(16, 32, 2), Seq(32, 1024), Seq(16, 32, 2),
        elementStrides = Seq(8, 1, 1),
        interleave = TmaV2Spec.Interleave32,
        swizzle = TmaV2Spec.Swizzle32)
      dut.io.in.valid.poke(false.B)
      startExecuteWindow(dut, words, Seq(0, 0, 0))

      var cycles = 0
      var rawWindows = 0
      while (!dut.io.out.valid.peekBoolean() && cycles < 192) {
        if (dut.io.rawPlannerFire.peekBoolean()) {
          rawWindows += 1
        }
        dut.clock.step()
        cycles += 1
      }
      assert(cycles < 192,
        s"compactor did not mark the only dense output as final: " +
          s"rawWindows=$rawWindows " +
          s"active=${dut.io.active.peekBoolean()} " +
          s"plannerProduced=${dut.io.plannerProduced.peekBoolean()} " +
          s"plannerStalled=${dut.io.plannerStalled.peekBoolean()} " +
          s"compiledStatus=${dut.io.compiledStatus.peekInt()} " +
          s"bindStatus=${dut.io.bindStatus.peekInt()}")
      dut.io.out.bits.last.expect(true.B)
      dut.io.out.bits.sharedBase.expect(0x2000.U)
      val held = (0 until 8).map { lane => (
        dut.io.out.bits.lanes(lane).globalAddress.peekInt(),
        dut.io.out.bits.lanes(lane).sharedAtomDelta.peekInt()) }
      for (row <- 0 until 2) {
        for (selected <- 0 until 2) {
          for (half <- 0 until 2) {
            val lane = row * 4 + selected * 2 + half
            val position = selected * 8
            dut.io.out.bits.lanes(lane).valid.expect(true.B)
            dut.io.out.bits.lanes(lane).globalAddress
              .expect((0x10000 + row * 1024 +
                position * 32 + half * 16).U)
            dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(lane.U)
          }
        }
      }
      dut.clock.step(3)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.last.expect(true.B)
      for (lane <- 0 until 8) {
        dut.io.out.bits.lanes(lane).globalAddress.expect(held(lane)._1.U)
        dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(held(lane)._2.U)
      }
      dut.io.out.ready.poke(true.B)
      dut.clock.step()
    }
  }

  "dim0 compactor uses an empty final raw window to close a pending full window" in {
    test(new TmaV2Dim0StrideCompactor) { dut =>
      dut.io.start.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.io.flush.poke(false.B)
      dut.clock.step(2)

      dut.io.start.valid.poke(true.B)
      dut.io.start.bits.sharedBase.poke(0x2000.U)
      dut.io.start.bits.strideMinus1.poke(7.U) // stride eight
      dut.io.start.bits.rawPositions.poke(8.U)
      dut.io.start.bits.interleave.poke(TmaV2Spec.Interleave32.U)
      dut.io.start.bits.swizzle.poke(TmaV2Spec.Swizzle32.U)
      dut.clock.step()
      dut.io.start.valid.poke(false.B)

      for (window <- 0 until 8) {
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.sharedBase.poke(0.U)
        dut.io.in.bits.last.poke((window == 7).B)
        for (lane <- 0 until 8) {
          val ordinal = window * 8 + lane
          dut.io.in.bits.lanes(lane).valid.poke(true.B)
          dut.io.in.bits.lanes(lane).globalAddress
            .poke((0x10000 + ordinal * 16).U)
          dut.io.in.bits.lanes(lane).globalBytes.poke(16.U)
          dut.io.in.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.in.bits.lanes(lane).sharedBytes.poke(16.U)
        }
        dut.io.in.ready.expect(true.B)
        if (window < 7) {
          dut.io.out.valid.expect(false.B)
        } else {
          dut.io.out.valid.expect(true.B)
          dut.io.out.bits.last.expect(true.B)
          for (lane <- 0 until 8) {
            val row = lane / 2
            val half = lane % 2
            val ordinal = row * 16 + half
            dut.io.out.bits.lanes(lane).valid.expect(true.B)
            dut.io.out.bits.lanes(lane).globalAddress
              .expect((0x10000 + ordinal * 16).U)
          }
        }
        dut.clock.step()
      }
      dut.io.in.valid.poke(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  "dim0 compactor resets its mask at a non-divisible raw row boundary" in {
    test(new TmaV2Dim0StrideCompactor) { dut =>
      dut.io.start.valid.poke(false.B)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.io.flush.poke(false.B)
      dut.clock.step(2)

      dut.io.start.valid.poke(true.B)
      dut.io.start.bits.sharedBase.poke(0x2000.U)
      dut.io.start.bits.strideMinus1.poke(2.U) // stride three
      dut.io.start.bits.rawPositions.poke(5.U)
      dut.io.start.bits.interleave.poke(TmaV2Spec.Interleave16.U)
      dut.io.start.bits.swizzle.poke(TmaV2Spec.SwizzleNone.U)
      dut.clock.step()
      dut.io.start.valid.poke(false.B)

      def pokeInput(baseOrdinal: Int, lanes: Int, last: Boolean): Unit = {
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.sharedBase.poke(0.U)
        dut.io.in.bits.last.poke(last.B)
        for (lane <- 0 until 8) {
          val active = lane < lanes
          dut.io.in.bits.lanes(lane).valid.poke(active.B)
          dut.io.in.bits.lanes(lane).globalAddress
            .poke((0x10000 + (baseOrdinal + lane) * 16).U)
          dut.io.in.bits.lanes(lane).globalBytes
            .poke((if (active) 16 else 0).U)
          dut.io.in.bits.lanes(lane).sharedAtomDelta.poke(lane.U)
          dut.io.in.bits.lanes(lane).sharedBytes
            .poke((if (active) 16 else 0).U)
        }
        if (!last) {
          while (!dut.io.in.ready.peekBoolean()) dut.clock.step()
          dut.clock.step()
          dut.io.in.valid.poke(false.B)
        } else {
          // The final partial output is flow-through.  Leave its input valid
          // while the downstream is stalled so the test can inspect the
          // Decoupled-stable combined payload before accepting it.
          dut.io.out.valid.expect(true.B)
        }
      }

      // Raw positions are [0..4][0..4].  With stride three the selected raw
      // atom ordinals must be 0,3,5,8; a free-running mask would incorrectly
      // choose 0,3,6,9.
      pokeInput(0, 8, last = false)
      pokeInput(8, 2, last = true)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.last.expect(true.B)
      for ((ordinal, lane) <- Seq(0, 3, 5, 8).zipWithIndex) {
        dut.io.out.bits.lanes(lane).valid.expect(true.B)
        dut.io.out.bits.lanes(lane).globalAddress
          .expect((0x10000 + ordinal * 16).U)
        dut.io.out.bits.lanes(lane).sharedAtomDelta.expect(lane.U)
      }
      for (lane <- 4 until 8)
        dut.io.out.bits.lanes(lane).valid.expect(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
      dut.io.busy.expect(false.B)
    }
  }

  "kill while the dim0 compactor output is stalled flushes and recovers" in {
    test(new TmaV2ExecuteWindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      val slow = descriptor(
        TmaV2Spec.DTypeU16, 3,
        Seq(8, 64, 4), Seq(16, 1024), Seq(8, 8, 4),
        elementStrides = Seq(2, 1, 1),
        interleave = TmaV2Spec.Interleave16)
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      startExecuteWindow(dut, slow, Seq(0, 0, 0))
      var waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 128) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 128, "strided compactor never reached its stalled output")

      dut.io.kill.poke(true.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step()
      dut.io.kill.poke(false.B)
      for (_ <- 0 until 12) {
        dut.io.out.valid.expect(false.B)
        dut.clock.step()
      }
      dut.io.active.expect(false.B)

      val recovery = descriptor(
        TmaV2Spec.DTypeU8, 1, Seq(16), Seq.empty, Seq(16))
      startExecuteWindow(dut, recovery, Seq(0))
      dut.io.out.ready.poke(true.B)
      waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 96) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 96, "post-compactor-kill command did not recover")
      dut.io.out.bits.last.expect(true.B)
      dut.io.out.bits.lanes(0).globalAddress.expect(0x10000.U)
      dut.clock.step()
    }
  }

  "kill during outer-stride setup drops the command and preserves recovery" in {
    test(new TmaV2ExecuteWindowHarness)
        .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.io.kill.poke(false.B)
      dut.clock.step(2)

      val strided = descriptor(
        TmaV2Spec.DTypeU8, 5,
        Seq(16, 8, 8, 8, 8), Seq(16, 128, 1024, 8192),
        Seq(16, 5, 5, 5, 5),
        elementStrides = Seq(1, 2, 3, 4, 5))
      startExecuteWindow(dut, strided, Seq.fill(5)(0))
      var waited = 0
      while (!dut.io.active.peekBoolean() && waited < 64) {
        dut.io.out.valid.expect(false.B)
        dut.clock.step()
        waited += 1
      }
      assert(waited < 64, "rank-5 outer-stride command never became active")
      dut.clock.step(2)
      dut.io.active.expect(true.B)
      dut.io.out.valid.expect(false.B)

      dut.io.kill.poke(true.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step()
      dut.io.kill.poke(false.B)
      for (_ <- 0 until 12) {
        dut.io.out.valid.expect(false.B)
        dut.clock.step()
      }
      dut.io.active.expect(false.B)

      val unit = descriptor(TmaV2Spec.DTypeU8, 1,
        Seq(16), Seq.empty, Seq(16), elementStrides = Seq(1))
      startExecuteWindow(dut, unit, Seq(0))
      waited = 0
      while (!dut.io.out.valid.peekBoolean() && waited < 64) {
        dut.clock.step()
        waited += 1
      }
      assert(waited < 64, "unit-stride recovery command never produced a window")
      dut.io.out.bits.last.expect(true.B)
      dut.io.out.bits.sharedBase.expect(0x2000.U)
      dut.clock.step()
    }
  }

  "window mappings match the independent C model for stride, swizzle, interleave and OOB" in {
    val cases = Seq(
      (descriptor(TmaV2Spec.DTypeU16, 3, Seq(32, 8, 3),
        Seq(64, 512), Seq(32, 4, 2),
        swizzle = TmaV2Spec.Swizzle64),
        TmaV2Spec.DirectionG2S, Seq(-8, 6, 2),
        "rank3_u16_swizzle64_oob"),
      (descriptor(TmaV2Spec.DTypeFp32, 3, Seq(4, 8, 2),
        Seq(16, 128), Seq(4, 8, 2),
        interleave = TmaV2Spec.Interleave16),
        TmaV2Spec.DirectionG2S, Seq(0, 0, 0),
        "rank3_fp32_interleave16"),
      (descriptor(TmaV2Spec.DTypeFp32, 3, Seq(8, 4, 2),
        Seq(32, 128), Seq(8, 4, 2),
        interleave = TmaV2Spec.Interleave32,
        swizzle = TmaV2Spec.Swizzle32),
        TmaV2Spec.DirectionS2G, Seq(0, 0, 0),
        "rank3_fp32_interleave32_swizzle32"))

    for (((words, direction, coordinates, caseName), index) <-
         cases.zipWithIndex) {
      test(new TmaV2WindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        dut.io.out.ready.poke(false.B)
        dut.io.in.valid.poke(false.B)
        startWindow(dut, words, direction, coordinates)
        val observed = collectWindows(dut, 0x3200 + index)._1
        val expected = golden(caseName)
        assert(observed.data == expected.data,
          s"$caseName data mapping mismatch")
        assert(observed.fill == expected.fill,
          s"$caseName fill mapping mismatch")
      }
    }
  }

  "FP6 exposes twelve global payload bytes in both directions" in {
    for (direction <- Seq(
        TmaV2Spec.DirectionG2S, TmaV2Spec.DirectionS2G)) {
      test(new TmaV2WindowHarness)
          .withAnnotations(Seq(CachingAnnotation)) { dut =>
        val words = descriptor(TmaV2Spec.DTypeB6, 1,
          Seq(128), Seq.empty, Seq(128))
        dut.io.out.ready.poke(true.B)
        dut.io.in.valid.poke(false.B)
        startWindow(dut, words, direction, Seq(0))
        while (!dut.io.out.valid.peekBoolean()) dut.clock.step()
        dut.io.out.bits.last.expect(true.B)
        for (lane <- 0 until 8) {
          dut.io.out.bits.lanes(lane).valid.expect(true.B)
          dut.io.out.bits.lanes(lane).globalBytes.expect(12.U)
        }
      }
    }
  }
}
