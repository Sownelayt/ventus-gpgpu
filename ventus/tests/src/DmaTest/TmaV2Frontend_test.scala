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
    val produced = Output(Bool())
    val stalled = Output(Bool())
  })

  val compiler = Module(new TmaV2DescriptorCompiler)
  val binder = Module(new TmaV2CommandBinder)
  val planner = Module(new TmaV2WindowPlanner)
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
    while (!dut.io.out.valid.peekBoolean() && compileCycles < 14) {
      assert(dut.io.busy.peekBoolean())
      dut.clock.step()
      compileCycles += 1
    }
    assert(compileCycles <= 13,
      s"compiler exceeded the 13-cycle pipelined budget: $compileCycles")
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
      compileOne(dut, interleave16)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.globalStrides(0).expect(16.U)
      dut.io.out.bits.globalStrides(1).expect(128.U)
      dut.io.out.bits.logicalBytes.expect(256.U)
      dut.io.out.bits.interleaveSliceStride.expect(128.U)
      dut.clock.step()

      val interleave32 = descriptor(
        TmaV2Spec.DTypeFp32, 3, Seq(8, 4, 2),
        Seq(32, 128), Seq(8, 4, 2),
        interleave = TmaV2Spec.Interleave32,
        swizzle = TmaV2Spec.Swizzle32)
      compileOne(dut, interleave32)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.logicalBytes.expect(256.U)
      dut.clock.step()

      val unitElementStride = descriptor(
        TmaV2Spec.DTypeU8, 2, Seq(128, 64),
        Seq(128), Seq(128, 64),
        elementStrides = Seq(1, 1))
      compileOne(dut, unitElementStride)
      dut.io.out.bits.status.expect(TmaV2Status.Ok)
      dut.io.out.bits.globalStrides(0).expect(128.U)
      dut.io.out.bits.logicalBytes.expect((128 * 64).U)
      dut.clock.step()

      for (elementStride <- 2 to 8) {
        val unsupportedElementStride = descriptor(
          TmaV2Spec.DTypeU8, 2, Seq(128, 64),
          Seq(128), Seq(128, 64),
          elementStrides = Seq(1, elementStride))
        compileOne(dut, unsupportedElementStride)
        dut.io.out.bits.status.expect(TmaV2Status.UnsupportedFeature)
        dut.clock.step()
      }

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

  "non-unit element-stride placeholder is rejected before binding" in {
    test(new TmaV2CompilerHarness) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(true.B)
      dut.clock.step(2)
      val words = descriptor(TmaV2Spec.DTypeU8, 2, Seq(128, 64),
        Seq(128), Seq(128, 64), elementStrides = Seq(1, 2))
      compileOne(dut, words)
      dut.io.out.bits.status.expect(TmaV2Status.UnsupportedFeature)
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
