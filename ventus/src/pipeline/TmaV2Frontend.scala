package pipeline

import chisel3._
import chisel3.util._

object TmaV2Status {
  val width = 4
  val Ok = 0.U(width.W)
  val BadMagic = 1.U(width.W)
  // Value 2 is reserved by the status encoding.
  val ReservedBits = 3.U(width.W)
  val UnsupportedDType = 4.U(width.W)
  val BadRank = 5.U(width.W)
  val BadLayout = 6.U(width.W)
  val BadAlignment = 7.U(width.W)
  val BadDimension = 8.U(width.W)
  val BadStride = 9.U(width.W)
  val BadCoordinate = 10.U(width.W)
  val AddressOverflow = 11.U(width.W)
  val UnsupportedFeature = 12.U(width.W)
}

class TmaV2Request extends Bundle {
  val copyDirection = UInt(1.W)
  val reduceMode = UInt(3.W)
  val sharedBase = UInt(32.W)
  val coordinates = Vec(TmaV2Spec.RankMax, SInt(32.W))
}

class TmaV2DescriptorPayload extends Bundle {
  // Semantic compaction of the 128B descriptor. lowWords retains the
  // frontend slots (magic/control/base-low/dims/stride-low)
  // while the newly widened fields are stored explicitly.
  val lowWords = Vec(12, UInt(32.W))
  val globalBaseHigh = UInt(32.W)
  val globalStrideHigh = Vec(TmaV2Spec.RankMax - 1, UInt(32.W))
  val boxDims = Vec(TmaV2Spec.RankMax, UInt(32.W))
  val elementStrides = Vec(TmaV2Spec.RankMax, UInt(32.W))
  val reservedBad = Bool()
}

/**
  * Descriptor-only state kept in the compiled store.
  *
  * No command-local address, direction or coordinate is retained here. In
  * particular logicalBytes and the interleave slice stride are computed
  * once per {ASID, TensorMap address}, not once per tensor command.
  * Static failures are entries too: status is cached alongside the
  * successfully compiled fields, without a redundant legal bit.
  */
class TmaV2CompiledDescriptor extends Bundle {
  val status = UInt(TmaV2Status.width.W)
  val dtype = UInt(5.W)
  val rank = UInt(3.W)
  val interleave = UInt(2.W)
  val swizzle = UInt(3.W)
  val oobFill = Bool()
  // Static validation rejects a non-zero high word.  Keeping only the
  // architecturally addressable low word removes 32 bits from every cached
  // descriptor and from every command snapshot.
  val globalBase = UInt(32.W)
  val globalDims = Vec(TmaV2Spec.RankMax, UInt(33.W))
  val globalStrides = Vec(TmaV2Spec.RankMax - 1, UInt(64.W))
  // boxDims contains the number of produced traversal positions, not the raw
  // descriptor box.  The original dim0 extent is retained for the
  // interleave-only atom scanner.
  val boxDims = Vec(TmaV2Spec.RankMax, UInt(9.W))
  val strideMinus1 = Vec(TmaV2Spec.RankMax, UInt(3.W))
  val rawDim0Box = UInt(9.W)
  val interleaveSliceStride = UInt(64.W)
  val logicalBytes = UInt(32.W)
}

/** Cheap fields deliberately derived instead of replicated in cached state. */
object TmaV2DescriptorDerived {
  def dtypeBits(dtype: UInt): UInt = MuxLookup(dtype, 0.U(7.W))(Seq(
    TmaV2Spec.DTypeU8.U -> 8.U,
    TmaV2Spec.DTypeU16.U -> 16.U,
    TmaV2Spec.DTypeU32.U -> 32.U,
    TmaV2Spec.DTypeS32.U -> 32.U,
    TmaV2Spec.DTypeU64.U -> 64.U,
    TmaV2Spec.DTypeS64.U -> 64.U,
    TmaV2Spec.DTypeFp16.U -> 16.U,
    TmaV2Spec.DTypeFp32.U -> 32.U,
    TmaV2Spec.DTypeFp32Ftz.U -> 32.U,
    TmaV2Spec.DTypeFp64.U -> 64.U,
    TmaV2Spec.DTypeBf16.U -> 16.U,
    TmaV2Spec.DTypeTf32.U -> 32.U,
    TmaV2Spec.DTypeTf32Ftz.U -> 32.U,
    TmaV2Spec.DTypeB4x16.U -> 4.U,
    TmaV2Spec.DTypeB4x16P64.U -> 4.U,
    TmaV2Spec.DTypeB6.U -> 6.U))

  def elementByteShift(dtypeBits: UInt): UInt =
    MuxLookup(dtypeBits, 0.U(3.W))(Seq(
      8.U -> 0.U, 16.U -> 1.U, 32.U -> 2.U, 64.U -> 3.U))

  def channelsLog2(interleave: UInt, byteShift: UInt): UInt =
    Mux(interleave === TmaV2Spec.Interleave16.U,
      4.U - byteShift, 5.U - byteShift)
}

/** Command-local state emitted by the coordinate binder. */
class TmaV2BoundCommand extends Bundle {
  val compiled = new TmaV2CompiledDescriptor
  val copyDirection = UInt(1.W)
  val sharedBase = UInt(32.W)
  val coordinates = Vec(TmaV2Spec.RankMax, SInt(32.W))
  val originComponents = Vec(TmaV2Spec.RankMax, SInt(64.W))
  // Metadata is part of the planner token.  There are no live sideband
  // registers whose value can change while a window is under backpressure.
  val reduceMode = UInt(3.W)
}

class TmaV2BindRequest extends Bundle {
  val compiled = new TmaV2CompiledDescriptor
  val request = new TmaV2Request
}

class TmaV2BindResponse extends Bundle {
  val legal = Bool()
  val status = UInt(TmaV2Status.width.W)
  val command = new TmaV2BoundCommand
}

/**
  * Single-lane static descriptor compiler.
  *
  * The rank counter alternates stride-span and logical-size micro-ops, then
  * performs the optional interleave-slice product.  All variable products
  * share `multiply`; rank 1..5 completes in at most eleven busy cycles.
  */
class TmaV2DescriptorCompiler extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new TmaV2DescriptorPayload))
    val out = Decoupled(new TmaV2CompiledDescriptor)
    val busy = Output(Bool())
  })

  val compiling = RegInit(false.B)
  val validatePending = RegInit(false.B)
  val strideSetupActive = RegInit(false.B)
  val strideSetupPending =
    RegInit(0.U(TmaV2Spec.RankMax.W))
  val validationFlags = Reg(UInt(10.W))
  val resultValid = RegInit(false.B)
  val rankCounter = RegInit(1.U(3.W))
  val logicalPhase = RegInit(false.B)
  val finishPhase = RegInit(false.B)
  val result = Reg(new TmaV2CompiledDescriptor)
  val resultWriteValid = WireDefault(false.B)
  val resultWrite = WireDefault(result)
  when(resultWriteValid) {
    result := resultWrite
  }
  // One overflow bit plus the full representable byte span is sufficient.
  // A set overflow bit compares greater than every 64-bit descriptor stride.
  val requiredSpan = Reg(UInt(65.W))
  val logicalBytes = Reg(UInt(64.W))
  val rawBoxDims =
    Reg(Vec(TmaV2Spec.RankMax, UInt(9.W)))

  val mulA = WireDefault(0.U(64.W))
  val mulB = WireDefault(0.U(33.W))
  when(finishPhase) {
    val sliceIndex = Mux(result.rank >= 3.U,
      result.rank - 3.U, 0.U)(1, 0)
    val sliceDimension = Mux(result.rank >= 2.U,
      result.rank - 2.U, 0.U)(2, 0)
    mulA := result.globalStrides(sliceIndex)
    mulB := result.globalDims(sliceDimension)
  }.elsewhen(logicalPhase) {
    mulA := logicalBytes
    mulB := result.boxDims(rankCounter)
  }.otherwise {
    mulA := result.globalStrides((rankCounter - 1.U)(1, 0))
    mulB := result.globalDims(rankCounter)(32, 0)
  }

  // Build the unsigned 64x33 product from 16x16 limbs.  The previous direct
  // multiplier was a separate failing setup family even after validation
  // was registered.  A one-entry result pipeline overlaps adjacent compiler
  // micro-ops, so cold compile latency grows by only the final drain cycle.
  val mulALimbs = Wire(Vec(4, UInt(16.W)))
  val mulBLimbs = Wire(Vec(3, UInt(16.W)))
  for (limb <- 0 until 4) {
    mulALimbs(limb) := mulA(limb * 16 + 15, limb * 16)
  }
  mulBLimbs(0) := mulB(15, 0)
  mulBLimbs(1) := mulB(31, 16)
  mulBLimbs(2) := Cat(0.U(15.W), mulB(32))
  val multiplyTerms = (0 until 4).flatMap { aLimb =>
    (0 until 3).map { bLimb =>
      val partial = mulALimbs(aLimb) * mulBLimbs(bLimb)
      (partial.pad(96) << ((aLimb + bLimb) * 16))(95, 0)
    }
  }
  def balancedProductSum(terms: Seq[UInt]): UInt = {
    if (terms.size == 1) terms.head
    else balancedProductSum(terms.grouped(2).map {
      case Seq(left, right) => (left +& right)(95, 0)
      case Seq(left) => left
    }.toSeq)
  }
  val launchedMultiply = balancedProductSum(multiplyTerms)
  val multiplyValid = RegInit(false.B)
  // No consumer needs all 96 product bits resident.  Keep low64 plus the
  // two exact overflow predicates used by stride/finish and logical-size
  // validation.  A high product saturates requiredSpan at 2^64, which is
  // greater than every representable following stride.
  val multiplyResult = Reg(UInt(64.W))
  val multiplyHighNonzero = Reg(Bool())
  val multiplyAbove32 = Reg(Bool())
  val multiplyLogical = Reg(Bool())
  val multiplyFinish = Reg(Bool())
  val multiplyRank = Reg(UInt(3.W))

  io.in.ready :=
    !validatePending &&
      !strideSetupActive && !compiling && !multiplyValid && !resultValid
  io.out.valid := resultValid
  io.out.bits := result
  io.busy := validatePending || strideSetupActive || compiling ||
    multiplyValid

  val control = io.in.bits.lowWords(1)
  val inputDtype = control(4, 0)
  val inputRank = control(7, 5)
  val inputInterleave = control(9, 8)
  val inputSwizzle = control(12, 10)
  val inputAtomicity = control(14, 13)
  val inputL2Promotion = control(17, 16)
  val inputOobFill = control(18)
  val inputAccessMode = control(20, 19)
  val inputDtypeBits = MuxLookup(inputDtype, 0.U)(Seq(
    TmaV2Spec.DTypeU8.U -> 8.U,
    TmaV2Spec.DTypeU16.U -> 16.U,
    TmaV2Spec.DTypeU32.U -> 32.U,
    TmaV2Spec.DTypeS32.U -> 32.U,
    TmaV2Spec.DTypeU64.U -> 64.U,
    TmaV2Spec.DTypeS64.U -> 64.U,
    TmaV2Spec.DTypeFp16.U -> 16.U,
    TmaV2Spec.DTypeFp32.U -> 32.U,
    TmaV2Spec.DTypeFp32Ftz.U -> 32.U,
    TmaV2Spec.DTypeFp64.U -> 64.U,
    TmaV2Spec.DTypeBf16.U -> 16.U,
    TmaV2Spec.DTypeTf32.U -> 32.U,
    TmaV2Spec.DTypeTf32Ftz.U -> 32.U,
    TmaV2Spec.DTypeB4x16.U -> 4.U,
    TmaV2Spec.DTypeB4x16P64.U -> 4.U,
    TmaV2Spec.DTypeB6.U -> 6.U))
  val supportedDtype = inputDtype <= TmaV2Spec.DTypeB6.U
  val floatingDtype = Seq(
    TmaV2Spec.DTypeFp16, TmaV2Spec.DTypeFp32,
    TmaV2Spec.DTypeFp32Ftz, TmaV2Spec.DTypeFp64,
    TmaV2Spec.DTypeBf16, TmaV2Spec.DTypeTf32,
    TmaV2Spec.DTypeTf32Ftz).map(inputDtype === _.U).reduce(_ || _)
  val subByteDtype = inputDtype >= TmaV2Spec.DTypeB4x16.U
  val paddedSubByteDtype =
    inputDtype === TmaV2Spec.DTypeB4x16P64.U ||
      inputDtype === TmaV2Spec.DTypeB6.U
  def rowBytesForBits(elements: UInt, dtypeBits: UInt): UInt =
    MuxLookup(dtypeBits, 0.U(64.W))(Seq(
    4.U -> ((elements + 1.U) >> 1),
    6.U -> ((((elements << 2) + (elements << 1)) + 7.U) >> 3),
    8.U -> elements,
    16.U -> (elements << 1),
    32.U -> (elements << 2),
    64.U -> (elements << 3)))
  def rowBytes(elements: UInt): UInt =
    rowBytesForBits(elements, inputDtypeBits)
  def interleaveBytes(positions: UInt, interleave: UInt): UInt =
    Mux(interleave === TmaV2Spec.Interleave16.U,
      (positions.pad(64) << 4)(63, 0),
      (positions.pad(64) << 5)(63, 0))
  def inputChannelSliceMetadata(dimension: Int): Bool =
    inputInterleave =/= TmaV2Spec.InterleaveNone.U &&
      (dimension + 2).U === inputRank

  // A valid box is at most 256 and an element stride is in 1..8.  Use one
  // nine-step restoring divider during the non-unit-stride setup state.  In
  // particular, do not apply `/` to the raw 32-bit descriptor fields: that
  // would replicate wide constant dividers once per dimension in the input
  // capture path.
  def ceilDivElementStride(elements: UInt, stride: UInt): UInt = {
    val roundedWide =
      elements.pad(10) +& stride.pad(10) - 1.U
    val rounded = roundedWide(8, 0)
    var remainder = 0.U(5.W)
    var quotient = 0.U(9.W)
    for (bit <- 8 to 0 by -1) {
      val shifted = Cat(remainder(3, 0), rounded(bit))
      val subtract = shifted >= stride.pad(5)
      remainder = Mux(subtract,
        (shifted - stride.pad(5))(4, 0), shifted)
      quotient = quotient | (subtract.asUInt << bit)
    }
    quotient
  }
  val rawBoxRowBytes = rowBytes(io.in.bits.boxDims(0))
  val initialLogicalBytes = Mux(
    inputInterleave === TmaV2Spec.InterleaveNone.U,
    rawBoxRowBytes,
    interleaveBytes(io.in.bits.boxDims(0), inputInterleave))
  val dim0Elements = Mux(io.in.bits.lowWords(3) === 0.U,
    (BigInt(1) << 32).U(33.W), io.in.bits.lowWords(3))
  val dim0RequiredSpan = MuxLookup(inputDtypeBits, 0.U(65.W))(Seq(
    4.U -> ((dim0Elements.pad(65) + 1.U) >> 1),
    6.U -> {
      val dimension = dim0Elements.pad(65)
      (((dimension << 2) + (dimension << 1) + 7.U) >> 3)
    },
    8.U -> dim0Elements.pad(65),
    16.U -> (dim0Elements.pad(65) << 1),
    32.U -> (dim0Elements.pad(65) << 2),
    64.U -> (dim0Elements.pad(65) << 3)))
  val activeFieldBad = (0 until TmaV2Spec.RankMax).map { dimension =>
    val active = dimension.U < inputRank
    val activeBad = io.in.bits.boxDims(dimension) === 0.U ||
      io.in.bits.boxDims(dimension) > TmaV2Spec.BoxDimMax.U
    val inactiveBad = io.in.bits.lowWords(3 + dimension) =/= 0.U ||
      io.in.bits.boxDims(dimension) =/= 0.U ||
      io.in.bits.elementStrides(dimension) =/= 0.U ||
      (if (dimension < TmaV2Spec.RankMax - 1)
        io.in.bits.lowWords(8 + dimension) =/= 0.U ||
          io.in.bits.globalStrideHigh(dimension) =/= 0.U
       else false.B)
    Mux(active, activeBad, inactiveBad)
  }.reduce(_ || _)
  val inactiveElementStrideBad =
    (0 until TmaV2Spec.RankMax).map { dimension =>
      inputRank >= TmaV2Spec.RankMin.U &&
        inputRank <= TmaV2Spec.RankMax.U &&
        dimension.U >= inputRank &&
        io.in.bits.elementStrides(dimension) =/= 0.U
    }.reduce(_ || _)
  val badElementStride =
    (0 until TmaV2Spec.RankMax).map { dimension =>
      val active = dimension.U < inputRank
      val stride = io.in.bits.elementStrides(dimension)
      active && (stride < TmaV2Spec.ElementStrideMin.U ||
        stride > TmaV2Spec.ElementStrideMax.U ||
        (if (dimension == 0)
          inputInterleave === TmaV2Spec.InterleaveNone.U && stride =/= 1.U
         else false.B))
    }.reduce(_ || _)
  val inputStrideSetupPending = VecInit(
    (0 until TmaV2Spec.RankMax).map { dimension =>
      val stride = io.in.bits.elementStrides(dimension)
      dimension.U < inputRank &&
        stride > TmaV2Spec.ElementStrideMin.U &&
        stride <= TmaV2Spec.ElementStrideMax.U &&
        !inputChannelSliceMetadata(dimension) &&
        (if (dimension == 0)
          inputInterleave =/= TmaV2Spec.InterleaveNone.U
         else true.B)
    }).asUInt
  val unusedStrideBad =
    (0 until TmaV2Spec.RankMax - 1).map { dimension =>
      (dimension + 1).U >= inputRank &&
        (io.in.bits.lowWords(8 + dimension) =/= 0.U ||
          io.in.bits.globalStrideHigh(dimension) =/= 0.U)
    }.reduce(_ || _)
  val swizzleSpan = 16.U << inputSwizzle
  val badLayout = inputInterleave > TmaV2Spec.Interleave32.U ||
    (inputInterleave =/= TmaV2Spec.InterleaveNone.U && inputRank < 3.U) ||
    (inputInterleave === TmaV2Spec.Interleave32.U &&
      inputSwizzle =/= TmaV2Spec.Swizzle32.U) ||
    (inputSwizzle =/= TmaV2Spec.SwizzleNone.U &&
      rawBoxRowBytes > swizzleSpan)
  val badSubByte = subByteDtype &&
    (inputOobFill =/= TmaV2Spec.OobZero.U ||
      inputInterleave =/= TmaV2Spec.InterleaveNone.U ||
      (inputDtype === TmaV2Spec.DTypeB6.U &&
        !(inputSwizzle === TmaV2Spec.SwizzleNone.U ||
          inputSwizzle === TmaV2Spec.Swizzle128.U)) ||
      (inputDtype === TmaV2Spec.DTypeB4x16P64.U &&
        !(inputSwizzle === TmaV2Spec.SwizzleNone.U ||
          inputSwizzle === TmaV2Spec.Swizzle128.U)) ||
      ((inputDtype === TmaV2Spec.DTypeB4x16P64.U ||
        inputDtype === TmaV2Spec.DTypeB6.U) &&
        (io.in.bits.boxDims(0) =/= 128.U ||
          io.in.bits.lowWords(3)(6, 0) =/= 0.U)))
  // CUDA's packed, unpadded 4-bit layout addresses pairs of elements as
  // bytes. An even dim0 prevents an architectural half-byte tensor edge;
  // the command binder separately enforces the 16B bounding-box origin.
  val badPackedFp4Dimension =
    inputDtype === TmaV2Spec.DTypeB4x16.U &&
      io.in.bits.lowWords(3)(0)
  val badOob = inputOobFill > TmaV2Spec.OobNaN.U ||
    (inputOobFill === TmaV2Spec.OobNaN.U && !floatingDtype)
  val alignmentMask = Mux(paddedSubByteDtype ||
    inputInterleave === TmaV2Spec.Interleave32.U, 31.U, 15.U)
  val unsupportedFeatureBad = inputL2Promotion =/= 0.U ||
      inputAccessMode =/= TmaV2Spec.AccessTiled.U ||
      inputAtomicity =/= TmaV2Spec.SwizzleAtom16.U ||
      inputSwizzle > TmaV2Spec.Swizzle128.U ||
      badOob || badSubByte
  val inputValidationFlags = VecInit(Seq(
    io.in.bits.lowWords(0) =/= TmaV2Spec.DescriptorMagic.U,
    control(31, 21) =/= 0.U || control(15) ||
      io.in.bits.reservedBad || unusedStrideBad ||
      inactiveElementStrideBad,
    !supportedDtype,
    inputRank < TmaV2Spec.RankMin.U ||
      inputRank > TmaV2Spec.RankMax.U,
    unsupportedFeatureBad,
    badLayout,
    (io.in.bits.lowWords(2) & alignmentMask) =/= 0.U,
    activeFieldBad || badPackedFp4Dimension || rawBoxRowBytes === 0.U ||
      rawBoxRowBytes(3, 0) =/= 0.U,
    badElementStride,
    io.in.bits.globalBaseHigh =/= 0.U))

  when(io.in.fire) {
    validationFlags := inputValidationFlags.asUInt
    validatePending := true.B
    resultWriteValid := true.B
    resultWrite := 0.U.asTypeOf(new TmaV2CompiledDescriptor)
    resultWrite.status := TmaV2Status.Ok
    resultWrite.dtype := inputDtype
    resultWrite.rank := inputRank
    resultWrite.interleave := inputInterleave
    resultWrite.swizzle := inputSwizzle
    resultWrite.oobFill := inputOobFill === TmaV2Spec.OobNaN.U
    resultWrite.globalBase := io.in.bits.lowWords(2)
    resultWrite.logicalBytes := initialLogicalBytes(31, 0)
    resultWrite.rawDim0Box := io.in.bits.boxDims(0)(8, 0)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      resultWrite.globalDims(dimension) := Mux(
        io.in.bits.lowWords(3 + dimension) === 0.U,
        (BigInt(1) << 32).U, io.in.bits.lowWords(3 + dimension))
      resultWrite.boxDims(dimension) := Mux(
        inputChannelSliceMetadata(dimension), 1.U,
        io.in.bits.boxDims(dimension)(8, 0))
      resultWrite.strideMinus1(dimension) := Mux(
        dimension.U < inputRank,
        (io.in.bits.elementStrides(dimension) - 1.U)(2, 0), 0.U)
      if (dimension < TmaV2Spec.RankMax - 1) {
        resultWrite.globalStrides(dimension) := Cat(
          io.in.bits.globalStrideHigh(dimension),
          io.in.bits.lowWords(8 + dimension))
      }
      rawBoxDims(dimension) := io.in.bits.boxDims(dimension)(8, 0)
    }
    requiredSpan := dim0RequiredSpan
    logicalBytes := initialLogicalBytes
    strideSetupPending := inputStrideSetupPending
  }

  // Each potentially broad predicate first ends at one flag bit. The
  // priority encoder therefore sees only ten registered inputs instead of
  // descriptor fields, dimension comparisons and layout decoding.
  val capturedStatus = WireDefault(TmaV2Status.Ok)
  when(validationFlags(0)) {
    capturedStatus := TmaV2Status.BadMagic
  }.elsewhen(validationFlags(1)) {
    capturedStatus := TmaV2Status.ReservedBits
  }.elsewhen(validationFlags(2)) {
    capturedStatus := TmaV2Status.UnsupportedDType
  }.elsewhen(validationFlags(3)) {
    capturedStatus := TmaV2Status.BadRank
  }.elsewhen(validationFlags(4)) {
    capturedStatus := TmaV2Status.UnsupportedFeature
  }.elsewhen(validationFlags(5)) {
    capturedStatus := TmaV2Status.BadLayout
  }.elsewhen(validationFlags(6)) {
    capturedStatus := TmaV2Status.BadAlignment
  }.elsewhen(validationFlags(7)) {
    capturedStatus := TmaV2Status.BadDimension
  }.elsewhen(validationFlags(8)) {
    capturedStatus := TmaV2Status.BadStride
  }.elsewhen(validationFlags(9)) {
    capturedStatus := TmaV2Status.AddressOverflow
  }
  when(validatePending) {
    // The broad validation predicates were captured on the preceding edge;
    // this compact priority decode can commit status and start the compiler
    // directly without a second initialization-only state.
    resultWriteValid := true.B
    resultWrite.status := capturedStatus
    rankCounter := 1.U
    logicalPhase := false.B
    finishPhase := false.B
    validatePending := false.B
    when(strideSetupPending.orR) {
      strideSetupActive := true.B
      compiling := false.B
    }.otherwise {
      compiling := true.B
    }
  }

  val strideSetupDimension = PriorityEncoder(strideSetupPending)
  val strideSetupStride =
    result.strideMinus1(strideSetupDimension).pad(4) + 1.U(4.W)
  val strideSetupBox = rawBoxDims(strideSetupDimension)
  val strideSetupEffective =
    ceilDivElementStride(strideSetupBox, strideSetupStride)
  val strideSetupGlobal = result.globalStrides(
    (strideSetupDimension - 1.U)(1, 0))
  val unsigned64Max = (BigInt(1) << 64) - 1
  val strideSetupMaxInput = MuxLookup(
    strideSetupStride, unsigned64Max.U(64.W))(
      (2 to TmaV2Spec.ElementStrideMax).map { factor =>
        factor.U -> (unsigned64Max / factor).U(64.W)
      })
  // CUDA limits global strides to less than 2^40, but the Ventus raw ABI
  // retains all 64 bits.  Reject a raw outer stride that would wrap the
  // Execute shift/add multiplier instead of silently changing its address.
  val strideSetupAddressOverflow = strideSetupDimension =/= 0.U &&
    strideSetupGlobal > strideSetupMaxInput
  val strideSetupBit =
    (1.U(TmaV2Spec.RankMax.W) << strideSetupDimension)(
      TmaV2Spec.RankMax - 1, 0)
  val strideSetupRemaining = strideSetupPending & ~strideSetupBit

  when(strideSetupActive) {
    resultWriteValid := true.B
    resultWrite.boxDims(strideSetupDimension) := strideSetupEffective
    when(result.status === TmaV2Status.Ok && strideSetupAddressOverflow) {
      resultWrite.status := TmaV2Status.AddressOverflow
    }
    when(strideSetupDimension === 0.U) {
      val effectiveRowBytes = Mux(
        result.interleave === TmaV2Spec.InterleaveNone.U,
        rowBytesForBits(
          strideSetupEffective,
          TmaV2DescriptorDerived.dtypeBits(result.dtype)),
        interleaveBytes(strideSetupEffective, result.interleave))
      logicalBytes := effectiveRowBytes
      resultWrite.logicalBytes := effectiveRowBytes(31, 0)
    }
    strideSetupPending := strideSetupRemaining
    when(!strideSetupRemaining.orR) {
      strideSetupActive := false.B
      compiling := true.B
    }
  }

  // Launch one micro-op every cycle and advance the small control sequence.
  // The registered result from the preceding launch commits below.
  multiplyValid := false.B
  when(compiling) {
    multiplyValid := true.B
    multiplyResult := launchedMultiply(63, 0)
    multiplyHighNonzero := launchedMultiply(95, 64).orR
    multiplyAbove32 := launchedMultiply(95, 32).orR
    multiplyLogical := logicalPhase
    multiplyFinish := finishPhase
    multiplyRank := rankCounter
    when(finishPhase) {
      compiling := false.B
    }.elsewhen(!logicalPhase) {
      logicalPhase := true.B
    }.otherwise {
      when(rankCounter === (TmaV2Spec.RankMax - 1).U) {
        finishPhase := true.B
      }.otherwise {
        rankCounter := rankCounter + 1.U
        logicalPhase := false.B
      }
    }
  }

  when(multiplyValid) {
    when(multiplyFinish) {
      resultWriteValid := true.B
      resultWrite.interleaveSliceStride := Mux(
        result.interleave === TmaV2Spec.InterleaveNone.U,
        0.U, multiplyResult)
      resultWrite.logicalBytes := logicalBytes(31, 0)
      when(result.status === TmaV2Status.Ok &&
          result.interleave =/= TmaV2Spec.InterleaveNone.U &&
          multiplyHighNonzero) {
        resultWrite.status := TmaV2Status.AddressOverflow
      }
      resultValid := true.B
    }.elsewhen(!multiplyLogical) {
      val active = multiplyRank < result.rank
      val stride = result.globalStrides((multiplyRank - 1.U)(1, 0))
      val resultDtypeBits = TmaV2DescriptorDerived.dtypeBits(result.dtype)
      val paddedSubByte =
        result.dtype === TmaV2Spec.DTypeB4x16P64.U ||
          result.dtype === TmaV2Spec.DTypeB6.U
      val alignMask = Mux(
        paddedSubByte || result.interleave === TmaV2Spec.Interleave32.U,
        31.U, 15.U)
      when(active && result.status === TmaV2Status.Ok &&
          (stride === 0.U || (stride & alignMask) =/= 0.U ||
            requiredSpan(64) || stride < requiredSpan(63, 0))) {
        resultWriteValid := true.B
        resultWrite.status := TmaV2Status.BadStride
      }
      when(active) {
        requiredSpan := Cat(multiplyHighNonzero, multiplyResult)
      }
    }.otherwise {
      val active = multiplyRank < result.rank
      when(active) {
        logicalBytes := multiplyResult
        resultWriteValid := true.B
        resultWrite.logicalBytes := multiplyResult(31, 0)
        when(result.status === TmaV2Status.Ok &&
            multiplyAbove32) {
          resultWrite.status := TmaV2Status.AddressOverflow
        }
      }
    }
  }
  when(io.out.fire) {
    resultValid := false.B
  }
}

/**
  * Binds command-local coordinates to a compiled descriptor.
  *
  * One coordinate product is issued per cycle. Static descriptor products
  * are consumed from the compiled entry. The response remains stable until
  * accepted.
  */
class TmaV2CommandBinder extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new TmaV2BindRequest))
    val out = Decoupled(new TmaV2BindResponse)
    val bindCycle = Output(Bool())
  })

  val launching = RegInit(false.B)
  val multiplyValid = RegInit(false.B)
  val assembleValid = RegInit(false.B)
  val done = RegInit(false.B)
  val dimension = RegInit(0.U(3.W))
  val origins = Reg(Vec(TmaV2Spec.RankMax, SInt(64.W)))
  val originTotal = Reg(SInt(68.W))
  val finalStatus = Reg(UInt(TmaV2Status.width.W))

  // Split the low-64-bit product into 16-bit limbs: four unsigned
  // low-limb products plus three signed high-limb products are sufficient
  // modulo 2^64.  The largest multiplier is now only 17x17, and dimensions
  // still launch at II=1.
  val multiplyDimension = Reg(UInt(3.W))
  val multiplyLast = Reg(Bool())
  val multiplyLowSum = Reg(UInt(68.W))
  val multiplyHighSum = Reg(UInt(68.W))
  val multiplyAddend = Reg(SInt(64.W))
  val assembleDimension = Reg(UInt(3.W))
  val assembleLast = Reg(Bool())
  val assembleProduct = Reg(SInt(64.W))
  val assembleAddend = Reg(SInt(64.W))

  // The LookaheadSlot owns and holds the input request from start through
  // response acceptance. The Binder
  // retains only its rank counter and five dynamic products instead of a
  // second ~800-bit descriptor/coordinate snapshot.
  val binding = io.in.bits
  val compiled = binding.compiled
  val compiledDtypeBits = TmaV2DescriptorDerived.dtypeBits(compiled.dtype)
  val compiledByteShift =
    TmaV2DescriptorDerived.elementByteShift(compiledDtypeBits)
  val coordinate = binding.request.coordinates(dimension)
  val strideIndex = Mux(dimension === 0.U, 0.U, dimension - 1.U)(1, 0)
  val dim0Origin = MuxLookup(compiledDtypeBits,
    (coordinate.pad(64) << compiledByteShift).asSInt)(Seq(
      4.U -> (coordinate >> 1).pad(64),
      6.U -> (((coordinate.pad(64) << 1) +
        coordinate.pad(64)) >> 2).asSInt))
  val interleaveAtomBytes = Mux(
    compiled.interleave === TmaV2Spec.Interleave16.U,
    16.U(64.W), 32.U(64.W))

  val launchUsesMultiply = dimension =/= 0.U ||
    compiled.interleave =/= TmaV2Spec.InterleaveNone.U
  val launchMultiplicand = coordinate
  val launchStride = Mux(
    dimension === 0.U,
    interleaveAtomBytes,
    compiled.globalStrides(strideIndex))
  val multiplicand33 =
    Cat(launchMultiplicand(31), launchMultiplicand.asUInt).asSInt
  val multiplicandLow16 = multiplicand33.asUInt(15, 0)
  val multiplicandHigh17 = multiplicand33.asUInt(32, 16).asSInt
  val strideLimbs = Wire(Vec(4, UInt(16.W)))
  for (limb <- 0 until 4) {
    strideLimbs(limb) := launchStride(limb * 16 + 15, limb * 16)
  }
  val launchLowProducts = Wire(Vec(4, UInt(32.W)))
  val launchHighProducts = Wire(Vec(3, SInt(34.W)))
  for (limb <- 0 until 4) {
    launchLowProducts(limb) := multiplicandLow16 * strideLimbs(limb)
    if (limb < 3) {
      launchHighProducts(limb) := multiplicandHigh17 *
        Cat(0.U(1.W), strideLimbs(limb)).asSInt
    }
  }
  val launchProductTerms = Wire(Vec(7, UInt(68.W)))
  for (limb <- 0 until 4) {
    launchProductTerms(limb) :=
      (launchLowProducts(limb).pad(68) << (limb * 16))(67, 0)
    if (limb < 3) {
      launchProductTerms(4 + limb) :=
        (launchHighProducts(limb).pad(68).asUInt <<
          ((limb + 1) * 16))(67, 0)
    }
  }
  val launchLow01 =
    (launchProductTerms(0) +& launchProductTerms(1))(67, 0)
  val launchLow23 =
    (launchProductTerms(2) +& launchProductTerms(3))(67, 0)
  val launchLowSum = (launchLow01 +& launchLow23)(67, 0)
  val launchHigh01 =
    (launchProductTerms(4) +& launchProductTerms(5))(67, 0)
  val launchHighSum =
    (launchHigh01 +& launchProductTerms(6))(67, 0)
  val launchAddend = 0.S(64.W)

  io.in.ready :=
    !launching && !multiplyValid && !assembleValid && !done
  io.out.valid := done
  io.out.bits := 0.U.asTypeOf(new TmaV2BindResponse)
  io.out.bits.command.compiled := compiled
  io.out.bits.command.copyDirection := binding.request.copyDirection
  io.out.bits.command.sharedBase := binding.request.sharedBase
  io.out.bits.command.coordinates := binding.request.coordinates
  io.out.bits.command.originComponents := origins
  io.out.bits.command.reduceMode := binding.request.reduceMode
  io.bindCycle := launching || multiplyValid || assembleValid

  val reduceRequested =
    binding.request.reduceMode =/= TmaV2Spec.ReduceCopy.U
  val reduceDTypeLegal =
    compiled.dtype === TmaV2Spec.DTypeU32.U ||
      compiled.dtype === TmaV2Spec.DTypeS32.U
  val badReduce = reduceRequested &&
    (binding.request.copyDirection =/= TmaV2Spec.DirectionS2G.U ||
      !reduceDTypeLegal ||
      binding.request.reduceMode === TmaV2Spec.ReduceReserved.U)
  val activeNegative = (0 until TmaV2Spec.RankMax).map { d =>
    d.U < compiled.rank && binding.request.coordinates(d) < 0.S
  }.reduce(_ || _)
  val badNegative = binding.request.copyDirection ===
    TmaV2Spec.DirectionS2G.U && activeNegative
  val paddedFp4 = compiled.dtype === TmaV2Spec.DTypeB4x16P64.U
  val paddedFp6 = compiled.dtype === TmaV2Spec.DTypeB6.U
  val badSubByteCommand =
    (paddedFp4 &&
      binding.request.copyDirection =/= TmaV2Spec.DirectionG2S.U) ||
      ((paddedFp4 || paddedFp6) &&
        binding.request.coordinates(0).asUInt(6, 0) =/= 0.U)
  val badOobDirection =
    compiled.oobFill &&
      binding.request.copyDirection === TmaV2Spec.DirectionS2G.U
  val badPackedFp4Coordinate =
    compiled.dtype === TmaV2Spec.DTypeB4x16.U &&
      binding.request.coordinates(0).asUInt(4, 0).orR
  io.out.bits.status := finalStatus
  io.out.bits.legal := finalStatus === TmaV2Status.Ok

  when(io.in.fire) {
    origins.foreach(_ := 0.S)
    originTotal := io.in.bits.compiled.globalBase.zext
    dimension := 0.U
    launching := true.B
  }

  // Stage 1: launch one rank dimension per cycle.
  multiplyValid := launching
  when(launching) {
    multiplyDimension := dimension
    multiplyLast := dimension + 1.U >= compiled.rank
    multiplyLowSum := Mux(
      launchUsesMultiply, launchLowSum, dim0Origin.pad(68).asUInt)
    multiplyHighSum := Mux(launchUsesMultiply, launchHighSum, 0.U)
    multiplyAddend := launchAddend
    when(dimension + 1.U >= compiled.rank) {
      launching := false.B
    }.otherwise {
      dimension := dimension + 1.U
    }
  }

  // Stage 2: assemble the low 64 bits of the signed 32x64 product.  The
  // architected address checks consume this same two's-complement result.
  assembleValid := multiplyValid
  when(multiplyValid) {
    assembleDimension := multiplyDimension
    assembleLast := multiplyLast
    val product = (multiplyLowSum +& multiplyHighSum)(67, 0)
    assembleProduct := product(63, 0).asSInt
    assembleAddend := multiplyAddend
  }

  // Stage 3: commit one origin and accumulate the command origin for the
  // final dynamic overflow check. Interleaved dim0 was multiplied directly
  // by its 16B/32B atom size in stage 1, so it needs no lane addend here.
  when(assembleValid) {
    val completedOriginWide = assembleProduct + assembleAddend
    val completedOrigin =
      completedOriginWide.asUInt(63, 0).asSInt
    val nextOriginTotal =
      originTotal + completedOrigin.pad(68)
    origins(assembleDimension) := completedOrigin
    originTotal := nextOriginTotal
    when(assembleLast) {
      val dynamicOriginOverflow = nextOriginTotal < 0.S ||
        nextOriginTotal.asUInt > "hffffffff".U
      val badBoundingBoxAlignment =
        nextOriginTotal.asUInt(3, 0).orR || badPackedFp4Coordinate
      val dynamicStatus = WireDefault(compiled.status)
      when(compiled.status === TmaV2Status.Ok) {
        dynamicStatus := TmaV2Status.Ok
        when(binding.request.sharedBase(6, 0) =/= 0.U) {
          dynamicStatus := TmaV2Status.BadAlignment
        }.elsewhen(badNegative) {
          dynamicStatus := TmaV2Status.BadCoordinate
        }.elsewhen(badReduce || badSubByteCommand || badOobDirection) {
          dynamicStatus := TmaV2Status.UnsupportedFeature
        }.elsewhen(dynamicOriginOverflow) {
          dynamicStatus := TmaV2Status.AddressOverflow
        }.elsewhen(badBoundingBoxAlignment) {
          dynamicStatus := TmaV2Status.BadAlignment
        }
      }
      finalStatus := dynamicStatus
      done := true.B
    }
  }
  when(io.out.fire) {
    done := false.B
  }
}
