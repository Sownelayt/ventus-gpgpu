// Ventus TMA v2 constants used by the gpgpu RTL and Chisel tests.
// This file is maintained independently in the gpgpu project.
package pipeline

import top.parameters.num_block

object TmaV2Spec {
  final val Schema = "ventus-tma-v2"
  final val Opcode = 66
  final val FunctBulkG2S = 1
  final val FunctTensorG2S = 2
  final val FunctBulkS2G = 3
  final val FunctTensorS2G = 4
  final val FunctPrefetchTensormap = 5
  final val TensorMapPrefetchSubop = 0
  final val TensorMapInvalidateSubop = 1
  final val TensorMapInvalidateAllSubop = 2
  final val FunctS2GGroup = 6
  final val FunctMbarrierProxy = 7
  final val S2GGroupCommit = 16
  final val S2GGroupWaitBase = 24
  final val S2GGroupWaitKeepMax = 3
  final val MbarrierInit = 0
  final val MbarrierArriveExpectTx = 1
  final val MbarrierWait = 2
  final val FenceProxyAsyncShared = 3
  final val StatusCsr = 2068
  final val S2GGroupsPerWarp = 4
  final val MbarrierEntriesPerWg = 4
  final val MbarrierEntries = MbarrierEntriesPerWg * num_block
  final val MbarrierBytes = 8
  final val MbarrierAlignment = 8
  final val ReduceCopy = 0
  final val ReduceAdd = 1
  final val ReduceMin = 2
  final val ReduceMax = 3
  final val ReduceAnd = 4
  final val ReduceOr = 5
  final val ReduceXor = 6
  final val ReduceReserved = 7
  // Bulk S2G reduce encodes its element type in inst[28:27].  Tensor
  // reduce obtains the type from the TensorMap descriptor instead.
  final val BulkReduceTypeU32 = 0
  final val BulkReduceTypeS32 = 1
  final val BulkReduceTypeB32 = 2
  final val BulkReduceTypeReserved = 3
  final val ReduceAtomic = true
  final val ReduceRequiresExclusiveDestination = false
  final val StatusOk = 0
  final val StatusInvalidDescriptor = 1
  final val StatusUnsupportedFeature = 2
  final val StatusAddressOverflow = 3
  final val StatusMbarrierProtocol = 4
  final val StatusInvalidGroupOperation = 5
  // The descriptor remains a fixed 128B storage object. Its size is an
  // interface property and is deliberately not encoded in the descriptor.
  final val DescriptorStorageBytes = 128
  final val DescriptorAlignment = 128
  final val DescriptorMagic = BigInt("544d4103", 16)
  final val RankMin = 1
  final val RankMax = 5
  final val AtomBytes = 16
  final val LineBytes = 128
  // windowEntries selects PayloadSlot capacity, while requestEntries selects
  // the narrower in-flight LineContext capacity.
  final val DefaultWindowEntries = 6
  final val DefaultGlobalRequestEntries = 40
  final val DefaultWriteAckEntries = 32
  final val DefaultSharedReadyEntries = 8
  // The one active direction reuses the source namespace: G2S indexes the
  // LineContext table and S2G indexes the compact ack scoreboard. The public
  // source width is the next power of two covering the larger active table.
  def cacheSourceEntries(requestEntries: Int, writeAckEntries: Int): Int = {
    val needed = math.max(requestEntries, writeAckEntries)
    1 << (32 - Integer.numberOfLeadingZeros(needed - 1))
  }
  final val TranslationEntriesPerCommand = 4
  // Payload translations are cached only for one accepted TMA command and
  // are discarded at command completion. Software must not remap, unmap,
  // migrate a touched page, or recycle the command ASID before completion
  // (including its mbarrier/group acknowledgement) becomes observable.
  final val DefaultDescriptorEntries = 4
  final val DefaultCommandQueueEntries = 16
  final val BoxDimMax = 256
  final val HeaderWord = 0
  final val ControlWord = 1
  final val GlobalBaseWord = 2
  final val GlobalDimsWord = 4
  final val GlobalStridesWord = 9
  final val BoxDimsWord = 17
  // CUDA-compatible traversal strides.  Raw Ventus descriptors require a
  // value in 1..8 for every active dimension and zero for every inactive
  // dimension.  Unlike cuTensorMapEncodeTiled, the raw ABI does not silently
  // canonicalize a non-interleaved dim0 value to one.
  final val ElementStridesWord = 22
  final val ElementStrideMin = 1
  final val ElementStrideMax = 8
  final val ReservedWords = 27 until 32
  final val GlobalBaseAlignment = 16
  final val GlobalStrideAlignment = 16
  final val Interleave32Alignment = 32
  final val DirectionG2S = 0
  final val DirectionS2G = 1
  final val InterleaveNone = 0
  final val Interleave16 = 1
  final val Interleave32 = 2
  final val SwizzleNone = 0
  final val Swizzle32 = 1
  final val Swizzle64 = 2
  final val Swizzle128 = 3
  final val Swizzle96 = 4
  final val SwizzleAtom16 = 0
  final val SwizzleAtom32 = 1
  final val SwizzleAtom32Flip8 = 2
  final val SwizzleAtom64 = 3
  final val OobZero = 0
  final val OobNaN = 1
  final val AccessTiled = 0
  final val DTypeU8 = 0
  final val DTypeU16 = 1
  final val DTypeU32 = 2
  final val DTypeS32 = 3
  final val DTypeU64 = 4
  final val DTypeS64 = 5
  final val DTypeFp16 = 6
  final val DTypeFp32 = 7
  final val DTypeFp32Ftz = 8
  final val DTypeFp64 = 9
  final val DTypeBf16 = 10
  final val DTypeTf32 = 11
  final val DTypeTf32Ftz = 12
  final val DTypeB4x16 = 13
  final val DTypeB4x16P64 = 14
  final val DTypeB6 = 15
  final val SupportedDTypes = (0 to 15).toSet
  final val FloatingDTypes = Set(
    DTypeFp16, DTypeFp32, DTypeFp32Ftz, DTypeFp64,
    DTypeBf16, DTypeTf32, DTypeTf32Ftz)
  final val SubByteDTypes = Set(DTypeB4x16, DTypeB4x16P64, DTypeB6)
  final val DTypeBits = Map(
    0 -> 8,
    1 -> 16,
    2 -> 32,
    3 -> 32,
    4 -> 64,
    5 -> 64,
    6 -> 16,
    7 -> 32,
    8 -> 32,
    9 -> 64,
    10 -> 16,
    11 -> 32,
    12 -> 32,
    13 -> 4,
    14 -> 4,
    15 -> 6
  )
}
