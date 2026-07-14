/*
 * Shared-memory to global-memory DMA datapath for CP_ASYNC_BULK_S2G.
 *
 * This implementation keeps the external DMA_core interface unchanged, but
 * internally uses bounded line/read/ack entries so shared reads, TLB requests,
 * L2 Puts, and AccessAck completion can overlap.
 */
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
  val data = Vec(num_thread, UInt(xLen.W))
  val activeMask = Vec(num_thread, Bool())
}

class S2GLineTask extends Bundle {
  val wid = UInt(depth_warp.W)
  val group = UInt(log2Ceil(dma_group_entries).W)
  val asid = UInt(SV32.asidLen.W)
  val src = UInt(xLen.W)
  val dst = UInt(xLen.W)
  val bytes = UInt((log2Ceil(l2cacheline) + 1).W)
  val dstWordStride = UInt(log2Ceil(dcache_BlockWords + 1).W)
  val swizzleMode = UInt(2.W)
  val swizzleBase = UInt(xLen.W)
  val swizzleRow = UInt(3.W)
  val earlyRelease = Bool()
  val first = Bool()
  val last = Bool()
}

class DmaS2G(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val from_fifo = Flipped(DecoupledIO(new vExeData))
    val line_task = Flipped(DecoupledIO(new S2GLineTask))
    val shared_req = DecoupledIO(new ShareMemCoreReq_np)
    val shared_rsp = Flipped(DecoupledIO(new DmaSharedRsp))
    val to_l2TLB = DecoupledIO(new L1TlbReq(SV32))
    val from_l2TLB = Flipped(DecoupledIO(new L1TlbRsp(SV32)))
    val to_l2cache = DecoupledIO(new DCacheMemReq_p)
    val from_l2cache = Flipped(DecoupledIO(new DCacheMemRsp))
    val inst_complete = DecoupledIO(new DmaCompletion)
    val perfEnable = Input(Bool())
    val perfReset = Input(Bool())
    val perf = if (PMU_DMA_S2G) Some(Output(new S2GPerfCounters)) else None
  })

  val lineOffsetBits = log2Ceil(l2cacheline)
  val lineByteCountWidth = lineOffsetBits + 1
  val wordOffsetBits = log2Ceil(dma_aligned_bulk)
  val sharedSetIdxHi = log2Ceil(sharedmem_depth) + dcache_BlockOffsetBits + dcache_WordOffsetBits - 1
  val sharedSetIdxLo = dcache_BlockOffsetBits + dcache_WordOffsetBits
  val dmaSourceLowBits = l1cache_sourceBits - log2Ceil(max_dma_tag) - log2Ceil(max_dma_inst)
  require(dmaSourceLowBits > 1, "DMA S2G source encoding needs a distinct non-meta low-bit pattern")
  require(lsu_nMshrEntry > 2, "DMA S2G dynamic shared-read IDs reserve 0/1 for legacy bulk/tensor routing")

  val lineEntries = s2g_line_entries
  val ackEntries = s2g_completion_entries
  val readIdBase = 2
  val readEntries = {
    val available = lsu_nMshrEntry - readIdBase
    if (s2g_shared_read_entries < available) s2g_shared_read_entries else available
  }
  require(lineEntries > 0, "DMA S2G needs at least one line entry")
  require(readEntries > 0, "DMA S2G needs at least one shared-read entry")
  require(ackEntries > 0, "DMA S2G needs at least one completion entry")
  require(ackEntries <= max_dma_tag, "DMA S2G completion entries must fit in the DMA source tag field")
  require((ackEntries & (ackEntries - 1)) == 0, "DMA S2G completion entries must be a power of two")

  val lineIdxWidth = log2Ceil(lineEntries).max(1)
  val readIdxWidth = log2Ceil(readEntries).max(1)
  val instIdxWidth = log2Ceil(max_dma_inst).max(1)
  val tagIdxWidth = log2Ceil(ackEntries).max(1)
  val pendingWidth = log2Ceil(lineEntries + ackEntries + 1).max(1)
  val ackSourceBase = dmaSourceLowBits + instIdxWidth
  val ackSourcePadBits = l1cache_sourceBits - ackSourceBase - tagIdxWidth
  require(ackSourcePadBits >= 0, "DMA S2G completion source encoding overflows d_source")
  val wordIdxWidth = log2Ceil(dcache_BlockWords).max(1)

  def alignToL2Line(addr: UInt): UInt =
    Cat(addr(xLen - 1, lineOffsetBits), 0.U(lineOffsetBits.W))

  def sharedSetIdx(addr: UInt): UInt =
    addr(sharedSetIdxHi, sharedSetIdxLo)

  def sharedBlockOffset(addr: UInt): UInt =
    addr(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits)

  def swizzleSharedAddr(logicalAddr: UInt, baseAddr: UInt, mode: UInt, rowLow: UInt): UInt = {
    val rel = logicalAddr - baseAddr
    val rel32 = Cat(rel(xLen - 1, 5), rel(4) ^ rowLow(0), rel(3, 0))
    val rel64 = Cat(rel(xLen - 1, 6), rel(5, 4) ^ rowLow(1, 0), rel(3, 0))
    val rel128 = Cat(rel(xLen - 1, 7), rel(6, 4) ^ rowLow(2, 0), rel(3, 0))
    baseAddr + MuxLookup(mode, rel)(Seq(
      1.U -> rel32,
      2.U -> rel64,
      3.U -> rel128
    ))
  }

  def minUInt(a: UInt, b: UInt): UInt = Mux(a < b, a, b)

  // Instruction table.
  val instValid = RegInit(VecInit(Seq.fill(max_dma_inst)(false.B)))
  val instWid = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(depth_warp.W))))
  val instGroup = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(log2Ceil(dma_group_entries).W))))
  val instAsid = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(SV32.asidLen.W))))
  // Bulk address generation keeps the current pointers and remaining bytes.
  // This removes the base+offset adders and size-offset subtractor from the
  // line-allocation path while reducing one 32-bit register per instruction.
  val instSrcCur = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(xLen.W))))
  val instDstCur = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(xLen.W))))
  val instBytesLeft = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(xLen.W))))
  val instExternal = RegInit(VecInit(Seq.fill(max_dma_inst)(false.B)))
  val instPendingLines = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(pendingWidth.W))))
  val instComplete = RegInit(VecInit(Seq.fill(max_dma_inst)(false.B)))
  val instSeq = RegInit(VecInit(Seq.fill(max_dma_inst)(0.U(xLen.W))))
  val instSeqNext = RegInit(0.U(xLen.W))

  // Remaining bytes are also the address-generation completion token. Tensor
  // line streams use 1 while open and 0 after their last line is accepted.
  val instAddrDone = VecInit((0 until max_dma_inst).map(i => instBytesLeft(i) === 0.U))

  // Line table. The 128B line image is the dominant storage; metadata stays in
  // flops, while line data is kept in word-banked SyncReadMem below.
  val lineValid = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineInst = RegInit(VecInit(Seq.fill(lineEntries)(0.U(instIdxWidth.W))))
  val lineSrc = RegInit(VecInit(Seq.fill(lineEntries)(0.U(xLen.W))))
  val lineDstVaddr = RegInit(VecInit(Seq.fill(lineEntries)(0.U(xLen.W))))
  val linePaddr = RegInit(VecInit(Seq.fill(lineEntries)(0.U(SV32.paLen.W))))
  val linePaddrValid = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineTlbReq = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineDstStartWord = RegInit(VecInit(Seq.fill(lineEntries)(0.U(wordIdxWidth.W))))
  val lineBytes = RegInit(VecInit(Seq.fill(lineEntries)(0.U(lineByteCountWidth.W))))
  val lineDstWordStride = RegInit(VecInit(Seq.fill(lineEntries)(1.U(log2Ceil(dcache_BlockWords + 1).W))))
  val linePutFull = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineReadIssued = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineSharedDone = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val linePutIssued = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineSwizzleMode = RegInit(VecInit(Seq.fill(lineEntries)(0.U(2.W))))
  val lineSwizzleBase = RegInit(VecInit(Seq.fill(lineEntries)(0.U(xLen.W))))
  val lineSwizzleRow = RegInit(VecInit(Seq.fill(lineEntries)(0.U(3.W))))
  val lineEarlyRelease = RegInit(VecInit(Seq.fill(lineEntries)(false.B)))
  val lineMask = RegInit(VecInit(Seq.fill(lineEntries)(
    VecInit(Seq.fill(dcache_BlockWords)(0.U(BytesOfWord.W)))
  )))
  val lineDataMem = Seq.fill(dcache_BlockWords)(SyncReadMem(lineEntries, UInt(xLen.W)))

  val putReadPending = RegInit(false.B)
  val putReadLine = RegInit(0.U(lineIdxWidth.W))
  val putDataValid = RegInit(false.B)
  val putDataLine = RegInit(0.U(lineIdxWidth.W))
  val putDataVec = Reg(Vec(dcache_BlockWords, UInt(xLen.W)))

  // Shared read table. instrId 0/1 are left to existing routes; bulk S2G uses
  // IDs [2, lsu_nMshrEntry).
  val readValid = RegInit(VecInit(Seq.fill(readEntries)(false.B)))
  val readLine = RegInit(VecInit(Seq.fill(readEntries)(0.U(lineIdxWidth.W))))
  val readPendingMask = RegInit(VecInit(Seq.fill(readEntries)(0.U(numgroupshared.W))))
  val readBaseWord = RegInit(VecInit(Seq.fill(readEntries)(0.U(wordIdxWidth.W))))
  val readWordStride = RegInit(VecInit(Seq.fill(readEntries)(1.U(log2Ceil(dcache_BlockWords + 1).W))))

  // L2 Put ack table.
  val ackValid = RegInit(VecInit(Seq.fill(ackEntries)(false.B)))
  val ackInst = RegInit(VecInit(Seq.fill(ackEntries)(0.U(instIdxWidth.W))))
  val ackLine = RegInit(VecInit(Seq.fill(ackEntries)(0.U(lineIdxWidth.W))))
  val ackReleaseLine = RegInit(VecInit(Seq.fill(ackEntries)(false.B)))

  val perfCycle = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfAckIssueCycle = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) {
    Some(RegInit(VecInit(Seq.fill(ackEntries)(0.U(64.W)))))
  } else {
    None
  }
  val perfInstIssued = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfLineIssued = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfPutFull = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfPutPart = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfBytesWritten = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfSharedReadReq = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfSharedReadRsp = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfTlbReq = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfAckCount = if (PMU_DMA_S2G) Some(RegInit(0.U(64.W))) else None
  val perfAckLatencySum = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfLineFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfReadEntryFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None
  val perfAckTagFullStallCycles = if (PMU_DMA_S2G && PMU_DMA_S2G_DETAIL) Some(RegInit(0.U(64.W))) else None

  val tlbBusy = RegInit(false.B)
  val tlbLineReg = RegInit(0.U(lineIdxWidth.W))

  val fullWordMask = Fill(BytesOfWord, 1.U)

  val freeInstVec = VecInit((0 until max_dma_inst).map(i => !instValid(i)))
  val freeInstValid = freeInstVec.asUInt.orR
  val freeInstIdx = PriorityEncoder(freeInstVec)

  val freeLineVec = VecInit((0 until lineEntries).map(i => !lineValid(i)))
  val freeLineValid = freeLineVec.asUInt.orR
  val freeLineIdx = PriorityEncoder(freeLineVec)

  val freeReadVec = VecInit((0 until readEntries).map(i => !readValid(i)))
  val freeReadValid = freeReadVec.asUInt.orR
  val freeReadIdx = PriorityEncoder(freeReadVec)

  val freeAckVec = VecInit((0 until ackEntries).map(i => !ackValid(i)))
  val freeAckValid = freeAckVec.asUInt.orR
  val freeAckIdx = PriorityEncoder(freeAckVec)

  val activeExtInstVec = VecInit((0 until max_dma_inst).map(i =>
    instValid(i) && instExternal(i) && !instAddrDone(i) && !instComplete(i)
  ))
  val activeExtInstValid = activeExtInstVec.asUInt.orR
  val activeExtInstIdx = PriorityEncoder(activeExtInstVec)
  when(!reset.asBool) {
    assert(PopCount(activeExtInstVec) <= 1.U,
      "DMA S2G backend supports only one open tensor line stream")
  }

  // -----------------------------
  // Instruction accept
  // -----------------------------
  val tensorTaskNeedsInst = io.line_task.bits.first && !activeExtInstValid
  val tensorTaskInstAvailable = Mux(
    tensorTaskNeedsInst,
    freeInstValid,
    activeExtInstValid
  )
  io.line_task.ready := freeLineValid && tensorTaskInstAvailable
  val tensorTaskFire = io.line_task.fire
  val tensorTaskInstIdx = Mux(tensorTaskNeedsInst, freeInstIdx, activeExtInstIdx)
  val tensorTaskAllocInst = tensorTaskFire && tensorTaskNeedsInst

  // Tensor line tasks use the same bounded writeback backend. Give a new
  // tensor task first priority for the shared instruction-table entry so it
  // cannot race a bulk S2G accept on the same free slot.
  io.from_fifo.ready := freeInstValid && !(io.line_task.valid && tensorTaskNeedsInst)
  val inputFire = io.from_fifo.fire
  when(inputFire) {
    instValid(freeInstIdx) := true.B
    instWid(freeInstIdx) := io.from_fifo.bits.ctrl.wid
    instGroup(freeInstIdx) := io.from_fifo.bits.ctrl.dma_group
    instAsid(freeInstIdx) := io.from_fifo.bits.ctrl.asid.getOrElse(0.U)
    instSrcCur(freeInstIdx) := io.from_fifo.bits.in1(0)
    instDstCur(freeInstIdx) := io.from_fifo.bits.in3(0)
    instBytesLeft(freeInstIdx) := io.from_fifo.bits.in2(0)
    instExternal(freeInstIdx) := false.B
    instPendingLines(freeInstIdx) := 0.U
    instComplete(freeInstIdx) := io.from_fifo.bits.in2(0) === 0.U
    instSeq(freeInstIdx) := instSeqNext
    instSeqNext := instSeqNext + 1.U
    when(!reset.asBool) {
      assert(io.from_fifo.bits.ctrl.funct === 3.U,
        "DMA S2G datapath received a non-bulk-S2G instruction")
      assert(io.from_fifo.bits.in1(0)(wordOffsetBits - 1, 0) === 0.U &&
             io.from_fifo.bits.in3(0)(wordOffsetBits - 1, 0) === 0.U &&
             io.from_fifo.bits.in2(0)(wordOffsetBits - 1, 0) === 0.U,
        "DMA S2G requires 4-byte aligned src, dst, and size")
    }
  }
  when(tensorTaskAllocInst) {
    instValid(freeInstIdx) := true.B
    instWid(freeInstIdx) := io.line_task.bits.wid
    instGroup(freeInstIdx) := io.line_task.bits.group
    instAsid(freeInstIdx) := io.line_task.bits.asid
    instSrcCur(freeInstIdx) := 0.U
    instDstCur(freeInstIdx) := 0.U
    instBytesLeft(freeInstIdx) := Mux(io.line_task.bits.last, 0.U, 1.U)
    instExternal(freeInstIdx) := true.B
    instPendingLines(freeInstIdx) := 0.U
    instComplete(freeInstIdx) := false.B
    instSeq(freeInstIdx) := instSeqNext
    instSeqNext := instSeqNext + 1.U
  }.elsewhen(tensorTaskFire && io.line_task.bits.last) {
    instBytesLeft(tensorTaskInstIdx) := 0.U
  }
  when(!reset.asBool && tensorTaskFire) {
    assert(io.line_task.bits.bytes =/= 0.U,
      "DMA S2G backend received an empty tensor line task")
    assert(io.line_task.bits.src(wordOffsetBits - 1, 0) === 0.U &&
           io.line_task.bits.dst(wordOffsetBits - 1, 0) === 0.U &&
           io.line_task.bits.bytes(wordOffsetBits - 1, 0) === 0.U,
      "DMA S2G tensor line task currently requires 4-byte aligned src, dst, and size")
    assert(io.line_task.bits.first || activeExtInstValid,
      "DMA S2G backend received a non-first tensor task without an active tensor instruction")
    assert(!(io.line_task.bits.first && activeExtInstValid),
      "DMA S2G backend received a new tensor first task before the previous tensor instruction ended")
  }

  // -----------------------------
  // Bulk address generation -> line allocation
  // -----------------------------
  val genInstVec = VecInit((0 until max_dma_inst).map(i =>
    instValid(i) && !instExternal(i) && !instAddrDone(i) && !instComplete(i) &&
      instBytesLeft(i) =/= 0.U
  ))
  val genInstValid = genInstVec.asUInt.orR
  val genInstIdx = PriorityEncoder(genInstVec)
  val genSrcCur = instSrcCur(genInstIdx)
  val genDstCur = instDstCur(genInstIdx)
  val genBytesLeft = instBytesLeft(genInstIdx)
  val genBytesToDstLine = l2cacheline.U - genDstCur(lineOffsetBits - 1, 0)
  val genBytesToSrcLine = l2cacheline.U - genSrcCur(lineOffsetBits - 1, 0)
  val genBytesPerSharedReq = (numgroupshared * dma_aligned_bulk).U
  val genChunkBytes = minUInt(genBytesLeft,
    minUInt(genBytesToDstLine, minUInt(genBytesToSrcLine, genBytesPerSharedReq)))
  val genBytesLeftNext = genBytesLeft - genChunkBytes
  val lineAllocBytes = Mux(
    tensorTaskFire,
    io.line_task.bits.bytes,
    genChunkBytes(lineByteCountWidth - 1, 0)
  )
  val bulkLineAllocFire = genInstValid && freeLineValid && !tensorTaskFire
  val lineAllocFire = bulkLineAllocFire || tensorTaskFire
  val lineAllocInstIdx = Mux(tensorTaskFire, tensorTaskInstIdx, genInstIdx)

  when(lineAllocFire) {
    val idx = freeLineIdx
    lineValid(idx) := true.B
    lineInst(idx) := lineAllocInstIdx
    lineSrc(idx) := Mux(tensorTaskFire, io.line_task.bits.src, genSrcCur)
    lineDstVaddr(idx) := alignToL2Line(Mux(tensorTaskFire, io.line_task.bits.dst, genDstCur))
    linePaddr(idx) := 0.U
    linePaddrValid(idx) := false.B
    lineTlbReq(idx) := false.B
    lineDstStartWord(idx) := Mux(tensorTaskFire, io.line_task.bits.dst, genDstCur)(lineOffsetBits - 1, wordOffsetBits)
    lineBytes(idx) := lineAllocBytes
    lineDstWordStride(idx) := Mux(tensorTaskFire, io.line_task.bits.dstWordStride, 1.U)
    linePutFull(idx) :=
      lineAllocBytes === l2cacheline.U &&
        Mux(tensorTaskFire, io.line_task.bits.dst, genDstCur)(lineOffsetBits - 1, wordOffsetBits) === 0.U &&
        Mux(tensorTaskFire, io.line_task.bits.dstWordStride, 1.U) === 1.U
    lineReadIssued(idx) := false.B
    lineSharedDone(idx) := false.B
    linePutIssued(idx) := false.B
    lineSwizzleMode(idx) := Mux(tensorTaskFire, io.line_task.bits.swizzleMode, 0.U)
    lineSwizzleBase(idx) := Mux(tensorTaskFire, io.line_task.bits.swizzleBase, 0.U)
    lineSwizzleRow(idx) := Mux(tensorTaskFire, io.line_task.bits.swizzleRow, 0.U)
    lineEarlyRelease(idx) := Mux(tensorTaskFire, io.line_task.bits.earlyRelease, true.B)
    for (w <- 0 until dcache_BlockWords) {
      lineMask(idx)(w) := 0.U
    }
    when(!tensorTaskFire) {
      instSrcCur(genInstIdx) := genSrcCur + genChunkBytes
      instDstCur(genInstIdx) := genDstCur + genChunkBytes
      instBytesLeft(genInstIdx) := genBytesLeftNext
    }
  }

  // -----------------------------
  // Shared read issue
  // -----------------------------
  val readIssueLineVec = VecInit((0 until lineEntries).map(i =>
    lineValid(i) && !lineReadIssued(i)
  ))
  val readIssueLineValid = readIssueLineVec.asUInt.orR
  val readIssueLineIdx = PriorityEncoder(readIssueLineVec)
  val readIssueFire = io.shared_req.fire

  val issueLineBytes = lineBytes(readIssueLineIdx)
  val issueLineDstStartWord = lineDstStartWord(readIssueLineIdx)
  val issueLineDstWordStride = lineDstWordStride(readIssueLineIdx)
  val issueSwizzleMode = lineSwizzleMode(readIssueLineIdx)
  val issueSwizzleBase = lineSwizzleBase(readIssueLineIdx)
  val issueSwizzleRow = lineSwizzleRow(readIssueLineIdx)
  val issueLaneActive = Wire(Vec(numgroupshared, Bool()))
  val issueSharedAddr = Wire(Vec(numgroupshared, UInt(xLen.W)))
  for (lane <- 0 until numgroupshared) {
    val rawSharedAddr = lineSrc(readIssueLineIdx) + (lane.U << wordOffsetBits)
    issueLaneActive(lane) := (lane * dma_aligned_bulk).U < issueLineBytes
    issueSharedAddr(lane) := Mux(
      issueSwizzleMode === 0.U,
      rawSharedAddr,
      swizzleSharedAddr(rawSharedAddr, issueSwizzleBase, issueSwizzleMode, issueSwizzleRow)
    )
  }
  val issueLaneMask = issueLaneActive.asUInt

  io.shared_req.valid := readIssueLineValid && freeReadValid
  io.shared_req.bits.instrId := (freeReadIdx + readIdBase.U)(log2Up(lsu_nMshrEntry) - 1, 0)
  io.shared_req.bits.isWrite := false.B
  io.shared_req.bits.setIdx := sharedSetIdx(issueSharedAddr(0))
  for (lane <- 0 until numgroupshared) {
    io.shared_req.bits.perLaneAddr(lane).activeMask := issueLaneActive(lane)
    io.shared_req.bits.perLaneAddr(lane).blockOffset := sharedBlockOffset(issueSharedAddr(lane))
    io.shared_req.bits.perLaneAddr(lane).wordOffset1H := Fill(BytesOfWord, 1.U)
    io.shared_req.bits.data(lane) := 0.U
  }

  when(readIssueFire) {
    readValid(freeReadIdx) := true.B
    readLine(freeReadIdx) := readIssueLineIdx
    readPendingMask(freeReadIdx) := issueLaneMask
    readBaseWord(freeReadIdx) := issueLineDstStartWord
    readWordStride(freeReadIdx) := issueLineDstWordStride
    lineReadIssued(readIssueLineIdx) := true.B
    when(!reset.asBool) {
      for (lane <- 0 until numgroupshared) {
        when(issueLaneActive(lane)) {
          assert(sharedSetIdx(issueSharedAddr(lane)) === sharedSetIdx(issueSharedAddr(0)),
            "DMA S2G swizzled line task crossed a shared-memory set")
        }
      }
    }
  }

  // -----------------------------
  // Shared read response collect
  // -----------------------------
  val rspStageValid = RegInit(false.B)
  val rspStageReadIdx = RegInit(0.U(readIdxWidth.W))
  val rspStageLineIdx = RegInit(0.U(lineIdxWidth.W))
  val rspStagePendingNext = RegInit(0.U(numgroupshared.W))
  val rspStageActiveMask = RegInit(0.U(numgroupshared.W))
  val rspStageBaseWord = RegInit(0.U(wordIdxWidth.W))
  val rspStageWordStride = RegInit(1.U(log2Ceil(dcache_BlockWords + 1).W))
  val rspStageData = Reg(Vec(numgroupshared, UInt(xLen.W)))

  io.shared_rsp.ready := true.B
  val incomingRspReadIdx =
    (io.shared_rsp.bits.instrId - readIdBase.U)(readIdxWidth - 1, 0)
  val incomingRspMask = io.shared_rsp.bits.activeMask.asUInt
  val incomingRspForwardsPending = rspStageValid &&
    rspStageReadIdx === incomingRspReadIdx
  val incomingRspPendingBase = Mux(
    incomingRspForwardsPending,
    rspStagePendingNext,
    readPendingMask(incomingRspReadIdx)
  )
  val incomingRspPendingNext = incomingRspPendingBase & ~incomingRspMask

  when(reset.asBool) {
    rspStageValid := false.B
  }.otherwise {
    rspStageValid := io.shared_rsp.fire
    when(io.shared_rsp.fire) {
      rspStageReadIdx := incomingRspReadIdx
      rspStageLineIdx := readLine(incomingRspReadIdx)
      rspStagePendingNext := incomingRspPendingNext
      rspStageActiveMask := incomingRspMask
      rspStageBaseWord := readBaseWord(incomingRspReadIdx)
      rspStageWordStride := readWordStride(incomingRspReadIdx)
      rspStageData := io.shared_rsp.bits.data
      assert(!io.shared_rsp.bits.isWrite,
        "DMA S2G shared response must be a read response")
      assert(readValid(incomingRspReadIdx),
        "DMA S2G shared response used an invalid read entry")
      assert((incomingRspMask & ~incomingRspPendingBase) === 0.U,
        "DMA S2G shared response returned lanes that were not pending")
    }
  }

  val rspStageDirect = rspStageWordStride === 1.U
  val rspLineMaskNext = Wire(Vec(dcache_BlockWords, UInt(BytesOfWord.W)))
  for (w <- 0 until dcache_BlockWords) {
    val directDelta = w.U((wordIdxWidth + 1).W) - rspStageBaseWord
    val directInRange = w.U >= rspStageBaseWord &&
      directDelta < numgroupshared.U
    val directLane = directDelta(log2Ceil(numgroupshared) - 1, 0)
    val directHit = directInRange && rspStageActiveMask(directLane)
    val genericHitVec = VecInit((0 until numgroupshared).map { lane =>
      val laneWord = (
        rspStageBaseWord.pad(xLen) +
          TmaPow2Math.scaleByPow2(lane.U(xLen.W), rspStageWordStride)
      )(wordIdxWidth - 1, 0)
      rspStageActiveMask(lane) && laneWord === w.U
    })
    val genericHit = genericHitVec.asUInt.orR
    val wordHit = Mux(rspStageDirect, directHit, genericHit)
    val wordData = Mux(
      rspStageDirect,
      rspStageData(directLane),
      Mux1H(genericHitVec, rspStageData)
    )
    rspLineMaskNext(w) := Mux(
      wordHit,
      lineMask(rspStageLineIdx)(w) | fullWordMask,
      lineMask(rspStageLineIdx)(w)
    )
    when(rspStageValid && wordHit) {
      lineDataMem(w).write(rspStageLineIdx, wordData)
    }
  }
  when(rspStageValid) {
    lineMask(rspStageLineIdx) := rspLineMaskNext
    readPendingMask(rspStageReadIdx) := rspStagePendingNext
    when(rspStagePendingNext === 0.U) {
      readValid(rspStageReadIdx) := false.B
      lineSharedDone(rspStageLineIdx) := true.B
    }
  }

  // -----------------------------
  // TLB/page translation
  // -----------------------------
  val tlbLineVec = VecInit((0 until lineEntries).map(i =>
    lineValid(i) && !linePaddrValid(i) && !lineTlbReq(i)
  ))
  val tlbLineValid = tlbLineVec.asUInt.orR
  val tlbLineIdx = PriorityEncoder(tlbLineVec)
  val tlbLineInstIdx = lineInst(tlbLineIdx)
  val tlbAsid = instAsid(tlbLineInstIdx)

  io.to_l2TLB.valid := tlbLineValid && !tlbBusy
  io.to_l2TLB.bits.vaddr := lineDstVaddr(tlbLineIdx)
  io.to_l2TLB.bits.asid := tlbAsid
  io.from_l2TLB.ready := tlbBusy

  when(io.to_l2TLB.fire) {
    tlbBusy := true.B
    tlbLineReg := tlbLineIdx
    lineTlbReq(tlbLineIdx) := true.B
  }

  when(io.from_l2TLB.fire) {
    tlbBusy := false.B
    linePaddr(tlbLineReg) := io.from_l2TLB.bits.paddr
    linePaddrValid(tlbLineReg) := true.B
  }

  // -----------------------------
  // L2 Put issue
  // -----------------------------
  val putLineVec = VecInit((0 until lineEntries).map(i =>
    lineValid(i) && lineSharedDone(i) && linePaddrValid(i) &&
      !linePutIssued(i) &&
      !(putDataValid && putDataLine === i.U) &&
      !(putReadPending && putReadLine === i.U)
  ))
  val putLineValid = putLineVec.asUInt.orR
  val putLineIdx = PriorityEncoder(putLineVec)
  val putReadStart = !putReadPending && !putDataValid && putLineValid
  val putDataRead = VecInit((0 until dcache_BlockWords).map(w =>
    lineDataMem(w).read(putLineIdx, putReadStart)
  ))

  when(putReadStart) {
    putReadPending := true.B
    putReadLine := putLineIdx
  }

  when(putReadPending) {
    putReadPending := false.B
    putDataValid := true.B
    putDataLine := putReadLine
    putDataVec := putDataRead
  }

  val putDataMask = lineMask(putDataLine)
  val putDataFullLine = linePutFull(putDataLine)
  val putDataHasBytes = VecInit((0 until dcache_BlockWords).map(w =>
    putDataMask(w).orR
  )).asUInt.orR
  val putDataReady = putDataValid && lineValid(putDataLine) &&
    !linePutIssued(putDataLine) && putDataHasBytes

  io.to_l2cache.valid := putDataReady && freeAckValid
  io.to_l2cache.bits.a_opcode := Mux(putDataFullLine, 0.U, 1.U)
  io.to_l2cache.bits.a_param := 0.U
  val ackSource = if (ackSourcePadBits == 0) {
    Cat(
      freeAckIdx(tagIdxWidth - 1, 0),
      lineInst(putDataLine)(instIdxWidth - 1, 0),
      2.U(dmaSourceLowBits.W)
    )
  } else {
    Cat(
      0.U(ackSourcePadBits.W),
      freeAckIdx(tagIdxWidth - 1, 0),
      lineInst(putDataLine)(instIdxWidth - 1, 0),
      2.U(dmaSourceLowBits.W)
    )
  }
  io.to_l2cache.bits.a_source := ackSource
  io.to_l2cache.bits.a_addr.foreach(_ := linePaddr(putDataLine))
  io.to_l2cache.bits.a_data := putDataVec
  io.to_l2cache.bits.a_mask := putDataMask
  io.to_l2cache.bits.spike_info.foreach(_ := io.to_l2cache.bits.defaultSpikeInfo)

  when(io.to_l2cache.fire) {
    ackValid(freeAckIdx) := true.B
    ackInst(freeAckIdx) := lineInst(putDataLine)
    ackLine(freeAckIdx) := putDataLine
    ackReleaseLine(freeAckIdx) := !lineEarlyRelease(putDataLine)
    if (PMU_DMA_S2G_DETAIL) {
      perfAckIssueCycle.foreach { cycles =>
        cycles(freeAckIdx) := perfCycle.get
      }
    }
    linePutIssued(putDataLine) := true.B
    when(lineEarlyRelease(putDataLine)) {
      lineValid(putDataLine) := false.B
    }
    putDataValid := false.B
    when(!reset.asBool) {
      assert(putDataHasBytes, "DMA S2G attempted to issue an empty L2 Put")
    }
  }

  // -----------------------------
  // L2 AccessAck collect
  // -----------------------------
  io.from_l2cache.ready := true.B
  val rspAckIdx = io.from_l2cache.bits.d_source(
    ackSourceBase + tagIdxWidth - 1,
    ackSourceBase
  )
  val rspAckInst = io.from_l2cache.bits.d_source(
    ackSourceBase - 1,
    dmaSourceLowBits
  )
  val ackInstIdx = ackInst(rspAckIdx)
  val l2AckFire = io.from_l2cache.fire

  when(l2AckFire) {
    when(!reset.asBool) {
      assert(io.from_l2cache.bits.d_opcode === 0.U,
        "DMA S2G L2 response must be AccessAck")
      assert(ackValid(rspAckIdx),
        "DMA S2G L2 ack used an invalid ack entry")
      assert(ackInst(rspAckIdx) === rspAckInst,
        "DMA S2G L2 ack source inst index mismatch")
    }
    ackValid(rspAckIdx) := false.B
    when(ackReleaseLine(rspAckIdx)) {
      lineValid(ackLine(rspAckIdx)) := false.B
    }
  }

  // -----------------------------
  // Pending-line accounting and completion
  // -----------------------------
  for (i <- 0 until max_dma_inst) {
    val inc = lineAllocFire && lineAllocInstIdx === i.U
    val dec = l2AckFire && ackInstIdx === i.U
    val pendingNext = instPendingLines(i) + inc.asUInt - dec.asUInt
    when(inc || dec) {
      instPendingLines(i) := pendingNext
    }
    when(dec && instAddrDone(i) && pendingNext === 0.U) {
      instComplete(i) := true.B
    }
    // Covers a final ack coincident with the delayed bulk address-done update.
    when(instValid(i) && instAddrDone(i) && !instComplete(i) &&
        instPendingLines(i) === 0.U) {
      instComplete(i) := true.B
    }
    when(!reset.asBool) {
      assert(!(dec && instPendingLines(i) === 0.U && !inc),
        "DMA S2G pending line counter underflow")
    }
  }

  val instCompleteInOrder = Wire(Vec(max_dma_inst, Bool()))
  for (i <- 0 until max_dma_inst) {
    val olderSameWarpValid = VecInit((0 until max_dma_inst).map { j =>
      if (i == j) {
        false.B
      } else {
        instValid(j) && instWid(j) === instWid(i) && instSeq(j) < instSeq(i)
      }
    }).asUInt.orR
    instCompleteInOrder(i) := instComplete(i) && !olderSameWarpValid
  }
  io.inst_complete.valid := instCompleteInOrder.asUInt.orR
  val completeIdx = PriorityEncoder(instCompleteInOrder)
  io.inst_complete.bits.wid := instWid(completeIdx)
  io.inst_complete.bits.group := instGroup(completeIdx)
  io.inst_complete.bits.is_s2g := true.B
  when(io.inst_complete.fire) {
    instValid(completeIdx) := false.B
    instExternal(completeIdx) := false.B
    instComplete(completeIdx) := false.B
    instBytesLeft(completeIdx) := 0.U
    instPendingLines(completeIdx) := 0.U
  }

  if (PMU_DMA_S2G) {
    val putBytes = PopCount(putDataMask.asUInt).pad(64)
    if (PMU_DMA_S2G_DETAIL) {
      val lineFullStall = (genInstValid || io.line_task.valid) && !freeLineValid
      val readEntryFullStall = readIssueLineValid && !freeReadValid
      val ackTagFullStall = putDataReady && !freeAckValid
      val ackLatency = perfCycle.get - perfAckIssueCycle.get(rspAckIdx)
      when(io.perfReset) {
        perfCycle.get := 0.U
        perfAckLatencySum.get := 0.U
        perfLineFullStallCycles.get := 0.U
        perfReadEntryFullStallCycles.get := 0.U
        perfAckTagFullStallCycles.get := 0.U
        for (i <- 0 until ackEntries) {
          perfAckIssueCycle.get(i) := 0.U
        }
      }.elsewhen(io.perfEnable) {
        perfCycle.get := perfCycle.get + 1.U
        when(l2AckFire) {
          perfAckLatencySum.get := perfAckLatencySum.get + ackLatency
        }
        when(lineFullStall) {
          perfLineFullStallCycles.get := perfLineFullStallCycles.get + 1.U
        }
        when(readEntryFullStall) {
          perfReadEntryFullStallCycles.get := perfReadEntryFullStallCycles.get + 1.U
        }
        when(ackTagFullStall) {
          perfAckTagFullStallCycles.get := perfAckTagFullStallCycles.get + 1.U
        }
      }
    }

    when(io.perfReset) {
      perfInstIssued.get := 0.U
      perfLineIssued.get := 0.U
      perfPutFull.get := 0.U
      perfPutPart.get := 0.U
      perfBytesWritten.get := 0.U
      perfSharedReadReq.get := 0.U
      perfSharedReadRsp.get := 0.U
      perfTlbReq.get := 0.U
      perfAckCount.get := 0.U
    }.elsewhen(io.perfEnable) {
      when(inputFire) {
        perfInstIssued.get := perfInstIssued.get + 1.U
      }
      when(lineAllocFire) {
        perfLineIssued.get := perfLineIssued.get + 1.U
      }
      when(readIssueFire) {
        perfSharedReadReq.get := perfSharedReadReq.get + 1.U
      }
      when(io.shared_rsp.fire) {
        perfSharedReadRsp.get := perfSharedReadRsp.get + 1.U
      }
      when(io.to_l2TLB.fire) {
        perfTlbReq.get := perfTlbReq.get + 1.U
      }
      when(io.to_l2cache.fire) {
        when(putDataFullLine) {
          perfPutFull.get := perfPutFull.get + 1.U
        }.otherwise {
          perfPutPart.get := perfPutPart.get + 1.U
        }
        perfBytesWritten.get := perfBytesWritten.get + putBytes
      }
      when(l2AckFire) {
        perfAckCount.get := perfAckCount.get + 1.U
      }
    }

    io.perf.get.instIssued := perfInstIssued.get
    io.perf.get.lineIssued := perfLineIssued.get
    io.perf.get.putFull := perfPutFull.get
    io.perf.get.putPart := perfPutPart.get
    io.perf.get.bytesWritten := perfBytesWritten.get
    io.perf.get.sharedReadReq := perfSharedReadReq.get
    io.perf.get.sharedReadRsp := perfSharedReadRsp.get
    io.perf.get.tlbReq := perfTlbReq.get
    io.perf.get.ackCount := perfAckCount.get
    if (PMU_DMA_S2G_DETAIL) {
      io.perf.get.ackLatencySum := perfAckLatencySum.get
      io.perf.get.lineFullStallCycles := perfLineFullStallCycles.get
      io.perf.get.readEntryFullStallCycles := perfReadEntryFullStallCycles.get
      io.perf.get.ackTagFullStallCycles := perfAckTagFullStallCycles.get
    } else {
      io.perf.get.ackLatencySum := 0.U
      io.perf.get.lineFullStallCycles := 0.U
      io.perf.get.readEntryFullStallCycles := 0.U
      io.perf.get.ackTagFullStallCycles := 0.U
    }
  }

  when(!reset.asBool) {
    assert(!(lineAllocFire && !freeLineValid),
      "DMA S2G line allocation fired with no free line entry")
    assert(!(readIssueFire && !freeReadValid),
      "DMA S2G shared read allocation fired with no free read entry")
    assert(!(io.to_l2cache.fire && !freeAckValid),
      "DMA S2G L2 Put allocation fired with no free ack entry")
  }
}
