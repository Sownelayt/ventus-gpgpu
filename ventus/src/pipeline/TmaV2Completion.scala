package pipeline

import chisel3._
import chisel3.util._
import top.parameters._

/**
  * Authoritative S2G bulk-group state.  The scheduler only submits group
  * control operations and consumes waitMask; issue assignment, accounting,
  * and ring reclamation stay with the TMA completion subsystem.
  */
class TmaV2S2GGroupTracker extends Module {
  require((dma_group_entries & (dma_group_entries - 1)) == 0,
    "S2G group ring expects a power-of-two group count")
  require(dma_group_entries == TmaV2Spec.S2GGroupsPerWarp,
    "public TMA v2 ABI fixes four S2G groups per warp")

  private val inflightWidth = log2Ceil(max_dma_inst + 1)
  private val groupWidth = log2Ceil(dma_group_entries)
  private val keepWidth = log2Ceil(dma_group_entries + 1)

  val io = IO(new Bundle {
    val control = Flipped(Decoupled(new DmaGroupCommand))
    val issue = Flipped(Valid(new DmaIssue))
    val completion = Flipped(Valid(new DmaCompletion))
    val warpReset = Flipped(Valid(UInt(depth_warp.W)))
    val clearAll = Input(Bool())
    val issueAllow = Output(Vec(num_warp, Bool()))
    val issueGroup = Output(Vec(num_warp, UInt(groupWidth.W)))
    val waitMask = Output(UInt(num_warp.W))
    val inflight = Output(Vec(num_warp, UInt(inflightWidth.W)))
    val status = Valid(new DmaStatusUpdate)
  })

  val inflight = RegInit(VecInit(Seq.fill(num_warp)(0.U(inflightWidth.W))))
  val issuePtr = RegInit(VecInit(Seq.fill(num_warp)(0.U(groupWidth.W))))
  val open = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val waitActive = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val waitKeep = RegInit(VecInit(Seq.fill(num_warp)(0.U(keepWidth.W))))
  val committed = RegInit(VecInit(Seq.fill(num_warp)(
    VecInit(Seq.fill(dma_group_entries)(false.B))
  )))
  val outstanding = RegInit(VecInit(Seq.fill(num_warp)(
    VecInit(Seq.fill(dma_group_entries)(0.U(inflightWidth.W)))
  )))

  val zimm = io.control.bits.zimm
  val commit = zimm === TmaV2Spec.S2GGroupCommit.U
  val waitValid = zimm >= TmaV2Spec.S2GGroupWaitBase.U &&
    zimm <= (TmaV2Spec.S2GGroupWaitBase + TmaV2Spec.S2GGroupWaitKeepMax).U
  val keep = (zimm - TmaV2Spec.S2GGroupWaitBase.U)(2, 0)
    .pad(keepWidth)(keepWidth - 1, 0)
  io.control.ready := true.B

  io.status.valid := io.control.fire && !commit && !waitValid
  io.status.bits.wid := io.control.bits.wid
  io.status.bits.code := TmaV2Spec.StatusInvalidGroupOperation.U
  io.status.bits.detail := zimm

  def groupBehind(ptr: UInt, distance: Int): UInt =
    (ptr - distance.U)(groupWidth - 1, 0)

  def groupIsRecent(ptr: UInt, group: UInt, keepCount: UInt): Bool =
    (1 to dma_group_entries).map { distance =>
      keepCount >= distance.U && group === groupBehind(ptr, distance)
    }.reduce(_ || _)

  val allow = Wire(Vec(num_warp, Bool()))
  for (wid <- 0 until num_warp) {
    val issueHit = io.issue.valid && io.issue.bits.wid === wid.U && io.issue.bits.isS2G
    val completeHit = io.completion.valid && io.completion.bits.wid === wid.U &&
      io.completion.bits.is_s2g
    val canIncrement = inflight(wid) =/= max_dma_inst.U(inflightWidth.W)
    val issueSlotFree = !committed(wid)(issuePtr(wid))
    val issueAllowed = (canIncrement || completeHit) && issueSlotFree
    val increment = issueHit && issueAllowed
    val decrement = completeHit && inflight(wid) =/= 0.U
    val inflightNext = inflight(wid) + increment.asUInt - decrement.asUInt

    val controlHere = io.control.fire && io.control.bits.wid === wid.U
    val commitHere = controlHere && commit
    val waitHere = controlHere && waitValid
    val countNext = Wire(Vec(dma_group_entries, UInt(inflightWidth.W)))
    val committedNext = Wire(Vec(dma_group_entries, Bool()))
    val openHasWork = open(wid) || outstanding(wid)(issuePtr(wid)) =/= 0.U
    val advance = commitHere && openHasWork
    val ptrNext = Mux(advance,
      (issuePtr(wid) + 1.U)(groupWidth - 1, 0), issuePtr(wid))

    for (group <- 0 until dma_group_entries) {
      val groupIncrement = increment && issuePtr(wid) === group.U
      val groupDecrement = completeHit && io.completion.bits.group === group.U &&
        outstanding(wid)(group) =/= 0.U
      countNext(group) := outstanding(wid)(group) + groupIncrement.asUInt - groupDecrement.asUInt
      committedNext(group) := committed(wid)(group)
      when(countNext(group) === 0.U) { committedNext(group) := false.B }
      when(commitHere && openHasWork && issuePtr(wid) === group.U && countNext(group) =/= 0.U) {
        committedNext(group) := true.B
      }
    }

    val savedWaitBlocked = VecInit((0 until dma_group_entries).map { group =>
      committedNext(group) && countNext(group) =/= 0.U &&
        !groupIsRecent(issuePtr(wid), group.U, waitKeep(wid))
    }).asUInt.orR
    val newWaitBlocked = VecInit((0 until dma_group_entries).map { group =>
      committedNext(group) && countNext(group) =/= 0.U &&
        !groupIsRecent(issuePtr(wid), group.U, keep)
    }).asUInt.orR
    val savedWaitDone = waitKeep(wid) >= dma_group_entries.U || !savedWaitBlocked
    val newWaitDone = keep >= dma_group_entries.U || !newWaitBlocked
    val openAfterCommit = Mux(advance, false.B, open(wid) || increment)
    val openNext = openAfterCommit && countNext(ptrNext) =/= 0.U
    val resetWarp = io.clearAll || (io.warpReset.valid && io.warpReset.bits === wid.U)

    allow(wid) := issueAllowed
    inflight(wid) := Mux(resetWarp, 0.U, inflightNext)
    issuePtr(wid) := Mux(resetWarp, 0.U, ptrNext)
    open(wid) := Mux(resetWarp, false.B, openNext)
    waitKeep(wid) := Mux(resetWarp, 0.U, Mux(waitHere, keep, waitKeep(wid)))
    waitActive(wid) := Mux(resetWarp, false.B,
      Mux(waitHere, !newWaitDone, Mux(savedWaitDone, false.B, waitActive(wid))))
    for (group <- 0 until dma_group_entries) {
      outstanding(wid)(group) := Mux(resetWarp, 0.U, countNext(group))
      committed(wid)(group) := Mux(resetWarp, false.B, committedNext(group))
    }

    when(!reset.asBool) {
      assert(outstanding(wid).reduce(_ +& _) === inflight(wid),
        s"S2G group counters must conserve inflight count on warp $wid")
      assert(!(issueHit && !issueAllowed), s"S2G group issue overflow on warp $wid")
      assert(!(completeHit && inflight(wid) === 0.U && !issueHit),
        s"S2G group completion underflow on warp $wid")
      assert(!(completeHit && outstanding(wid)(io.completion.bits.group) === 0.U &&
        !(increment && issuePtr(wid) === io.completion.bits.group)),
        s"S2G group counter underflow on warp $wid")
      when(io.warpReset.valid && io.warpReset.bits === wid.U) {
        assert(inflight(wid) === 0.U, s"Warp $wid reused before S2G completion drains")
        assert(!waitActive(wid), s"Warp $wid reused while S2G wait is active")
      }
    }
  }

  io.issueAllow := allow
  io.issueGroup := issuePtr
  io.waitMask := waitActive.asUInt
  io.inflight := inflight
}

class TmaV2MbarrierController(entries: Int = TmaV2Spec.MbarrierEntries) extends Module {
  require(entries == num_block * TmaV2Spec.MbarrierEntriesPerWg,
    "mbarrier table is statically partitioned into four entries per resident WG")
  private val entryWidth = log2Ceil(entries)
  private val ownerWidth = log2Ceil(num_block)
  private val localEntryWidth =
    log2Ceil(TmaV2Spec.MbarrierEntriesPerWg)
  private val sharedSetIdxBits = log2Ceil(sharedmem_depth)
  private val sharedSetIdxLo = dcache_BlockOffsetBits + dcache_WordOffsetBits
  private val sharedSetIdxHi = sharedSetIdxLo + sharedSetIdxBits - 1

  val io = IO(new Bundle {
    val syncCommand = Flipped(Decoupled(new DmaSyncCommand))
    val reserveRequest = Flipped(Decoupled(new DmaTxReserveRequest))
    val reserveResponse = Decoupled(new DmaTxReserveResponse)
    val completion = Flipped(Valid(new DmaCompletion))
    val warpReset = Flipped(Valid(UInt(depth_warp.W)))
    val wgRelease = Flipped(Valid(UInt(ownerWidth.W)))
    val ownerBusy = Output(UInt(num_block.W))
    val waitMask = Output(UInt(num_warp.W))
    val sharedRequest = Decoupled(new ShareMemCoreReq_np)
    val sharedResponse = Flipped(Decoupled(new DmaSharedRsp))
    val status = Valid(new DmaStatusUpdate)
  })

  val valid = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val address = Reg(Vec(entries, UInt(32.W)))
  val pendingBytes = RegInit(VecInit(Seq.fill(entries)(0.U(32.W))))
  val expectedArrivals = RegInit(VecInit(Seq.fill(entries)(0.U(8.W))))
  val pendingArrivals = RegInit(VecInit(Seq.fill(entries)(0.U(8.W))))
  val phase = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val generation = RegInit(VecInit(Seq.fill(entries)(0.U(8.W))))
  val phaseObserved = RegInit(VecInit(Seq.fill(entries)(false.B)))
  val dirty = RegInit(VecInit(Seq.fill(entries)(false.B)))

  val bindingValid = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val bindingEntry = Reg(Vec(num_warp, UInt(entryWidth.W)))
  val bindingGeneration = Reg(Vec(num_warp, UInt(8.W)))
  val bindingRemaining = Reg(Vec(num_warp, UInt(32.W)))
  val waitActive = RegInit(VecInit(Seq.fill(num_warp)(false.B)))
  val waitEntry = Reg(Vec(num_warp, UInt(entryWidth.W)))
  val waitPhase = Reg(Vec(num_warp, Bool()))
  val writerActive = RegInit(false.B)
  val writerEntry = Reg(UInt(entryWidth.W))
  val writerWord = RegInit(false.B)
  val writerAddress = Reg(UInt(32.W))
  val writerPendingBytes = Reg(UInt(32.W))
  val writerState = Reg(UInt(32.W))
  val writerRequestOutstanding = RegInit(false.B)

  io.waitMask := waitActive.asUInt

  val commandAddress = io.syncCommand.bits.address
  val commandOwner = io.syncCommand.bits.owner
  def ownedEntry(owner: UInt, local: Int): UInt =
    Cat(owner, local.U(localEntryWidth.W))
  val hitVec = VecInit((0 until TmaV2Spec.MbarrierEntriesPerWg).map { local =>
    val entry = ownedEntry(commandOwner, local)
    valid(entry) && address(entry) === commandAddress
  })
  val hit = hitVec.asUInt.orR
  val hitLocal = PriorityEncoder(hitVec)
  val freeVec = VecInit((0 until TmaV2Spec.MbarrierEntriesPerWg).map { local =>
    !valid(ownedEntry(commandOwner, local))
  })
  val free = freeVec.asUInt.orR
  val freeLocal = PriorityEncoder(freeVec)
  val selectedLocal = Mux(hit, hitLocal, freeLocal)
  val selectedEntry = Cat(commandOwner, selectedLocal)
  val hitEntry = Cat(commandOwner, hitLocal)
  val commandWid = io.syncCommand.bits.wid
  val commandOp = io.syncCommand.bits.op
  val addressAligned = commandAddress(2, 0) === 0.U

  // Completion updates have priority over a new control operation. This keeps
  // a single write point for the table and avoids a same-entry add/subtract race.
  val reserveQ = Module(new Queue(new DmaTxReserveResponse, 2))
  io.reserveResponse <> reserveQ.io.deq
  reserveQ.io.enq.valid := false.B
  reserveQ.io.enq.bits := 0.U.asTypeOf(new DmaTxReserveResponse)

  io.syncCommand.ready := !io.completion.valid
  io.reserveRequest.ready := !io.completion.valid && !io.syncCommand.valid && reserveQ.io.enq.ready
  val commandFire = io.syncCommand.fire
  val reserveFire = io.reserveRequest.fire

  val statusValid = WireDefault(false.B)
  val statusWid = WireDefault(0.U(depth_warp.W))
  val statusCode = WireDefault(TmaV2Spec.StatusMbarrierProtocol.U(8.W))
  val statusDetail = WireDefault(0.U(8.W))
  io.status.valid := statusValid
  io.status.bits.wid := statusWid
  io.status.bits.code := statusCode
  io.status.bits.detail := statusDetail

  when(commandFire) {
    when(commandOp === TmaV2Spec.MbarrierInit.U) {
      val arrivals = io.syncCommand.bits.value(7, 0)
      when(!addressAligned || arrivals === 0.U || (!hit && !free) ||
          (hit && pendingBytes(hitEntry) =/= 0.U)) {
        statusValid := true.B
        statusWid := commandWid
        statusDetail := 0.U
      }.otherwise {
        valid(selectedEntry) := true.B
        address(selectedEntry) := commandAddress
        pendingBytes(selectedEntry) := 0.U
        expectedArrivals(selectedEntry) := arrivals
        pendingArrivals(selectedEntry) := arrivals
        phase(selectedEntry) := false.B
        generation(selectedEntry) := 0.U
        phaseObserved(selectedEntry) := true.B
        dirty(selectedEntry) := true.B
      }
    }.elsewhen(commandOp === TmaV2Spec.MbarrierArriveExpectTx.U) {
      val bytes = io.syncCommand.bits.value
      val overflow = hit && (pendingBytes(hitEntry) +& bytes)(32)
      when(!addressAligned || !hit || bytes === 0.U || overflow ||
          pendingArrivals(hitEntry) === 0.U || bindingValid(commandWid) ||
          !phaseObserved(hitEntry)) {
        statusValid := true.B
        statusWid := commandWid
        statusDetail := 1.U
      }.otherwise {
        pendingBytes(hitEntry) := pendingBytes(hitEntry) + bytes
        pendingArrivals(hitEntry) := pendingArrivals(hitEntry) - 1.U
        dirty(hitEntry) := true.B
        bindingValid(commandWid) := true.B
        bindingEntry(commandWid) := hitEntry
        bindingGeneration(commandWid) := generation(hitEntry)
        bindingRemaining(commandWid) := bytes
      }
    }.elsewhen(commandOp === TmaV2Spec.MbarrierWait.U) {
      when(!addressAligned || !hit) {
        statusValid := true.B
        statusWid := commandWid
        statusDetail := 2.U
      }.otherwise {
        val waitSatisfied = phase(hitEntry) =/= io.syncCommand.bits.value(0) &&
          !dirty(hitEntry) && !(writerActive && writerEntry === hitEntry)
        waitEntry(commandWid) := hitEntry
        waitPhase(commandWid) := io.syncCommand.bits.value(0)
        waitActive(commandWid) := !waitSatisfied
        when(waitSatisfied) { phaseObserved(hitEntry) := true.B }
      }
    }.otherwise {
      statusValid := true.B
      statusWid := commandWid
      statusDetail := commandOp
    }
  }

  when(reserveFire) {
    val wid = io.reserveRequest.bits.wid
    val bytes = io.reserveRequest.bits.bytes
    val bound = bindingValid(wid)
    val entry = bindingEntry(wid)
    val bindingCurrent = bound && valid(entry) &&
      bindingGeneration(wid) === generation(entry)
    val accepted = !bound || (bindingCurrent && bytes =/= 0.U &&
      bytes <= bindingRemaining(wid))
    reserveQ.io.enq.valid := true.B
    reserveQ.io.enq.bits.wid := wid
    reserveQ.io.enq.bits.accepted := accepted
    reserveQ.io.enq.bits.barrierValid := bound && accepted
    reserveQ.io.enq.bits.barrierId := Mux(bound, entry, 0.U)
    reserveQ.io.enq.bits.generation := Mux(bound, bindingGeneration(wid), 0.U)
    when(bound && accepted) {
      val remaining = bindingRemaining(wid) - bytes
      bindingRemaining(wid) := remaining
      when(remaining === 0.U) { bindingValid(wid) := false.B }
    }.elsewhen(bound) {
      statusValid := true.B
      statusWid := wid
      statusDetail := 4.U
    }
  }

  when(io.completion.valid && !io.completion.bits.is_s2g &&
      io.completion.bits.barrierValid) {
    val entry = io.completion.bits.barrierId
    val bytes = io.completion.bits.transactionBytes
    when(!valid(entry) || io.completion.bits.barrierGeneration =/= generation(entry) ||
        bytes === 0.U || pendingBytes(entry) < bytes) {
      statusValid := true.B
      statusWid := io.completion.bits.wid
      statusDetail := 3.U
    }.otherwise {
      val bytesNext = pendingBytes(entry) - bytes
      pendingBytes(entry) := bytesNext
      dirty(entry) := true.B
      when(bytesNext === 0.U && pendingArrivals(entry) === 0.U) {
        phase(entry) := !phase(entry)
        generation(entry) := generation(entry) + 1.U
        phaseObserved(entry) := false.B
        pendingArrivals(entry) := expectedArrivals(entry)
      }
    }
  }

  when(io.warpReset.valid) {
    bindingValid(io.warpReset.bits) := false.B
    waitActive(io.warpReset.bits) := false.B
  }

  // A dirty entry is mirrored into the architected 64-bit shared-memory
  // mbarrier object. Updates may coalesce, but a waiter is not released until
  // both words of the newest visible state have received shared-memory acks.
  val dirtyVec = dirty.asUInt
  val dirtyEntry = PriorityEncoder(dirtyVec)

  val ownerBusyVec = Wire(Vec(num_block, Bool()))
  for (wg <- 0 until num_block) {
    val tableBusy = VecInit((0 until TmaV2Spec.MbarrierEntriesPerWg).map { local =>
      val entry = wg * TmaV2Spec.MbarrierEntriesPerWg + local
      valid(entry) &&
        (pendingBytes(entry) =/= 0.U || dirty(entry) ||
          (writerActive && writerEntry === entry.U))
    }).asUInt.orR
    val bindingBusy = VecInit((0 until num_warp).map { wid =>
      (bindingValid(wid) &&
        bindingEntry(wid)(entryWidth - 1, localEntryWidth) === wg.U) ||
        (waitActive(wid) &&
          waitEntry(wid)(entryWidth - 1, localEntryWidth) === wg.U)
    }).asUInt.orR
    ownerBusyVec(wg) := tableBusy || bindingBusy
  }
  io.ownerBusy := ownerBusyVec.asUInt

  when(!writerActive && dirtyVec.orR) {
    writerActive := true.B
    writerEntry := dirtyEntry
    writerWord := false.B
    writerAddress := address(dirtyEntry)
    writerPendingBytes := pendingBytes(dirtyEntry)
    writerState := Cat(0.U(8.W), pendingArrivals(dirtyEntry),
      expectedArrivals(dirtyEntry), 0.U(7.W), phase(dirtyEntry))
    dirty(dirtyEntry) := false.B
  }

  io.sharedRequest.valid := writerActive && !writerRequestOutstanding
  io.sharedRequest.bits := 0.U.asTypeOf(new ShareMemCoreReq_np)
  val writeAddress = writerAddress + Mux(writerWord, 4.U, 0.U)
  io.sharedRequest.bits.instrId := 0.U
  io.sharedRequest.bits.isWrite := true.B
  io.sharedRequest.bits.isMBarrier := true.B
  io.sharedRequest.bits.setIdx := writeAddress(sharedSetIdxHi, sharedSetIdxLo)
  io.sharedRequest.bits.perLaneAddr(0).activeMask := true.B
  io.sharedRequest.bits.perLaneAddr(0).blockOffset :=
    writeAddress(dcache_BlockOffsetBits + dcache_WordOffsetBits - 1, dcache_WordOffsetBits)
  io.sharedRequest.bits.perLaneAddr(0).wordOffset1H := Fill(BytesOfWord, 1.U)
  io.sharedRequest.bits.data(0) := Mux(writerWord, writerState, writerPendingBytes)

  when(io.sharedRequest.fire) {
    writerRequestOutstanding := true.B
  }

  io.sharedResponse.ready := writerActive && writerRequestOutstanding &&
    io.sharedResponse.bits.isMBarrier
  when(io.sharedResponse.fire) {
    assert(io.sharedResponse.bits.isWrite)
    writerRequestOutstanding := false.B
    when(!writerWord) {
      writerWord := true.B
    }.otherwise {
      writerActive := false.B
      writerWord := false.B
    }
  }

  for (wid <- 0 until num_warp) {
    val entry = waitEntry(wid)
    val mirrored = !dirty(entry) && !(writerActive && writerEntry === entry)
    when(waitActive(wid) && valid(entry) && phase(entry) =/= waitPhase(wid) && mirrored) {
      waitActive(wid) := false.B
      phaseObserved(entry) := true.B
    }
  }

  when(io.wgRelease.valid) {
    val releasedOwner = io.wgRelease.bits
    assert(!ownerBusyVec(releasedOwner), "WG mbarrier owner released before its state drained")
    for (local <- 0 until TmaV2Spec.MbarrierEntriesPerWg) {
      val entry = ownedEntry(releasedOwner, local)
      valid(entry) := false.B
      pendingBytes(entry) := 0.U
      expectedArrivals(entry) := 0.U
      pendingArrivals(entry) := 0.U
      generation(entry) := 0.U
      phaseObserved(entry) := false.B
      dirty(entry) := false.B
    }
  }

  when(!reset.asBool) {
    assert(!(io.sharedResponse.valid && io.sharedResponse.bits.isMBarrier && !writerActive))
    for (wid <- 0 until num_warp) {
      when(bindingValid(wid)) {
        assert(valid(bindingEntry(wid)))
        assert(bindingGeneration(wid) === generation(bindingEntry(wid)))
        assert(bindingRemaining(wid) =/= 0.U)
      }
      when(waitActive(wid)) { assert(valid(waitEntry(wid))) }
    }
  }
}
