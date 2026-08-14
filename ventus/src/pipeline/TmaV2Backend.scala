package pipeline

import chisel3._
import chisel3.util._
import mmu.SV32
import top.parameters._

class TmaV2DescriptorRequest extends Bundle {
  val address = UInt(32.W)
  val asid = UInt(SV32.asidLen.W)
  val wantResponse = Bool()
  val invalidate = Bool()
}

class TmaV2DescriptorResponse extends Bundle {
  val compiled = new TmaV2CompiledDescriptor
}

class TmaV2DescriptorMemoryRequest extends Bundle {
  val address = UInt(32.W)
  val asid = UInt(SV32.asidLen.W)
}

class TmaV2DescriptorMemoryResponse extends Bundle {
  val words = Vec(32, UInt(32.W))
}

class TmaV2DescriptorEvents extends Bundle {
  val demandHit = Bool()
  val prefetchHit = Bool()
  val demandMiss = Bool()
  val prefetchMiss = Bool()
  val demandCoalesce = Bool()
  val prefetchCoalesce = Bool()
  val invalidateKill = Bool()
  val compile = Bool()
  val compileCycle = Bool()
  val eviction = Bool()
}

class TmaV2DescriptorService(
    entries: Int = TmaV2Spec.DefaultDescriptorEntries) extends Module {
  require(entries == 2 || entries == 4,
    "compiled descriptor store supports two or four entries")
  private val entryWidth = log2Ceil(entries)
  private val tagWidth = 32 - log2Ceil(TmaV2Spec.DescriptorAlignment)

  val io = IO(new Bundle {
    val invalidateAll = Input(Bool())
    // Client 0 is the implicit demand path. Client 1 is the explicit
    // prefetch/invalidate control path. Both use the same physical store.
    val request = Flipped(Vec(2, Decoupled(new TmaV2DescriptorRequest)))
    val response = Vec(2, Decoupled(new TmaV2DescriptorResponse))
    val memoryRequest = Decoupled(new TmaV2DescriptorMemoryRequest)
    val memoryResponse = Flipped(Decoupled(new TmaV2DescriptorMemoryResponse))
    val events = Output(new TmaV2DescriptorEvents)
  })

  object EntryState {
    val Invalid = 0.U(2.W)
    val Fetching = 1.U(2.W)
    val Compiling = 2.U(2.W)
    val Valid = 3.U(2.W)
  }

  val state = RegInit(VecInit(Seq.fill(entries)(EntryState.Invalid)))
  val tag = Reg(Vec(entries, UInt(tagWidth.W)))
  val tagAsid = Reg(Vec(entries, UInt(SV32.asidLen.W)))
  val data = Reg(Vec(entries, new TmaV2CompiledDescriptor))
  // Each bit names the subtree to evict next. Three bits implement a
  // four-way tree; the two-entry area point uses bit zero only.
  val plru = RegInit(0.U((entries - 1).W))

  val compilerEntry = Reg(UInt(entryWidth.W))
  val refillWait = RegInit(false.B)
  val compileKill = RegInit(false.B)
  val demandWaiter = RegInit(false.B)
  val retryDemand = RegInit(false.B)

  // Demand pins a compiled-store entry until Binder captures it, so the
  // 612-bit compiled payload is never copied into a second response register.
  val responseValid = RegInit(VecInit(Seq.fill(2)(false.B)))
  val demandResponseEntry = Reg(UInt(entryWidth.W))

  val events = WireDefault(0.U.asTypeOf(new TmaV2DescriptorEvents))
  io.events := events

  def compact(words: Vec[UInt]): TmaV2DescriptorPayload = {
    val payload = Wire(new TmaV2DescriptorPayload)
    payload.lowWords(0) := words(TmaV2Spec.HeaderWord)
    payload.lowWords(1) := words(TmaV2Spec.ControlWord)
    payload.lowWords(2) := words(TmaV2Spec.GlobalBaseWord)
    payload.globalBaseHigh := words(TmaV2Spec.GlobalBaseWord + 1)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      payload.lowWords(3 + dimension) :=
        words(TmaV2Spec.GlobalDimsWord + dimension)
      payload.boxDims(dimension) :=
        words(TmaV2Spec.BoxDimsWord + dimension)
      payload.elementStrides(dimension) :=
        words(TmaV2Spec.ElementStridesWord + dimension)
    }
    for (stride <- 0 until TmaV2Spec.RankMax - 1) {
      payload.lowWords(8 + stride) :=
        words(TmaV2Spec.GlobalStridesWord + stride * 2)
      payload.globalStrideHigh(stride) :=
        words(TmaV2Spec.GlobalStridesWord + stride * 2 + 1)
    }
    payload.reservedBad := TmaV2Spec.ReservedWords
      .map(word => words(word) =/= 0.U).reduce(_ || _)
    payload
  }

  val invalidatePending = io.request.map(request =>
    request.valid && request.bits.invalidate).reduce(_ || _)
  for (client <- 0 until 2) {
    io.response(client).valid :=
      responseValid(client) && !invalidatePending && !io.invalidateAll
    io.response(client).bits.compiled := Mux(client.U === 0.U,
      data(demandResponseEntry),
      0.U.asTypeOf(new TmaV2CompiledDescriptor))
    when(io.response(client).fire) {
      responseValid(client) := false.B
    }
    io.request(client).ready := false.B
  }

  // Invalidate > demand > prefetch. A demand hit is therefore never hidden
  // behind a speculative control lookup.
  val selectInvalidate0 = io.request(0).valid &&
    io.request(0).bits.invalidate
  val selectInvalidate1 = !selectInvalidate0 &&
    io.request(1).valid && io.request(1).bits.invalidate
  val selectDemand = !invalidatePending && io.request(0).valid
  val selectPrefetch = !invalidatePending && !io.request(0).valid &&
    io.request(1).valid
  val selectedValid =
    selectInvalidate0 || selectInvalidate1 || selectDemand || selectPrefetch
  val selected = Mux(selectInvalidate1 || selectPrefetch, 1.U, 0.U)
  val selectedRequest = Mux(selected === 0.U,
    io.request(0).bits, io.request(1).bits)
  val selectedTag = selectedRequest.address(31,
    log2Ceil(TmaV2Spec.DescriptorAlignment))
  val selectedAddress = Cat(selectedTag,
    0.U(log2Ceil(TmaV2Spec.DescriptorAlignment).W))
  val selectedDemand = selected === 0.U &&
    selectedRequest.wantResponse && !selectedRequest.invalidate
  val selectedPrefetch = !selectedRequest.wantResponse &&
    !selectedRequest.invalidate

  val validHitVec = VecInit((0 until entries).map { entry =>
    state(entry) === EntryState.Valid &&
      tag(entry) === selectedTag &&
      tagAsid(entry) === selectedRequest.asid
  })
  val validHit = validHitVec.asUInt.orR
  val validHitEntry = PriorityEncoder(validHitVec)
  val inFlightHitVec = VecInit((0 until entries).map { entry =>
    (state(entry) === EntryState.Fetching ||
      state(entry) === EntryState.Compiling) &&
      tag(entry) === selectedTag &&
      tagAsid(entry) === selectedRequest.asid
  })
  val inFlightHit = inFlightHitVec.asUInt.orR
  val inFlightEntry = PriorityEncoder(inFlightHitVec)
  val invalidVec = VecInit(state.map(_ === EntryState.Invalid))
  val hasInvalid = invalidVec.asUInt.orR
  val invalidEntry = PriorityEncoder(invalidVec)
  val replaceableVec = VecInit((0 until entries).map { entry =>
    state(entry) === EntryState.Valid &&
      !(responseValid(0) && demandResponseEntry === entry.U)
  })
  val hasReplaceable = replaceableVec.asUInt.orR
  val plruVictim = if (entries == 2) {
    plru(0).asUInt
  } else {
    Mux(plru(0), Cat(1.U(1.W), plru(2)),
      Cat(0.U(1.W), plru(1)))
  }
  val plruReplaceable = replaceableVec(plruVictim)
  val replacementEntry = Mux(
    plruReplaceable, plruVictim, PriorityEncoder(replaceableVec))
  val allocationEntry = Mux(hasInvalid, invalidEntry, replacementEntry)
  val allocationAvailable = hasInvalid || hasReplaceable

  def touch(entry: UInt): Unit = {
    if (entries == 2) {
      plru := ~entry(0)
    } else {
      plru := Cat(
        Mux(entry(1), ~entry(0), plru(2)),
        Mux(!entry(1), ~entry(0), plru(1)),
        ~entry(1))
    }
  }

  val refillPayload = compact(io.memoryResponse.bits.words)
  // Keep the compiler working set local until the descriptor is complete.
  // This gives the compiled store a single registered commit boundary and
  // keeps validation and multiplier logic away from every store D pin.
  val compiler = Module(new TmaV2DescriptorCompiler)
  compiler.io.in.valid := io.memoryResponse.valid && refillWait
  compiler.io.in.bits := refillPayload
  io.memoryResponse.ready := refillWait && compiler.io.in.ready
  val compilerOccupied =
    refillWait || compiler.io.busy || compiler.io.out.valid
  compiler.io.out.ready :=
    (!demandWaiter || !responseValid(0)) &&
      !invalidatePending && !io.invalidateAll

  io.memoryRequest.valid := false.B
  io.memoryRequest.bits.address := selectedAddress
  io.memoryRequest.bits.asid := selectedRequest.asid

  // A killed demand is retried internally after the invalidation ordering
  // point. Software never observes a stale compiled result or a spurious
  // descriptor error.
  when(retryDemand && !compilerOccupied && allocationAvailable) {
    io.memoryRequest.valid := true.B
    io.memoryRequest.bits.address := Cat(tag(compilerEntry),
      0.U(log2Ceil(TmaV2Spec.DescriptorAlignment).W))
    io.memoryRequest.bits.asid := tagAsid(compilerEntry)
    when(io.memoryRequest.fire) {
      state(allocationEntry) := EntryState.Fetching
      tag(allocationEntry) := tag(compilerEntry)
      tagAsid(allocationEntry) := tagAsid(compilerEntry)
      compilerEntry := allocationEntry
      refillWait := true.B
      compileKill := false.B
      retryDemand := false.B
      when(!hasInvalid) { events.eviction := true.B }
    }
  }

  when(selectedValid && !retryDemand) {
    when(selectedRequest.invalidate) {
      val responseSpace =
        !selectedRequest.wantResponse || !responseValid(selected)
      io.request(selected).ready := responseSpace
      when(io.request(selected).fire) {
        for (entry <- 0 until entries) {
          when(state(entry) =/= EntryState.Invalid &&
              tag(entry) === selectedTag &&
              tagAsid(entry) === selectedRequest.asid) {
            state(entry) := EntryState.Invalid
            when(entry.U === compilerEntry && compilerOccupied &&
                !compileKill) {
              compileKill := true.B
              events.invalidateKill := true.B
            }
          }
        }
        when(responseValid(0) &&
            tag(demandResponseEntry) === selectedTag &&
            tagAsid(demandResponseEntry) === selectedRequest.asid) {
          responseValid(0) := false.B
          demandWaiter := true.B
          retryDemand := true.B
        }
        when(selectedRequest.wantResponse) {
          responseValid(selected) := true.B
        }
      }
    }.elsewhen(validHit) {
      val responseSpace = !selectedDemand || !responseValid(0)
      io.request(selected).ready := responseSpace
      when(io.request(selected).fire) {
        touch(validHitEntry)
        when(selectedDemand) {
          demandResponseEntry := validHitEntry
          responseValid(0) := true.B
          events.demandHit := true.B
        }.otherwise {
          events.prefetchHit := true.B
        }
      }
    }.elsewhen(inFlightHit) {
      val waiterSpace = !selectedDemand || !demandWaiter
      io.request(selected).ready := waiterSpace
      when(io.request(selected).fire) {
        when(selectedDemand) {
          demandWaiter := true.B
          events.demandCoalesce := true.B
        }.otherwise {
          events.prefetchCoalesce := true.B
        }
        assert(inFlightEntry === compilerEntry)
      }
    }.otherwise {
      val demandSpace = !selectedDemand ||
        (!demandWaiter && !responseValid(0))
      val canAllocate =
        !compilerOccupied && demandSpace && allocationAvailable
      when(canAllocate) {
        io.memoryRequest.valid := true.B
        io.memoryRequest.bits.address := selectedAddress
        io.memoryRequest.bits.asid := selectedRequest.asid
      }
      io.request(selected).ready :=
        canAllocate && io.memoryRequest.ready
      when(io.request(selected).fire) {
        state(allocationEntry) := EntryState.Fetching
        tag(allocationEntry) := selectedTag
        tagAsid(allocationEntry) := selectedRequest.asid
        compilerEntry := allocationEntry
        refillWait := true.B
        compileKill := false.B
        when(selectedDemand) {
          demandWaiter := true.B
          events.demandMiss := true.B
        }.otherwise {
          events.prefetchMiss := true.B
        }
        when(!hasInvalid) { events.eviction := true.B }
      }
    }
  }

  when(compiler.io.in.fire) {
    refillWait := false.B
    state(compilerEntry) := EntryState.Compiling
    events.compile := true.B
  }
  when(compiler.io.busy) {
    events.compileCycle := true.B
  }
  when(compiler.io.out.fire) {
    when(compileKill) {
      state(compilerEntry) := EntryState.Invalid
      when(demandWaiter) { retryDemand := true.B }
    }.otherwise {
      data(compilerEntry) := compiler.io.out.bits
      state(compilerEntry) := EntryState.Valid
      touch(compilerEntry)
      when(demandWaiter) {
        demandResponseEntry := compilerEntry
        responseValid(0) := true.B
        demandWaiter := false.B
      }
    }
    compileKill := false.B
  }

  when(io.invalidateAll) {
    state.foreach(_ := EntryState.Invalid)
    plru := 0.U
    responseValid.foreach(_ := false.B)
    when(compilerOccupied && !compileKill) {
      compileKill := true.B
      events.invalidateKill := true.B
    }
    when(demandWaiter) { retryDemand := true.B }
  }

  when(responseValid(0) && !invalidatePending && !io.invalidateAll) {
    assert(state(demandResponseEntry) === EntryState.Valid,
      "a pending demand response must pin one valid compiled entry")
  }
  for (entry <- 0 until entries; other <- 0 until entry) {
    assert(state(entry) === EntryState.Invalid ||
      state(other) === EntryState.Invalid ||
      tag(entry) =/= tag(other) || tagAsid(entry) =/= tagAsid(other),
      "TMA compiled descriptor slots must never hold duplicate tags")
  }
}

class TmaV2EngineCommand extends Bundle {
  val wid = UInt(3.W)
  val copyDirection = UInt(1.W)
  val dtype = UInt(5.W)
  val oobFill = Bool()
  val reduceMode = UInt(3.W)
  val asid = UInt(SV32.asidLen.W)
  val group = UInt(log2Ceil(TmaV2Spec.S2GGroupsPerWarp).W)
  val barrierValid = Bool()
  val barrierId = UInt(log2Ceil(TmaV2Spec.MbarrierEntries).W)
  val barrierGeneration = UInt(8.W)
  val transactionBytes = UInt(32.W)
}

class TmaV2TlbRequest(lineEntries: Int) extends Bundle {
  val source = UInt(log2Ceil(lineEntries).W)
  val virtualAddress = UInt(32.W)
  val asid = UInt(SV32.asidLen.W)
}

class TmaV2TlbResponse(lineEntries: Int) extends Bundle {
  val source = UInt(log2Ceil(lineEntries).W)
  val physicalAddress = UInt(32.W)
}

class TmaV2SharedRequest(lineEntries: Int) extends Bundle {
  val write = Bool()
  val source = UInt(log2Ceil(lineEntries).W)
  val sharedSetIdx = UInt(log2Ceil(sharedmem_depth).W)
  val sharedAtomIndex = Vec(8, UInt(3.W))
  val data = UInt(1024.W)
  val mask = UInt(128.W)
}

class TmaV2SharedResponse(lineEntries: Int) extends Bundle {
  val source = UInt(log2Ceil(lineEntries).W)
  val data = UInt(1024.W)
  // SharedMemory may replay a bank-conflicting 128B request over multiple
  // response beats. One bit identifies each returned 4B bank lane.
  val wordMask = UInt(32.W)
}

class TmaV2CacheRequest(lineEntries: Int) extends Bundle {
  val write = Bool()
  val reduceMode = UInt(3.W)
  val signed = Bool()
  val source = UInt(log2Ceil(lineEntries).W)
  val physicalAddress = UInt(32.W)
  val data = Vec(8, UInt(128.W))
  val mask = UInt(128.W)
}

class TmaV2CacheResponse(lineEntries: Int) extends Bundle {
  val source = UInt(log2Ceil(lineEntries).W)
  val data = Vec(8, UInt(128.W))
}

class TmaV2EngineCompletion extends Bundle {
  val wid = UInt(3.W)
  val copyDirection = UInt(1.W)
  val group = UInt(log2Ceil(TmaV2Spec.S2GGroupsPerWarp).W)
  val barrierValid = Bool()
  val barrierId = UInt(log2Ceil(TmaV2Spec.MbarrierEntries).W)
  val barrierGeneration = UInt(8.W)
  val transactionBytes = UInt(32.W)
}
