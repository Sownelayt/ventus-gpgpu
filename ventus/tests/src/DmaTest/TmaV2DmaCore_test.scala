package DmaTest

import chisel3._
import chiseltest._
import chiseltest.internal.CachingAnnotation
import config.config.Parameters
import L1Cache.MyConfig
import org.scalatest.freespec.AnyFreeSpec
import pipeline._
import top.parameters._

import scala.collection.mutable

class TmaV2AtomicEncodingHarness extends Module {
  val io = IO(new Bundle {
    val mode = Input(UInt(3.W))
    val signed = Input(Bool())
    val opcode = Output(UInt(3.W))
    val param = Output(UInt(3.W))
  })
  io.opcode := TmaV2AtomicEncoding.opcode(io.mode)
  io.param := TmaV2AtomicEncoding.param(io.mode, io.signed)
}

class TmaV2DmaCore_test extends AnyFreeSpec with ChiselScalatestTester {
  implicit val p: Parameters = (new MyConfig).toInstance
  private def compactCore =
    new TmaV2DmaCore(windowEntries = 8, requestEntries = 4,
      sharedEntries = 4, writeAckEntries = 8)

  private case class Completion(wid: BigInt, group: BigInt, s2g: Boolean,
                                barrierValid: Boolean, barrierId: BigInt,
                                bytes: BigInt)
  private case class Reservation(wid: Int, barrierValid: Boolean, barrierId: Int)
  private val pendingReservations = mutable.Queue.empty[Reservation]

  private def initialize(dut: TmaV2DmaCore): Unit = {
    pendingReservations.clear()
    dut.io.dma_req.valid.poke(false.B)
    dut.io.dma_cache_rsp.valid.poke(false.B)
    dut.io.shared_rsp.valid.poke(false.B)
    dut.io.dma_cache_req.ready.poke(true.B)
    dut.io.shared_req.ready.poke(true.B)
    dut.io.tma_completion.ready.poke(true.B)
    dut.io.txReserve.ready.poke(true.B)
    dut.io.txReserveResponse.valid.poke(false.B)
    dut.io.txReserveResponse.bits.wid.poke(0.U)
    dut.io.txReserveResponse.bits.accepted.poke(false.B)
    dut.io.txReserveResponse.bits.barrierValid.poke(false.B)
    dut.io.txReserveResponse.bits.barrierId.poke(0.U)
    dut.io.txReserveResponse.bits.generation.poke(0.U)
    dut.io.to_l2TLB.ready.poke(true.B)
    dut.io.from_l2TLB.valid.poke(false.B)
    dut.io.from_l2TLB.bits.paddr.poke(0.U)
    dut.io.perfEnable.poke(false.B)
    dut.io.perfReset.poke(false.B)
    dut.io.killReq.valid.poke(false.B)
    dut.io.killReq.bits.asid.poke(0.U)
    dut.clock.step(2)
  }

  private def issue(dut: TmaV2DmaCore, funct: Int, wid: Int,
                    in1: Int, in2: Seq[Int], in3: Int,
                    group: Int = 0, barrierValid: Boolean = false,
                    barrierId: Int = 0, transactionBytes: Int = 0,
                    reduceMode: Int = TmaV2Spec.ReduceCopy,
                    bulkReduceType: Int = TmaV2Spec.BulkReduceTypeU32,
                    tensorMapSubop: Int =
                      TmaV2Spec.TensorMapPrefetchSubop,
                    asid: Int = 0): Unit = {
    if (funct == TmaV2Spec.FunctBulkG2S ||
        funct == TmaV2Spec.FunctTensorG2S) {
      pendingReservations.enqueue(Reservation(wid, barrierValid, barrierId))
    }
    for (lane <- 0 until num_thread) {
      dut.io.dma_req.bits.in1(lane).poke((if (lane == 0) in1 else 0).U)
      val in2Value = if (lane < in2.size) in2(lane) else 0
      dut.io.dma_req.bits.in2(lane).poke(
        (BigInt(in2Value) & BigInt("ffffffff", 16)).U)
      dut.io.dma_req.bits.in3(lane).poke((if (lane == 0) in3 else 0).U)
      dut.io.dma_req.bits.mask(lane).poke(true.B)
    }
    dut.io.dma_req.bits.ctrl.dma.poke(true.B)
    val instructionControl =
      if (funct == TmaV2Spec.FunctBulkS2G)
        (BigInt(reduceMode) << 29) | (BigInt(bulkReduceType) << 27)
      else if (funct == TmaV2Spec.FunctTensorS2G)
        BigInt(reduceMode) << 29
      else if (funct == TmaV2Spec.FunctPrefetchTensormap)
        BigInt(tensorMapSubop) << 27
      else BigInt(0)
    dut.io.dma_req.bits.ctrl.inst.poke(instructionControl.U)
    dut.io.dma_req.bits.ctrl.funct.poke(funct.U)
    dut.io.dma_req.bits.ctrl.wid.poke(wid.U)
    dut.io.dma_req.bits.ctrl.dma_group.poke(group.U)
    dut.io.dma_req.bits.ctrl.asid.foreach(_.poke(asid.U))
    dut.io.dma_req.valid.poke(true.B)
    var cycles = 0
    while (!dut.io.dma_req.ready.peekBoolean() && cycles < 100) {
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < 100, s"funct=$funct request never became ready")
    dut.clock.step()
    dut.io.dma_req.valid.poke(false.B)
  }

  private def descriptorWords(dtype: Int = TmaV2Spec.DTypeU8): Seq[BigInt] = {
    val words = Array.fill[BigInt](32)(0)
    words(0) = TmaV2Spec.DescriptorMagic
    words(1) = dtype | (1 << 5)
    words(2) = 0x1000
    words(4) = 16
    words(17) = 16
    words(22) = 1
    words.toSeq
  }

  private def pokeDescriptorPayload(port: TmaV2DescriptorResponse,
                                    words: Seq[BigInt]): Unit = {
    val control = words(1).toInt
    val dtype = control & 0x1f
    val rank = (control >> 5) & 0x7
    val interleave = (control >> 8) & 0x3
    val swizzle = (control >> 10) & 0x7
    val dtypeBits = TmaV2Spec.DTypeBits.getOrElse(dtype, 0)
    val legal = words(0) == TmaV2Spec.DescriptorMagic &&
      rank >= TmaV2Spec.RankMin && rank <= TmaV2Spec.RankMax &&
      TmaV2Spec.SupportedDTypes.contains(dtype)
    port.compiled.status.poke(
      (if (legal) TmaV2Status.Ok.litValue else
        TmaV2Status.BadMagic.litValue).U)
    port.compiled.dtype.poke(dtype.U)
    port.compiled.rank.poke(rank.U)
    port.compiled.interleave.poke(interleave.U)
    port.compiled.swizzle.poke(swizzle.U)
    port.compiled.oobFill.poke(((control >> 18) & 0x1) != 0)
    port.compiled.globalBase.poke(
      words(2).U)
    port.compiled.rawDim0Box.poke(words(17).U)
    var logicalBytes = BigInt(0)
    for (dimension <- 0 until TmaV2Spec.RankMax) {
      val globalDim = if (words(4 + dimension) == 0)
        BigInt(1) << 32 else words(4 + dimension)
      val rawElementStride = words(22 + dimension)
      val elementStride =
        if (dimension < rank && rawElementStride >= 1 && rawElementStride <= 8)
          rawElementStride else BigInt(1)
      val effectiveBox =
        if (dimension < rank)
          (words(17 + dimension) + elementStride - 1) / elementStride
        else BigInt(0)
      port.compiled.globalDims(dimension).poke(globalDim.U)
      port.compiled.boxDims(dimension).poke(effectiveBox.U)
      port.compiled.strideMinus1(dimension).poke(
        (if (dimension < rank) elementStride - 1 else BigInt(0)).U)
      if (dimension < TmaV2Spec.RankMax - 1) {
        val stride = (words(10 + dimension * 2) << 32) |
          words(9 + dimension * 2)
        port.compiled.globalStrides(dimension).poke(stride.U)
      }
    }
    val dim0Stride = if (words(22) >= 1 && words(22) <= 8)
      words(22) else BigInt(1)
    val effectiveDim0 = (words(17) + dim0Stride - 1) / dim0Stride
    val rowBytes = dtypeBits match {
      case 4 => (effectiveDim0 + 1) / 2
      case 6 => (effectiveDim0 * 6 + 7) / 8
      case bits if bits > 0 => effectiveDim0 * bits / 8
      case _ => BigInt(0)
    }
    logicalBytes = rowBytes
    for (dimension <- 1 until rank) {
      val step = if (words(22 + dimension) >= 1 &&
          words(22 + dimension) <= 8) words(22 + dimension) else BigInt(1)
      logicalBytes *= (words(17 + dimension) + step - 1) / step
    }
    port.compiled.logicalBytes.poke(logicalBytes.U)
    val sliceStride =
      if (interleave != TmaV2Spec.InterleaveNone && rank >= 3) {
        val strideIndex = rank - 3
        val stride = (words(10 + strideIndex * 2) << 32) |
          words(9 + strideIndex * 2)
        stride * (if (words(4 + rank - 2) == 0)
          BigInt(1) << 32 else words(4 + rank - 2))
      } else BigInt(0)
    port.compiled.interleaveSliceStride.poke(sliceStride.U)
  }

  private def serviceMany(dut: TmaV2DmaCore, descriptor: Seq[BigInt] = Seq.empty,
                          expectedCompletions: Int, limit: Int = 1000,
                          observeCacheRequest:
                            (Int, Int, BigInt, Seq[BigInt], Seq[BigInt]) => Unit =
                              (_, _, _, _, _) => (),
                          observeSharedRequest:
                            (Boolean, Seq[BigInt], Seq[BigInt],
                              Seq[Boolean], Seq[BigInt]) => Unit =
                              (_, _, _, _, _) => (),
                          observeTlbRequest: BigInt => Unit = _ => ()):
      (Seq[Completion], Seq[Int], Int, Int, Seq[(BigInt, BigInt)], Seq[BigInt]) = {
    val tlbResponses = mutable.Queue.empty[BigInt]
    val cacheResponses = mutable.Queue.empty[(BigInt, Seq[BigInt])]
    val sharedResponses = mutable.Queue.empty[(BigInt, Boolean, Seq[BigInt], Seq[Boolean])]
    val cacheOpcodes = mutable.ArrayBuffer.empty[Int]
    val statuses = mutable.ArrayBuffer.empty[(BigInt, BigInt)]
    val sharedAddresses = mutable.ArrayBuffer.empty[BigInt]
    var sharedReads = 0
    var sharedWrites = 0
    val completions = mutable.ArrayBuffer.empty[Completion]
    var cycles = 0

    while (completions.size < expectedCompletions && cycles < limit) {
      dut.io.txReserveResponse.valid.poke(pendingReservations.nonEmpty.B)
      if (pendingReservations.nonEmpty) {
        val reservation = pendingReservations.front
        dut.io.txReserveResponse.bits.wid.poke(reservation.wid.U)
        dut.io.txReserveResponse.bits.accepted.poke(true.B)
        dut.io.txReserveResponse.bits.barrierValid.poke(reservation.barrierValid.B)
        dut.io.txReserveResponse.bits.barrierId.poke(reservation.barrierId.U)
        dut.io.txReserveResponse.bits.generation.poke(0.U)
      }
      dut.io.from_l2TLB.valid.poke(tlbResponses.nonEmpty.B)
      if (tlbResponses.nonEmpty) dut.io.from_l2TLB.bits.paddr.poke(tlbResponses.front.U)

      dut.io.dma_cache_rsp.valid.poke(cacheResponses.nonEmpty.B)
      if (cacheResponses.nonEmpty) {
        val (source, data) = cacheResponses.front
        dut.io.dma_cache_rsp.bits.d_opcode.poke(0.U)
        dut.io.dma_cache_rsp.bits.d_param.poke(0.U)
        dut.io.dma_cache_rsp.bits.d_addr.poke(0.U)
        dut.io.dma_cache_rsp.bits.d_source.poke(source.U)
        for (word <- 0 until dcache_BlockWords)
          dut.io.dma_cache_rsp.bits.d_data(word).poke(data(word).U)
      }

      dut.io.shared_rsp.valid.poke(sharedResponses.nonEmpty.B)
      if (sharedResponses.nonEmpty) {
        val (source, isWrite, data, mask) = sharedResponses.front
        dut.io.shared_rsp.bits.instrId.poke(source.U)
        dut.io.shared_rsp.bits.isWrite.poke(isWrite.B)
        dut.io.shared_rsp.bits.isMBarrier.poke(false.B)
        for (lane <- 0 until num_thread) {
          dut.io.shared_rsp.bits.data(lane).poke(data(lane).U)
          dut.io.shared_rsp.bits.activeMask(lane).poke(mask(lane).B)
        }
      }

      val tlbReqFire = dut.io.to_l2TLB.valid.peekBoolean()
      val cacheReqFire = dut.io.dma_cache_req.valid.peekBoolean()
      val sharedReqFire = dut.io.shared_req.valid.peekBoolean()
      val tlbRspFire = tlbResponses.nonEmpty && dut.io.from_l2TLB.ready.peekBoolean()
      val cacheRspFire = cacheResponses.nonEmpty && dut.io.dma_cache_rsp.ready.peekBoolean()
      val sharedRspFire = sharedResponses.nonEmpty && dut.io.shared_rsp.ready.peekBoolean()
      val reserveRspFire = pendingReservations.nonEmpty &&
        dut.io.txReserveResponse.ready.peekBoolean()

      val newTlb = if (tlbReqFire) {
        val address = dut.io.to_l2TLB.bits.vaddr.peekInt()
        observeTlbRequest(address)
        Some(address)
      } else None
      val newCache = if (cacheReqFire) {
        val source = dut.io.dma_cache_req.bits.a_source.peekInt()
        val opcode = dut.io.dma_cache_req.bits.a_opcode.peekInt().toInt
        val param = dut.io.dma_cache_req.bits.a_param.peekInt().toInt
        val address = dut.io.dma_cache_req.bits.a_addr.get.peekInt()
        val data = (0 until dcache_BlockWords).map(
          word => dut.io.dma_cache_req.bits.a_data(word).peekInt())
        val mask = (0 until dcache_BlockWords).map(
          word => dut.io.dma_cache_req.bits.a_mask(word).peekInt())
        observeCacheRequest(opcode, param, address, data, mask)
        cacheOpcodes += opcode
        val isDescriptor = source == ((BigInt(1) << l1cache_sourceBits) - 1)
        val payload = if (isDescriptor) descriptor
        else (0 until dcache_BlockWords).map(word => BigInt(0x10000000L + word))
        assert(payload.size == dcache_BlockWords)
        Some((source, payload))
      } else None
      val newShared = if (sharedReqFire) {
        val source = dut.io.shared_req.bits.instrId.peekInt()
        val isWrite = dut.io.shared_req.bits.isWrite.peekBoolean()
        val set = dut.io.shared_req.bits.setIdx.peekInt()
        val block = dut.io.shared_req.bits.perLaneAddr(0).blockOffset.peekInt()
        val addresses = (0 until num_thread).map(lane =>
          dut.io.shared_req.bits.perLaneAddr(lane).blockOffset.peekInt())
        val requestData = (0 until num_thread).map(lane =>
          dut.io.shared_req.bits.data(lane).peekInt())
        val requestByteMasks = (0 until num_thread).map(lane =>
          dut.io.shared_req.bits.perLaneAddr(lane).wordOffset1H.peekInt())
        sharedAddresses += ((set << (dcache_BlockOffsetBits + dcache_WordOffsetBits)) |
          (block << dcache_WordOffsetBits))
        if (isWrite) sharedWrites += 1 else sharedReads += 1
        val data = (0 until num_thread).map(lane => BigInt(0x20000000L + lane))
        val mask = (0 until num_thread).map(
          lane => dut.io.shared_req.bits.perLaneAddr(lane).activeMask.peekBoolean())
        observeSharedRequest(isWrite, addresses, requestData, mask,
          requestByteMasks)
        Some((source, isWrite, data, mask))
      } else None
      if (dut.io.tma_completion.valid.peekBoolean()) {
        completions += Completion(
          dut.io.tma_completion.bits.wid.peekInt(),
          dut.io.tma_completion.bits.group.peekInt(),
          dut.io.tma_completion.bits.is_s2g.peekBoolean(),
          dut.io.tma_completion.bits.barrierValid.peekBoolean(),
          dut.io.tma_completion.bits.barrierId.peekInt(),
          dut.io.tma_completion.bits.transactionBytes.peekInt())
      }
      if (dut.io.status.valid.peekBoolean())
        statuses += ((dut.io.status.bits.code.peekInt(), dut.io.status.bits.detail.peekInt()))

      dut.clock.step()
      if (tlbRspFire) tlbResponses.dequeue()
      if (cacheRspFire) cacheResponses.dequeue()
      if (sharedRspFire) sharedResponses.dequeue()
      if (reserveRspFire) pendingReservations.dequeue()
      newTlb.foreach(tlbResponses.enqueue(_))
      newCache.foreach(cacheResponses.enqueue(_))
      newShared.foreach(sharedResponses.enqueue(_))
      cycles += 1
    }
    if (cycles >= limit) {
      println(s"TMA_V2_TIMEOUT ingressState=${dut.ingress.state.peekInt()} " +
        s"tensorEmit=${dut.execute.tensorEmitActive.peekBoolean()} " +
        s"tlbBusy=${dut.tlbBusy.peekBoolean()} descriptorNeedCache=${dut.descriptorNeedCache.peekBoolean()} " +
        s"descriptorCacheWait=${dut.descriptorCacheWait.peekBoolean()} " +
        s"commandValid=${dut.subsystem.engine.commandValid.peekBoolean()} " +
        s"commandSealed=${dut.subsystem.engine.commandSealed.peekBoolean()} " +
        s"slotOutstanding=${dut.subsystem.engine.slotOutstanding.map(_.peekInt()).mkString(",")} " +
        s"slotState=${dut.subsystem.engine.slotState.map(_.peekInt()).mkString(",")} " +
        s"queuedRsp=tlb:${tlbResponses.size},cache:${cacheResponses.size},shared:${sharedResponses.size} " +
        s"completions=${completions.size}")
    }
    assert(cycles < limit, "TMA v2 physical core did not complete")
    (completions.toSeq, cacheOpcodes.toSeq, sharedReads, sharedWrites,
      statuses.toSeq, sharedAddresses.toSeq)
  }

  private def service(dut: TmaV2DmaCore, descriptor: Seq[BigInt] = Seq.empty,
                      limit: Int = 1000):
      (Completion, Seq[Int], Int, Int, Seq[(BigInt, BigInt)], Seq[BigInt]) = {
    val (completions, opcodes, reads, writes, statuses, addresses) =
      serviceMany(dut, descriptor, expectedCompletions = 1, limit)
    (completions.head, opcodes, reads, writes, statuses, addresses)
  }

  "TMA reduce maps every supported 32-bit operation to the AMO encoding" in {
    test(new TmaV2AtomicEncodingHarness) { dut =>
      def check(mode: Int, signed: Boolean, opcode: Int, param: Int): Unit = {
        dut.io.mode.poke(mode.U)
        dut.io.signed.poke(signed.B)
        dut.clock.step()
        dut.io.opcode.expect(opcode.U)
        dut.io.param.expect(param.U)
      }

      check(TmaV2Spec.ReduceAdd, signed = false, opcode = 2, param = 4)
      check(TmaV2Spec.ReduceMin, signed = true, opcode = 2, param = 0)
      check(TmaV2Spec.ReduceMin, signed = false, opcode = 2, param = 2)
      check(TmaV2Spec.ReduceMax, signed = true, opcode = 2, param = 1)
      check(TmaV2Spec.ReduceMax, signed = false, opcode = 2, param = 3)
      check(TmaV2Spec.ReduceAnd, signed = false, opcode = 3, param = 2)
      check(TmaV2Spec.ReduceOr, signed = false, opcode = 3, param = 1)
      check(TmaV2Spec.ReduceXor, signed = false, opcode = 3, param = 0)
    }
  }

  "physical core executes aligned bulk G2S and retires after shared write ack" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 1,
        in1 = 0x1000, in2 = Seq(16), in3 = 0x2000)
      val (done, opcodes, reads, writes, statuses, _) = service(dut)
      assert(done == Completion(1, 0, s2g = false, barrierValid = false, 0, 0))
      assert(opcodes == Seq(4))
      assert(reads == 0 && writes == 1)
      assert(statuses.isEmpty)
    }
  }

  "physical core merges a cross-line bulk G2S into one shared window" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 1,
        in1 = 0x1070, in2 = Seq(32), in3 = 0x2000)
      val (done, opcodes, reads, writes, statuses, _) = service(dut)
      assert(done == Completion(1, 0, s2g = false,
        barrierValid = false, 0, 0))
      assert(opcodes == Seq(4, 4),
        "one lane crossing a global boundary must issue two unique lines")
      assert(reads == 0 && writes == 2,
        "each returned line segment must retire through one partial write")
      assert(statuses.isEmpty)
    }
  }

  "physical core executes one full 128B G2S line with eight shared atoms" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 1,
        in1 = 0x1000, in2 = Seq(128), in3 = 0x2000,
        barrierValid = true, barrierId = 2, transactionBytes = 128)
      var observed = false
      val (completions, opcodes, reads, writes, statuses, _) =
        serviceMany(dut, expectedCompletions = 1, limit = 1000,
          observeSharedRequest = {
            (isWrite, addresses, data, active, byteMasks) =>
              assert(isWrite)
              assert(addresses == (0 until 32).map(BigInt(_)))
              assert(data == (0 until 32).map(word =>
                BigInt(0x10000000L + word)))
              assert(active.forall(identity))
              assert(byteMasks.forall(_ == 0xf))
              observed = true
          })
      val done = completions.head
      assert(done == Completion(1, 0, s2g = false,
        barrierValid = true, 2, 128))
      assert(opcodes == Seq(4))
      assert(reads == 0 && writes == 1,
        "one full 128B window must issue one 32-bank shared request")
      assert(observed, "full G2S shared request data was not observed")
      assert(statuses.isEmpty)
    }
  }

  "bulk shared chunks never cross a 128B set" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 1,
        in1 = 0x1000, in2 = Seq(128), in3 = 0x2010)
      val (_, _, _, g2sWrites, g2sStatus, g2sShared) = service(dut)
      assert(g2sWrites == 2)
      assert(g2sShared == Seq(0x2010, 0x2080),
        "+0x10/128B must split into 112B plus 16B")
      assert(g2sStatus.isEmpty)

      val reductions = Seq(
        (TmaV2Spec.ReduceCopy, TmaV2Spec.BulkReduceTypeU32),
        (TmaV2Spec.ReduceAdd, TmaV2Spec.BulkReduceTypeU32),
        (TmaV2Spec.ReduceMin, TmaV2Spec.BulkReduceTypeS32),
        (TmaV2Spec.ReduceMax, TmaV2Spec.BulkReduceTypeU32),
        (TmaV2Spec.ReduceAnd, TmaV2Spec.BulkReduceTypeB32))
      reductions.zipWithIndex.foreach { case ((mode, dtype), index) =>
        issue(dut, TmaV2Spec.FunctBulkS2G, wid = index + 2,
          in1 = 0x2070, in2 = Seq(32), in3 = 0x3000 + index * 0x100,
          reduceMode = mode, bulkReduceType = dtype)
        val (_, _, sharedReads, _, statuses, sharedAddresses) = service(dut)
        assert(sharedReads == 2)
        assert(sharedAddresses == Seq(0x2070, 0x2080),
          s"mode=$mode +0x70/32B must split into 16B plus 16B")
        assert(statuses.isEmpty)
      }
    }
  }

  "physical core executes bulk S2G and waits for cache AccessAck" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(16), in3 = 0x1000, group = 3)
      val (done, opcodes, reads, writes, statuses, _) = service(dut)
      assert(done == Completion(2, 3, s2g = true, barrierValid = false, 0, 0))
      assert(opcodes.size == 1 && (opcodes.head == 0 || opcodes.head == 1))
      assert(reads == 1 && writes == 0)
      assert(statuses.isEmpty)
    }
  }

  "physical core emits one AMO per 32-bit bulk reduce element" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val cacheRequests = mutable.ArrayBuffer.empty[
        (Int, Int, BigInt, Seq[BigInt], Seq[BigInt])]
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(32), in3 = 0x1000, group = 3,
        reduceMode = TmaV2Spec.ReduceAdd,
        bulkReduceType = TmaV2Spec.BulkReduceTypeU32)
      val (done, _, reads, writes, statuses, _) = serviceMany(
        dut, expectedCompletions = 1, limit = 3000,
        observeCacheRequest = (opcode, param, address, data, mask) =>
          cacheRequests += ((opcode, param, address, data, mask)))

      assert(done == Seq(Completion(2, 3, s2g = true,
        barrierValid = false, 0, 0)))
      assert(cacheRequests.size == 8,
        s"32B u32 reduce must issue eight AMOs, saw ${cacheRequests.size}")
      assert(cacheRequests.forall(request =>
        request._1 == 2 && request._2 == 4),
        s"bulk add did not use arithmetic-add AMOs: $cacheRequests")
      assert(cacheRequests.forall(request =>
        request._5.count(_ != 0) == 1 &&
          request._5.exists(_ == 0xf)),
        s"bulk reduce emitted a non-word mask: $cacheRequests")
      assert(reads == 1 && writes == 0)
      assert(statuses.isEmpty)
    }
  }

  "physical core carries bulk reduce across cache lines at a 4KiB boundary" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val cacheRequests = mutable.ArrayBuffer.empty[(Int, BigInt)]
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(32), in3 = 0x1ff0, group = 1,
        reduceMode = TmaV2Spec.ReduceXor,
        bulkReduceType = TmaV2Spec.BulkReduceTypeB32)
      val (done, _, reads, writes, statuses, _) = serviceMany(
        dut, expectedCompletions = 1, limit = 3000,
        observeCacheRequest = (opcode, _, address, _, _) =>
          cacheRequests += ((opcode, address)))

      assert(done == Seq(Completion(2, 1, s2g = true,
        barrierValid = false, 0, 0)))
      assert(cacheRequests.size == 8 && cacheRequests.forall(_._1 == 3),
        s"cross-page xor must issue eight logical AMOs: $cacheRequests")
      assert(cacheRequests.map(_._2).distinct.sorted == Seq(0x1f80, 0x2000),
        s"boundary-crossing reduce touched unexpected cache lines: $cacheRequests")
      assert(reads == 1 && writes == 0)
      assert(statuses.isEmpty)
    }
  }

  "physical core propagates signed bulk min to the AMO endpoint" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val encodings = mutable.ArrayBuffer.empty[(Int, Int)]
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(16), in3 = 0x1000,
        reduceMode = TmaV2Spec.ReduceMin,
        bulkReduceType = TmaV2Spec.BulkReduceTypeS32)
      val (_, _, _, _, statuses, _) = serviceMany(
        dut, expectedCompletions = 1, limit = 2000,
        observeCacheRequest = (opcode, param, _, _, _) =>
          encodings += ((opcode, param)))
      assert(encodings.size == 4 && encodings.forall(_ == (2, 0)),
        s"signed min AMO encoding mismatch: $encodings")
      assert(statuses.isEmpty)
    }
  }

  "physical core rejects illegal bulk reduce op-type pairs without traffic" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      val invalid = Seq(
        (TmaV2Spec.ReduceCopy, TmaV2Spec.BulkReduceTypeS32),
        (TmaV2Spec.ReduceAdd, TmaV2Spec.BulkReduceTypeB32),
        (TmaV2Spec.ReduceAnd, TmaV2Spec.BulkReduceTypeU32),
        (TmaV2Spec.ReduceReserved, TmaV2Spec.BulkReduceTypeU32),
        (TmaV2Spec.ReduceAdd, TmaV2Spec.BulkReduceTypeReserved))
      invalid.zipWithIndex.foreach { case ((mode, elementType), index) =>
        initialize(dut)
        issue(dut, TmaV2Spec.FunctBulkS2G, wid = index,
          in1 = 0x2000, in2 = Seq(16), in3 = 0x1000,
          reduceMode = mode, bulkReduceType = elementType)
        var tlbRequests = 0
        val (done, opcodes, reads, writes, statuses, _) = serviceMany(
          dut, expectedCompletions = 1,
          observeTlbRequest = _ => tlbRequests += 1)
        assert(done.size == 1)
        assert(tlbRequests == 0 && opcodes.isEmpty &&
          reads == 0 && writes == 0,
          s"illegal bulk reduce encoding $mode/$elementType generated traffic")
        assert(statuses == Seq((
          BigInt(TmaV2Spec.StatusUnsupportedFeature),
          BigInt(TmaV2Spec.FunctBulkS2G))))
      }
    }
  }

  "physical core collects an aligned partial-line bulk S2G into one cache line" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2010, in2 = Seq(16), in3 = 0x1010, group = 1)
      val (done, opcodes, reads, writes, statuses, _) = service(dut)
      assert(done == Completion(2, 1, s2g = true, barrierValid = false, 0, 0))
      assert(opcodes.size == 1 && opcodes.head == 1)
      assert(reads == 1 && writes == 0,
        "partial S2G must remain one masked full-window shared read")
      assert(statuses.isEmpty)
    }
  }

  "physical core splits a cross-line bulk S2G after one shared read" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(32), in3 = 0x1070, group = 1)
      val (done, opcodes, reads, writes, statuses, _) = service(dut)
      assert(done == Completion(2, 1, s2g = true,
        barrierValid = false, 0, 0))
      assert(opcodes.size == 2 && opcodes.forall(opcode =>
        opcode == 0 || opcode == 1))
      assert(reads == 1 && writes == 0,
        "both global line segments must reuse one complete shared read")
      assert(statuses.isEmpty)
    }
  }

  "physical core rejects invalid bulk alignment and length without memory traffic" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      val invalidRequests = Seq(
        (TmaV2Spec.FunctBulkG2S, 0x1004, 16, 0x2000),
        (TmaV2Spec.FunctBulkG2S, 0x1000, 16, 0x2004),
        (TmaV2Spec.FunctBulkG2S, 0x1000, 20, 0x2000),
        (TmaV2Spec.FunctBulkG2S, 0x1000, 0, 0x2000),
        (TmaV2Spec.FunctBulkS2G, 0x2004, 16, 0x1000),
        (TmaV2Spec.FunctBulkS2G, 0x2000, 16, 0x1004),
        (TmaV2Spec.FunctBulkS2G, 0x2000, 20, 0x1000),
        (TmaV2Spec.FunctBulkS2G, 0x2000, 0, 0x1000))

      invalidRequests.zipWithIndex.foreach { case ((funct, in1, bytes, in3), index) =>
        initialize(dut)
        issue(dut, funct, wid = index & 7, in1 = in1,
          in2 = Seq(bytes), in3 = in3)
        var tlbRequests = 0
        val (done, opcodes, reads, writes, statuses, _) = serviceMany(
          dut, expectedCompletions = 1,
          observeTlbRequest = _ => tlbRequests += 1)
        assert(done.size == 1)
        assert(tlbRequests == 0 && opcodes.isEmpty && reads == 0 && writes == 0,
          s"invalid bulk request $index generated memory traffic")
        assert(statuses == Seq((
          BigInt(TmaV2Spec.StatusUnsupportedFeature), BigInt(funct))))
      }
    }
  }

  "unknown funct0 completes unsupported without memory traffic" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, funct = 0, wid = 1, in1 = 0x1000,
        in2 = Seq(16), in3 = 0x2000)
      var tlbRequests = 0
      val (done, opcodes, reads, writes, statuses, _) = serviceMany(
        dut, expectedCompletions = 1,
        observeTlbRequest = _ => tlbRequests += 1)
      assert(done == Seq(
        Completion(1, 0, s2g = false, barrierValid = false, 0, 0)))
      assert(tlbRequests == 0 && opcodes.isEmpty && reads == 0 && writes == 0)
      assert(statuses == Seq((
        BigInt(TmaV2Spec.StatusUnsupportedFeature), BigInt(0))))
    }
  }

  "physical core executes one full 128B S2G line with eight shared atoms" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(128), in3 = 0x1000, group = 3)
      var observed = false
      val (completions, opcodes, reads, writes, statuses, _) =
        serviceMany(dut, expectedCompletions = 1, limit = 1000,
          observeCacheRequest = { (opcode, _, _, data, mask) =>
            assert(opcode == 0)
            assert(data == (0 until 32).map(word =>
              BigInt(0x20000000L + word)))
            assert(mask.forall(_ == 0xf))
            observed = true
          })
      val done = completions.head
      assert(done == Completion(2, 3, s2g = true, barrierValid = false, 0, 0))
      assert(opcodes.size == 1 && (opcodes.head == 0 || opcodes.head == 1))
      assert(reads == 1 && writes == 0,
        "one full 128B window must issue one 32-bank shared request")
      assert(observed, "full S2G cache request data was not observed")
      assert(statuses.isEmpty)
    }
  }

  "physical core fetches a v2 descriptor and executes tensor G2S" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000)
      val (done, opcodes, reads, writes, statuses, _) = service(dut, descriptorWords())
      assert(done.wid == 3 && !done.s2g)
      assert(opcodes == Seq(4, 4))
      assert(reads == 0 && writes == 1)
      assert(statuses.isEmpty)
    }
  }

  "physical core executes an outer element stride with dense shared packing" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val descriptor = Array.fill[BigInt](32)(0)
      descriptor(0) = TmaV2Spec.DescriptorMagic
      descriptor(1) = TmaV2Spec.DTypeU8 | (2 << 5)
      descriptor(2) = 0x1000
      descriptor(4) = 128
      descriptor(5) = 64
      descriptor(9) = 128
      descriptor(17) = 128
      descriptor(18) = 64
      descriptor(22) = 1
      descriptor(23) = 2
      val payloadAddresses = mutable.ArrayBuffer.empty[BigInt]

      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000,
        barrierValid = true, barrierId = 2, transactionBytes = 4096)
      val (done, opcodes, reads, writes, statuses, _) = serviceMany(
        dut, descriptor.toSeq, expectedCompletions = 1, limit = 10000,
        observeCacheRequest = (_, _, address, _, _) => {
          if (address != 0x4000) payloadAddresses += address
        })

      assert(done == Seq(
        Completion(3, 0, s2g = false, barrierValid = true, 2, 4096)))
      assert(opcodes.count(_ == 4) == 33,
        s"descriptor plus 32 strided source lines were expected: $opcodes")
      val expectedAddresses = (0 until 32).map(index =>
        BigInt(0x1000 + index * 256))
      assert(payloadAddresses == expectedAddresses,
        s"outer stride addresses were $payloadAddresses")
      assert(reads == 0 && writes == 32)
      assert(statuses.isEmpty)
    }
  }

  "physical core completes a two-window interleave16 tensor G2S" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val descriptor = Array.fill[BigInt](32)(0)
      descriptor(0) = TmaV2Spec.DescriptorMagic
      descriptor(1) = TmaV2Spec.DTypeFp32 | (3 << 5) | (1 << 8)
      descriptor(2) = 0x1000
      descriptor(4) = 8
      descriptor(5) = 3
      descriptor(6) = 2
      descriptor(9) = 32
      descriptor(11) = 192
      descriptor(17) = 8
      descriptor(18) = 3
      descriptor(19) = 2
      descriptor(22) = 1
      descriptor(23) = 1
      descriptor(24) = 1
      val payloadAddresses = mutable.ArrayBuffer.empty[BigInt]

      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000,
        barrierValid = true, barrierId = 2, transactionBytes = 256)
      val (done, _, reads, writes, statuses, _) = serviceMany(
        dut, descriptor.toSeq, expectedCompletions = 1, limit = 5000,
        observeCacheRequest = (_, _, address, _, _) => {
          if (address != 0x4000) payloadAddresses += address
        })

      assert(done.size == 1 && !done.head.s2g)
      assert(done.head.barrierValid && done.head.barrierId == 2 &&
        done.head.bytes == 256,
        s"interleave16 must account eight 16B atoms in each of two slices: " +
          s"${done.head}")
      assert(payloadAddresses.distinct.size == 3,
        s"interleave16 expected three unique payload lines, saw " +
          payloadAddresses.map(_.toString(16)))
      assert(payloadAddresses.size == 3,
        s"interleave16 duplicated payload lines: " +
          payloadAddresses.map(_.toString(16)))
      assert(reads == 0 && writes == 3,
        "corrected channel-metadata traversal consumes each cache line once")
      assert(statuses.isEmpty)
    }
  }

  "physical core prefetches, invalidates and refetches one TensorMap address" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctPrefetchTensormap, wid = 3,
        in1 = 0x4000, in2 = Seq.empty, in3 = 0,
        tensorMapSubop = TmaV2Spec.TensorMapPrefetchSubop)
      issue(dut, TmaV2Spec.FunctPrefetchTensormap, wid = 3,
        in1 = 0x4000, in2 = Seq.empty, in3 = 0,
        tensorMapSubop = TmaV2Spec.TensorMapInvalidateSubop)
      val (firstDone, firstOpcodes, _, _, firstStatus, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 2)
      assert(firstDone.size == 2 && firstDone.forall(_.wid == 3))
      assert(firstOpcodes == Seq(4),
        "invalidate must wait for the in-flight prefetch refill")
      assert(firstStatus.isEmpty)

      issue(dut, TmaV2Spec.FunctPrefetchTensormap, wid = 3,
        in1 = 0x4000, in2 = Seq.empty, in3 = 0,
        tensorMapSubop = TmaV2Spec.TensorMapPrefetchSubop)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000)
      val (secondDone, refillOpcodes, _, writes, refillStatus, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 2)
      assert(secondDone.size == 2 && secondDone.forall(_.wid == 3))
      assert(refillOpcodes == Seq(4, 4),
        "prefetch after targeted invalidation must issue a fresh 128B Get")
      assert(writes == 1)
      assert(refillStatus.isEmpty)
    }
  }

  "physical core invalidates every local TensorMap without an address operand" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 1,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000)
      val (_, firstOpcodes, _, _, firstStatus, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 1)
      assert(firstOpcodes.count(_ == 4) == 2,
        "first tensor command must fetch descriptor and payload")
      assert(firstStatus.isEmpty)

      // IVALL has no address operand; an intentionally unaligned x-register
      // value must not be interpreted as a descriptor address.
      issue(dut, TmaV2Spec.FunctPrefetchTensormap, wid = 1,
        in1 = 3, in2 = Seq.empty, in3 = 0,
        tensorMapSubop = TmaV2Spec.TensorMapInvalidateAllSubop)
      val (invalidateDone, invalidateOpcodes, _, _, invalidateStatus, _) =
        serviceMany(dut, expectedCompletions = 1)
      assert(invalidateDone.size == 1)
      assert(invalidateOpcodes.isEmpty)
      assert(invalidateStatus.isEmpty)

      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 1,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2200)
      val (_, secondOpcodes, _, _, secondStatus, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 1)
      assert(secondOpcodes.count(_ == 4) == 2,
        "IVALL must force the next tensor demand to refill its descriptor")
      assert(secondStatus.isEmpty)
    }
  }

  "negative Tensor S2G coordinates stop after descriptor validation" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      for ((mode, index) <- Seq(
          TmaV2Spec.ReduceCopy, TmaV2Spec.ReduceAdd).zipWithIndex) {
        initialize(dut)
        val requests = mutable.ArrayBuffer.empty[(Int, BigInt)]
        issue(dut, TmaV2Spec.FunctTensorS2G, wid = index + 1,
          in1 = 0x4000, in2 = Seq(-1, 0, 0, 0, 0), in3 = 0x2000,
          reduceMode = mode)
        val (done, _, reads, writes, statuses, _) = serviceMany(
          dut, descriptorWords(), expectedCompletions = 1,
          observeCacheRequest = (opcode, _, address, _, _) =>
            requests += ((opcode, address)))
        assert(done.size == 1)
        assert(requests.forall(_ == (4, BigInt(0x4000))) && requests.size <= 1,
          s"negative Tensor S2G mode=$mode issued payload traffic: $requests")
        assert(reads == 0 && writes == 0)
        assert(statuses.exists { case (code, detail) =>
          code == TmaV2Spec.StatusInvalidDescriptor &&
            detail == TmaV2Status.BadCoordinate.litValue
        }, s"negative Tensor S2G mode=$mode statuses=$statuses")
      }
    }
  }

  "physical core uses x10 as tensor S2G shared base" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorS2G, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000, group = 2)
      val (done, opcodes, reads, writes, statuses, sharedAddresses) =
        service(dut, descriptorWords())
      assert(done.wid == 3 && done.s2g && done.group == 2)
      assert(opcodes.size == 2 && opcodes.head == 4 &&
        (opcodes.last == 0 || opcodes.last == 1))
      assert(reads == 1 && writes == 0)
      assert(sharedAddresses == Seq(0x2000))
      assert(statuses.isEmpty)
    }
  }

  "physical core emits one AMO per tensor reduce element" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val cacheRequests = mutable.ArrayBuffer.empty[
        (Int, Int, BigInt, Seq[BigInt], Seq[BigInt])]
      issue(dut, TmaV2Spec.FunctTensorS2G, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000, group = 2,
        reduceMode = TmaV2Spec.ReduceAdd)
      val (done, opcodes, reads, writes, statuses, _) = serviceMany(
        dut, descriptorWords(TmaV2Spec.DTypeU32), expectedCompletions = 1,
        limit = 5000,
        observeCacheRequest = (opcode, param, address, data, mask) =>
          cacheRequests += ((opcode, param, address, data, mask)))

      assert(done.size == 1 && done.head.wid == 3 && done.head.s2g &&
        done.head.group == 2)
      val elementRequests = cacheRequests.filter(_._3 == 0x1000)
      assert(elementRequests.size == 16,
        s"16 elements must each issue one AMO, saw ${elementRequests.size}")
      assert(elementRequests.forall(request => request._1 == 2 && request._2 == 4),
        s"reduce requests were not arithmetic-add AMOs: $elementRequests")
      val requestsByWord = elementRequests.sortBy(request =>
        request._5.indexWhere(_ != 0))
      requestsByWord.zipWithIndex.foreach { case (request, word) =>
        assert(request._5.count(_ != 0) == 1 && request._5(word) == 0xf,
          s"word $word AMO mask was ${request._5}")
        assert(request._4(word) == BigInt(0x20000000L + word),
          s"word $word operand was 0x${request._4(word).toString(16)}")
      }
      assert(!elementRequests.exists(request => request._1 == 0 ||
        request._1 == 1 || request._1 == 4),
        s"reduce unexpectedly used ordinary Get/Put: $elementRequests")
      assert(reads == 1 && writes == 0)
      assert(statuses.isEmpty)
    }
  }

  "physical core reduces compacted interleave16 stride atoms" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      val descriptor = Array.fill[BigInt](32)(0)
      descriptor(0) = TmaV2Spec.DescriptorMagic
      descriptor(1) = TmaV2Spec.DTypeU32 | (3 << 5) |
        (TmaV2Spec.Interleave16 << 8)
      descriptor(2) = 0x1000
      descriptor(4) = 8
      descriptor(5) = 4
      descriptor(6) = 1
      descriptor(9) = 128
      descriptor(11) = 512
      descriptor(17) = 8
      descriptor(18) = 4
      descriptor(19) = 1
      descriptor(22) = 3
      descriptor(23) = 1
      descriptor(24) = 1

      val cacheRequests = mutable.ArrayBuffer.empty[
        (Int, Int, BigInt, Seq[BigInt], Seq[BigInt])]
      issue(dut, TmaV2Spec.FunctTensorS2G, wid = 4,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000, group = 1,
        reduceMode = TmaV2Spec.ReduceAdd)
      val (done, _, reads, writes, statuses, _) = serviceMany(
        dut, descriptor.toSeq, expectedCompletions = 1, limit = 5000,
        observeCacheRequest = (opcode, param, address, data, mask) =>
          cacheRequests += ((opcode, param, address, data, mask)))

      assert(done == Seq(
        Completion(4, 1, s2g = true, barrierValid = false, 0, 0)))
      val elementRequests = cacheRequests.filter(_._3 == 0x1000)
      assert(elementRequests.size == 12,
        s"three selected 16B atoms must issue 12 U32 AMOs, saw " +
          elementRequests.size)
      assert(elementRequests.forall(request => request._1 == 2 && request._2 == 4),
        s"interleave reduction did not use arithmetic-add AMOs: $elementRequests")

      val expectedTargetWords =
        Seq(0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27)
      val requestsByTarget = elementRequests.sortBy(request =>
        request._5.indexWhere(_ != 0))
      requestsByTarget.zip(expectedTargetWords).zipWithIndex.foreach {
        case ((request, targetWord), denseWord) =>
          assert(request._5.count(_ != 0) == 1 &&
            request._5(targetWord) == 0xf,
            s"target word $targetWord AMO mask was ${request._5}")
          assert(request._4(targetWord) == BigInt(0x20000000L + denseWord),
            s"target word $targetWord operand was " +
              s"0x${request._4(targetWord).toString(16)}")
      }
      assert(reads == 1 && writes == 0,
        "the three densely packed atoms must use one shared-memory read")
      assert(statuses.isEmpty)
    }
  }

  "physical core completes tensor G2S before accepting bulk S2G" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000,
        barrierValid = true, barrierId = 2, transactionBytes = 16)
      val (tensorDone, tensorOpcodes, tensorReads, tensorWrites,
        tensorStatuses, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 1,
          limit = 300)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 3,
        in1 = 0x2200, in2 = Seq(32), in3 = 0x3000, group = 1)

      val (bulkDone, bulkOpcodes, bulkReads, bulkWrites,
        bulkStatuses, _) =
        serviceMany(dut, expectedCompletions = 1, limit = 300)
      val done = tensorDone ++ bulkDone
      val opcodes = tensorOpcodes ++ bulkOpcodes
      assert(done == Seq(
        Completion(3, 0, s2g = false, barrierValid = true, 2, 16),
        Completion(3, 1, s2g = true, barrierValid = false, 0, 0)))
      assert(opcodes.count(_ == 4) == 2)
      assert(opcodes.count(opcode => opcode == 0 || opcode == 1) == 1)
      assert(tensorReads + bulkReads == 1 &&
        tensorWrites + bulkWrites == 1)
      assert((tensorStatuses ++ bulkStatuses).isEmpty)
    }
  }

  "unsupported tensor G2S seals without payload and does not consume barrier credit" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 4,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000,
        barrierValid = true, barrierId = 2, transactionBytes = 64)
      val unsupported = descriptorWords().toArray
      unsupported(1) = unsupported(1) | (BigInt(1) << 16)
      val (done, opcodes, reads, writes, statuses, _) =
        service(dut, unsupported.toSeq)
      assert(done == Completion(4, 0, s2g = false, barrierValid = false, 0, 0))
      assert(opcodes == Seq(4))
      assert(reads == 0 && writes == 0)
      assert(statuses.exists(_._1 == TmaV2Spec.StatusUnsupportedFeature))
    }
  }

  "ASID kill cancels an active command and every queued command exactly once" in {
    test(new TmaV2DmaCore(windowEntries = 4, requestEntries = 4,
      sharedEntries = 4, writeAckEntries = 8, commandEntries = 4))
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      dut.io.shared_req.ready.poke(false.B)
      dut.io.dma_cache_req.ready.poke(false.B)
      for (wid <- 0 until 3) {
        issue(dut, TmaV2Spec.FunctBulkS2G, wid = wid,
          in1 = 0x2000 + wid * 0x100, in2 = Seq(16),
          in3 = 0x1000 + wid * 0x100, group = wid)
      }

      var waitActive = 0
      while (!dut.io.shared_req.valid.peekBoolean() && waitActive < 80) {
        dut.clock.step()
        waitActive += 1
      }
      assert(waitActive < 80)
      val heldSharedSource = dut.io.shared_req.bits.instrId.peekInt()
      val heldSharedWrite = dut.io.shared_req.bits.isWrite.peekBoolean()
      val heldSharedSet = dut.io.shared_req.bits.setIdx.peekInt()
      val heldSharedActive = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.perLaneAddr(lane).activeMask.peekBoolean())
      val heldSharedBlocks = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.perLaneAddr(lane).blockOffset.peekInt())
      val heldSharedMasks = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.perLaneAddr(lane).wordOffset1H.peekInt())
      val heldSharedData = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.data(lane).peekInt())
      def expectHeldShared(): Unit = {
        dut.io.shared_req.valid.expect(true.B)
        dut.io.shared_req.bits.instrId.expect(heldSharedSource.U)
        dut.io.shared_req.bits.isWrite.expect(heldSharedWrite.B)
        dut.io.shared_req.bits.setIdx.expect(heldSharedSet.U)
        for (lane <- 0 until num_thread) {
          dut.io.shared_req.bits.perLaneAddr(lane).activeMask.expect(
            heldSharedActive(lane).B)
          dut.io.shared_req.bits.perLaneAddr(lane).blockOffset.expect(
            heldSharedBlocks(lane).U)
          dut.io.shared_req.bits.perLaneAddr(lane).wordOffset1H.expect(
            heldSharedMasks(lane).U)
          dut.io.shared_req.bits.data(lane).expect(heldSharedData(lane).U)
        }
      }
      // Make the request externally irrevocable before accepting kill.
      dut.clock.step()

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.io.killReq.ready.expect(true.B)
      expectHeldShared()
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)

      var completions = Vector.empty[(BigInt, BigInt)]
      def sampleCompletion(): Unit = {
        if (dut.io.tma_completion.valid.peekBoolean())
          completions :+= (
            dut.io.tma_completion.bits.wid.peekInt(),
            dut.io.tma_completion.bits.group.peekInt())
      }
      // valid and every payload bit must remain stable while the external
      // sink is stalled, even though the owning command is now killed.
      for (_ <- 0 until 3) {
        expectHeldShared()
        dut.io.killDone.valid.expect(false.B)
        sampleCompletion()
        dut.clock.step()
      }
      dut.io.shared_req.ready.poke(true.B)
      expectHeldShared()
      sampleCompletion()
      dut.clock.step()
      dut.io.shared_req.ready.poke(false.B)
      dut.io.shared_req.valid.expect(false.B)

      // The irrevocable read owns its response after the delayed request
      // handshake.  Drain that response; kill must discard its data rather
      // than creating a new cache request.
      dut.io.shared_rsp.bits.instrId.poke(heldSharedSource.U)
      dut.io.shared_rsp.bits.isWrite.poke(false.B)
      dut.io.shared_rsp.bits.isMBarrier.poke(false.B)
      for (lane <- 0 until num_thread) {
        dut.io.shared_rsp.bits.data(lane).poke((0x300 + lane).U)
        dut.io.shared_rsp.bits.activeMask(lane).poke(
          heldSharedActive(lane).B)
      }
      dut.io.shared_rsp.valid.poke(true.B)
      var responseWait = 0
      while (!dut.io.shared_rsp.ready.peekBoolean() && responseWait < 20) {
        sampleCompletion()
        dut.clock.step()
        responseWait += 1
      }
      assert(responseWait < 20)
      sampleCompletion()
      dut.clock.step()
      dut.io.shared_rsp.valid.poke(false.B)

      var sawDone = false
      var cycles = 0
      while (!sawDone && cycles < 120) {
        assert(!dut.io.shared_req.valid.peekBoolean(),
          "killed command emitted a new shared request")
        assert(!dut.io.dma_cache_req.valid.peekBoolean(),
          "killed command emitted a new cache request")
        sampleCompletion()
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone, "queued/active TMA kill did not drain")
      assert(completions.sorted == Vector((BigInt(0), BigInt(0)),
        (BigInt(1), BigInt(1)), (BigInt(2), BigInt(2))))
    }
  }

  "ASID kill wins over a same-cycle idle-Prepare dequeue" in {
    test(new TmaV2DmaCore(windowEntries = 4, requestEntries = 4,
      sharedEntries = 4, writeAckEntries = 8, commandEntries = 1))
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      // issue() returns on the enqueue edge.  The following cycle therefore
      // has a resident FIFO head and an idle, ready Prepare stage.
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 6,
        in1 = 0x1000, in2 = Seq(128), in3 = 0x2000,
        group = 2, asid = 0)
      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.io.killReq.ready.expect(true.B)
      dut.io.txReserve.valid.expect(false.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)

      var completion: Option[Completion] = None
      var sawDone = false
      var cycles = 0
      while (!sawDone && cycles < 40) {
        assert(!dut.io.txReserve.valid.peekBoolean(),
          "same-cycle killed head escaped into Prepare reservation")
        assert(!dut.io.dma_cache_req.valid.peekBoolean(),
          "same-cycle killed head issued descriptor or payload traffic")
        assert(!dut.io.shared_req.valid.peekBoolean(),
          "same-cycle killed head issued shared-memory traffic")
        if (dut.io.tma_completion.valid.peekBoolean()) {
          assert(completion.isEmpty, "killed queued command completed twice")
          completion = Some(Completion(
            dut.io.tma_completion.bits.wid.peekInt(),
            dut.io.tma_completion.bits.group.peekInt(),
            dut.io.tma_completion.bits.is_s2g.peekBoolean(),
            dut.io.tma_completion.bits.barrierValid.peekBoolean(),
            dut.io.tma_completion.bits.barrierId.peekInt(),
            dut.io.tma_completion.bits.transactionBytes.peekInt()))
        }
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }

      assert(sawDone, "same-cycle queued kill did not drain")
      assert(completion.contains(Completion(6, 2, s2g = false,
        barrierValid = false, barrierId = 0, bytes = 0)))
    }
  }

  "active tensor kill flushes a backpressured planner and the next tensor recovers" in {
    test(new TmaV2DmaCore(windowEntries = 2, requestEntries = 4,
      sharedEntries = 4, writeAckEntries = 8, commandEntries = 2))
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)

      val longDescriptor = Array.fill[BigInt](32)(0)
      longDescriptor(0) = TmaV2Spec.DescriptorMagic
      longDescriptor(1) = TmaV2Spec.DTypeU8 | (2 << 5)
      longDescriptor(2) = 0x1000
      longDescriptor(4) = 256
      longDescriptor(5) = 128
      longDescriptor(9) = 256
      longDescriptor(17) = 256
      longDescriptor(18) = 128
      longDescriptor(22) = 1
      longDescriptor(23) = 1

      // Populate the descriptor cache first so the command reaches Execute
      // without a descriptor-transport dependency during the kill scenario.
      issue(dut, TmaV2Spec.FunctPrefetchTensormap, wid = 1,
        in1 = 0x4000, in2 = Seq.empty, in3 = 0,
        tensorMapSubop = TmaV2Spec.TensorMapPrefetchSubop)
      val (prefetchDone, _, _, _, prefetchStatus, _) = serviceMany(
        dut, longDescriptor.toSeq, expectedCompletions = 1, limit = 1000)
      assert(prefetchDone.size == 1 && prefetchStatus.isEmpty)
      // serviceMany may leave the just-consumed producer payload driven for
      // its final sampled cycle; manual stepping below must not replay it.
      dut.io.from_l2TLB.valid.poke(false.B)
      dut.io.dma_cache_rsp.valid.poke(false.B)
      dut.io.shared_rsp.valid.poke(false.B)
      dut.io.txReserveResponse.valid.poke(false.B)

      dut.io.dma_cache_req.ready.poke(false.B)
      dut.io.shared_req.ready.poke(false.B)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 5,
        in1 = 0x4000, in2 = Seq(0, 0), in3 = 0x2000,
        group = 3)

      var plannerBackpressured = false
      var cycles = 0
      while (!plannerBackpressured && cycles < 200) {
        dut.io.txReserveResponse.valid.poke(pendingReservations.nonEmpty.B)
        if (pendingReservations.nonEmpty) {
          val reservation = pendingReservations.front
          dut.io.txReserveResponse.bits.wid.poke(reservation.wid.U)
          dut.io.txReserveResponse.bits.accepted.poke(true.B)
          dut.io.txReserveResponse.bits.barrierValid.poke(false.B)
          dut.io.txReserveResponse.bits.barrierId.poke(0.U)
          dut.io.txReserveResponse.bits.generation.poke(0.U)
        }
        val reserveFire = pendingReservations.nonEmpty &&
          dut.io.txReserveResponse.ready.peekBoolean()
        plannerBackpressured = dut.io.dma_cache_req.valid.peekBoolean() &&
          !dut.io.dma_cache_req.ready.peekBoolean()
        dut.clock.step()
        if (reserveFire) pendingReservations.dequeue()
        cycles += 1
      }
      assert(plannerBackpressured,
        "long tensor never reached the stalled cache/engine boundary")
      // Leave the cache stalled long enough to fill both engine window slots
      // and propagate backpressure through every elastic planner stage.
      dut.clock.step(16)
      dut.io.txReserveResponse.valid.poke(false.B)
      val heldCacheOpcode = dut.io.dma_cache_req.bits.a_opcode.peekInt()
      val heldCacheParam = dut.io.dma_cache_req.bits.a_param.peekInt()
      val heldCacheSource = dut.io.dma_cache_req.bits.a_source.peekInt()
      val heldCacheAddress = dut.io.dma_cache_req.bits.a_addr.get.peekInt()
      val heldCacheData = (0 until dcache_BlockWords).map(word =>
        dut.io.dma_cache_req.bits.a_data(word).peekInt())
      val heldCacheMasks = (0 until dcache_BlockWords).map(word =>
        dut.io.dma_cache_req.bits.a_mask(word).peekInt())
      def expectHeldCache(): Unit = {
        dut.io.dma_cache_req.valid.expect(true.B)
        dut.io.dma_cache_req.bits.a_opcode.expect(heldCacheOpcode.U)
        dut.io.dma_cache_req.bits.a_param.expect(heldCacheParam.U)
        dut.io.dma_cache_req.bits.a_source.expect(heldCacheSource.U)
        dut.io.dma_cache_req.bits.a_addr.get.expect(heldCacheAddress.U)
        for (word <- 0 until dcache_BlockWords) {
          dut.io.dma_cache_req.bits.a_data(word).expect(heldCacheData(word).U)
          dut.io.dma_cache_req.bits.a_mask(word).expect(heldCacheMasks(word).U)
        }
      }

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.io.killReq.ready.expect(true.B)
      expectHeldCache()
      dut.io.shared_req.valid.expect(false.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)

      var killedCompletions = Vector.empty[Completion]
      def sampleKilledCompletion(): Unit = {
        if (dut.io.tma_completion.valid.peekBoolean()) {
          killedCompletions :+= Completion(
            dut.io.tma_completion.bits.wid.peekInt(),
            dut.io.tma_completion.bits.group.peekInt(),
            dut.io.tma_completion.bits.is_s2g.peekBoolean(),
            dut.io.tma_completion.bits.barrierValid.peekBoolean(),
            dut.io.tma_completion.bits.barrierId.peekInt(),
            dut.io.tma_completion.bits.transactionBytes.peekInt())
        }
      }
      for (_ <- 0 until 3) {
        expectHeldCache()
        dut.io.killDone.valid.expect(false.B)
        sampleKilledCompletion()
        dut.clock.step()
      }
      // Only the request which was already visible before kill may cross the
      // boundary.  Its response is drained and discarded by kill mode.
      dut.io.dma_cache_req.ready.poke(true.B)
      expectHeldCache()
      sampleKilledCompletion()
      dut.clock.step()
      dut.io.dma_cache_req.ready.poke(false.B)
      dut.io.dma_cache_req.valid.expect(false.B)

      dut.io.dma_cache_rsp.bits.d_opcode.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_param.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_addr.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_source.poke(heldCacheSource.U)
      dut.io.dma_cache_rsp.bits.d_data.foreach(_.poke(0.U))
      dut.io.dma_cache_rsp.valid.poke(true.B)
      var responseWait = 0
      while (!dut.io.dma_cache_rsp.ready.peekBoolean() && responseWait < 20) {
        sampleKilledCompletion()
        dut.clock.step()
        responseWait += 1
      }
      assert(responseWait < 20)
      sampleKilledCompletion()
      dut.clock.step()
      dut.io.dma_cache_rsp.valid.poke(false.B)

      var sawDone = false
      cycles = 0
      while (!sawDone && cycles < 120) {
        assert(!dut.io.dma_cache_req.valid.peekBoolean(),
          "killed tensor issued a new cache request")
        assert(!dut.io.shared_req.valid.peekBoolean(),
          "killed tensor issued a new shared request")
        sampleKilledCompletion()
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone, "active tensor kill did not drain")
      assert(killedCompletions == Vector(
        Completion(5, 3, s2g = false, barrierValid = false, 0, 0)))

      // A stale token in any planner queue used to leave in.ready low
      // permanently.  Complete a fresh tensor to prove post-kill recovery.
      dut.io.dma_cache_req.ready.poke(true.B)
      dut.io.shared_req.ready.poke(true.B)
      issue(dut, TmaV2Spec.FunctTensorG2S, wid = 7,
        in1 = 0x5000, in2 = Seq.fill(5)(0), in3 = 0x3000,
        group = 1)
      val (recovered, _, _, recoveredWrites, recoveredStatus, _) =
        serviceMany(dut, descriptorWords(), expectedCompletions = 1,
          limit = 1000)
      assert(recovered == Seq(
        Completion(7, 1, s2g = false, barrierValid = false, 0, 0)))
      assert(recoveredWrites == 1 && recoveredStatus.isEmpty)
    }
  }

  "ASID kill drains an outstanding Prepare descriptor refill" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctTensorS2G, wid = 3,
        in1 = 0x4000, in2 = Seq.fill(5)(0), in3 = 0x2000,
        group = 1)

      var pendingTlb: Option[BigInt] = None
      var descriptorSource: Option[BigInt] = None
      var cycles = 0
      while (descriptorSource.isEmpty && cycles < 100) {
        dut.io.from_l2TLB.valid.poke(pendingTlb.nonEmpty.B)
        pendingTlb.foreach(address =>
          dut.io.from_l2TLB.bits.paddr.poke(address.U))
        val tlbFire = dut.io.to_l2TLB.valid.peekBoolean()
        val tlbAddress = if (tlbFire)
          Some(dut.io.to_l2TLB.bits.vaddr.peekInt()) else None
        val tlbRspFire = pendingTlb.nonEmpty &&
          dut.io.from_l2TLB.ready.peekBoolean()
        if (dut.io.dma_cache_req.valid.peekBoolean()) {
          dut.io.dma_cache_req.bits.a_opcode.expect(4.U)
          dut.io.dma_cache_req.bits.a_addr.get.expect(0x4000.U)
          descriptorSource = Some(
            dut.io.dma_cache_req.bits.a_source.peekInt())
        }
        dut.clock.step()
        if (tlbRspFire) pendingTlb = None
        if (tlbFire) pendingTlb = tlbAddress
        cycles += 1
      }
      assert(descriptorSource.nonEmpty)
      dut.io.from_l2TLB.valid.poke(false.B)

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)
      dut.clock.step(4)
      dut.io.killDone.valid.expect(false.B)
      assert(!dut.io.shared_req.valid.peekBoolean())

      val words = descriptorWords()
      dut.io.dma_cache_rsp.bits.d_opcode.poke(1.U)
      dut.io.dma_cache_rsp.bits.d_param.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_addr.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_source.poke(descriptorSource.get.U)
      for (word <- 0 until dcache_BlockWords)
        dut.io.dma_cache_rsp.bits.d_data(word).poke(words(word).U)
      dut.io.dma_cache_rsp.valid.poke(true.B)
      while (!dut.io.dma_cache_rsp.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.dma_cache_rsp.valid.poke(false.B)

      var completions = 0
      var sawDone = false
      cycles = 0
      while (!sawDone && cycles < 100) {
        if (dut.io.tma_completion.valid.peekBoolean()) completions += 1
        sawDone = dut.io.killDone.valid.peekBoolean()
        assert(!dut.io.shared_req.valid.peekBoolean())
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone && completions == 1)
    }
  }

  "stalled Prepare reservation is stable across kill and then drains to barrier cleanup" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      dut.io.txReserve.ready.poke(false.B)
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 5,
        in1 = 0x1000, in2 = Seq(64), in3 = 0x2000,
        barrierValid = true, barrierId = 2)
      var cycles = 0
      while (!dut.io.txReserve.valid.peekBoolean() && cycles < 60) {
        dut.clock.step()
        cycles += 1
      }
      assert(cycles < 60, "Prepare did not present the G2S reservation")
      dut.io.txReserve.bits.wid.expect(5.U)
      dut.io.txReserve.bits.bytes.expect(64.U)
      // Cross one stalled edge so the top-level irrevocable marker owns this
      // exact payload before kill arrives.
      dut.clock.step()

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.io.txReserve.valid.expect(true.B)
      dut.io.txReserve.bits.wid.expect(5.U)
      dut.io.txReserve.bits.bytes.expect(64.U)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)
      for (_ <- 0 until 3) {
        dut.io.txReserve.valid.expect(true.B)
        dut.io.txReserve.bits.wid.expect(5.U)
        dut.io.txReserve.bits.bytes.expect(64.U)
        dut.io.killDone.valid.expect(false.B)
        dut.clock.step()
      }

      // The pre-kill presented beat is allowed to complete exactly once.
      dut.io.txReserve.ready.poke(true.B)
      dut.io.txReserve.valid.expect(true.B)
      dut.clock.step()
      dut.io.txReserve.ready.poke(false.B)
      dut.io.txReserve.valid.expect(false.B)
      dut.io.killDone.valid.expect(false.B)

      dut.io.txReserveResponse.bits.wid.poke(5.U)
      dut.io.txReserveResponse.bits.accepted.poke(true.B)
      dut.io.txReserveResponse.bits.barrierValid.poke(true.B)
      dut.io.txReserveResponse.bits.barrierId.poke(2.U)
      dut.io.txReserveResponse.bits.generation.poke(7.U)
      dut.io.txReserveResponse.valid.poke(true.B)
      while (!dut.io.txReserveResponse.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.txReserveResponse.valid.poke(false.B)

      var completion: Option[Completion] = None
      var sawDone = false
      cycles = 0
      while (!sawDone && cycles < 80) {
        if (dut.io.tma_completion.valid.peekBoolean()) {
          completion = Some(Completion(
            dut.io.tma_completion.bits.wid.peekInt(),
            dut.io.tma_completion.bits.group.peekInt(),
            dut.io.tma_completion.bits.is_s2g.peekBoolean(),
            dut.io.tma_completion.bits.barrierValid.peekBoolean(),
            dut.io.tma_completion.bits.barrierId.peekInt(),
            dut.io.tma_completion.bits.transactionBytes.peekInt()))
        }
        sawDone = dut.io.killDone.valid.peekBoolean()
        assert(!dut.io.dma_cache_req.valid.peekBoolean())
        assert(!dut.io.shared_req.valid.peekBoolean())
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone)
      assert(completion.contains(Completion(5, 0, s2g = false,
        barrierValid = true, barrierId = 2, bytes = 64)))
    }
  }

  "ASID kill drains an accepted shared read and drops its returned payload" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 2,
        in1 = 0x2000, in2 = Seq(16), in3 = 0x1000, group = 2)

      var waitShared = 0
      while (!dut.io.shared_req.valid.peekBoolean() && waitShared < 80) {
        dut.clock.step()
        waitShared += 1
      }
      assert(waitShared < 80)
      val source = dut.io.shared_req.bits.instrId.peekInt()
      val active = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.perLaneAddr(lane).activeMask.peekBoolean())
      dut.clock.step() // shared request fires

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)
      dut.io.killDone.valid.expect(false.B)

      dut.io.shared_rsp.bits.instrId.poke(source.U)
      dut.io.shared_rsp.bits.isWrite.poke(false.B)
      dut.io.shared_rsp.bits.isMBarrier.poke(false.B)
      for (lane <- 0 until num_thread) {
        dut.io.shared_rsp.bits.data(lane).poke((0x100 + lane).U)
        dut.io.shared_rsp.bits.activeMask(lane).poke(active(lane).B)
      }
      dut.io.shared_rsp.valid.poke(true.B)
      var waitResponse = 0
      while (!dut.io.shared_rsp.ready.peekBoolean() && waitResponse < 20) {
        dut.clock.step()
        waitResponse += 1
      }
      assert(waitResponse < 20)
      dut.clock.step()
      dut.io.shared_rsp.valid.poke(false.B)

      var completions = 0
      var sawDone = false
      var cycles = 0
      while (!sawDone && cycles < 80) {
        assert(!dut.io.dma_cache_req.valid.peekBoolean(),
          "shared payload returned after kill must not become a Put/AMO")
        if (dut.io.tma_completion.valid.peekBoolean()) completions += 1
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone && completions == 1)
    }
  }

  "ASID kill drains an accepted cache Get and drops its returned payload" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkG2S, wid = 3,
        in1 = 0x1000, in2 = Seq(16), in3 = 0x2000,
        barrierValid = true, barrierId = 1)

      // Complete the existing transaction-reservation handshake.
      var waitReserve = 0
      while (!dut.io.txReserve.valid.peekBoolean() && waitReserve < 40) {
        dut.clock.step()
        waitReserve += 1
      }
      assert(waitReserve < 40)
      dut.clock.step()
      dut.io.txReserveResponse.bits.wid.poke(3.U)
      dut.io.txReserveResponse.bits.accepted.poke(true.B)
      dut.io.txReserveResponse.bits.barrierValid.poke(true.B)
      dut.io.txReserveResponse.bits.barrierId.poke(1.U)
      dut.io.txReserveResponse.bits.generation.poke(0.U)
      dut.io.txReserveResponse.valid.poke(true.B)
      while (!dut.io.txReserveResponse.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.txReserveResponse.valid.poke(false.B)

      var pendingTlb: Option[BigInt] = None
      var cacheSource: Option[BigInt] = None
      var cycles = 0
      while (cacheSource.isEmpty && cycles < 100) {
        dut.io.from_l2TLB.valid.poke(pendingTlb.nonEmpty.B)
        pendingTlb.foreach(address =>
          dut.io.from_l2TLB.bits.paddr.poke(address.U))
        val tlbFire = dut.io.to_l2TLB.valid.peekBoolean()
        val tlbAddress = if (tlbFire)
          Some(dut.io.to_l2TLB.bits.vaddr.peekInt()) else None
        val tlbRspFire = pendingTlb.nonEmpty &&
          dut.io.from_l2TLB.ready.peekBoolean()
        if (dut.io.dma_cache_req.valid.peekBoolean())
          cacheSource = Some(dut.io.dma_cache_req.bits.a_source.peekInt())
        dut.clock.step()
        if (tlbRspFire) pendingTlb = None
        if (tlbFire) pendingTlb = tlbAddress
        cycles += 1
      }
      assert(cacheSource.nonEmpty)
      dut.io.from_l2TLB.valid.poke(false.B)

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)
      dut.io.killDone.valid.expect(false.B)

      dut.io.dma_cache_rsp.bits.d_opcode.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_param.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_addr.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_source.poke(cacheSource.get.U)
      dut.io.dma_cache_rsp.bits.d_data.foreach(_.poke(0x5a5a5a5aL.U))
      dut.io.dma_cache_rsp.valid.poke(true.B)
      while (!dut.io.dma_cache_rsp.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.dma_cache_rsp.valid.poke(false.B)

      var completions = Vector.empty[Completion]
      var sawDone = false
      cycles = 0
      while (!sawDone && cycles < 80) {
        assert(!dut.io.shared_req.valid.peekBoolean(),
          "cache payload returned after kill must not become a shared write")
        if (dut.io.tma_completion.valid.peekBoolean()) {
          completions :+= Completion(
            dut.io.tma_completion.bits.wid.peekInt(),
            dut.io.tma_completion.bits.group.peekInt(),
            dut.io.tma_completion.bits.is_s2g.peekBoolean(),
            dut.io.tma_completion.bits.barrierValid.peekBoolean(),
            dut.io.tma_completion.bits.barrierId.peekInt(),
            dut.io.tma_completion.bits.transactionBytes.peekInt())
        }
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone && completions.size == 1)
      assert(completions.head.barrierValid && completions.head.bytes == 16)
    }
  }

  "ASID kill waits for an already accepted S2G write acknowledgement" in {
    test(compactCore).withAnnotations(Seq(CachingAnnotation)) { dut =>
      initialize(dut)
      issue(dut, TmaV2Spec.FunctBulkS2G, wid = 4,
        in1 = 0x2000, in2 = Seq(16), in3 = 0x1000, group = 1)

      var cycles = 0
      while (!dut.io.shared_req.valid.peekBoolean() && cycles < 80) {
        dut.clock.step()
        cycles += 1
      }
      assert(cycles < 80)
      val sharedSource = dut.io.shared_req.bits.instrId.peekInt()
      val active = (0 until num_thread).map(lane =>
        dut.io.shared_req.bits.perLaneAddr(lane).activeMask.peekBoolean())
      dut.clock.step()

      dut.io.shared_rsp.bits.instrId.poke(sharedSource.U)
      dut.io.shared_rsp.bits.isWrite.poke(false.B)
      dut.io.shared_rsp.bits.isMBarrier.poke(false.B)
      for (lane <- 0 until num_thread) {
        dut.io.shared_rsp.bits.data(lane).poke((0x400 + lane).U)
        dut.io.shared_rsp.bits.activeMask(lane).poke(active(lane).B)
      }
      dut.io.shared_rsp.valid.poke(true.B)
      while (!dut.io.shared_rsp.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.shared_rsp.valid.poke(false.B)

      var pendingTlb: Option[BigInt] = None
      var cacheSource: Option[BigInt] = None
      cycles = 0
      while (cacheSource.isEmpty && cycles < 100) {
        dut.io.from_l2TLB.valid.poke(pendingTlb.nonEmpty.B)
        pendingTlb.foreach(address =>
          dut.io.from_l2TLB.bits.paddr.poke(address.U))
        val tlbFire = dut.io.to_l2TLB.valid.peekBoolean()
        val tlbAddress = if (tlbFire)
          Some(dut.io.to_l2TLB.bits.vaddr.peekInt()) else None
        val tlbRspFire = pendingTlb.nonEmpty &&
          dut.io.from_l2TLB.ready.peekBoolean()
        if (dut.io.dma_cache_req.valid.peekBoolean()) {
          cacheSource = Some(dut.io.dma_cache_req.bits.a_source.peekInt())
          val opcode = dut.io.dma_cache_req.bits.a_opcode.peekInt()
          assert(opcode == 0 || opcode == 1)
        }
        dut.clock.step()
        if (tlbRspFire) pendingTlb = None
        if (tlbFire) pendingTlb = tlbAddress
        cycles += 1
      }
      assert(cacheSource.nonEmpty)
      dut.io.from_l2TLB.valid.poke(false.B)

      dut.io.killReq.bits.asid.poke(0.U)
      dut.io.killReq.valid.poke(true.B)
      dut.clock.step()
      dut.io.killReq.valid.poke(false.B)
      for (_ <- 0 until 4) {
        dut.io.killDone.valid.expect(false.B)
        assert(!dut.io.dma_cache_req.valid.peekBoolean())
        dut.clock.step()
      }

      dut.io.dma_cache_rsp.bits.d_opcode.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_param.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_addr.poke(0.U)
      dut.io.dma_cache_rsp.bits.d_source.poke(cacheSource.get.U)
      dut.io.dma_cache_rsp.bits.d_data.foreach(_.poke(0.U))
      dut.io.dma_cache_rsp.valid.poke(true.B)
      while (!dut.io.dma_cache_rsp.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.dma_cache_rsp.valid.poke(false.B)

      var completions = 0
      var sawDone = false
      cycles = 0
      while (!sawDone && cycles < 80) {
        if (dut.io.tma_completion.valid.peekBoolean()) completions += 1
        sawDone = dut.io.killDone.valid.peekBoolean()
        dut.clock.step()
        cycles += 1
      }
      assert(sawDone && completions == 1)
    }
  }

  "one-entry command FIFO plus Prepare accepts one lookahead and blocks the third data command" in {
    test(new TmaV2Ingress(commandEntries = 1)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.prepared.ready.poke(false.B)
      dut.io.directCompletion.ready.poke(true.B)
      dut.io.kill.valid.poke(false.B)
      dut.io.kill.bits.asid.poke(0.U)
      dut.io.killActive.poke(false.B)
      dut.io.killAsid.poke(0.U)
      dut.io.txReserve.ready.poke(true.B)
      dut.io.txReserveHeld.poke(false.B)
      dut.io.txReserveResponse.valid.poke(true.B)
      dut.io.txReserveResponse.bits.wid.poke(0.U)
      dut.io.txReserveResponse.bits.accepted.poke(true.B)
      dut.io.txReserveResponse.bits.barrierValid.poke(false.B)
      dut.io.txReserveResponse.bits.barrierId.poke(0.U)
      dut.io.txReserveResponse.bits.generation.poke(0.U)
      for (client <- 0 until 2) {
        dut.io.descriptorRequest(client).ready.poke(true.B)
        dut.io.descriptorResponse(client).valid.poke(false.B)
        pokeDescriptorPayload(dut.io.descriptorResponse(client).bits,
          Seq.fill(32)(BigInt(0)))
      }

      def drive(bytes: Int, global: Int, shared: Int): Unit = {
        dut.io.in.bits.ctrl.dma.poke(true.B)
        dut.io.in.bits.ctrl.funct.poke(TmaV2Spec.FunctBulkG2S.U)
        dut.io.in.bits.ctrl.wid.poke(0.U)
        dut.io.in.bits.ctrl.dma_group.poke(0.U)
        dut.io.in.bits.ctrl.asid.foreach(_.poke(0.U))
        for (lane <- 0 until num_thread) {
          dut.io.in.bits.in1(lane).poke((if (lane == 0) global else 0).U)
          dut.io.in.bits.in2(lane).poke((if (lane == 0) bytes else 0).U)
          dut.io.in.bits.in3(lane).poke((if (lane == 0) shared else 0).U)
          dut.io.in.bits.mask(lane).poke(true.B)
        }
      }

      dut.clock.step(2)
      drive(128, 0x1000, 0x2000)
      dut.io.in.valid.poke(true.B)
      while (!dut.io.in.ready.peekBoolean()) dut.clock.step()
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
      while (!dut.io.prepared.valid.peekBoolean()) dut.clock.step()

      // The control lane must not wait for the stalled bulk window.  A
      // prefetch completes when the descriptor service accepts the hint.
      dut.io.in.bits.ctrl.inst.poke(0.U)
      dut.io.in.bits.ctrl.funct.poke(
        TmaV2Spec.FunctPrefetchTensormap.U)
      dut.io.in.bits.in1(0).poke(0x5000.U)
      dut.io.in.valid.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
      var controlCycles = 0
      var sawPrefetch = false
      var sawPrefetchCompletion = false
      while (!(sawPrefetch && sawPrefetchCompletion) &&
          controlCycles < 12) {
        sawPrefetch ||= dut.io.descriptorRequest(1).valid.peekBoolean()
        sawPrefetchCompletion ||=
          dut.io.directCompletion.valid.peekBoolean()
        assert(dut.io.prepared.valid.peekBoolean(),
          "prepared bulk command must remain stable while control lane progresses")
        dut.clock.step()
        controlCycles += 1
      }
      assert(sawPrefetch && sawPrefetchCompletion,
        "descriptor control lane was blocked behind active bulk data")

      drive(128, 0x8000, 0x6000)
      dut.io.in.valid.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)

      drive(64, 0xa000, 0x7000)
      dut.io.in.valid.poke(true.B)
      for (_ <- 0 until 8) {
        assert(!dut.io.in.ready.peekBoolean(),
          "third data command must wait for the lookahead slot")
        dut.clock.step()
      }
      dut.io.prepared.ready.poke(true.B)
      while (!dut.io.prepared.valid.peekBoolean()) dut.clock.step()
      dut.clock.step()
      var thirdIssueLatency = 0
      while (!dut.io.in.ready.peekBoolean() && thirdIssueLatency < 12) {
        dut.clock.step()
        thirdIssueLatency += 1
      }
      assert(thirdIssueLatency < 12)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
    }
  }

  "queued TensorMap control kill wins over same-cycle control dequeue" in {
    test(new TmaV2Ingress(commandEntries = 1))
      .withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.prepared.ready.poke(true.B)
      dut.io.directCompletion.ready.poke(true.B)
      dut.io.kill.valid.poke(false.B)
      dut.io.kill.bits.asid.poke(0.U)
      dut.io.killActive.poke(false.B)
      dut.io.killAsid.poke(0.U)
      dut.io.txReserve.ready.poke(true.B)
      dut.io.txReserveHeld.poke(false.B)
      dut.io.txReserveResponse.valid.poke(false.B)
      dut.io.txReserveResponse.bits.wid.poke(0.U)
      dut.io.txReserveResponse.bits.accepted.poke(false.B)
      dut.io.txReserveResponse.bits.barrierValid.poke(false.B)
      dut.io.txReserveResponse.bits.barrierId.poke(0.U)
      dut.io.txReserveResponse.bits.generation.poke(0.U)
      for (client <- 0 until 2) {
        dut.io.descriptorRequest(client).ready.poke(true.B)
        dut.io.descriptorResponse(client).valid.poke(false.B)
        pokeDescriptorPayload(dut.io.descriptorResponse(client).bits,
          Seq.fill(32)(BigInt(0)))
      }
      dut.clock.step(2)

      dut.io.in.bits.ctrl.dma.poke(true.B)
      dut.io.in.bits.ctrl.inst.poke(
        (BigInt(TmaV2Spec.TensorMapPrefetchSubop) << 27).U)
      dut.io.in.bits.ctrl.funct.poke(
        TmaV2Spec.FunctPrefetchTensormap.U)
      dut.io.in.bits.ctrl.wid.poke(2.U)
      dut.io.in.bits.ctrl.dma_group.poke(0.U)
      dut.io.in.bits.ctrl.asid.foreach(_.poke(0.U))
      for (lane <- 0 until num_thread) {
        dut.io.in.bits.in1(lane).poke((if (lane == 0) 0x4000 else 0).U)
        dut.io.in.bits.in2(lane).poke(0.U)
        dut.io.in.bits.in3(lane).poke(0.U)
        dut.io.in.bits.mask(lane).poke(true.B)
      }
      dut.io.in.valid.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)

      // The control queue is ready to dequeue here.  controlKillNow must
      // dominate the cPrefetchReq transition produced by that same edge.
      dut.io.kill.valid.poke(true.B)
      dut.io.kill.bits.asid.poke(0.U)
      dut.io.killActive.poke(true.B)
      dut.io.killAsid.poke(0.U)
      dut.io.descriptorRequest(1).valid.expect(false.B)
      dut.clock.step()
      dut.io.kill.valid.poke(false.B)

      dut.io.descriptorRequest(1).valid.expect(false.B)
      dut.io.directCompletion.valid.expect(true.B)
      dut.io.directCompletion.bits.wid.expect(2.U)
      dut.io.directCompletion.bits.transactionBytes.expect(0.U)
      dut.clock.step()
      dut.io.directCompletion.valid.expect(false.B)
      dut.io.descriptorRequest(1).valid.expect(false.B)
    }
  }

  "active tensor permits descriptor prefetch plus one prepared data command" in {
    test(new TmaV2Ingress(commandEntries = 1)).withAnnotations(Seq(CachingAnnotation)) { dut =>
      dut.io.in.valid.poke(false.B)
      dut.io.prepared.ready.poke(false.B)
      dut.io.directCompletion.ready.poke(true.B)
      dut.io.kill.valid.poke(false.B)
      dut.io.kill.bits.asid.poke(0.U)
      dut.io.killActive.poke(false.B)
      dut.io.killAsid.poke(0.U)
      dut.io.txReserve.ready.poke(true.B)
      dut.io.txReserveHeld.poke(false.B)
      dut.io.txReserveResponse.valid.poke(true.B)
      dut.io.txReserveResponse.bits.wid.poke(0.U)
      dut.io.txReserveResponse.bits.accepted.poke(true.B)
      dut.io.txReserveResponse.bits.barrierValid.poke(true.B)
      dut.io.txReserveResponse.bits.barrierId.poke(1.U)
      dut.io.txReserveResponse.bits.generation.poke(0.U)
      for (client <- 0 until 2) {
        dut.io.descriptorRequest(client).ready.poke(true.B)
        dut.io.descriptorResponse(client).valid.poke(false.B)
        pokeDescriptorPayload(dut.io.descriptorResponse(client).bits,
          Seq.fill(32)(BigInt(0)))
      }
      dut.clock.step(2)

      def prepare(funct: Int, in1: Int, in2: Seq[Int], in3: Int): Unit = {
        dut.io.in.bits.ctrl.dma.poke(true.B)
        dut.io.in.bits.ctrl.inst.poke(0.U)
        dut.io.in.bits.ctrl.funct.poke(funct.U)
        dut.io.in.bits.ctrl.wid.poke(0.U)
        dut.io.in.bits.ctrl.dma_group.poke(0.U)
        dut.io.in.bits.ctrl.asid.foreach(_.poke(0.U))
        for (lane <- 0 until num_thread) {
          dut.io.in.bits.in1(lane).poke((if (lane == 0) in1 else 0).U)
          dut.io.in.bits.in2(lane).poke((if (lane < in2.size) in2(lane) else 0).U)
          dut.io.in.bits.in3(lane).poke((if (lane == 0) in3 else 0).U)
          dut.io.in.bits.mask(lane).poke(true.B)
        }
      }

      def drive(funct: Int, in1: Int, in2: Seq[Int], in3: Int): Unit = {
        prepare(funct, in1, in2, in3)
        dut.io.in.valid.poke(true.B)
        var waited = 0
        while (!dut.io.in.ready.peekBoolean() && waited < 32) {
          dut.clock.step()
          waited += 1
        }
        assert(waited < 32)
        dut.clock.step()
        dut.io.in.valid.poke(false.B)
      }

      drive(TmaV2Spec.FunctTensorG2S, 0x4000, Seq.fill(5)(0), 0x2000)
      drive(TmaV2Spec.FunctPrefetchTensormap, 0x5000,
        Seq.fill(5)(0), 0)

      var descriptorPending = false
      var tensorLast = false
      var prefetchRequest = false
      var prefetchCompletion = false
      var cycles = 0
      while (!(tensorLast && prefetchRequest && prefetchCompletion) &&
          cycles < 200) {
        dut.io.descriptorResponse(0).valid.poke(descriptorPending.B)
        if (descriptorPending) {
          pokeDescriptorPayload(dut.io.descriptorResponse(0).bits,
            descriptorWords())
        }
        val requestFire = dut.io.descriptorRequest(0).valid.peekBoolean()
        prefetchRequest ||= dut.io.descriptorRequest(1).valid.peekBoolean()
        prefetchCompletion ||= dut.io.directCompletion.valid.peekBoolean()
        tensorLast ||= dut.io.prepared.valid.peekBoolean() &&
          dut.io.prepared.bits.kind.peekInt() ==
            TmaV2PreparedKind.Tensor.litValue
        dut.clock.step()
        descriptorPending = requestFire
        cycles += 1
      }

      assert(cycles < 200,
        "tensor/prefetch control lanes did not make independent progress")

      prepare(TmaV2Spec.FunctBulkS2G, 0x2200, Seq(32), 0x3000)
      dut.io.in.valid.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)

      prepare(TmaV2Spec.FunctBulkS2G, 0x2400, Seq(32), 0x3200)
      dut.io.in.valid.poke(true.B)
      for (_ <- 0 until 4) {
        assert(!dut.io.in.ready.peekBoolean())
        dut.clock.step()
      }
      dut.io.prepared.ready.poke(true.B)
      dut.clock.step()
      var bulkWait = 0
      while (!dut.io.in.ready.peekBoolean() && bulkWait < 12) {
        dut.clock.step()
        bulkWait += 1
      }
      assert(bulkWait < 12)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
    }
  }
}
