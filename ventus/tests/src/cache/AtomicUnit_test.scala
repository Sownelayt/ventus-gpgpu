package play.cache

import chisel3._
import chiseltest._
import config.config.Parameters
import L1Cache.AtomicUnit.AtomicUnit
import L1Cache.MyConfig
import L2cache._
import org.scalatest.freespec.AnyFreeSpec
import top.parameters._

class AtomicUnitTest extends AnyFreeSpec with ChiselScalatestTester {
  implicit val p: Parameters = (new MyConfig).toInstance
  private val params = InclusiveCacheParameters_lite(
    CacheParameters(2, l2cache_NSets, l2cache_NWays, num_l2cache,
      blockBytes = l2cache_BlockWords << 2,
      beatBytes = l2cache_BlockWords << 2),
    InclusiveCacheMicroParameters(l2cache_writeBytes, l2cache_memCycles,
      l2cache_portFactor, num_warp, num_sm, num_sm_in_cluster, num_cluster,
      dcache_MshrEntry, dcache_NSets, atuns_NInfWriteEntry),
    control = false, mmu = MMU_ENABLED)

  private def initialize(dut: AtomicUnit): Unit = {
    dut.io.L12ATUmemReq.valid.poke(false.B)
    dut.io.L22ATUmemRsp.valid.poke(false.B)
    dut.io.ATU2L2memReq.ready.poke(true.B)
    dut.io.ATU2L1memRsp.ready.poke(true.B)
    dut.io.L12ATUmemReq.bits.opcode.poke(0.U)
    dut.io.L12ATUmemReq.bits.param.poke(0.U)
    dut.io.L12ATUmemReq.bits.size.poke(0.U)
    dut.io.L12ATUmemReq.bits.source.poke(0.U)
    dut.io.L12ATUmemReq.bits.address.poke(0.U)
    dut.io.L12ATUmemReq.bits.mask.poke(0.U)
    dut.io.L12ATUmemReq.bits.data.poke(0.U)
    dut.io.L22ATUmemRsp.bits.opcode.poke(0.U)
    dut.io.L22ATUmemRsp.bits.param.poke(0.U)
    dut.io.L22ATUmemRsp.bits.size.poke(0.U)
    dut.io.L22ATUmemRsp.bits.source.poke(0.U)
    dut.io.L22ATUmemRsp.bits.address.poke(0.U)
    dut.io.L22ATUmemRsp.bits.data.poke(0.U)
    dut.clock.step(2)
  }

  private def driveRequest(dut: AtomicUnit, opcode: Int, param: Int,
                           operand: BigInt, source: Int,
                           address: BigInt = 0x100): Unit = {
    dut.io.L12ATUmemReq.bits.opcode.poke(opcode.U)
    dut.io.L12ATUmemReq.bits.param.poke(param.U)
    dut.io.L12ATUmemReq.bits.size.poke(2.U)
    dut.io.L12ATUmemReq.bits.source.poke(source.U)
    dut.io.L12ATUmemReq.bits.address.poke(address.U)
    dut.io.L12ATUmemReq.bits.mask.poke(0xf.U)
    dut.io.L12ATUmemReq.bits.data.poke(operand.U)
    dut.io.L12ATUmemReq.valid.poke(true.B)
  }

  private def driveL2Response(dut: AtomicUnit, source: BigInt,
                              data: BigInt, opcode: Int): Unit = {
    dut.io.L22ATUmemRsp.bits.opcode.poke(opcode.U)
    dut.io.L22ATUmemRsp.bits.param.poke(0.U)
    dut.io.L22ATUmemRsp.bits.size.poke(2.U)
    dut.io.L22ATUmemRsp.bits.source.poke(source.U)
    dut.io.L22ATUmemRsp.bits.address.poke(0x100.U)
    dut.io.L22ATUmemRsp.bits.data.poke(data.U)
    dut.io.L22ATUmemRsp.valid.poke(true.B)
  }

  private def runAtomic(dut: AtomicUnit, opcode: Int, param: Int,
                        oldValue: BigInt, operand: BigInt,
                        expected: BigInt, source: Int): Unit = {
    driveRequest(dut, opcode, param, operand, source)
    dut.io.L12ATUmemReq.ready.expect(true.B)
    dut.io.ATU2L2memReq.valid.expect(true.B)
    dut.io.ATU2L2memReq.bits.opcode.expect(4.U)
    val internalSource = dut.io.ATU2L2memReq.bits.source.peekInt()
    val expectedInternalSource =
      (BigInt(3) << l1cache_sourceBits) | source
    assert(internalSource == expectedInternalSource,
      "serial AMO must use the reserved internal source namespace")
    dut.clock.step()
    dut.io.L12ATUmemReq.valid.poke(false.B)
    dut.io.ATU2L1memRsp.valid.expect(false.B)

    driveL2Response(dut, internalSource, oldValue, opcode = 1)
    dut.io.L22ATUmemRsp.ready.expect(true.B)
    dut.io.ATU2L2memReq.valid.expect(false.B)
    dut.clock.step()
    dut.io.L22ATUmemRsp.valid.poke(false.B)
    dut.io.L22ATUmemRsp.bits.data.poke("hdeadbeef".U)
    dut.io.ATU2L2memReq.valid.expect(true.B)
    dut.io.ATU2L2memReq.bits.opcode.expect(1.U)
    dut.io.ATU2L2memReq.bits.mask.expect(0xf.U)
    val putWord = dut.io.ATU2L2memReq.bits.data.peekInt() & 0xffffffffL
    assert(putWord == (expected & 0xffffffffL),
      f"unexpected atomic result 0x$putWord%x expected 0x$expected%x")
    dut.io.ATU2L1memRsp.valid.expect(false.B)
    dut.clock.step()
    dut.io.ATU2L1memRsp.valid.expect(false.B)

    driveL2Response(dut, internalSource, 0, opcode = 0)
    dut.io.ATU2L1memRsp.valid.expect(true.B)
    dut.io.ATU2L1memRsp.bits.source.expect(source.U)
    val returnedOld = dut.io.ATU2L1memRsp.bits.data.peekInt() & 0xffffffffL
    assert(returnedOld == (oldValue & 0xffffffffL))
    dut.clock.step()
    dut.io.L22ATUmemRsp.valid.poke(false.B)
  }

  "AtomicUnit serializes every integer AMO and responds after the Put ack" in {
    test(new AtomicUnit(params)) { dut =>
      initialize(dut)
      runAtomic(dut, opcode = 2, param = 4,
        oldValue = 0xffffffffL, operand = 1, expected = 0, source = 1)
      runAtomic(dut, opcode = 2, param = 0,
        oldValue = 0x80000000L, operand = 0x7fffffffL,
        expected = 0x80000000L, source = 2)
      runAtomic(dut, opcode = 2, param = 2,
        oldValue = 0x80000000L, operand = 0x7fffffffL,
        expected = 0x7fffffffL, source = 3)
      runAtomic(dut, opcode = 2, param = 1,
        oldValue = 0x80000000L, operand = 0x7fffffffL,
        expected = 0x7fffffffL, source = 4)
      runAtomic(dut, opcode = 2, param = 3,
        oldValue = 0x80000000L, operand = 0x7fffffffL,
        expected = 0x80000000L, source = 5)
      runAtomic(dut, opcode = 3, param = 2,
        oldValue = 0x33333333L, operand = 0x0ff00ff0L,
        expected = 0x03300330L, source = 6)
      runAtomic(dut, opcode = 3, param = 1,
        oldValue = 0x30303030L, operand = 0x0ff00ff0L,
        expected = 0x3ff03ff0L, source = 7)
      runAtomic(dut, opcode = 3, param = 0,
        oldValue = 0x33333333L, operand = 0x0ff00ff0L,
        expected = 0x3cc33cc3L, source = 1)
    }
  }

  "AtomicUnit backpressures a second AMO and same-line write until final ack" in {
    test(new AtomicUnit(params)) { dut =>
      initialize(dut)
      driveRequest(dut, opcode = 2, param = 4, operand = 3, source = 1)
      val internalSource = dut.io.ATU2L2memReq.bits.source.peekInt()
      assert(internalSource ==
        ((BigInt(3) << l1cache_sourceBits) | 1))
      dut.clock.step()

      driveRequest(dut, opcode = 2, param = 4, operand = 5, source = 2)
      dut.io.L12ATUmemReq.ready.expect(false.B)
      dut.io.ATU2L2memReq.valid.expect(false.B)

      driveRequest(dut, opcode = 1, param = 0, operand = 5, source = 2)
      dut.io.L12ATUmemReq.ready.expect(false.B)
      dut.io.ATU2L2memReq.valid.expect(false.B)
      dut.io.L12ATUmemReq.valid.poke(false.B)

      driveL2Response(dut, internalSource, 7, opcode = 1)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      dut.io.L22ATUmemRsp.bits.data.poke("hdeadbeef".U)
      dut.io.ATU2L2memReq.valid.expect(true.B)
      dut.clock.step()
      dut.io.ATU2L1memRsp.valid.expect(false.B)

      dut.io.ATU2L1memRsp.ready.poke(false.B)
      driveL2Response(dut, internalSource, 0, opcode = 0)
      dut.io.ATU2L1memRsp.valid.expect(true.B)
      dut.io.L22ATUmemRsp.ready.expect(false.B)
      dut.clock.step(2)
      dut.io.ATU2L1memRsp.valid.expect(true.B)

      dut.io.ATU2L1memRsp.ready.poke(true.B)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      driveRequest(dut, opcode = 2, param = 4, operand = 5, source = 2)
      dut.io.L12ATUmemReq.ready.expect(true.B)
      val secondInternalSource = dut.io.ATU2L2memReq.bits.source.peekInt()
      assert(secondInternalSource ==
        ((BigInt(3) << l1cache_sourceBits) | 2))
      dut.clock.step()
      dut.io.L12ATUmemReq.valid.poke(false.B)

      // The second source observes the first source's committed value.  This
      // models two independent requesters updating one address and proves that
      // the single-entry endpoint serializes them without losing an update.
      driveL2Response(dut, secondInternalSource, 10, opcode = 1)
      dut.io.ATU2L2memReq.valid.expect(false.B)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      dut.io.L22ATUmemRsp.bits.data.poke("hdeadbeef".U)
      dut.io.ATU2L2memReq.valid.expect(true.B)
      val secondPutWord =
        dut.io.ATU2L2memReq.bits.data.peekInt() & 0xffffffffL
      assert(secondPutWord == 15)
      dut.clock.step()

      driveL2Response(dut, secondInternalSource, 0, opcode = 0)
      dut.io.ATU2L1memRsp.valid.expect(true.B)
      dut.io.ATU2L1memRsp.bits.source.expect(2.U)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
    }
  }

  "AtomicUnit drains the Get response even while the L2 A channel is blocked" in {
    test(new AtomicUnit(params)) { dut =>
      initialize(dut)
      driveRequest(dut, opcode = 2, param = 4, operand = 3, source = 1)
      val internalSource = dut.io.ATU2L2memReq.bits.source.peekInt()
      dut.clock.step()
      dut.io.L12ATUmemReq.valid.poke(false.B)

      // Model an L2 with all MSHRs occupied.  The Get response must still
      // drain; otherwise it can block the responses that would free an MSHR.
      dut.io.ATU2L2memReq.ready.poke(false.B)
      driveL2Response(dut, source = internalSource, data = 7, opcode = 1)
      dut.io.L22ATUmemRsp.ready.expect(true.B)
      dut.io.ATU2L2memReq.valid.expect(false.B)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      dut.io.L22ATUmemRsp.bits.data.poke("hdeadbeef".U)

      dut.io.ATU2L2memReq.valid.expect(true.B)
      dut.io.ATU2L2memReq.bits.opcode.expect(1.U)
      val putWord = dut.io.ATU2L2memReq.bits.data.peekInt() & 0xffffffffL
      assert(putWord == 10)
      dut.clock.step(2)
      dut.io.ATU2L2memReq.valid.expect(true.B)

      dut.io.ATU2L2memReq.ready.poke(true.B)
      dut.clock.step()
      driveL2Response(dut, source = internalSource, data = 0, opcode = 0)
      dut.io.ATU2L1memRsp.valid.expect(true.B)
      dut.io.ATU2L1memRsp.bits.source.expect(1.U)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
    }
  }

  "AtomicUnit does not consume an ordinary response with the same architectural source" in {
    test(new AtomicUnit(params)) { dut =>
      initialize(dut)
      driveRequest(dut, opcode = 2, param = 4, operand = 3, source = 1)
      val internalSource = dut.io.ATU2L2memReq.bits.source.peekInt()
      assert(internalSource != 1)
      dut.clock.step()
      dut.io.L12ATUmemReq.valid.poke(false.B)

      // Ordinary traffic remains legal while this AMO waits for its Get.
      // A response that reuses source 1 must be forwarded to L1 and must not
      // advance the atomic FSM or become the AMO's old value.
      driveL2Response(dut, source = 1, data = 0x55, opcode = 1)
      dut.io.L22ATUmemRsp.ready.expect(true.B)
      dut.io.ATU2L1memRsp.valid.expect(true.B)
      dut.io.ATU2L1memRsp.bits.source.expect(1.U)
      dut.io.ATU2L2memReq.valid.expect(false.B)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      dut.io.ATU2L1memRsp.valid.expect(false.B)
      dut.io.ATU2L2memReq.valid.expect(false.B)

      driveL2Response(dut, source = internalSource, data = 7, opcode = 1)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
      dut.io.ATU2L2memReq.valid.expect(true.B)
      val putWord = dut.io.ATU2L2memReq.bits.data.peekInt() & 0xffffffffL
      assert(putWord == 10, s"ordinary response corrupted AMO result: $putWord")
      dut.clock.step()

      driveL2Response(dut, source = internalSource, data = 0, opcode = 0)
      dut.io.ATU2L1memRsp.valid.expect(true.B)
      dut.io.ATU2L1memRsp.bits.source.expect(1.U)
      val returnedOld = dut.io.ATU2L1memRsp.bits.data.peekInt() & 0xffffffffL
      assert(returnedOld == 7)
      dut.clock.step()
      dut.io.L22ATUmemRsp.valid.poke(false.B)
    }
  }
}
