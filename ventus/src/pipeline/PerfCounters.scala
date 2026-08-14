/*
 * Copyright (c) 2021-2022 International Innovation Center of Tsinghua University, Shanghai
 * Ventus is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details. */
package pipeline

import chisel3._

class PipelinePerfCounters extends Bundle {
  val activeCycles = UInt(64.W)
  val totalScalarIssued = UInt(64.W)
  val totalVectorIssued = UInt(64.W)

  val execStructuralHazardCyclesX = UInt(64.W)
  val execStructuralHazardCyclesV = UInt(64.W)
  val dataDepStallCycles = UInt(64.W)
  val barrierStallCycles = UInt(64.W)
  val controlHazardFlushCount = UInt(64.W)
  val frontendStallCycles = UInt(64.W)
  val lsuBackpressureCycles = UInt(64.W)
  val ibufferFullCycles = UInt(64.W)
  val tmaWaitStallCycles = UInt(64.W)
}

class InstClassPerfCounters extends Bundle {
  val computeIssued = UInt(64.W)
  val memIssued = UInt(64.W)
  val ctrlIssued = UInt(64.W)
}

class TmaPerfCounters extends Bundle {
  val instIssued = UInt(64.W)
  val lineIssued = UInt(64.W)
  val putFull = UInt(64.W)
  val putPart = UInt(64.W)
  val bytesWritten = UInt(64.W)
  val sharedReadReq = UInt(64.W)
  val sharedReadRsp = UInt(64.W)
  val tlbReq = UInt(64.W)
  val lineFullStallCycles = UInt(64.W)
  val ackTagFullStallCycles = UInt(64.W)

  // TMA diagnostic counters. They are removed entirely when
  // VENTUS_PMU_TMA=0 and therefore do not affect the PMU-off build.
  val commandSlotFullStallCycles = UInt(64.W)
  val pendingFullStallCycles = UInt(64.W)
  val tlbStallCycles = UInt(64.W)
  val sharedStallCycles = UInt(64.W)
  val cacheReqStallCycles = UInt(64.W)
  val g2sLineCount = UInt(64.W)
  val s2gLineCount = UInt(64.W)
  val g2sCompletionCount = UInt(64.W)
  val s2gCompletionCount = UInt(64.W)
  val g2sCacheResponseCount = UInt(64.W)
  val s2gCacheResponseCount = UInt(64.W)
  val g2sCacheLatencySum = UInt(64.W)
  val s2gCacheLatencySum = UInt(64.W)
  val activeCommandCycles = UInt(64.W)
  val activeLineCycles = UInt(64.W)
  val sharedActiveLineCycles = UInt(64.W)
  val requestActiveLineCycles = UInt(64.W)

  // Full-window backend counters.
  val windowIssued = UInt(64.W)
  val windowRetired = UInt(64.W)
  val uniqueLineWaves = UInt(64.W)
  val translationHits = UInt(64.W)
  val translationMisses = UInt(64.W)
  val translationCoalesces = UInt(64.W)
  val windowRobFullCycles = UInt(64.W)
  val globalRequestFullCycles = UInt(64.W)
  val sharedQueueFullCycles = UInt(64.W)
  val cacheToSharedLatencyCount = UInt(64.W)
  val cacheToSharedLatencySum = UInt(64.W)

  // Descriptor service, binder and elastic-window diagnostics.
  val descriptorDemandHits = UInt(64.W)
  val descriptorDemandMisses = UInt(64.W)
  val descriptorPrefetchHits = UInt(64.W)
  val descriptorPrefetchMisses = UInt(64.W)
  val descriptorEvictions = UInt(64.W)
  val descriptorCompiles = UInt(64.W)
  val descriptorCoalesces = UInt(64.W)
  val descriptorInvalidateKills = UInt(64.W)
  val descriptorCompileCycles = UInt(64.W)
  val bindCycles = UInt(64.W)
  val plannerProduced = UInt(64.W)
  val plannerFire = UInt(64.W)
  val plannerStallCycles = UInt(64.W)
  val maxActiveWindows = UInt(64.W)
  val maxActiveRequests = UInt(64.W)
  val maxActiveShared = UInt(64.W)
  val maxActiveWriteAcks = UInt(64.W)
  val longestWindowFireRun = UInt(64.W)
}
