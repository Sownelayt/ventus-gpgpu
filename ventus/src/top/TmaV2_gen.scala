package top

import circt.stage.ChiselStage
import config.config.Parameters
import L1Cache.MyConfig
import pipeline.{TmaV2CommandQueue, TmaV2DescriptorService, TmaV2DmaCore, TmaV2WindowEngine, TmaV2WindowSubsystem, TmaV2WindowPlanner, tma}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

object TmaV2_gen extends App {
  implicit val p: Parameters = (new MyConfig).toInstance
  val outputDir: Path = Paths.get(args.headOption.getOrElse("generated-tma-v2"))
  private val optionArgs = args.drop(1)
  private def intOption(name: String, default: Int): Int = {
    val index = optionArgs.indexOf(name)
    if (index < 0) {
      default
    } else {
      require(index + 1 < optionArgs.length, s"missing value for $name")
      optionArgs(index + 1).toInt
    }
  }
  private val windowEntries = intOption(
    "--window-entries", pipeline.TmaV2Spec.DefaultWindowEntries)
  private val requestEntries = intOption(
    "--request-entries", pipeline.TmaV2Spec.DefaultGlobalRequestEntries)
  private val sharedEntries = intOption(
    "--shared-entries", pipeline.TmaV2Spec.DefaultSharedReadyEntries)
  private val writeAckEntries = intOption(
    "--write-ack-entries", pipeline.TmaV2Spec.DefaultWriteAckEntries)
  private val descriptorEntries = intOption(
    "--descriptor-entries", pipeline.TmaV2Spec.DefaultDescriptorEntries)
  private val commandEntries = intOption(
    "--command-entries", pipeline.TmaV2Spec.DefaultCommandQueueEntries)
  Files.createDirectories(outputDir)

  val options = Array(
    "--disable-mem-randomization",
    "--disable-reg-randomization",
    "-lowering-options=disallowLocalVariables"
  )
  val modules = Seq(
    "TmaV2CommandQueue.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2CommandQueue(commandEntries), firtoolOpts = options),
    "TmaV2WindowPlanner.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2WindowPlanner, firtoolOpts = options),
    "TmaV2DescriptorService.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2DescriptorService(descriptorEntries), firtoolOpts = options),
    "TmaV2WindowEngine.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2WindowEngine(
          windowEntries = windowEntries,
          requestEntries = requestEntries,
          sharedEntries = sharedEntries,
          writeAckEntries = writeAckEntries),
        firtoolOpts = options),
    "TmaV2WindowSubsystem.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2WindowSubsystem(
          windowEntries = windowEntries,
          requestEntries = requestEntries,
          sharedEntries = sharedEntries,
          writeAckEntries = writeAckEntries,
          descriptorEntries = descriptorEntries),
        firtoolOpts = options),
    "TmaV2DmaCore.sv" ->
      ChiselStage.emitSystemVerilog(
        new TmaV2DmaCore(
          windowEntries = windowEntries,
          requestEntries = requestEntries,
          sharedEntries = sharedEntries,
          writeAckEntries = writeAckEntries,
          descriptorEntries = descriptorEntries,
          commandEntries = commandEntries),
        firtoolOpts = options),
    "tma.sv" ->
      ChiselStage.emitSystemVerilog(
        new tma(
          windowEntries = windowEntries,
          requestEntries = requestEntries,
          sharedEntries = sharedEntries,
          writeAckEntries = writeAckEntries,
          descriptorEntries = descriptorEntries,
          commandEntries = commandEntries),
        firtoolOpts = options)
  )
  modules.foreach { case (name, contents) =>
    Files.writeString(outputDir.resolve(name), contents, StandardCharsets.UTF_8)
  }
}
