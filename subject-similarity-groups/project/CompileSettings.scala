import sbt.Keys._
import sbt._

object CompileSettings extends AutoPlugin {
  override def trigger = allRequirements

  override def projectSettings = Seq(
    scalacOptions ++= Seq(
      "-optimize",
      "-deprecation", // Emit warning and location for usages of deprecated APIs.
      "-feature", // Emit warning and location for usages of features that should be imported explicitly.
      "-unchecked", // Enable additional warnings where generated code depends on assumptions.
      "-Xfatal-warnings", // Fail the compilation if there are any warnings.
      "-Xlint", // Enable recommended additional warnings.
      "-Ywarn-adapted-args", // Warn if an argument list is modified to match the receiver.
      "-Ywarn-dead-code", // Warn when dead code is identified.
      "-Ywarn-inaccessible", // Warn about inaccessible types in method signatures.
      "-Ywarn-nullary-override", // Warn when non-nullary overrides nullary, e.g. def foo() over def foo.
      "-Ywarn-numeric-widen", // Warn when numerics are widened.
      "-Yinline-warnings", //
      "-language:postfixOps", // See the Scala docs for value scala.language.postfixOps for a discussion
      "-Ybackend:GenBCode"
      //,"-target:jvm-1.8" // force use jvm 1.8
    ),
    //javacOptions in compile ++= Seq("-target", "1.8", "-source", "1.8"), // force use jvm 1.8
    compileOrder in Compile := CompileOrder.Mixed,
    compileOrder in Test := CompileOrder.Mixed,
    testOptions in Test += Tests.Argument(TestFrameworks.Specs2, "junitxml", "console"),
    scalacOptions in Test ~= { (options: Seq[String]) =>
      options filterNot (_ == "-Ywarn-dead-code") // Allow dead code in tests (to support using mockito).
    },
    parallelExecution in Test := false,
    unmanagedBase := baseDirectory.value / "lib")
}
