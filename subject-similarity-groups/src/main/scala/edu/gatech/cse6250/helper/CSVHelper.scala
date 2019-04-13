package edu.gatech.cse6250.helper

import org.apache.spark.sql.{ DataFrame, SparkSession }

object CSVHelper {
  def loadCSVAsTable(spark: SparkSession, path: String, tableName: String): DataFrame = {
    val data = spark.read.format("com.databricks.spark.csv").
      option("header", "true").
      option("mode", "DROPMALFORMED").
      option("delimiter", ",").
      load(path)
    data.createOrReplaceTempView(tableName)
    data
  }

  def loadCSVAsTable(spark: SparkSession, path: String): DataFrame = {
    loadCSVAsTable(spark, path, inferTableNameFromPath(path))
  }

  private val pattern = "(\\w+)(\\.csv)?$".r.unanchored

  def inferTableNameFromPath(path: String): String = path match {
    case pattern(filename, extension) => filename
    case _                            => path
  }
}
