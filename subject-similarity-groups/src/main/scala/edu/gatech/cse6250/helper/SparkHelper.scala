package edu.gatech.cse6250.helper

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.{ SQLContext, SparkSession }
import org.apache.spark.{ SparkConf, SparkContext }

object SparkHelper {
  lazy val sparkMasterURL = "local[*]"

  lazy val spark: SparkSession = SparkHelper.createSparkSession(
    appName = "CSE 6250 Homework 4 Application",
    masterUrl = sparkMasterURL,
    cfg = {
      _.set("spark.executor.memory", "1G")
        .set("spark.driver.memory", "1G")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer", "24")
        .set("spark.sql.shuffle.partitions", "15")
    })

  lazy val sc: SparkContext = spark.sparkContext

  lazy val sqlContext: SQLContext = spark.sqlContext

  def hdfs(sc: SparkContext = sc): FileSystem = {
    val hadoopConf: Configuration = sc.hadoopConfiguration
    org.apache.hadoop.fs.FileSystem.get(new java.net.URI("hdfs://localhost:9000"), hadoopConf)
  }

  def createSparkSession(
    appName:   String,
    masterUrl: String                 = sparkMasterURL,
    cfg:       SparkConf => SparkConf = { in => in }): SparkSession = {
    val session = SparkSession.builder().config(sparkConf(appName, masterUrl, cfg)).getOrCreate()
    session
  }

  def sparkConf(appName: String, masterUrl: String, cfg: SparkConf => SparkConf): SparkConf = {
    cfg(new SparkConf()
      .setAppName(appName)
      .setMaster(masterUrl)
      .set("spark.executor.memory", "1G")
      .set("spark.driver.memory", "500M"))
  }

  def createSparkSession: SparkSession = createSparkSession("CSE 6250 Homework Three Application")
}