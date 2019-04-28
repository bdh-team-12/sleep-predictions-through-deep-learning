package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.clustering.PowerIterationClustering
import edu.gatech.cse6250.graphconstruct.GraphLoader
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import edu.gatech.cse6250.jaccard.Jaccard
import edu.gatech.cse6250.model.{ SubjectProperty, Demographic, MedicalHistory, Medication }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._

    /** initialize loading of data */
    val (subject, demographics, medical_history, medication) = loadRddRawData(spark)

    val subjectGraph = GraphLoader.load(subject, demographics, medical_history, medication)

    val similarities = Jaccard.jaccardSimilarityAllPatients(subjectGraph)

    // save to SubjectSimilarities
    similarities.toDF().coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("SubjectSimilarities.scala.csv")
    //    hdfs dfs -get hdfs://bootcamp.local:9000/user/root/SubjectSimilarities.scala.csv SubjectSimilarities.scala.csv                    

    // PIC needed too much resource -> use python instead
    //val PICLabels = PowerIterationClustering.runPIC(similarities)

    sc.stop()
  }

  def loadRddRawData(spark: SparkSession): (RDD[SubjectProperty], RDD[Demographic], RDD[MedicalHistory], RDD[Medication]) = {
    import spark.implicits._
    val sqlContext = spark.sqlContext
    import org.apache.spark.sql.functions._

    // example of host file path
    var host_dir = "/mnt/host/c/Users/OMSCS/Documents/GaTech/sleep-predictions-through-deep-learning/subject-similarity-groups/";
    var base = "file://" + host_dir;

    List(base + "data/SUBJECTS.csv", base + "data/DEMOGRAPHICS.csv", base + "data/MEDICAL_HISTORY.csv", base + "data/MEDICATION.csv").foreach(CSVHelper.loadCSVAsTable(spark, _))

    val subject = sqlContext.sql(
      """
        |SELECT nsrrid FROM SUBJECTS
      """.stripMargin).map(r =>
        SubjectProperty(r(0).toString))

    val demographics = sqlContext.sql(
      """
        |SELECT nsrrid, classifier, value FROM DEMOGRAPHICS
      """.stripMargin).map(r => Demographic(r(0).toString, r(1).toString, r(2).toString))

    val medical_history = sqlContext.sql(
      """
        |SELECT nsrrid, condition, value FROM MEDICAL_HISTORY
      """.stripMargin).map(r => MedicalHistory(r(0).toString, r(1).toString, r(2).toString))

    val medication = sqlContext.sql(
      """
        |SELECT nsrrid, medicine, value FROM MEDICATION
      """.stripMargin).map(r => Medication(r(0).toString, r(1).toString, r(2).toString))

    (subject.rdd, demographics.rdd, medical_history.rdd, medication.rdd)

  }

}
