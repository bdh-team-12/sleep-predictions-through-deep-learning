package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.graphconstruct.GraphLoader
import edu.gatech.cse6250.helper.SparkHelper
import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer
import scala.util.Random

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */
    val srcPatientOnly = graph.triplets.filter(x => x.srcAttr.isInstanceOf[PatientProperty])
    val patientSet: RDD[(Long, String)] = srcPatientOnly.map { triplet =>
      triplet.dstAttr match {
        case l: LabResultProperty =>
          (triplet.srcId, l.testName)
        case d: DiagnosticProperty =>
          (triplet.srcId, d.icd9code)
        case m: MedicationProperty =>
          (triplet.srcId, m.medicine)
        case _ =>
          (-1L, "")
      }
    }

    //    val srcIdDestIdGrpd: RDD[(Long, Set[String])] = srcIdDestId.groupByKey().asInstanceOf[RDD[(Long, Set[String])]]
    //    srcIdDestIdGrpd.cache()

    val srcIdDestIdGrpd = patientSet.groupByKey()
    val srcIdDestIdGrpdCar = srcIdDestIdGrpd.cartesian(srcIdDestIdGrpd)

    val allJaccardScores: RDD[(Long, Double)] = srcIdDestIdGrpdCar.map { x =>
      if (x._1._1 == patientID.toLong && x._1._1 != x._2._1) { // don't compare desired patient to itself
        (x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet))
      } else {
        (-1L, -1L)
      }
    }.filter(x => x._1 != -1L).asInstanceOf[RDD[(Long, Double)]]

    val jaccardRanked: List[Long] = allJaccardScores.sortBy(x => x._2, ascending = false).map(x => x._1).take(10).toList
    jaccardRanked
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    val srcPatientOnly = graph.triplets.filter(x => x.srcAttr.isInstanceOf[PatientProperty])
    val patientSet: RDD[(Long, String)] = srcPatientOnly.map { triplet =>
      triplet.dstAttr match {
        case l: LabResultProperty =>
          (triplet.srcId, l.testName)
        case d: DiagnosticProperty =>
          (triplet.srcId, d.icd9code)
        case m: MedicationProperty =>
          (triplet.srcId, m.medicine)
        case _ =>
          (-1L, "")
      }
    }

    val srcIdDestIdGrpd = patientSet.groupByKey()
    val srcIdDestIdGrpdCar = srcIdDestIdGrpd.cartesian(srcIdDestIdGrpd)

    val allJaccardScores: RDD[(Long, Long, Double)] = srcIdDestIdGrpdCar.map { x =>
      if (x._1._1 < x._2._1) { // don't compare patients to themselves and prevent duplicates
        (x._1._1, x._2._1, jaccard(x._1._2.toSet, x._2._2.toSet))
      } else {
        (-1L, -1L, -1L)
      }
    }.filter(x => x._1 != -1L).asInstanceOf[RDD[(Long, Long, Double)]]
    allJaccardScores
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */
    val intersection_set = a.intersect(b)
    val union_set = a.union(b)
    if (union_set.isEmpty) {
      0.0
    } else {
      intersection_set.size.toDouble / union_set.size.toDouble
    }
  }
}
