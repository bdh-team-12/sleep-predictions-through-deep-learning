package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for subject, demographics, medical_history, medication
   * @return: Constructed Graph
   *
   */
  def load(subject: RDD[SubjectProperty], demographics: RDD[Demographic],
           medical_history: RDD[MedicalHistory], medication: RDD[Medication]): Graph[VertexProperty, EdgeProperty] = {

    val spark = SparkHelper.spark
    import spark.implicits._
    val sqlContext = spark.sqlContext
    val sc = subject.sparkContext
    //
    //    subject.cache()
    //    demographics.cache()
    //    medical_history.cache()
    //    medication.cache()

    // INDEXES
    val subjectVertexIdRDD = subject.map(_.nsrrid).distinct.zipWithIndex.map {
      case (res, zeroBasedIndex) =>
        (res, res.toLong)
    }
    val subject2VertexId = subjectVertexIdRDD.collect.toMap

    val startIndex1 = if (subject2VertexId.nonEmpty) subject2VertexId.maxBy(x => x._2)._2 + 1 else 1
    val demographicsVertexIdRDD = demographics.map(_.classifier).distinct.zipWithIndex.map {
      case (res, zeroBasedIndex) =>
        (res, zeroBasedIndex + startIndex1)
    }
    val demographics2VertexId = demographicsVertexIdRDD.collect.toMap

    val startIndex2 = if (demographics2VertexId.nonEmpty) demographics2VertexId.maxBy(x => x._2)._2 + 1 else startIndex1 + 1
    val medical_historyVertexIdRDD = medical_history.map(_.condition).distinct.zipWithIndex.map {
      case (med, zeroBasedIndex) =>
        (med, zeroBasedIndex + startIndex2)
    }
    val medical_history2VertexId = medical_historyVertexIdRDD.collect.toMap

    val startIndex3 = if (medical_history2VertexId.nonEmpty) medical_history2VertexId.maxBy(x => x._2)._2 + 1 else startIndex2 + 1
    val medicationVertexIdRDD = medication.map(_.medicine).distinct.zipWithIndex.map {
      case (diag, zeroBasedIndex) =>
        (diag, zeroBasedIndex + startIndex3)
    }
    val medication2VertexId = medicationVertexIdRDD.collect.toMap

    // VERTICES //////////////////////////////
    // SubjectProperty
    val vertexSubject: RDD[(VertexId, VertexProperty)] = subject.map(subject =>
      (subject.nsrrid.toLong, subject.asInstanceOf[VertexProperty]))

    // convert Demographic to DemographicProperty
    val bcDemographic2VertexId = sc.broadcast(demographics2VertexId)
    val demoZip = demographics.map(demo => demo.classifier).distinct.zipWithIndex
    val demoVertex = demoZip.map {
      case (demoName, index) => (bcDemographic2VertexId.value(demoName), DemographicProperty(demoName))
    }.asInstanceOf[RDD[(VertexId, VertexProperty)]]
    val vertexDemographics: RDD[(VertexId, VertexProperty)] = demoVertex.map(demo =>
      (demo._1, demo._2))

    // convert MedicalHistory to MedicalHistoryProperty
    val bcMedicalHistory2VertexId = sc.broadcast(medical_history2VertexId)
    val medicalHistoryZip = medical_history.map(medicalHistory => medicalHistory.condition).distinct.zipWithIndex
    val medicalHistoryVertex = medicalHistoryZip.map {
      case (condition, index) => (bcMedicalHistory2VertexId.value(condition), MedicalHistoryProperty(condition))
    }.asInstanceOf[RDD[(VertexId, VertexProperty)]]
    val vertexMedicalHistory: RDD[(VertexId, VertexProperty)] = medicalHistoryVertex.map(hist =>
      (hist._1, hist._2))

    // convert Medication to MedicationProperty
    val bcMedResult2VertexId = sc.broadcast(medication2VertexId)
    val medZip = medication.map(med => med.medicine).distinct.zipWithIndex
    val medVertex = medZip.map {
      case (medicine, index) => (bcMedResult2VertexId.value(medicine), MedicationProperty(medicine))
    }.asInstanceOf[RDD[(VertexId, VertexProperty)]]
    val vertexMedResults: RDD[(VertexId, VertexProperty)] = medVertex.map(med =>
      (med._1, med._2))

    // FILTER DATA //////////////////////////////
    // filter the data for latest date and only one per date
    //    val bcPatient2VertexId = sc.broadcast(subject2VertexId)
    //
    //    val labResultList = demographics.toDF(Seq("nsrrid", "date", "classifier", "value"): _*)
    //    labResultList.createOrReplaceTempView("df_lab")
    //    val demographicsFiltered = sqlContext.sql(
    //      """
    //        |SELECT t.nsrrid, FIRST(t.date) as date, t.classifier, FIRST(t.value) as value FROM df_lab t
    //        |INNER JOIN ( SELECT nsrrid, classifier, MAX(date) as MaxDate
    //        |FROM df_lab GROUP BY nsrrid, classifier) tm ON t.nsrrid = tm.nsrrid
    //        |AND t.date = tm.MaxDate AND t.classifier = tm.classifier
    //        |GROUP BY t.nsrrid, t.classifier
    //      """.stripMargin).as[Demographic].rdd
    //    demographicsFiltered.cache()
    //
    //    val medList = medical_history.toDF(Seq("nsrrid", "date", "medicine"): _*)
    //    medList.createOrReplaceTempView("df_med")
    //    val medical_historyFiltered = sqlContext.sql(
    //      """
    //        |SELECT t.nsrrid, FIRST(t.date) as date, t.medicine FROM df_med t
    //        |INNER JOIN ( SELECT nsrrid, medicine, MAX(date) as MaxDate
    //        |FROM df_med GROUP BY nsrrid, medicine) tm ON t.nsrrid = tm.nsrrid
    //        |AND t.date = tm.MaxDate AND t.medicine = tm.medicine
    //        |GROUP BY t.nsrrid, t.medicine
    //      """.stripMargin).as[Medication].rdd
    //    medical_historyFiltered.cache()
    //
    //    val diagList = medication.toDF(Seq("nsrrid", "date", "condition", "sequence"): _*)
    //    diagList.createOrReplaceTempView("df_diag")
    //    val medicationFiltered = sqlContext.sql(
    //      """
    //        |SELECT t.nsrrid, FIRST(t.date) as date, t.condition, FIRST(t.sequence) as sequence FROM df_diag t
    //        |INNER JOIN ( SELECT nsrrid, condition, MAX(date) as MaxDate
    //        |FROM df_diag GROUP BY nsrrid, condition) tm ON t.nsrrid = tm.nsrrid
    //        |AND t.date = tm.MaxDate AND t.condition = tm.condition
    //        |GROUP BY t.nsrrid, t.condition
    //      """.stripMargin).as[MedicalHistory].rdd
    //    medicationFiltered.cache()

    // EDGES ONE WAY//////////////////////////////
    val edgeSubjectDemographic: RDD[Edge[EdgeProperty]] = demographics.map({ p =>
      Edge(p.nsrrid.toLong, bcDemographic2VertexId.value(p.classifier), SubjectDemographicEdgeProperty(Demographic(p.nsrrid, p.classifier, p.value)).asInstanceOf[EdgeProperty])
    })

    val edgeSubjectMedicalHistory: RDD[Edge[EdgeProperty]] = medical_history.map({ p =>
      Edge(p.nsrrid.toLong, bcMedicalHistory2VertexId.value(p.condition), SubjectMedicalHistoryEdgeProperty(MedicalHistory(p.nsrrid, p.condition, p.value)).asInstanceOf[EdgeProperty])
    })

    val edgeSubjectMedication: RDD[Edge[EdgeProperty]] = medication.map({ p =>
      Edge(p.nsrrid.toLong, bcMedResult2VertexId.value(p.medicine), SubjectMedicationEdgeProperty(Medication(p.nsrrid, p.medicine, p.value)).asInstanceOf[EdgeProperty])
    })

    // EDGES REVERSE WAY//////////////////////////////
    val edgeDemographicSubject: RDD[Edge[EdgeProperty]] = demographics.map({ p =>
      Edge(bcDemographic2VertexId.value(p.classifier), p.nsrrid.toLong, SubjectDemographicEdgeProperty(Demographic(p.nsrrid, p.classifier, p.value)).asInstanceOf[EdgeProperty])
    })

    val edgeMedicalHistorySubject: RDD[Edge[EdgeProperty]] = medical_history.map({ p =>
      Edge(bcMedicalHistory2VertexId.value(p.condition), p.nsrrid.toLong, SubjectMedicalHistoryEdgeProperty(MedicalHistory(p.nsrrid, p.condition, p.value)).asInstanceOf[EdgeProperty])
    })

    val edgeMedicationSubject: RDD[Edge[EdgeProperty]] = medication.map({ p =>
      Edge(bcMedResult2VertexId.value(p.medicine), p.nsrrid.toLong, SubjectMedicationEdgeProperty(Medication(p.nsrrid, p.medicine, p.value)).asInstanceOf[EdgeProperty])
    })

    // Making Graph
    val combinedVertex = vertexSubject.union(vertexDemographics).union(vertexMedicalHistory).union(vertexMedResults)
    val combinedEdge = edgeSubjectDemographic.union(edgeSubjectMedicalHistory).union(edgeSubjectMedication).union(edgeDemographicSubject).union(edgeMedicalHistorySubject).union(edgeMedicationSubject)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph[VertexProperty, EdgeProperty](combinedVertex, combinedEdge)
    graph
  }
}