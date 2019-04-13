package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ EdgeProperty, PatientProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.storage.StorageLevel

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */
    var rankGraph: Graph[Double, Double] = graph.outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }.mapTriplets(e => 1.0 / e.srcAttr, TripletFields.Src).mapVertices { (id, attr) =>
      if (!(id != patientID)) 1.0 else 0.0
    }

    def delta(u: VertexId, v: VertexId): Double = { if (u == v) 1.0 else 0.0 }

    var iteration = 0
    var prevRankGraph: Graph[Double, Double] = null
    while (iteration < numIter) {
      rankGraph.cache()
      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)

      prevRankGraph = rankGraph
      val rPrb = (src: VertexId, id: VertexId) => alpha * delta(src, id)

      rankGraph = rankGraph.outerJoinVertices(rankUpdates) {
        (id, oldRank, msgSumOpt) => rPrb(patientID.asInstanceOf[VertexId], id) + (1.0 - alpha) * msgSumOpt.getOrElse(0.0)
      }.cache()

      rankGraph.edges.foreachPartition(x => {})
      prevRankGraph.vertices.unpersist()
      prevRankGraph.edges.unpersist()

      iteration += 1
    }

    val maxPatientId = graph.vertices.filter(x => x._2.isInstanceOf[PatientProperty]).map(x => x._1).max()
    val normed = normalizeRankSum(rankGraph)
    val filteredNormedPatients = normed.vertices.filter(x => x._1 <= maxPatientId && x._1 != patientID)
    val rankedPatients = filteredNormedPatients.sortBy(x => x._2, ascending = false).map(x => x._1.toLong).collect.toList
    rankedPatients
  }

  def normalizeRankSum(rankGraph: Graph[Double, Double]) = {
    val rankSum = rankGraph.vertices.values.sum()
    rankGraph.mapVertices((id, rank) => rank / rankSum)
  }
}
