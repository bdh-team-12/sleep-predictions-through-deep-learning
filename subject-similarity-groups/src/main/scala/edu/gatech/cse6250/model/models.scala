/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.model

case class Demographic(nsrrid: String, classifier: String, value: String)

case class MedicalHistory(nsrrid: String, condition: String, value: String)

case class Medication(nsrrid: String, medicine: String, value: String)

//case class Measurment(nsrrid: String, measurement: String, value: String)

abstract class VertexProperty

case class SubjectProperty(nsrrid: String) extends VertexProperty

case class DemographicProperty(classifier: String) extends VertexProperty

case class MedicalHistoryProperty(condition: String) extends VertexProperty

case class MedicationProperty(medicine: String) extends VertexProperty

//case class MeasurmentProperty(measurement: String) extends VertexProperty

abstract class EdgeProperty

case class SubjectDemographicEdgeProperty(classifier: Demographic) extends EdgeProperty

case class SubjectMedicalHistoryEdgeProperty(condition: MedicalHistory) extends EdgeProperty

case class SubjectMedicationEdgeProperty(medication: Medication) extends EdgeProperty

//case class SubjectMeasurmentEdgeProperty(measurement: Measurment) extends EdgeProperty

