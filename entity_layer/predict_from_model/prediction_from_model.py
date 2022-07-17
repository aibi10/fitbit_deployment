from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
from project_library_layer.initializer.initializer import Initializer
from controller.project_controller.projects.WaferFaultDetection_new.prediction_Validation_Insertion import \
    PredictionValidation
from controller.project_controller.projects.WaferFaultDetection_new.predictFromModel import Prediction
from exception_layer.generic_exception.generic_exception import GenericException as PredictFromModelException
from controller.project_controller.projects.mushroom.predict_from_model_mushroom import \
    Prediction as PredictionOfMushroom
from project_library_layer.project_training_prediction_mapper.project_training_prediction_mapper import \
    get_prediction_validation_and_prediction_model_class_name

#from entity_layer.streaming.azure_event_hub_sent_data import start_call
import sys


class PredictFromModel:

    def __init__(self, project_id, executed_by, execution_id, socket_io=None):
        try:
            self.project_id = project_id
            self.executed_by = executed_by
            self.execution_id = execution_id
            self.project_detail = Project()
            self.project_config = ProjectConfiguration()
            self.initializer = Initializer()
            self.socket_io = socket_io
        except Exception as e:
            predict_from_model_exception = PredictFromModelException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.__init__.__name__))
            raise Exception(predict_from_model_exception.error_message_detail(str(e), sys)) from e

    def prediction_from_model(self):
        try:


            if self.project_id is None:
                raise Exception("Project id not found")
            project_detail = self.project_detail.get_project_detail(project_id=self.project_id)
            if not project_detail['status']:
                project_detail.update(
                    {'is_failed':True,'message':"Project detail not found",'message_status': 'info', 'project_id': self.project_id})
                return project_detail
            project_detail=project_detail['project_detail']
            project_config_detail = self.project_config.get_project_configuration_detail(project_id=self.project_id)
            if not project_config_detail['status']:
                project_config_detail.update(
                    {'is_failed':True,'message':"Project configuration not found",'message_status': 'info', 'project_id': self.project_id})
                return project_config_detail
            if 'project_config_detail' in project_config_detail:
                project_config_detail = project_config_detail['project_config_detail']
            if project_config_detail is None:
                response = {'status': False, 'message': 'project configuration not found',
                            'message_status': 'info', 'project_id': self.project_id,'is_failed':True}

                return response
            prediction_file_path = Initializer().get_prediction_batch_file_path(project_id=self.project_id)
            cloud_storage = None
            if 'cloud_storage' in project_config_detail:
                cloud_storage = project_config_detail['cloud_storage']
            if cloud_storage is None:
                result = {'status': False,
                          'message': 'Cloud Storage location not found',
                          'message_status': 'info', 'project_id': self.project_id,
                          'is_failed': True,
                          }

                return result
            PredictionValidation, Prediction = get_prediction_validation_and_prediction_model_class_name(
                self.project_id)
            if PredictionValidation is not None:
                pred_val = PredictionValidation(project_id=self.project_id,
                                                prediction_file_path=prediction_file_path,
                                                executed_by=self.executed_by,
                                                execution_id=self.execution_id,
                                                cloud_storage=cloud_storage,
                                                socket_io=self.socket_io
                                                )  # object initialization

                pred_val.prediction_validation()  # calling the training_validation function

                pred = Prediction(project_id=self.project_id,
                                  executed_by=self.executed_by,
                                  execution_id=self.execution_id,
                                  cloud_storage=cloud_storage,
                                  socket_io=self.socket_io
                                  )  # object initialization
                prediction_generated_file = pred.prediction_from_model()  # training the model for the files in the table

                response = {'status': True,'is_failed':False,
                            'message': 'Prediction completed at path {}'.format(prediction_generated_file),
                            'message_status': 'info', 'project_id': self.project_id}

            else:

                prediction_data=MongoDBOperation().get_record("sentiment_data_prediction", "sentiment_input", {
                    'execution_id': self.execution_id,
                })
                if prediction_data is None:
                    raise Exception("Prediction data not found")
                text=prediction_data['sentiment_data']
                sentiment_user_id = int(prediction_data['sentiment_user_id'])

                sentiment_project_id = int(prediction_data['sentiment_project_id'])
                pred = Prediction(self.project_id,execution_id=self.execution_id,executed_by=self.executed_by)

                res = pred.predictRoute(global_project_id=self.project_id,
                                        projectId=sentiment_project_id,
                                        userId=sentiment_user_id,
                                        text=text)
                print(res)
                #send data to azure event hub
                #start_call(prediction_label=res.__str__(),project_name=project_detail['project_name'],execution_id=self.execution_id)

                response = {'status': True,'is_failed':False, 'message': 'Predicted label {}'.format(res),
                            'message_status': 'info', 'project_id': self.project_id}
            return response
        except Exception as e:

            predict_from_model_exception = PredictFromModelException(
                "Failed during prediction from in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.prediction_from_model.__name__))
            raise Exception(predict_from_model_exception.error_message_detail(str(e), sys)) from e
