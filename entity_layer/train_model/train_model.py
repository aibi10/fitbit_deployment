import json

from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
from project_library_layer.initializer.initializer import Initializer
from exception_layer.generic_exception.generic_exception import GenericException as TrainModelException
from controller.project_controller.projects.mushroom.train_model_murshroom import TrainingModel as TrainModelMushroom
import sys
from project_library_layer.project_training_prediction_mapper.project_training_prediction_mapper import \
    get_training_validation_and_training_model_class_name
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from entity_layer.email_sender.email_sender import EmailSender


class TrainModel:

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
            train_model_exception = TrainModelException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.__init__.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def training_model(self):
        try:

            if self.project_id is None:
                raise Exception("Project id not found")
            project_detail = self.project_detail.get_project_detail(project_id=self.project_id)
            if not project_detail['status']:
                project_detail.update(
                    {'is_failed':True,'message':"Project detail not found",'message_status': 'info', 'project_id': self.project_id})
                return project_detail

            project_config_detail = self.project_config.get_project_configuration_detail(project_id=self.project_id)
            if not project_config_detail['status']:
                project_config_detail.update(
                    {'is_failed':True,'message':"Project configuration not found",'message_status': 'info', 'project_id': self.project_id})
                return project_config_detail
            if 'project_config_detail' in project_config_detail:
                project_config_detail = project_config_detail['project_config_detail']
            if project_config_detail is None:
                response = {'is_failed':True,'status': False, 'message': 'project configuration not found',
                            'message_status': 'info', 'project_id': self.project_id}

                return response
            training_file_path = self.initializer.get_training_batch_file_path(project_id=self.project_id)
            cloud_storage = None
            if 'cloud_storage' in project_config_detail:
                cloud_storage = project_config_detail['cloud_storage']
            if cloud_storage is None:
                result = {'status': False,'is_failed':True,
                          'message': 'Cloud Storage location not found',
                          'message_status': 'info', 'project_id': self.project_id}

                return result

            TrainingValidation, TrainingModel = get_training_validation_and_training_model_class_name(self.project_id)
            if TrainingValidation is not None:
                train_validation_object = TrainingValidation(project_id=self.project_id,
                                                             training_file_path=training_file_path,
                                                             executed_by=self.executed_by,
                                                             execution_id=self.execution_id,
                                                             cloud_storage=cloud_storage,
                                                             socket_io=self.socket_io

                                                             )  # object initialization

                train_validation_object.train_validation()  # calling the training_validation function

                training_model_object = TrainingModel(project_id=self.project_id,
                                                      executed_by=self.executed_by,
                                                      execution_id=self.execution_id,
                                                      cloud_storage=cloud_storage,
                                                      socket_io=self.socket_io

                                                      )  # object initialization
                training_model_object.training_model()  # training the model for the files in the table

                response = {'status': True, 'message': 'Training completed successfully', 'is_failed': False,
                            'message_status': 'info', 'project_id': self.project_id}

            else:
                training_data = MongoDBOperation().get_record("sentiment_data_training", "sentiment_input",
                                                              {'execution_id': self.execution_id})
                print(training_data)
                if training_data is None:
                    raise Exception("Training data not found")
                sentiment_user_id = int(training_data['sentiment_user_id'])
                sentiment_data = json.loads(training_data['sentiment_data'])
                sentiment_project_id = int(training_data['sentiment_project_id'])
                train_model = TrainingModel(self.project_id, execution_id=self.execution_id,
                                            executed_by=self.executed_by)

                res = train_model.trainModel(global_project_id=self.project_id,
                                             projectId=sentiment_project_id,
                                             userId=sentiment_user_id,
                                             data=sentiment_data,
                                             )
                if res:
                    response = {'status': True, 'message': 'Training completed successfully', 'is_failed': False,
                                'message_status': 'info', 'project_id': self.project_id}

                else:
                    response = {'status': False, 'message': 'Training Failed',
                                'message_status': 'info', 'project_id': self.project_id, 'is_failed': True, }
            return response
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during model training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.training_model.__name__))

            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e
