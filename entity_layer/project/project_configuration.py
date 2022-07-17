from exception_layer.generic_exception.generic_exception import GenericException as ProjectConfigurationException
import sys
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer
from entity_layer.project.project import Project
import json
class ProjectConfiguration:

    def __init__(self,project_id=None,cloud_storage=None,machine_learning_type=None,file_name_pattern=None,
                 training_schema_definition_json=None,prediction_schema_definition_json=None):
        try:
            self.project_id=project_id
            self.cloud_storage=cloud_storage
            self.machine_learning_type=machine_learning_type
            self.file_name_pattern=file_name_pattern
            self.training_schema_definition_json=training_schema_definition_json
            self.prediction_schema_definition_json=prediction_schema_definition_json
            self.mongo_db = MongoDBOperation()
            self.initial = Initializer()
            self.database = self.initial.get_project_system_database_name()
            self.collection = self.initial.get_project_configuration_collection_name()
            self.project_detail=Project()

        except Exception as e:
            project_config_exception = ProjectConfigurationException("Failed object initialization in class [{0}] method [{1}]"
                                                           .format(ProjectConfiguration.__name__,
                                                                   "__init__"))
            raise Exception(project_config_exception.error_message_detail(str(e), sys)) from e

    def get_project_configuration_detail(self,project_id):
        """

        :param project_id: accept project id
        :return: project configuration detail
         return {'status':True,'message':'Project configuration found','project_config_detail':project_config_detail}
        """
        try:
            project_config_query={'project_id':project_id}
            project_config_detail=self.mongo_db.get_record(self.database,self.collection,project_config_query)
            if project_config_detail is None:
                return {'status':False,'message':'project configuration not found'}
            return {'status':True,'message':'Project configuration found','project_config_detail':project_config_detail}

        except Exception as e:
            project_config_exception = ProjectConfigurationException(
                "Not able to retrive project configuration in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.get_project_configuration_detail.__name__))
            raise Exception(project_config_exception.error_message_detail(str(e), sys)) from e


    def save_project_configuration(self):

        try:
            project_detail=self.project_detail.get_project_detail(self.project_id)
            if not project_detail['status']:
                return project_detail
            if 'project_detail' not in project_detail:
                return {'status':False,'message':'Project not found'}
            project_detail=project_detail['project_detail']
            project_name=project_detail['project_name']
            project_config={'project_id': self.project_id,
            }
            records = self.mongo_db.get_record(self.database,self.collection,project_config)

            if records is None:
                project_config.update(
                    {'machine_learning_type':self.machine_learning_type,
                     'file_name_pattern':self.file_name_pattern,
                      'cloud_storage': self.cloud_storage})
                result = self.mongo_db.insert_record_in_collection(
                    self.database,self.collection,project_config
                )
                if result > 0:
                    project_schema={'project_id':self.project_id,'schema':self.training_schema_definition_json}
                    schema_training_inserted=self.mongo_db.insert_record_in_collection(self.initial.get_project_system_database_name(),
                                           self.initial.get_schema_training_collection_name(),project_schema)
                    msg=""
                    error_msg=""
                    if schema_training_inserted>0:
                        msg=msg+'Training schema definition updated'
                    else:
                        error_msg=error_msg+"Failed to save training schema definition"
                    project_schema = {'project_id': self.project_id,
                                      'schema': self.prediction_schema_definition_json}
                    schema_prediction_inserted = self.mongo_db.insert_record_in_collection(
                        self.initial.get_project_system_database_name(),
                        self.initial.get_schema_prediction_collection_name(), project_schema)
                    if schema_prediction_inserted>0:
                        msg=msg+ " Prediction schema definition updated"
                    else:
                        error_msg=error_msg+"failed to save prediction schema definition"
                    if len(error_msg)>1:
                        return {'status':False,'message':error_msg}
                    return {'status': True, 'message': 'Project configuration {} added. {}'.format(project_name,msg)}
            else:
                return {'status': False, 'message': 'Project  configuration {} already present. '.format(project_name)}
        except Exception as e:
            project_config_exception = ProjectConfigurationException(
                "Failed during saving project configuration in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ProjectConfiguration.__name__,
                            self.save_project_configuration.__name__))
            raise Exception(project_config_exception.error_message_detail(str(e), sys)) from e

