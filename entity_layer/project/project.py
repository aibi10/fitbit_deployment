from exception_layer.generic_exception.generic_exception import GenericException as ProjectException
import sys
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer


class Project:

    def __init__(self, project_name=None, project_description=None):
        try:
            self.project_name = project_name
            self.project_description = project_description
            self.mongo_db = MongoDBOperation()
            self.initial = Initializer()
            self.database = self.initial.get_project_system_database_name()
            self.collection = self.initial.get_project_collection_name()

        except Exception as e:
            registration_exception = ProjectException("Failed email address validation in class [{0}] method [{1}]"
                                                      .format(Project.__name__,
                                                              "__init__"))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def save_project(self):

        try:

            project = {'project_name': self.project_name, 'project_description': self.project_description}
            records = self.mongo_db.get_record(self.database, self.collection, project)
            if records is None:
                project_id = self.mongo_db.get_max_value_of_column(self.database,
                                                              self.collection,
                                                              query={},
                                                              column='project_id'
                                                              )
                if project_id is None:
                    project_id = 1
                else:
                    project_id = project_id + 1
                project.update({'project_id': project_id})
                result = self.mongo_db.insert_record_in_collection(
                    self.database, self.collection, project
                )
                if result > 0:
                    return {'status': True, 'message': 'Project {} added. '.format(self.project_name)}
            else:
                return {'status': False, 'message': 'Project {} already present. '.format(self.project_name)}
        except Exception as e:
            project_exception = ProjectException(
                "Save project in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Project.__name__,
                            self.save_project.__name__))
            raise Exception(project_exception.error_message_detail(str(e), sys)) from e

    def list_project(self):
        try:
            project_list=[project for project in self.mongo_db.get_records(self.database, self.collection, {})]
            if len(project_list)>0:
                return {'status':True,'message':"Project list found",'project_list':project_list}
            else:
                return {'status':False,'message':"Project list not found"}
        except Exception as e:
            project_exception = ProjectException(
                "list project in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Project.__name__,
                            self.list_project.__name__))
            raise Exception(project_exception.error_message_detail(str(e), sys)) from e

    def get_project_detail(self,project_id):
        """

        :param project_id:
        :return: return {'status':True,'message':'Project detail found','project_detail':project_detail}
        """
        try:
            project_query = {'project_id': project_id}
            project_detail = self.mongo_db.get_record(self.database, self.collection, project_query)
            if project_detail is None:
                result = {'status': False, 'message': 'Project not found'}
                return result
            else:
                return {'status':True,'message':'Project detail found','project_detail':project_detail}

        except Exception as e:
            project_exception = ProjectException(
                "Failed in get project in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Project.__name__,
                            self.get_project_detail.__name__))
            raise Exception(project_exception.error_message_detail(str(e), sys)) from e

