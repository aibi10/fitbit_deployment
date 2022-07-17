from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from exception_layer.generic_exception.generic_exception import GenericException as SchedulerStorageException
import sys

from project_library_layer.initializer.initializer import Initializer


class SchedulerStorage():

    def __init__(self):
        try:
            self.mongo_db = MongoDBOperation()
            self.database_name = Initializer().get_scheduler_database_name()
            self.collection_name = Initializer().get_scheduler_collection_name()
        except Exception as e:
            train_model_exception = SchedulerStorageException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, SchedulerStorage.__name__,
                            "__init__"))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def update_job_record(self, job_id, status):
        """

        :param job_id:
        :param status:
        :return:
        """
        try:

            query = {'job_id': job_id}
            record =self.mongo_db.get_record(self.database_name, self.collection_name, query)
            if record is not None:
                record.update({'status': status})
                self.mongo_db.update_record_in_collection(self.database_name, self.collection_name, query, record)
        except Exception as e:
            raise e

