from datetime import datetime
from project_library_layer.datetime_libray import  date_time
from exception_layer.generic_exception.generic_exception import GenericException as AppLoggerException
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import uuid
import sys

class LogExceptionDetail:
    def __init__(self, executed_by, execution_id, log_database=None, log_collection_name=None):
        if log_database is None:
            self.log_database='exception_log'
        else:
            self.log_database=log_database
        if log_collection_name is None:
            self.log_collection_name='exception_collection'
        else:
            self.log_collection_name=log_collection_name
        self.executed_by=executed_by
        self.execution_id=execution_id
        self.mongo_db_object=MongoDBOperation()

    def log(self,  log_message):
        try:
            log_writer_id=str(uuid.uuid4())

            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log_data = {
                'log_updated_date': date_time.get_date(),
                'log_update_time': date_time.get_time(),
                'execution_id': self.execution_id,
                'message':log_message,
                'executed_by':self.executed_by,

                'log_writer_id':log_writer_id
            }
            self.mongo_db_object.insert_record_in_collection(
                self.log_database, self.log_collection_name, log_data)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(LogExceptionDetail.__module__.__str__(), LogExceptionDetail.__name__,
                            self.log.__name__,log_data))
            message= Exception(app_logger_exception.error_message_detail(str(e),sys))
            aws=AmazonSimpleStorageService()
            file_name='log_'+log_writer_id+'.txt'
            aws.write_file_content('failed_exception_log',file_name,message.__str__())
