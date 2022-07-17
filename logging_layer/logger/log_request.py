from datetime import datetime
from project_library_layer.datetime_libray import  date_time
from exception_layer.generic_exception.generic_exception import GenericException as AppLoggerException
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import uuid
import sys

class LogRequest:
    def __init__(self,executed_by=None, execution_id=None):
        self.log_database="log_request"
        self.log_collection_name="requests"
        self.executed_by=executed_by
        self.execution_id=execution_id
        self.mongo_db_object = MongoDBOperation()
        self.log_start_date=date_time.get_date()
        self.log_start_time=date_time.get_time()


    def get_log_data(self, request_data):
        try:
            if isinstance(request_data,dict):
                return request_data
            log_params = {
                          'Body':request_data.get_data().decode(),
                          'url':request_data.path,
                          }
            return log_params
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed in creating request data to log in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(LogRequest.__module__.__str__(), LogRequest.__name__,
                            self.get_log_data.__name__, request_data))
            message = Exception(app_logger_exception.error_message_detail(str(e), sys))
            aws = AmazonSimpleStorageService()
            file_name = 'log_' +self.log_writer_id + '.txt'
            aws.write_file_content('failed_log', file_name, message.__str__())

    def log_start(self,request):
        log_data=None
        try:
            self.log_writer_id=str(uuid.uuid4())
            file_object=None
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log_data = {
                'log_start_date':self.log_start_date,
                'log_start_time': self.log_start_time,
                'execution_id': self.execution_id,
                'executed_by':self.executed_by,
                'log_writer_id':self.log_writer_id,
                'request':self.get_log_data(request),
                'log_updated_time':datetime.now()
            }
            #print(log_data)
            self.mongo_db_object.insert_record_in_collection(self.log_database, self.log_collection_name, log_data)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(LogRequest.__module__.__str__(), LogRequest.__name__,
                            self.log_start.__name__,log_data))
            message= Exception(app_logger_exception.error_message_detail(str(e),sys))
            aws=AmazonSimpleStorageService()
            file_name='log_'+self.log_writer_id+'.txt'
            aws.write_file_content('failed_log',file_name,message.__str__())

    def log_stop(self,response:dict):
        log_data=None
        try:

            file_object = None
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")

            log_stop_date= date_time.get_date()
            log_stop_time=date_time.get_time()
            future_date="{} {}".format(log_stop_date,log_stop_time)
            past_date="{} {}".format(self.log_start_date,self.log_start_time)
            log_data = {
                'log_stop_date':log_stop_date,
                'log_stop_time': log_stop_time,
                'log_writer_id': self.log_writer_id,
                'execution_time_milisecond':date_time.get_difference_in_milisecond(future_date,past_date)

            }
            log_data.update(response)
            query={'execution_id': self.execution_id,}
            self.mongo_db_object.update_record_in_collection(self.log_database, self.log_collection_name, query,log_data)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(LogRequest.__module__.__str__(), LogRequest.__name__,
                            self.log_stop.__name__, log_data))
            message = Exception(app_logger_exception.error_message_detail(str(e), sys))
            aws = AmazonSimpleStorageService()
            file_name = 'log_' + self.log_writer_id + '.txt'
            aws.write_file_content('failed_log', file_name, message.__str__())

