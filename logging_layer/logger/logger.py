from datetime import datetime
from project_library_layer.datetime_libray import date_time
from exception_layer.generic_exception.generic_exception import GenericException as AppLoggerException
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer
import uuid
import sys
import time
from dateutil.parser import parse
import pandas as pd


class AppLogger:
    def __init__(self, project_id=None, log_database=None, log_collection_name=None, executed_by=None,
                 execution_id=None, socket_io=None):
        self.log_database = log_database
        self.log_collection_name = log_collection_name
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.mongo_db_object = MongoDBOperation()
        self.project_id = project_id
        self.socket_io = socket_io

    def log(self, log_message):
        log_writer_id = str(uuid.uuid4())
        log_data = None
        try:
            if self.socket_io is not None:
                if self.log_database == Initializer().get_training_database_name():
                    self.socket_io.emit("started_training" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}
                                        , namespace="/training_model")

                if self.log_database == Initializer().get_prediction_database_name():
                    self.socket_io.emit("prediction_started" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}
                                        , namespace="/training_model")

            file_object = None
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log_data = {
                'log_updated_date': date_time.get_date(),
                'log_update_time': date_time.get_time(),
                'execution_id': self.execution_id,
                'message': log_message,
                'executed_by': self.executed_by,
                'project_id': self.project_id,
                'log_writer_id': log_writer_id,
                'updated_date_and_time': datetime.now()
            }
            with open("log.txt", "a+") as f:
                f.write("<p style='color:red'>{} {}</P>: {} {} > {}\n".format(self.execution_id, self.executed_by,date_time.get_date(), date_time.get_time(),

                                                 log_message))
            self.mongo_db_object.insert_record_in_collection(
                self.log_database, self.log_collection_name, log_data)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(AppLogger.__module__.__str__(), AppLogger.__name__,
                            self.log.__name__, log_data))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys))

    def get_log(self, project_id, execution_id, process_type=None, data_time_value=None):
        try:
            self.line_data = []
            yield "<h3>Please find log detail..</h3><br>"
            while True:
                with open("log.txt") as f:
                    for line in f.readlines():
                        if execution_id in line and line not in self.line_data:
                            self.line_data.append(line)
                            yield line + "</br>"
                    time.sleep(5)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}]"
                    .format(AppLogger.__module__.__str__(), AppLogger.__name__,
                            self.log.__name__))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys))
