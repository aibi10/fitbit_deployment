from threading import Thread
from exception_layer.generic_exception.generic_exception import GenericException as  AppLoggerException
from logging_layer.logger.logger import AppLogger
import sys


class LogDataThread(Thread):

    def __init__(self, project_id, execution_id, data_time_value=None):
        Thread.__init__(self)
        try:
            self.project_id = project_id
            self.execution_id = execution_id
            self.data_time_value = data_time_value

        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, LogDataThread.__name__,
                            "__init__"))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys)) from e

    def run(self):
        try:
            return AppLogger().get_log(project_id=self.project_id,execution_id=self.execution_id,data_time_value=self.data_time_value)
        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, LogDataThread.__name__,
                            "run"))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys)) from e


