from threading import Thread
from exception_layer.generic_exception.generic_exception import GenericException as AppLoggerException
from logging_layer.logger.logger import AppLogger
import sys


class WatcherThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        try:
           pass

        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, WatcherThread.__name__,
                            "__init__"))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys)) from e

    def run(self):
        try:
            pass

        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, WatcherThread.__name__,
                            "run"))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys)) from e


