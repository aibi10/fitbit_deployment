from threading import Thread

from entity_layer.email_sender.email_sender import EmailSender
from entity_layer.predict_from_model.prediction_from_model import PredictFromModel, PredictFromModelException
from logging_layer.logger.log_exception import LogExceptionDetail
from project_library_layer.initializer.initializer import Initializer
from project_library_layer.datetime_libray.date_time import get_date, get_time
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import sys


class PredictFromModelThread(Thread):

    def __init__(self, project_id, executed_by, execution_id, socket_io=None, log_writer=None):
        """

        :param project_id: project_id
        :param executed_by: executed_by
        :param execution_id: execution_id
        """
        Thread.__init__(self)
        self.project_id = project_id
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.initialize = Initializer()
        self.mongo_db = MongoDBOperation()
        self.prediction_thread_database_name = self.initialize.get_prediction_thread_database_name()
        self.thread_status_collection_name = self.initialize.get_thread_status_collection_name()
        self.socket_io = socket_io
        self.log_writer = log_writer

    def get_max_status_id(self):
        try:
            max_status_id = None
            if self.mongo_db.is_database_present(self.mongo_db.get_database_client_object(),
                                                 self.prediction_thread_database_name):
                database_obj = self.mongo_db.create_database(self.mongo_db.get_database_client_object(),
                                                             self.prediction_thread_database_name)
                if self.mongo_db.is_collection_present(self.thread_status_collection_name, database_obj):
                    max_status_id = self.mongo_db.get_max_value_of_column(self.prediction_thread_database_name,
                                                                          self.thread_status_collection_name,
                                                                          {'project_id': self.project_id},
                                                                          'status_id')

            return max_status_id
        except Exception as e:
            predict_model_exception = PredictFromModelException(
                "Failed during get_max_status_id in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictFromModelThread.__name__,
                            self.get_max_status_id.__name__))
            raise Exception(predict_model_exception.error_message_detail(str(e), sys)) from e

    def get_running_status_of_prediction_thread(self):
        try:
            max_status_id = self.get_max_status_id()
            if max_status_id is not None:
                response = self.mongo_db.get_record(self.prediction_thread_database_name,
                                                    self.thread_status_collection_name,
                                                    {'project_id': self.project_id, 'status_id': max_status_id})

                return response
            else:
                return None
        except Exception as e:
            predict_model_exception = PredictFromModelException(
                "Failed during get_running_status_of_training_thread in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictFromModelThread.__name__,
                            self.get_running_status_of_prediction_thread.__name__))
            raise Exception(predict_model_exception.error_message_detail(str(e), sys)) from e

    def run(self):
        record = None
        status_id = None
        try:
            is_prediction_already_running = False
            max_status_id = self.get_max_status_id()
            if max_status_id is not None:
                response = self.get_running_status_of_prediction_thread()
                if 'is_running' in response:
                    is_prediction_already_running = response['is_running']

            if is_prediction_already_running:
                result = {'status': True, 'message': "Training/prediction is already in progress please wait..."}
                EmailSender().send_email(
                    mail_text=f"""

                Training/prediction is already in progress. Try again once current operation finished.
                
                Execution Detail:
                Execution_id:{self.execution_id}
                Executed by:{self.executed_by}
                project_id:{self.project_id}""",
                    subject=f"Prediction notification email of project_id {self.project_id}")
                if self.socket_io is not None:
                    self.socket_io.emit("prediction_started" + str(self.project_id),
                                        {
                                            'message': result['message']}
                                        , namespace="/training_model")
                    self.log_writer.log_stop(result)
                return result

            if max_status_id is None:
                status_id = 1
            else:
                status_id = max_status_id + 1
            record = {'project_id': self.project_id,
                      'execution_id': self.execution_id,
                      'executed_by': self.executed_by,
                      'status_id': status_id,
                      'is_running': True,
                      'start_date': get_date(),
                      'start_time': get_time(),
                      'message': 'prediction running...',
                      'is_Failed': None,
                      'process_type': 'prediction'
                      }
            EmailSender().send_email(f"""

                                        Prediction is in  progress ....

                                        Execution Detail:
                                        Execution_id:{self.execution_id}
                                        Executed by:{self.executed_by}
                                        project_id:{self.project_id}


                                        """, subject=f"Prediction notification email of project_id {self.project_id}")
            if self.socket_io is not None:
                self.socket_io.emit("prediction_started" + str(self.project_id),
                                    {
                                        'message': record['message']}
                                    , namespace="/training_model")
            self.mongo_db.insert_record_in_collection(self.prediction_thread_database_name,
                                                      self.thread_status_collection_name,
                                                      record
                                                      )
            prediction_from_model_obj = PredictFromModel(project_id=self.project_id, executed_by=self.executed_by,
                                                         execution_id=self.execution_id, socket_io=self.socket_io)
            prediction_result = prediction_from_model_obj.prediction_from_model()
            record.update({
                'is_running': False,
                'is_Failed': prediction_result['is_failed'],
                'message': str(prediction_result['message']),
                'stop_time': get_time(),
                'stop_date': get_date()

            })
            if self.socket_io is not None:
                self.socket_io.emit("prediction_started" + str(self.project_id),
                                    {
                                        'message': record['message']}
                                    , namespace="/training_model")
            if '_id' in record:
                record.pop('_id')
            self.mongo_db.update_record_in_collection(self.prediction_thread_database_name,
                                                      self.thread_status_collection_name,
                                                      {'status_id': status_id, 'project_id': self.project_id}, record)

            response = {'message': 'Prediction completed', 'status': True, 'message_status': 'info',
                        'project_id': self.project_id}
            if self.socket_io is not None:
                self.socket_io.emit("prediction_completed" + str(self.project_id),
                                    {'message': response['message']}
                                    , namespace="/training_model")
            self.log_writer.log_stop(response)
            EmailSender().send_email(f"""

                                                    Message: {response['message']}

                                                    Execution Detail:
                                                    Execution_id:{self.execution_id}
                                                    Executed by:{self.executed_by}
                                                    project_id:{self.project_id}


                                                    """,
                                     subject=f"Prediction notification email of project_id {self.project_id}")
            return response
        except Exception as e:
            EmailSender().send_email(f"""

                                                    Error message: {str(e)}

                                                    Execution Detail:
                                                    Execution_id:{self.execution_id}
                                                    Executed by:{self.executed_by}
                                                    project_id:{self.project_id}


                                                    """,
                                     subject=f"Training notification email of project_id {self.project_id}")
            if record is not None and status_id is not None:
                record.update({
                    'is_running': False,
                    'is_Failed': True,
                    'message': 'prediction failed due to :{}'.format(str(e)),
                    'stop_time': get_time(),
                    'stop_date': get_date()

                })
                if '_id' in record:
                    record.pop('_id')
                self.mongo_db.update_record_in_collection(self.prediction_thread_database_name,
                                                          self.thread_status_collection_name,
                                                          {'status_id': status_id, 'project_id': self.project_id},
                                                          record)

            train_model_exception = PredictFromModelException(
                "Failed during model prediction in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictFromModelThread.__name__,
                            self.run.__name__))
            if self.socket_io is not None:
                self.socket_io.emit("prediction_completed" + str(self.project_id),
                                    {'message': train_model_exception.error_message_detail(str(e), sys)}
                                    , namespace="/training_model")
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(train_model_exception.error_message_detail(str(e), sys))
            # raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e
