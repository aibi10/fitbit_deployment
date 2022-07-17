import uuid
from datetime import datetime, timedelta

from exception_layer.generic_exception.generic_exception import GenericException as SchedulerException
import sys
from flask_apscheduler import APScheduler
from project_library_layer.initializer.initializer import Initializer
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.datetime_libray.date_time import get_time, get_date, is_future_date
from entity_layer.project.project import Project
from entity_layer.scheduler.scheduler_task import ScheduleTask

class Scheduler(APScheduler):

    def __init__(self,socket_io=None):
        try:
            super(Scheduler, self).__init__()
            self.api_enabled = True
            self.initializer = Initializer()
            self.database_name = self.initializer.get_scheduler_database_name()
            self.collection_name = self.initializer.get_scheduler_collection_name()
            self.mongo_db = MongoDBOperation()
            self.reoccurring = "reoccurring"
            self.socket_io=socket_io
            self.non_reoccurring = "non-reoccurring"
            job_list = self.mongo_db.get_records(self.database_name, self.collection_name, {})
            possible_job_to_add = []
            if job_list is not None:
                for job in job_list:
                    job_schedule_time = str(job['start_time'])[:16]
                    current_time = str(get_date() + " " + get_time())
                    if is_future_date(job_schedule_time, current_time):
                        if job['job_type'] == self.reoccurring:
                            interval_type = job['interval']
                            if interval_type == 'hour':
                                self.add_recurring_job_in_hour(hour=job['interval_value'], job_name=job['job_name']
                                                               , project_id=job['project_id'],
                                                               email_address=job['submitted_by'],
                                                               is_record_inserted=True,
                                                               job_id=job['job_id'],
                                                               action_name=job['Action']

                                                               )
                            if interval_type == 'minute':
                                self.add_recurring_job_in_minute(minute=job['interval_value'], job_name=job['job_name']
                                                                 , project_id=job['project_id'],
                                                                 email_address=job['submitted_by'],
                                                                 is_record_inserted=True,
                                                                 job_id=job['job_id'],
                                                                 action_name=job['Action']

                                                                 )
                            if interval_type == 'second':
                                self.add_recurring_job_in_second(second=job['interval_value'], job_name=job['job_name']
                                                                 , project_id=job['project_id'],
                                                                 email_address=job['submitted_by'],
                                                                 is_record_inserted=True,
                                                                 job_id=job['job_id'],
                                                                 action_name=job['Action']

                                                                 )
                            if interval_type=='week':
                                self.add_recurring_job_weekly_basis(is_reoccurring=self.reoccurring,days_of_week=
                                                                    job['interval_value'],job_name=job['job_name'],
                                                                    project_id=job['project_id'],
                                                                    email_address=job['submitted_by'],
                                                                    is_record_inserted=True,
                                                                    job_id=job['job_id'],
                                                                    action_name=job['Action']

                                                                    )

                        else:
                            self.add_job_at_time(job_schedule_time, job['job_name']
                                                 , project_id=job['project_id'],
                                                 email_address=job['submitted_by'],
                                                 is_record_inserted=True,
                                                 job_id=job['job_id'],
                                                 action_name=job['Action']
                                                 )
        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            "__init__"))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def get_all_job(self):
        try:
            records = self.mongo_db.get_records(self.database_name, self.collection_name, {})
            job_detail = []
            for record in records:
                job_detail.append(record)
            if len(job_detail) > 0:
                return {'status': True, 'message': 'Job detail found', 'job_list': job_detail}
            else:
                return {'status': False, 'message': 'Job job_detail not found'}
        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.get_all_job.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e



    def create_job_record(self, job_name, job_id, scheduled_time, project_id, type_of_job=None, interval=None,
                          interval_value=None, status="waiting", submitted_by=None, action_name=None):
        """

        :param action_name:
        :param job_name: Job name
        :param job_id: Job id unique identifier
        :param scheduled_time: scheduled time
        :param project_id: project id
        :param type_of_job: type of job eg: {reoccurring,non-reoccurring}
        :param interval: minute/hour/second
        :param interval_value:  interval value in integer
        :param status: waiting ,running, failed, success
        :param submitted_by: user email id
        :return:
        """
        try:
            project_detail = Project()
            result = project_detail.get_project_detail(project_id=project_id)
            project_name = None
            if result['status']:
                project_detail = result.get('project_detail', None)
                if project_detail is not None:
                    project_name = project_detail.get('project_name', None)
            if project_name is None:
                raise Exception("Project name not found")
            record = {'created_date': get_date(),
                      'create_time': get_time(),
                      'job_name': job_name,
                      'job_id': job_id,
                      'start_time': scheduled_time,
                      'project_id': project_id,
                      'project_name': project_name,
                      'job_type': type_of_job,
                      'interval': interval,
                      'interval_value': interval_value,
                      'status': status,
                      'submitted_by': submitted_by,
                      'Action': action_name
                      }
            self.mongo_db.insert_record_in_collection(self.database_name, self.collection_name, record)
        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.create_job_record.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def schedule_task(self, second):
        """

        :param second:
        :return:
        """
        print("This test runs every {} seconds".format(second))

    def add_recurring_job_in_second(self, second, job_name, project_id,
                                    email_address,action_name=None, is_record_inserted=False, job_id=None):
        """

        :param job_id:
        :param email_address:
        :param project_id:
        :param job_name:
        :param is_record_inserted:
        :param action_name:
        :param second:
        :return:
        """
        try:
            if job_id is None:
                job_id = str(uuid.uuid4())

            if not is_record_inserted:
                self.create_job_record(job_name=job_name, job_id=job_id,
                                       scheduled_time=datetime.now() + timedelta(seconds=second), project_id=project_id,
                                       type_of_job=self.reoccurring, interval='second', interval_value=second,
                                       status="waiting", submitted_by=email_address,action_name=action_name
                                       )
            sch_task = ScheduleTask(project_id=int(project_id),
                                    executed_by=email_address,
                                    execution_id=job_id,
                                    socket_io=self.socket_io

                                    )
            if 'training' in action_name and 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training_prediction_both, trigger="interval", seconds=second,
                             )
                return True
            if 'training' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training, trigger="interval", seconds=second,
                            )
                return True
            if 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_prediction, trigger="interval", seconds=second,
                             )
                return True

        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed during model training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.add_recurring_job_in_second.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def schedule_task_in_minute(self, minute):
        try:
            print("This test runs every {} minute".format(minute))
        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.schedule_task_in_minute.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def add_recurring_job_in_minute(self, minute, job_name, project_id,
                                    email_address,action_name, is_record_inserted=False,job_id=None):
        try:
            if job_id is None:
                job_id = str(uuid.uuid4())

            if not is_record_inserted:
                self.create_job_record(job_name=job_name, job_id=job_id,
                                       scheduled_time=datetime.now() + timedelta(minutes=minute), project_id=project_id,
                                       type_of_job=self.reoccurring, interval='minute', interval_value=minute,
                                       status="waiting", submitted_by=email_address,
                                       action_name=action_name
                                       )
            sch_task = ScheduleTask(project_id=int(project_id),
                                    executed_by=email_address,
                                    execution_id=job_id,
                                    socket_io=self.socket_io
                                    )
            if 'training' in action_name and 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training_prediction_both, trigger="interval", minutes=minute,
                            )
                return True
            if 'training' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training, trigger="interval", minutes=minute,
                             )
                return True
            if 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_prediction, trigger="interval", minutes=minute,
                             )
                return True

        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.add_recurring_job_in_minute.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def add_recurring_job_in_hour(self, hour, job_name, project_id,
                                  email_address, action_name,is_record_inserted=False, job_id=None):
        try:
            if job_id is None:
                job_id = str(uuid.uuid4())
            if not is_record_inserted:
                self.create_job_record(job_name=job_name, job_id=job_id,
                                       scheduled_time=datetime.now() + timedelta(hours=hour), project_id=project_id,
                                       type_of_job=self.reoccurring, interval='hour', interval_value=hour,
                                       status="waiting", submitted_by=email_address,action_name=action_name
                                       )
            sch_task = ScheduleTask(project_id=int(project_id),
                                    executed_by=email_address,
                                    execution_id=job_id,
                                    socket_io=self.socket_io
                                    )
            if 'training' in action_name and 'prediction' in action_name:
                self.add_job(id='job_id', func=sch_task.start_training_prediction_both, trigger="interval", hours=hour,)
                return True
            if 'training' in action_name:
                self.add_job(id='job_id', func=sch_task.start_training, trigger="interval", hours=hour,)
                return True
            if 'prediction' in action_name:
                self.add_job(id='job_id', func=sch_task.start_prediction, trigger="interval", hours=hour, )
                return True


        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.add_recurring_job_in_hour.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def my_job(self, arg):
        print("This is called at {}".format(arg))

    def add_job_at_time(self, date_time: str, job_name, project_id, email_address,
                        action_name, is_record_inserted=False,job_id=None):
        try:
            if job_id is None:
                job_id = str(uuid.uuid4())
            if not is_record_inserted:
                self.create_job_record(job_name=job_name, job_id=job_id, scheduled_time=date_time,
                                       project_id=project_id,
                                       type_of_job=self.non_reoccurring, interval=None, interval_value=None,
                                       status="waiting", submitted_by=email_address,action_name=action_name
                                       )
            sch_task = ScheduleTask(project_id=int(project_id),
                                    executed_by=email_address,
                                    execution_id=job_id,
                                    socket_io=self.socket_io
                                    )
            if 'training' in action_name and 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training_prediction_both, trigger='date', run_date=date_time,)
                return True
            if 'training' in action_name:
                self.add_job(id=job_id, func=sch_task.start_training, trigger='date', run_date=date_time, )
                return True
            if 'prediction' in action_name:
                self.add_job(id=job_id, func=sch_task.start_prediction, trigger='date', run_date=date_time, )
                return True

        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.add_job_at_time.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e



    def add_recurring_job_weekly_basis(self,is_reoccurring, days_of_week,job_name, project_id, email_address,action_name,
                                       is_record_inserted=False, job_id=None):
        try:

            week_day_name=days_of_week[:3]
            week = ['sun',
                    'mon',
                    'tue',
                    'wed',
                    'thur',
                    'fri',
                    'sat']
            week_day_number=week.index(week_day_name)+1
            week_day_number_of_today= datetime.today().weekday()+1
            n_day_diff=0
            if week_day_number>week_day_number_of_today:
                n_day_diff=week_day_number-week_day_number_of_today
            else:
                n_day_diff=7-week_day_number_of_today+week_day_number
            date_time=str(datetime.now()+timedelta(days=n_day_diff))
            date_time=date_time[:11]+" 05:30"
            if job_id is None:
                job_id = str(uuid.uuid4())
            if is_reoccurring=='No':
                is_reoccurring=self.non_reoccurring
            else:
                is_reoccurring=self.reoccurring
            if not is_record_inserted:
                self.create_job_record(job_name=job_name, job_id=job_id, scheduled_time=date_time,
                                       project_id=project_id,
                                       type_of_job=is_reoccurring, interval='week', interval_value=days_of_week,
                                       status="waiting", submitted_by=email_address,action_name=action_name
                                       )
            sch_task = ScheduleTask(project_id=int(project_id),
                                    executed_by=email_address,
                                    execution_id=job_id,
                                    socket_io=self.socket_io


                                    )
            if is_reoccurring==self.reoccurring:
                if 'training' in action_name and 'prediction' in action_name:

                    self.add_job(id=job_id, func=sch_task.start_training_prediction_both, trigger='cron', week="*", day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True
                if 'training' in action_name:
                    self.add_job(id=job_id, func=sch_task.start_training, trigger='cron', week="*",
                                 day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True
                if 'prediction' in action_name:
                    self.add_job(id=job_id, func=sch_task.start_prediction, trigger='cron', week="*",
                                 day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True
            else:
                if 'training' in action_name and 'prediction' in action_name:
                    self.add_job(id=job_id, func=sch_task.start_training_prediction_both, trigger='cron',
                                 day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True
                if 'training' in action_name:
                    self.add_job(id=job_id, func=sch_task.start_training, trigger='cron',
                                 day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True
                if 'prediction' in action_name:
                    self.add_job(id=job_id, func=sch_task.start_prediction, trigger='cron',
                                 day_of_week=days_of_week, hour=5,
                                 minute=30)
                    return True



        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.add_job_at_time.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e



    def remove_job_by_id(self, job_id):
        try:
            query = {'job_id': job_id}
            job_detail = self.mongo_db.get_record(self.database_name, self.collection_name, query)

            if job_detail is not None:
                job_detail['status'] = 'cancel'
                self.mongo_db.update_record_in_collection(self.database_name, self.collection_name, query, job_detail)
            self.remove_job(id=job_id)

            return True
        except Exception as e:
            train_model_exception = SchedulerException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Scheduler.__name__,
                            self.remove_job_by_id.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e
