import os
import sys
from datetime import datetime, timedelta
from os import abort
from flask import render_template, redirect, url_for, jsonify, session, request
import threading
import json
import time
from project_library_layer.initializer.initializer import Initializer

from integration_layer.file_management.file_manager import FileManager
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from entity_layer.registration.registration import Register
from logging_layer.logger.log_request import LogRequest
from logging_layer.logger.log_exception import LogExceptionDetail
from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
import json
from entity_layer.scheduler.scheduler import Scheduler
import uuid

global process_value


class SchedulerController:
    def __init__(self):
        self.registration_obj = Register()
        self.WRITE = "WRITE"
        self.READ = "READ"


    def get_scheduler_object(self):
        self.scheduler = Scheduler()
        return self.scheduler

    def scheduler_index(self):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' not in session:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))
            log_writer.executed_by = session['email_address']
            log_writer.log_start(request)
            project_data = Project()
            result = project_data.list_project()
            project_list = None
            if result['status']:
                project_list = result.get('project_list', None)
            result = {'message': None, 'message_status': 'info', 'status': 'True'}
            if project_list is not None:
                result.update({'project_list': project_list})
            sch = Scheduler(socket_io=None)
            job_result = sch.get_all_job()
            job_detail = None

            if job_result['status']:
                job_detail = job_result.get('job_list', None)

            is_job_detail_found = False
            if job_detail is not None:
                is_job_detail_found = True
            result.update({'is_job_detail_found': is_job_detail_found, 'job_detail': job_detail})

            log_writer.log_stop(result)
            return render_template("scheduler_manager.html",
                                   context=result)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            if log_writer is not None:
                log_writer.log_stop({'status': False, 'error_message': str(e)})
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
            return render_template('error.html',
                                   context={'message': None,'status ':False,'message_status': 'info', 'error_message': exception_detail.__str__()})

    def scheduler_ajax_index(self):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' not in session:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))
            log_writer.executed_by = session['email_address']
            log_writer.log_start(request)
            project_data = Project()
            result = project_data.list_project()
            project_list = None
            if result['status']:
                project_list = result.get('project_list', None)
            result = {'message': None, 'message_status': 'info', 'status': 'True'}
            if project_list is not None:
                result.update({'project_list': project_list})
            sch = Scheduler(socket_io=None)
            job_result = sch.get_all_job()
            job_detail = None

            if job_result['status']:
                job_detail = job_result.get('job_list', None)

            is_job_detail_found = False
            if job_detail is not None:
                is_job_detail_found = True
            result.update({'is_job_detail_found': is_job_detail_found, 'job_detail': job_detail})

            log_writer.log_stop(result)
            return render_template("scheduler_manager_ajax.html",
                                   context=result)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            if log_writer is not None:
                log_writer.log_stop({'status': False, 'error_message': str(e)})
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})
    def add_job_at_specific_time(self):
        log_writer = None
        execution_id = str(uuid.uuid4())
        try:
            if 'email_address' in session:
                log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                log_writer.executed_by = session['email_address']
                log_writer.execution_id = execution_id
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                del result
                data = json.loads(request.data)
                project_id = int(data['project_id'])
                job_name = data['job_name']
                print(job_name)
                date_time = data['date_time']
                executed_by = session['email_address']
                action_name = data['action_name'].split(',')[:-1]
                log_writer = LogRequest(executed_by=executed_by, execution_id=execution_id)
                log_writer.log_start(dict(data))

                res = self.scheduler.add_job_at_time(date_time=date_time, job_name=job_name, project_id=project_id,
                                                     email_address=executed_by, action_name=action_name)
                if res:
                    log_writer.log_stop(
                        {'status': True, 'message': "Job <{}> created at <{}>".format(job_name, date_time)})
                    return jsonify({'status': True, 'message': "Job <{}> created at <{}>".format(job_name, date_time)})
                else:
                    return jsonify({'status': False, 'message': 'Failed while creating job'})
            else:
                return jsonify({'status': True, 'message': "Please login to your account"})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)

            if log_writer is not None:
                log_writer.log_stop({'status': False, 'error_message': str(e)})
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            return jsonify({'status': False, 'message': 'Error occurred [{}]'.format(str(exception_detail))})



    def add_job_within_a_day(self):
        log_writer = None
        execution_id = str(uuid.uuid4())
        try:
            if 'email_address' in session:
                log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                log_writer.executed_by = session['email_address']
                log_writer.execution_id = execution_id
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                del result
                data = json.loads(request.data)
                project_id = int(data['project_id'])
                job_name = data['job_name']
                time_type = data['time_type']
                time_value = int(data['time_value'])
                is_reoccurring = data['is_reoccurring']
                executed_by = session['email_address']
                action_name = data['action_name'].split(',')[:-1]
                log_writer = LogRequest(executed_by=executed_by, execution_id=execution_id)
                log_writer.log_start(dict(data))
                if is_reoccurring == 'No':
                    date_time_val = None
                    if time_type == 'hour':
                        date_time_val = datetime.now() + timedelta(hours=time_value)
                    if time_type == 'minute':
                        date_time_val = datetime.now() + timedelta(minutes=time_value)
                    if time_type == 'second':
                        date_time_val = datetime.now() + timedelta(seconds=time_value)
                        if date_time_val is None:
                            raise Exception("Date time required!")
                    date_time_val = str(date_time_val)
                    res = self.scheduler.add_job_at_time(date_time=date_time_val, job_name=job_name,
                                                         project_id=project_id,
                                                         email_address=executed_by,
                                                         action_name=data['action_name'].split(',')[:-1])
                    if res:
                        log_writer.log_stop(
                            {'status': True, 'message': "Job <{}> created at <{}>".format(job_name, date_time_val)})
                        return jsonify(
                            {'status': True, 'message': "Job <{}> created at <{}>".format(job_name, date_time_val)})
                    else:
                        return jsonify({'status': False, 'message': 'Failed while creating job'})

                else:
                    res = False
                    if time_type == 'hour':
                        res = self.scheduler.add_recurring_job_in_hour(time_value, job_name=job_name,
                                                                       project_id=project_id,
                                                                       email_address=executed_by,
                                                                       action_name=data['action_name'].split(',')[:-1])
                    if time_type == 'minute':
                        res = self.scheduler.add_recurring_job_in_minute(time_value, job_name=job_name,
                                                                         project_id=project_id,
                                                                         email_address=executed_by,
                                                                         action_name=data['action_name'].split(',')[
                                                                                     :-1])
                    if time_type == 'second':
                        res = self.scheduler.add_recurring_job_in_second(time_value, job_name=job_name,
                                                                         project_id=project_id,
                                                                         email_address=executed_by,
                                                                         action_name=data['action_name'].split(',')[
                                                                                     :-1])

                    if res:
                        log_writer.log_stop(
                            {'status': True,
                             'message': "Recurring job <{}> created at interval of <{}>".format(job_name, time_value)})
                        return jsonify(
                            {'status': True,
                             'message': "Recurring job <{}> created at  interval of <{}>".format(job_name, time_value)})
                    else:
                        return jsonify({'status': False, 'message': 'Failed while creating job'})
            else:
                return jsonify({'status': True, 'message': "Please login to your account"})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            if log_writer is not None:
                log_writer.log_stop({'status': False, 'error_message': str(exception_detail)})
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            return jsonify({'status': False, 'message': 'Error occurred [{}]'.format(str(exception_detail))})

    def add_job_in_week_day(self):
        print("job_on_week_day")
        log_writer = None
        execution_id = str(uuid.uuid4())
        try:
            if 'email_address' in session:
                log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                log_writer.executed_by = session['email_address']
                log_writer.execution_id = execution_id
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                del result
                data = json.loads(request.data)
                project_id = int(data['project_id'])
                job_name = data['job_name']
                week_day_names = data['week_day_names'][:-1]
                is_reoccurring = data['is_reoccurring']
                executed_by = session['email_address']
                log_writer = LogRequest(executed_by=executed_by, execution_id=execution_id)
                log_writer.log_start(dict(data))

                res = self.scheduler.add_recurring_job_weekly_basis(is_reoccurring=is_reoccurring,
                                                                    days_of_week=week_day_names,
                                                                    job_name=job_name, project_id=project_id,
                                                                    email_address=executed_by,
                                                                    is_record_inserted=False,
                                                                    action_name=data['action_name'].split(',')[:-1]
                                                                    )
                if res:
                    log_writer.log_stop(
                        {'status': True, 'message': "Job <{}> created at for week days <{}>"
                            .format(job_name, week_day_names)})
                    return jsonify({'status': True, 'message': "Job <{}> created at for week days <{}>"
                                   .format(job_name, week_day_names)})
                else:
                    return jsonify({'status': False, 'message': 'Failed while creating job'})
            else:
                return jsonify({'status': True, 'message': "Please login to your account"})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            try:
                if log_writer is not None:
                    log_writer.log_stop({'status': False, 'error_message': str(exception_detail)})
                    log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                    log_exception.log(str(exception_detail))
                return jsonify({'status': False, 'message': 'Error occurred [{}]'.format(str(exception_detail))})

            except Exception as e:
                return jsonify({'status': False, 'message': str(e)})

    def remove_existing_job(self):
        print("job_on_week_day")
        log_writer = None
        execution_id = str(uuid.uuid4())
        try:
            if 'email_address' in session:
                log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                log_writer.executed_by = session['email_address']
                log_writer.execution_id = execution_id
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                del result
                data = json.loads(request.data)
                job_id = data['job_id']
                res=self.scheduler.remove_job_by_id(job_id=job_id)
                if res:
                    result={'status':True,'message':f"Job id:<{job_id}> has been canceled"}
                    log_writer.log_stop(result)
                    return jsonify(result)
                else:
                    return jsonify({'status': False, 'message': 'Failed while creating job'})
            else:
                return jsonify({'status': True, 'message': "Please login to your account"})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            if log_writer is not None:
                log_writer.log_stop({'status': False, 'error_message': str(exception_detail)})
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            return jsonify({'status': False, 'message': 'Error occurred [{}]'.format(str(exception_detail))})