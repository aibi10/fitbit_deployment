import os
import sys
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
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import json

import uuid

global process_value
import json


class ProjectController:
    def __init__(self):
        self.registration_obj = Register()
        self.WRITE = "WRITE"
        self.READ = "READ"

    def projects(self):
        log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
        try:

            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                context = {'message': None, 'message_status': 'info'}
                log_writer.log_stop(context)
                return render_template('project.html', context=context)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))
        except Exception as e:
            log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
            log_exception.log(str(e))
            context = {'status': False, 'message': str(e)}
            return jsonify(context)

    def project_detail(self):
        log_writer = None

        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                project_id = request.args.get('project_id')
                log_writer.executed_by = session['email_address']
                project_data = Project()
                project_data = project_data.get_project_detail(int(project_id))
                log_writer.log_start(request)
                database_name = Initializer().get_training_thread_database_name()
                collection_name = Initializer().get_thread_status_collection_name()
                project_id=int(project_id)
                query = {'project_id': project_id, 'is_running': True}
                result = MongoDBOperation().get_record(database_name=database_name, collection_name=collection_name,
                                                       query=query)
                query.pop('is_running')
                execution_records=MongoDBOperation().get_records(database_name=database_name, collection_name=collection_name,
                                                       query=query)
                if result is not None:
                    execution_id = result['execution_id']
                else:
                    execution_id = None
                execution_details_of_projects=[]
                for exec_record in execution_records:
                    execution_details_of_projects.append(exec_record)



                context = {'message': None, 'message_status': 'info', 'project_id': project_id,
                           'project_data': project_data,
                           'execution_id': execution_id,'execution_records':execution_details_of_projects}
                log_writer.log_stop(context)
                return render_template('project_operation_ajax.html', context=context)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))

        except Exception as e:
            if log_writer is not None:
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
            context = {'status': False, 'message': str(e)}

            return jsonify(context)

    def project_list(self):
        log_writer = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                project = Project()
                result = project.list_project()
                return render_template('project.html', context=result)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))

        except Exception as e:
            #context = {'status': False, 'message': str(e)}
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            if log_writer is not None:
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(f"{exception_detail}")
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': f"{exception_detail}"})

    def save_project_data(self):
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                project_data = json.loads(request.data)
                msg = ""
                project_name = None
                if 'project_name' in project_data:
                    project_name = project_data['project_name']
                else:
                    msg = msg + " Project name not found in request"
                if project_name is None:
                    msg = msg + " Project name required"
                project_description = None
                if 'project_description' in project_data:
                    project_description = project_data['project_description']
                else:
                    msg = msg + " Project description not found in request"
                if project_description is None:
                    msg = msg + " Project description required"
                if len(msg) > 0:
                    response = {'status': True, 'message': msg}
                    log_writer.log_stop(response)
                    return jsonify(response)
                project = Project(project_name=project_name, project_description=project_description)
                response = project.save_project()
                log_writer.log_stop(response)
                return jsonify(response)

            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))
        except Exception as e:
            log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
            log_exception.log(str(e))
            context = {'status': False, 'message': str(e)}
            return jsonify(context)

    def save_project_configuration(self):
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                project_data = json.loads(request.data)
                msg = ""
                project_id = int(project_data.get('project_id', None))
                machine_learning_type = project_data.get('machine_learning_type', None)
                file_name_pattern = project_data.get('file_name_pattern', None)
                cloud_storage = project_data.get('cloud_storage', None)
                training_schema_definition_json = project_data.get('schema_definition_json', None)
                prediction_schema_definition_json = project_data.get('prediction_definition_json', None)
                if project_id is None:
                    msg = msg + " Project id required"
                if cloud_storage is None:
                    msg = msg + " cloud_storage id required"
                if machine_learning_type is None:
                    msg = msg + " machine_learning_type id required"
                if file_name_pattern is None:
                    msg = msg + " file_name_pattern  required"
                if training_schema_definition_json is None:
                    msg = msg + " schema_definition_json  required"
                if prediction_schema_definition_json is None:
                    msg = msg + " prediction_definition_json required"

                if len(msg) > 0:
                    response = {'status': True, 'message': msg}
                    log_writer.log_stop(response)
                    return jsonify(response)
                training_schema_definition_json = training_schema_definition_json.replace("\'", "\"")
                training_schema_definition_json = dict(json.loads(training_schema_definition_json))
                prediction_schema_definition_json = prediction_schema_definition_json.replace("\'", "\"")
                prediction_schema_definition_json = dict(json.loads(prediction_schema_definition_json))
                project_config = ProjectConfiguration(project_id=project_id,
                                                      machine_learning_type=machine_learning_type,
                                                      training_schema_definition_json=training_schema_definition_json,
                                                      prediction_schema_definition_json=prediction_schema_definition_json,
                                                      file_name_pattern=file_name_pattern,
                                                      cloud_storage=cloud_storage
                                                      )
                response = project_config.save_project_configuration()
                log_writer.log_stop(response)
                return jsonify(response)

            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))
        except Exception as e:
            log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
            log_exception.log(str(e))
            context = {'status': False, 'message': str(e)}
            return jsonify(context)
