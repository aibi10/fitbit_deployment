import os
import sys
import asyncio
from flask import render_template, redirect, url_for, jsonify, session
from flask import request
from werkzeug.utils import secure_filename
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import json, uuid
from integration_layer.file_management.file_manager import FileManager
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from entity_layer.registration.registration import Register
from logging_layer.logger.log_request import LogRequest
from logging_layer.logger.log_exception import LogExceptionDetail
#from thread_layer.watcher_thread.watcher_thread import WatcherThread
global process_value
from entity_layer.watcher.watcher import start_call
from threading import Thread

class WatcherController:
    def __init__(self):
        self.registration_obj = Register()
        self.WRITE = "WRITE"
        self.READ = "READ"

        self.watcher_database_name = "watcher_db"
        self.watcher_collection_name = "watcher_events"
        self.mongo_db=MongoDBOperation()
        self._thread = Thread(target=start_call)
        #self.watcher_thread=WatcherThread()

    def display_captured_event(self):
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                log_writer.executed_by = session['email_address']
                log_writer.log_start(request)
                result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)
                if not result['status']:
                    log_writer.log_stop(result)
                    return jsonify(result)
                del result
                thread_msg=""
                if not self._thread.is_alive():
                    thread_msg="Event started"
                    self._thread.start()
                df=self.mongo_db.get_dataframe_of_collection(self.watcher_database_name,self.watcher_collection_name)
                context={'status':True,'watcher_table':df.to_html(header="true"),'message':'{} watcher detail retrived'.format(thread_msg),}
                log_writer.log_stop(context)
                return render_template('watcher_detail.html',context=context)
            else:
                log_writer.log_start(request)
                result = {'status': True, 'message': 'Please login to your account'}
                log_writer.log_stop(result)
                return jsonify(result)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            return render_template('error.html',
                                   context={'message': None,'status ':False,'message_status': 'info', 'error_message': exception_detail.__str__()})