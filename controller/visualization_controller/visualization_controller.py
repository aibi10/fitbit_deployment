import os
import sys
import uuid
from os import abort
from flask import render_template, redirect, url_for, jsonify, session, request
import threading
import json
import time

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from project_library_layer.initializer.initializer import Initializer

from integration_layer.file_management.file_manager import FileManager
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from entity_layer.registration.registration import Register
from logging_layer.logger.log_request import LogRequest
from logging_layer.logger.log_exception import LogExceptionDetail
from entity_layer.project.project import Project
from entity_layer.project.project_configuration import ProjectConfiguration
from plotly_dash.accuracy_graph.accuracy_graph import AccurayGraph
from integration_layer.file_management.file_manager import FileManager
from project_library_layer.initializer.initializer import Initializer
import plotly
import plotly.express as px
import plotly.graph_objs as go
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import pandas as pd
import numpy as np
import json
from plotly import subplots


class VisualizationController:

    def __init__(self):
        self.registration_obj = Register()
        self.WRITE = "WRITE"
        self.READ = "READ"
        self.initializer = Initializer()
        self.mongo_db = MongoDBOperation()

    def get_failed_success_running_count(self):
        try:
            database_name=self.initializer.get_training_thread_database_name()
            collection_name=self.initializer.get_thread_status_collection_name()

            df=self.mongo_db.get_dataframe_of_collection(database_name,collection_name=collection_name)

            res = df['is_Failed'].value_counts()
            label=list(res.index)
            counts=list(res.values)
            marker_colors_value=[]
            if label[0]:
                marker_colors_value.append('rgb(0, 230, 0)')
                marker_colors_value.append('rgb(255, 153, 0)')
            else:
                marker_colors_value.append('rgb(255, 153, 0)')
                marker_colors_value.append('rgb(0, 230, 0)')



            fig = go.Figure(data=[go.Pie(labels=label,
                                         values=counts,
                                         hole=.3,
                                         marker_colors=marker_colors_value
                                         )])
            fig.update_layout(title_text="Project status on Failure vs Success ")
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return graph_json

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            raise Exception(f"{exception_detail}")

    def running_projects(self):
        try:
            database_name=self.initializer.get_training_thread_database_name()
            collection_name=self.initializer.get_thread_status_collection_name()
            query={'is_running':True}
            df=self.mongo_db.get_dataframe_of_collection(database_name,collection_name=collection_name,query=query)
            return df
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            raise Exception(f"{exception_detail}")

    def performance_graph(self, feature=None):
        try:

            df = self.mongo_db.get_dataframe_of_collection("log_request", "requests")
            if df is None:
                fig = go.Figure()
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            if 'execution_time_milisecond' not in df.columns:
                fig = go.Figure()
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            if df.shape[0] > 0:

                df = df[['execution_time_milisecond', 'log_start_date', 'log_start_time']]
                start_index = df.shape[0] - 50 if df.shape[0] > 50 else 0
                df = df.iloc[start_index:, :]
                print(start_index)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['log_start_date'] + '_' + df['log_start_time'],  # assign x as the dataframe column 'x'
                    y=df['execution_time_milisecond'],
                    mode='lines+markers',
                )
                )
                fig.update_layout(
                    xaxis_title='Request time',
                    yaxis_title='Request process time in millisecond',
                    title={'text': "Performace of website"}
                )
            else:
                fig = go.Figure()
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            raise Exception(exception_detail.__str__())

    def success_vs_failure(self, feature=None):
        try:
            df = self.mongo_db.get_dataframe_of_collection("log_request", "requests")
            if df.shape[0]==0:
                fig = go.Figure()
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            df_exception=self.mongo_db.get_dataframe_of_collection("exception_log","exception_collection")
            if df_exception.shape[0]!=0:

                fail_record=pd.merge(df,df_exception,on='execution_id')
                fail_count=fail_record.shape[0]
            else:
                fail_count=0
            success_count=df.shape[0]-fail_count
            if df is None:
                fig = go.Figure()
                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            if df.shape[0] > 0:
                fig = go.Figure(data=[go.Pie(labels=["Success", "Failed"],
                                             values=[success_count,fail_count],
                                             hole=.3,
                                             marker_colors=['rgb(0, 230, 0)', 'rgb(255, 153, 0)']
                                             )])
                fig.update_layout(title_text="Request and response failure and success rate")
                graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graph_json
            else:
                fig = go.Figure()
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            raise Exception(exception_detail.__str__())



    def dashboard(self):
        try:
            perfor_graph = self.performance_graph()
            success_vs_failure = self.success_vs_failure()
            running_project_df=self.running_projects()
            project_success_graph=self.get_failed_success_running_count()

            context = {'message': None, 'running_project_table':running_project_df.to_html(header=True),'message_status': 'info', 'project_success_graph':project_success_graph,'plot': perfor_graph, 'success_graph': success_vs_failure}
            return render_template('dashboard.html', context=context)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})


    def visualization_project_list(self):
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
                return render_template('report.html', context=result)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            if log_writer is not None:
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            #context = {'status': False, 'message': str(e)}
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})

    def report_detail(self):
        log_writer = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                project_id = request.args.get('project_id')
                log_writer.executed_by = session['email_address']
                project_data = Project()
                project_id = int(project_id)
                project_data = project_data.get_project_detail(project_id=project_id)
                project_config_obj = ProjectConfiguration()
                project_config_detail = project_config_obj.get_project_configuration_detail(project_id=project_id)
                if not project_config_detail['status']:
                    project_config_detail.update(
                        {'message_status': 'info', 'project_id': project_id})

                if 'project_config_detail' in project_config_detail:
                    project_config_detail = project_config_detail['project_config_detail']
                if project_config_detail is None:
                    pass
                cloud_storage = None
                if 'cloud_storage' in project_config_detail:
                    cloud_storage = project_config_detail['cloud_storage']
                file_manager = FileManager(cloud_storage)
                result = file_manager.list_directory(
                    self.initializer.get_project_report_graph_path(project_id=project_id))

                context = {'message': None, 'message_status': 'info', 'project_id': project_id,
                           'project_data': project_data}
                if result['status']:
                    folder_list = result.get('directory_list', None)
                    if folder_list is not None:
                        if 'initial.txt.dat' in folder_list:
                            folder_list.remove('initial.txt.dat')
                            result['directory_list'] = folder_list
                    context.update(result)
                log_writer.log_start(request)
                log_writer.log_stop(context)
                return render_template('report_detail.html', context=context)
            else:
                log_writer.log_start(request)
                log_writer.log_stop({'navigating': 'login'})
                return redirect(url_for('login'))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            if log_writer is not None:
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            #context = {'status': False, 'message': str(e)}
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})

    def display_graph(self):
        log_writer = None

        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            if 'email_address' in session:
                project_id = int(request.args.get('project_id'))
                execution_id = request.args.get('dir_name')
                project_data = Project()
                graph_file_path = self.initializer.get_project_report_graph_file_path(project_id=project_id,
                                                                                      execution_id=execution_id)
                project_data = project_data.get_project_detail(project_id=project_id)
                project_config_obj = ProjectConfiguration()
                project_config_detail = project_config_obj.get_project_configuration_detail(project_id=project_id)
                if not project_config_detail['status']:
                    project_config_detail.update(
                        {'message_status': 'info', 'project_id': project_id})

                if 'project_config_detail' in project_config_detail:
                    project_config_detail = project_config_detail['project_config_detail']
                if project_config_detail is None:
                    pass
                cloud_storage = None
                if 'cloud_storage' in project_config_detail:
                    cloud_storage = project_config_detail['cloud_storage']
                file_manager = FileManager(cloud_storage)
                list_graph_file = file_manager.list_files(graph_file_path)
                graphs = {}
                graph_name = []
                if list_graph_file['status']:
                    files = list_graph_file.get('files_list', None)
                    graph_number = 0
                    for file in files:
                        if file in 'initial.txt.data':
                            continue
                        data = file_manager.read_file_content(graph_file_path, file)
                        graph_data = data.get('file_content', None)
                        graphs.update({'graph{}'.format(graph_number): graph_data})
                        graph_name.append('graph{}'.format(graph_number))
                        graph_number = graph_number + 1

                result = {'project_id': project_id, 'execution_id': execution_id}
                result.update(graphs)
                result.update({'graph_name': graph_name})

                return render_template('graph.html', context=result)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            if log_writer is not None:
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(exception_detail))
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})
