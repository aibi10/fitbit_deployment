import uuid

from sklearn.metrics import roc_curve, roc_auc_score

from exception_layer.generic_exception.generic_exception import GenericException as  PlotlyDashException
from project_library_layer.initializer.initializer import Initializer
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.datetime_libray.date_time import get_time, get_date
import sys
import plotly.figure_factory as ff
import json
import pandas as pd
import plotly
import plotly.graph_objs as go
import random

class AccurayGraph:

    def __init__(self, project_id=None, model_accuracy_dict: dict = None):
        try:
            self.project_id = project_id
            self.model_accuray = model_accuracy_dict
            self.initializer = Initializer()
            self.mongo_db = MongoDBOperation()
            self.accuracy_score_database_name = self.initializer.get_accuracy_metric_database_name()
            self.accuracy_score_collection_name = self.initializer.get_accuracy_metric_collection_name()
            self.colors = ['slategray', 'aquamarine', 'darkturquoise', 'deepskyblue', 'orange',  'green',   'purple']
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            "__init__"))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def get_random_color_name(self):
        """
        :return: Name of a color
        """
        try:
            n_colors=len(self.colors)
            index=random.randint(0,n_colors-1)
            return self.colors[index]
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.get_random_color_name.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_accuracy(self):
        try:
            self.model_accuray.update(
                {'project_id': self.project_id, 'stored_date': get_date(), 'stored_time': get_time()})
            is_inserted = self.mongo_db.insert_record_in_collection(self.accuracy_score_database_name,
                                                                    self.accuracy_score_collection_name,
                                                                    self.model_accuray)
            if is_inserted > 0:
                return {'status': True, 'message': 'Model accuracy stored '}
            else:
                return {'status': False, 'message': 'Model accuracy failed to store'}
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_accuracy.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def get_accuray_score_of_trained_model(self, project_id):
        try:
            records = self.mongo_db.get_records(self.accuracy_score_database_name, self.accuracy_score_collection_name,
                                                {'project_id': project_id})
            if records is not None:
                return {'status': True, 'message': 'accuracy record found', 'accuracy_data': records}
            else:
                return {'status': False, 'message': 'accuracy record not found'}
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.get_accuray_score_of_trained_model.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def get_training_execution_id_with_project_id(self):
        try:
            response = {'status': False, 'message': "We don't have project with execution id"}
            df = self.mongo_db.get_dataframe_of_collection(self.accuracy_score_database_name,
                                                           self.accuracy_score_collection_name)
            if df is None:
                return response
            training_execution_with_project_id = df[['project_id', 'training_execution_id']].copy()
            training_execution_with_project_id.drop_duplicates(inplace=True)
            training_execution_with_project_id_list = list(training_execution_with_project_id.T.to_dict().values())
            if len(training_execution_with_project_id_list) > 0:
                return {'status': True, 'message': 'We have project with execution id',
                        'training_execution_with_project_id_list': training_execution_with_project_id_list}
            else:
                return response

        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.get_training_execution_id_with_project_id.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_accuracy_bar_graph(self, model_name_list, accuracy_score_list, project_id, execution_id, file_object,
                                title=None,x_label=None,y_label=None):
        """
        :param model_name_list: model_name_list
        :param accuracy_score_list: accuracy_score_list
        :return: graph_json
        """
        try:
            x_label='Model Name' if x_label is None else x_label
            y_label = 'Score' if x_label is None else y_label

            if len(model_name_list) != len(accuracy_score_list):
                return False
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=model_name_list,  # assign x as the dataframe column 'x'
                y=accuracy_score_list,
                marker_color=self.get_random_color_name()
            )
            )
            fig.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                title={'text': title}
            )

            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)

        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_accuracy_bar_graph.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_roc_curve_plot_binary_classification(self, fpr, tpr, project_id=None, execution_id=None, file_object=None,
                                                  title=None):
        """

        :param fpr: False +ve rate
        :param tpr: True +ve rate
        :param project_id: project id
        :param execution_id: execution id
        :param file_object: file object
        :param title: title
        :return: nothing
        """
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy',fillcolor=self.get_random_color_name()))  # fill down to xaxis
            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title={'text': title}
            )
            json_graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=json_graph)

        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_roc_curve_plot_binary_classification.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_plot_multiclass_roc_curve(self,y, y_scores, model, project_id=None, execution_id=None, file_object=None,
                                                  title=None):
        """

        :param y: truth value of y
        :param y_scores: predict proba score
        :param model: trained model
        :param project_id: project id
        :param execution_id: execution id
        :param file_object: file object
        :param title: title of graph
        :return: nothing
        """
        try:
            y_onehot = pd.get_dummies(y, columns=model.classes_)

            # Create an empty figure, and iteratively add new lines
            # every time we compute a new class
            fig = go.Figure()
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )

            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                title={'text': title}


            )
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_plot_multiclass_roc_curve.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e


    def save_scatter_plot(self, x_axis_data, y_axis_data, project_id, execution_id, file_object, x_label=None,
                          y_label=None, title=None):
        """

        :param x_axis_data: X axis data
        :param y_axis_data: Y axis data
        :param project_id: project id
        :param execution_id: execution_id
        :param file_object: file object
        :param x_label: x label name
        :param y_label:  ylabel name
        :return: nothing
        """
        try:
            x_axis_label = x_label if x_label is not None else 'X axis'
            y_axis_label = y_label if y_label is not None else 'Y axis'
            if len(x_axis_data) != len(y_axis_data):
                return False
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_axis_data,  # assign x as the dataframe column 'x'
                y=y_axis_data,
                fillcolor=self.get_random_color_name(),
                mode='markers'

            )
            )
            fig.update_layout(
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                title={'text': title}
            )

            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)

        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_scatter_plot.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e


    def save_line_plot(self, x_axis_data, y_axis_data, project_id, execution_id, file_object, x_label=None,
                       y_label=None, title=None):
        """

        :param x_axis_data: X axis data
        :param y_axis_data: Y axis data
        :param project_id: project id
        :param execution_id: execution_id
        :param file_object: file object
        :param x_label: x label name
        :param y_label:  ylabel name
        :return: nothing
        """
        try:
            x_axis_label = x_label if x_label is not None else 'X axis'
            y_axis_label = y_label if y_label is not None else 'Y axis'
            if len(x_axis_data) != len(y_axis_data):
                return False
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_axis_data,  # assign x as the dataframe column 'x'
                y=y_axis_data,
                mode='lines+markers',
                fillcolor=self.get_random_color_name()
            )
            )
            fig.update_layout(
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                title={'text': title}
            )

            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)

        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed to instantiate mongo_db_object in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_line_plot.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_distribution_plot(self, data,label, project_id, execution_id, file_object, x_label=None,
                               y_label=None, title=None):
        """

        :param data: data kind of array
        :param label: list of label
        :param project_id: project id
        :param execution_id: execution id
        :param file_object: file object
        :param x_label: x label
        :param y_label: y label
        :param title: title
        :return: nothing
        """
        try:
            fig = ff.create_distplot([data], group_labels=[label], bin_size=.5,
                                     curve_type='normal',  # override default 'kde'
                                    )
            fig.update_layout(title_text=title)
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_distribution_plot.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def save_pie_plot(self, data, label, project_id, execution_id, file_object, title=None):
        """

        :param data: data
        :param label: label
        :param project_id: project id
        :param execution_id:  execution id
        :param file_object: file object
        :param title: title
        :return: nothing
        """
        try:
            fig = go.Figure(data=[go.Pie(labels=label, values=data)])
            fig.update_layout(title_text=title)
            graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            file_object.write_file_content(
                directory_full_path=self.initializer.get_project_report_graph_file_path(project_id, execution_id),
                file_name=str(uuid.uuid4()) + '.graph',
                content=graph_json)
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.save_pie_plot.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e

    def get_training_execution_id_of_project(self, project_id):
        """
        :param project_id: accpet project id
        :return: return {'status':True/False,'training_execution_id_list':training_execution_id_list }
        """
        try:
            df = self.mongo_db.get_dataframe_of_collection(self.accuracy_score_database_name,
                                                           self.accuracy_score_collection_name)
            if df is not None:
                df = df[df['project_id'] == project_id]
                if df.shape[0] == 0:
                    return {'status': False}
                training_execution_id_list = list(df['training_execution_id'].unique())
                if len(training_execution_id_list) > 0:
                    return {'status': True, 'training_execution_id_list': training_execution_id_list}
                else:
                    return {'status': False}
            else:
                return {'status': False}
        except Exception as e:
            plotly_dash = PlotlyDashException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(AccurayGraph.__module__.__str__(), AccurayGraph.__name__,
                            self.get_training_execution_id_of_project.__name__))
            raise Exception(plotly_dash.error_message_detail(str(e), sys)) from e
