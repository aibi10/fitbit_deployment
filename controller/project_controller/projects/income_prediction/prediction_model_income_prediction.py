

import numpy as np
import pandas
from controller.project_controller.projects.WaferFaultDetection_new.file_operations import file_methods
from controller.project_controller.projects.WaferFaultDetection_new.data_preprocessing import preprocessing
from controller.project_controller.projects.WaferFaultDetection_new.data_ingestion import data_loader_prediction
# from controller.project_controller.projects.WaferFaultDetection_new.application_logging import logger
from controller.project_controller.projects.WaferFaultDetection_new.Prediction_Raw_Data_Validation.predictionDataValidation import \
    PredictionDataValidation
from logging_layer.logger.logger import AppLogger
from project_library_layer.initializer.initializer import Initializer
from integration_layer.file_management.file_manager import FileManager
from exception_layer.generic_exception.generic_exception import GenericException as PredictFromModelException
import sys
import pandas as pd


class Prediction:

    def __init__(self, project_id, executed_by, execution_id, cloud_storage, socket_io=None):
        try:
            # self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
            self.log_writer = AppLogger(project_id=project_id, executed_by=executed_by,
                                        execution_id=execution_id, socket_io=socket_io)
            self.file_object = FileManager(cloud_storage)
            self.initializer = Initializer()
            self.log_writer.log_database = self.initializer.get_prediction_database_name()
            self.log_writer.log_collection_name = self.initializer.get_prediction_main_log_collection_name()
            self.project_id = project_id
            self.socket_io = socket_io
            self.pred_data_val = PredictionDataValidation(project_id=project_id, prediction_file_path=None,
                                                          executed_by=executed_by, execution_id=execution_id,
                                                          cloud_storage=cloud_storage, socket_io=socket_io)
        except Exception as e:
            predict_model_exception = PredictFromModelException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Prediction.__name__,
                            self.__init__.__name__))
            raise Exception(predict_model_exception.error_message_detail(str(e), sys)) from e

    def prediction_from_model(self):

        try:
            self.pred_data_val.delete_prediction_file()  # deletes the existing prediction file from last run!
            self.log_writer.log('Start of Prediction')
            data_getter = data_loader_prediction.DataGetterPrediction(project_id=self.project_id,
                                                                      file_object=self.file_object,
                                                                      logger_object=self.log_writer)
            data = data_getter.get_data()

            if not isinstance(data, pandas.DataFrame):
                raise Exception("prediction data not loaded successfully into pandas data frame.")


            preprocessor = preprocessing.Preprocessor(file_object=self.file_object, logger_object=self.log_writer,
                                                      project_id=self.project_id)

            data=preprocessor.remove_columns(data, ['education'])
            data=preprocessor.remove_unwanted_spaces(data)
            data.replace('?',np.NAN,inplace=True)
            is_null_present,cols_with_missing_values = preprocessor.is_null_present_in_columns(data)
            if is_null_present:
                data=preprocessor.impute_missing_values_mushroom(data,cols_with_missing_values)

            scaled_num_df = preprocessor.scale_numerical_columns_credit_default_or_income_prediction(data)

            cat_df=preprocessor.encode_categorical_columns_income_prediction(data)

            X = pd.concat([scaled_num_df, cat_df], axis=1)
            file_loader = file_methods.FileOperation(project_id=self.project_id, file_object=self.file_object,
                                                     logger_object=self.log_writer)

            kmean_folder_name = self.initializer.get_kmean_folder_name()
            kmeans = file_loader.load_model(kmean_folder_name)

            clusters = kmeans.predict(X)
            X['clusters'] = clusters
            clusters = X['clusters'].unique()
            prediction_file_path = self.initializer.get_prediction_output_file_path(self.project_id)
            prediction_file_name = self.initializer.get_prediction_output_file_name()
            predictions = []
            for i in clusters:
                cluster_data = X[X['clusters'] == i]
                # wafer_names = list(cluster_data['Wafer'])
                cluster_data = cluster_data.drop(['clusters'], axis=1)
                model_name = file_loader.find_correct_model_file(str(i))
                model = file_loader.load_model(model_name)
                result = (model.predict(cluster_data))
                for res in result:
                    if res == 0:
                        predictions.append('<=50K')
                    else:
                        predictions.append('>50K')

            final= pd.DataFrame(list(zip(predictions)),columns=['Predictions'])
            final.reset_index(drop=True, inplace=True)
            self.file_object.write_file_content(prediction_file_path, prediction_file_name, final, over_write=True)
            self.log_writer.log('End of Prediction')
            return "{}/{}".format(prediction_file_path, prediction_file_name)
        except Exception as e:
            predict_model_exception = PredictFromModelException(
                "Failed during prediction in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Prediction.__name__,
                            self.prediction_from_model.__name__))
            raise Exception(predict_model_exception.error_message_detail(str(e), sys)) from e
