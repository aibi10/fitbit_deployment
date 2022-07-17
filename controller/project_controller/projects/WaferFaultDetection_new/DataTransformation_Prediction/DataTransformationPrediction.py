from datetime import datetime
from os import listdir
import pandas
from logging_layer.logger.logger import AppLogger
from project_library_layer.initializer.initializer import Initializer
from integration_layer.file_management.file_manager import FileManager
from exception_layer.generic_exception.generic_exception import GenericException as \
    DataTransformationPredictionException
import sys


class DataTransformPrediction:
    """
                  This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

                  Written By: iNeuron Intelligence
                  Version: 1.0
                  Revisions: None

                  """
    """
     def __init__(self):
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger()
     """

    def __init__(self, project_id, executed_by, execution_id, cloud_storage, socket_io=None):
        try:
            self.initializer = Initializer()
            self.logger = AppLogger(project_id=project_id, executed_by=executed_by,
                                    execution_id=execution_id, socket_io=socket_io)
            self.goodDataPath = self.initializer.get_prediction_good_raw_data_file_path(project_id)
            self.project_id = project_id
            self.logger.log_database = self.initializer.get_prediction_database_name()
            self.file_manager = FileManager(cloud_provider=cloud_storage)
        except Exception as e:
            data_transformation_exception = DataTransformationPredictionException(
                "Failed during instantiation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataTransformPrediction.__name__,
                            "__init__"))
            raise Exception(data_transformation_exception.error_message_detail(str(e), sys)) from e

    def replace_missing_with_null_back_order(self):
        """
                                             Method Name: replaceMissingWithNull
                                             Description: This method replaces the missing values in columns with "NULL" to
                                                          store in the table. We are using substring in the first column to
                                                          keep only "Integer" data for ease up the loading.
                                                          This column is anyways going to be removed during training.

                                              Written By: iNeuron Intelligence
                                             Version: 1.0
                                             Revisions: None

                                                     """

        # log_file = open("Training_Logs/dataTransformLog.txt", 'a+')
        try:
            # les = [f for f in listdir(self.goodDataPath)]
            self.logger.log_collection_name = self.initializer.get_data_transform_log_collection_name()
            response = self.file_manager.list_files(self.goodDataPath)
            if not response['status']:
                return True

            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                self.logger.log(response['message'])
                return True

            for file in files:
                # csv = pandas.read_csv(self.goodDataPath + "/" + file)
                response = self.file_manager.read_file_content(self.goodDataPath, file)
                if not response['status']:
                    continue
                csv = response['file_content']
                if not isinstance(csv, pandas.DataFrame):
                    continue
                csv.fillna('NULL', inplace=True)
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))

                csv.reset_index(drop=True, inplace=True)
                # csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.file_manager.write_file_content(self.goodDataPath, file, csv, over_write=True)
                self.logger.log(" %s: File Transformed successfully!!" % file)
            # log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")
        except Exception as e:
            data_transformation_exception = DataTransformationPredictionException(
                "Failed during data transformation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataTransformPrediction.__name__,
                            self.replace_missing_with_null_back_order.__name__))

    def replace_missing_with_null(self):

        """
                                  Method Name: replaceMissingWithNull
                                  Description: This method replaces the missing values in columns with "NULL" to
                                               store in the table. We are using substring in the first column to
                                               keep only "Integer" data for ease up the loading.
                                               This column is anyways going to be removed during prediction.

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                                          """
        """  
        try:
            log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath + "/" + file)
                csv.fillna('NULL', inplace=True)
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))
                csv['Wafer'] = csv['Wafer'].str[6:]
                csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(log_file, " %s: File Transformed successfully!!" % file)
            # log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")

        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            # log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
            log_file.close()
            raise e
        log_file.close()
        """
        try:
            # les = [f for f in listdir(self.goodDataPath)]
            self.logger.log_collection_name = self.initializer.get_data_transform_log_collection_name()
            response = self.file_manager.list_files(self.goodDataPath)
            if not response['status']:
                return True

            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                self.logger.log(response['message'])
                return True

            for file in files:
                # csv = pandas.read_csv(self.goodDataPath + "/" + file)
                response = self.file_manager.read_file_content(self.goodDataPath, file)
                if not response['status']:
                    continue
                csv = response['file_content']
                if not isinstance(csv, pandas.DataFrame):
                    continue
                csv.fillna('NULL', inplace=True)
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))
                csv['Wafer'] = csv['Wafer'].str[6:]
                csv.reset_index(drop=True, inplace=True)
                # csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.file_manager.write_file_content(self.goodDataPath, file, csv, over_write=True)
                self.logger.log(" %s: File Transformed successfully!!" % file)
            # log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")
        except Exception as e:
            data_transformation_exception = DataTransformationPredictionException(
                "Failed during data transformation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataTransformPrediction.__name__,
                            self.replace_missing_with_null.__name__))
            raise Exception(data_transformation_exception.error_message_detail(str(e), sys)) from e

    def add_quotes_to_string_values_in_column(self):
        """
         Method Name: addQuotesToStringValuesInColumn
         Description: This method converts all the columns with string datatype such that
                     each value for that column is enclosed in quotes. This is done
                     to avoid the error while inserting string values in table as varchar.

          Written By: iNeuron Intelligence
         Version: 1.0
         Revisions: None

                 """

        #log_file = open("Training_Logs/addQuotesToStringValuesInColumn.txt", 'a+')
        try:
            self.logger.log_collection_name=self.initializer.get_add_quotes_to_string_values_in_column_collection_name()
            self.logger.log("converts all the columns with string datatype such that each value"
                            " for that column is enclosed in quotes")
            response = self.file_manager.list_files(self.goodDataPath)
            if not response['status']:
                return True

            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                self.logger.log(response['message'])
                return True

            #onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in files:
                response = self.file_manager.read_file_content(self.goodDataPath,file)
                if not response['status']:
                    continue
                data = response['file_content']
                if not isinstance(data, pandas.DataFrame):
                    continue
                # list of columns with string datatype variables
                column = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick',
                              'pregnant',
                              'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                              'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'T3_measured',
                              'TT4_measured',
                              'T4U_measured', 'FTI_measured', 'TBG_measured', 'TBG', 'referral_source', 'Class']

                for col in data.columns:
                    if col in column:  # add quotes in string value
                        data[col] = data[col].apply(lambda x: "'" + str(x) + "'")
                    if col not in column:  # add quotes to '?' values in integer/float columns
                        data[col] = data[col].replace('?', "'?'")
                # #csv.update("'"+ csv['Wafer'] +"'")
                # csv.update(csv['Wafer'].astype(str))
                # csv['Wafer'] = csv['Wafer'].str[6:]
                data.reset_index(drop=True,inplace=True)
                self.file_manager.write_file_content(self.goodDataPath,file,data,over_write=True)
                #data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(" %s: Quotes added successfully!!" % file)
            # log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")
        except Exception as e:
            data_transformation_exception = DataTransformationPredictionException(
                "Failed during add_quotes_to_string_values_in_column of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataTransformPrediction.__name__,
                            self.add_quotes_to_string_values_in_column.__name__))
            raise Exception(data_transformation_exception.error_message_detail(str(e), sys)) from e

