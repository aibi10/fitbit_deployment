from datetime import datetime
from controller.project_controller.projects.WaferFaultDetection_new.Prediction_Raw_Data_Validation.predictionDataValidation import PredictionDataValidation
from controller.project_controller.projects.WaferFaultDetection_new.DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import DbOperationMongoDB
from controller.project_controller.projects.WaferFaultDetection_new.DataTransformation_Prediction.DataTransformationPrediction import DataTransformPrediction

from logging_layer.logger.logger import AppLogger
from project_library_layer.initializer.initializer import Initializer
from exception_layer.generic_exception.generic_exception import GenericException as PredictionValidationException
import sys

class PredictionValidation:
    def __init__(self,project_id,prediction_file_path,  executed_by,execution_id,cloud_storage,socket_io=None):
        try:
            self.project_id=project_id
            self.raw_data = PredictionDataValidation(project_id, prediction_file_path,
                                                     executed_by, execution_id, cloud_storage, socket_io=socket_io)
            self.dataTransform = DataTransformPrediction(project_id, executed_by, execution_id, cloud_storage, socket_io=socket_io)
            self.dBOperation = DbOperationMongoDB(project_id,executed_by,execution_id,cloud_storage,socket_io=socket_io)
            self.initializer = Initializer()
            #self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
            self.log_writer = AppLogger(project_id=project_id, executed_by=executed_by, execution_id=execution_id,
                                        socket_io=socket_io)
            #self.log_writer = logger.App_Logger()
            self.log_writer.log_database = self.initializer.get_prediction_database_name()
            self.socket_io = socket_io
        except Exception as e:
            prediction_validation_exception = PredictionValidationException(
                "Failed during instantiation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionValidation.__name__,
                            "__init__"))
            raise Exception(prediction_validation_exception.error_message_detail(str(e), sys)) from e

    def prediction_validation(self):

        try:
            self.log_writer.log_collection_name = self.initializer.get_prediction_main_log_collection_name()
            self.log_writer.log('Start of Validation on files for prediction!!')
            #extracting values from prediction schema
            length_of_date_stamp_in_file,length_of_time_stamp_in_file,column_names,no_of_columns = self.raw_data.values_from_schema()
            #getting the regex defined to validate filename
            regex = self.raw_data.manual_regex_creation()
            #validating filename of prediction files
            self.raw_data.validation_file_name_raw_start_with_index_two(regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file)
            #validating column length in the file
            self.raw_data.validate_column_length(no_of_columns)
            #validating if any column has all values missing
            self.raw_data.validate_missing_values_in_whole_column()
            self.log_writer.log("Raw Data Validation Complete!!")

            self.log_writer.log(("Starting Data Transforamtion!!"))
            #replacing blanks in the csv file with "Null" values to insert in table

                #self.dataTransform.add_quotes_to_string_values_in_column()

            self.log_writer.log("DataTransformation Completed!!!")

            self.log_writer.log("Creating Prediction_Database and tables on the basis of given schema!!!")
            #create database with given name, if present open the connection! Create table with columns given in schema
            #self.dBOperation.createTableDb('Prediction',column_names)
            #self.log_writer.log("Table creation Completed!!")
            #self.log_writer.log("Insertion of Data into Table started!!!!")
            #insert csv files in the table

            self.log_writer.log("Creating database and collection if not exist then create and insert record")
            self.dBOperation.insert_into_table_good_data(column_names)
            self.log_writer.log("Insertion in Table completed!!!")
            self.log_writer.log("Deleting Good Data Folder!!!")
            #Delete the good data folder after loading files in table
            self.raw_data.delete_existing_good_data_prediction_folder()
            self.log_writer.log("Good_Data folder deleted!!!")
            self.log_writer.log("Moving bad files to Archive and deleting Bad_Data folder!!!")
            #Move the bad files to archive folder
            self.raw_data.move_bad_files_to_archive_bad()
            self.log_writer.log("Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log("Validation Operation completed!!")
            self.log_writer.log("Extracting csv file from table")
            #export data in table to csvfile
            self.dBOperation.selecting_data_from_table_into_csv()
            self.log_writer.log("Extraction csv file from table completed.")
        except Exception as e:
            prediction_validation_exception = PredictionValidationException(
                "Failed during instantiation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionValidation.__name__,
                            self.prediction_validation.__name__))
            raise Exception(prediction_validation_exception.error_message_detail(str(e), sys)) from e








