import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from entity_layer.project.project_configuration import ProjectConfiguration

from logging_layer.logger.logger import AppLogger
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.initializer.initializer import Initializer
from integration_layer.file_management.file_manager import FileManager
from exception_layer.generic_exception.generic_exception import GenericException as PredictionDataValidationException
import sys
class PredictionDataValidation:
    """
               This class shall be used for handling all the validation done on the Raw Prediction Data!!.

               Written By: iNeuron Intelligence
               Version: 1.0
               Revisions: None

               """
    def __init__(self, project_id, prediction_file_path, executed_by, execution_id, cloud_storage,socket_io=None):
        try:
            self.Batch_Directory = prediction_file_path
            self.project_id = project_id
            self.initializer = Initializer()
            self.database_name = self.initializer.get_prediction_database_name()
            self.mongo_db = MongoDBOperation()
            self.logger = AppLogger(project_id=project_id, executed_by=executed_by,
                                    execution_id=execution_id,socket_io=socket_io)
            self.logger.log_database=self.database_name
            self.file_manager = FileManager(cloud_storage)
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed during instantiation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            "__init__"))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    """
    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = 'schema_prediction.json'
        self.logger = App_Logger()
    """

    def values_from_schema(self):
        """
                                Method Name: valuesFromSchema
                                Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                                Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                                On Failure: Raise ValueError,KeyError,Exception

                                 Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                                        """
        try:
            self.logger.log_collection_name = self.initializer.get_values_from_schema_validation_collection_name()
            dic = self.mongo_db.get_record(self.initializer.get_project_system_database_name(),
                                           self.initializer.get_schema_prediction_collection_name(),
                                           {'project_id': self.project_id})
            if dic is None:
                msg = "Schema definition not found for project id:{}".format(self.project_id)
                raise Exception(msg)
            if 'schema' not in dic:
                msg = "Schema definition not found for project id:{}".format(self.project_id)
                raise Exception(msg)
            dic = dic['schema']
            """
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
                """
            pattern = dic['SampleFileName']
            length_of_date_stamp_in_file = dic['LengthOfDateStampInFile']
            length_of_time_stamp_in_file = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            number_of_columns = dic['NumberofColumns']

            #file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            message ="LengthOfDateStampInFile:: %s" %length_of_date_stamp_in_file + "\t" + "LengthOfTimeStampInFile:: %s" % length_of_time_stamp_in_file +"\t " + "NumberofColumns:: %s" % number_of_columns + "\n"
            self.logger.log(message)

            #file.close()
            return length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns

        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.values_from_schema.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e




    def manual_regex_creation(self):

        """
                                      Method Name: manualRegexCreation
                                      Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                                  This Regex is used to validate the filename of the prediction data.
                                      Output: Regex pattern
                                      On Failure: None

                                       Written By: iNeuron Intelligence
                                      Version: 1.0
                                      Revisions: None

                                              """
        try:
            project_config_detail = ProjectConfiguration().get_project_configuration_detail(project_id=self.project_id)
            if not project_config_detail['status']:
                raise Exception("Project configuration not found")
            if 'project_config_detail' in project_config_detail:
                project_config_detail = project_config_detail['project_config_detail']
            if project_config_detail is None:
                raise Exception('Project configuration not found')
            file_name_pattern = project_config_detail.get('prediction_file_name_pattern', None)
            if file_name_pattern is None:
                file_name_pattern = project_config_detail.get('file_name_pattern', None)
            if file_name_pattern is None:
                raise Exception('File pattern name not found')
            regex = file_name_pattern
            return regex
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.manual_regex_creation.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e


    def create_directory_for_good_bad_raw_data(self):

        """
                                        Method Name: createDirectoryForGoodBadRawData
                                        Description: This method creates directories to store the Good Data and Bad Data
                                                      after validating the prediction data.

                                        Output: None
                                        On Failure: OSError

                                         Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None


                                                """
        try:
            self.logger.log_collection_name = self.initializer.get_general_log_collection_name()
            self.logger.log("Good raw and bad raw directory creation begin..")
            good_raw_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            bad_raw_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            self.file_manager.create_directory(good_raw_file_path, over_write=True)
            self.file_manager.create_directory(bad_raw_file_path, over_write=True)
            self.logger.log("{} and {} directory created successfully".format(good_raw_file_path, bad_raw_file_path))
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.create_directory_for_good_bad_raw_data.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    """
        try:
            path = os.path.join("Prediction_Raw_Files_Validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Prediction_Raw_Files_Validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while creating Directory %s:" % ex)
            file.close()
            raise OSError
        """

    def delete_existing_good_data_prediction_folder(self):
        """
                                            Method Name: deleteExistingGoodDataTrainingFolder
                                            Description: This method deletes the directory made to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None


                                                    """
        """
        try:
            path = 'Prediction_Raw_Files_Validated/'
            # if os.path.isdir("ids/" + userName):
            # if os.path.isdir(path + 'Bad_Raw/'):
            #     shutil.rmtree(path + 'Bad_Raw/')
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open("Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"GoodRaw directory deleted successfully!!!")
                file.close()
        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while Deleting Directory : %s" %s)
            file.close()
            raise OSError
        """
        try:
            self.logger.log_collection_name = self.initializer.get_general_log_collection_name()
            self.logger.log("Removing good raw directory")
            good_raw_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            self.file_manager.remove_directory(good_raw_file_path)
            self.logger.log("{} directory deleted successfully".format(good_raw_file_path))
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.delete_existing_good_data_prediction_folder.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e


    def delete_existing_bad_data_prediction_folder(self):

        """
                                            Method Name: deleteExistingBadDataTrainingFolder
                                            Description: This method deletes the directory made to store the bad Data.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        """
        try:
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"BadRaw directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while Deleting Directory : %s" %s)
            file.close()
            raise OSError
        """
        try:
            self.logger.log_collection_name = self.initializer.get_general_log_collection_name()
            self.logger.log("Removing bad raw directory")
            bad_raw_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            self.file_manager.remove_directory(bad_raw_file_path)
            self.logger.log("{} directory deleted successfully".format(bad_raw_file_path))
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.delete_existing_bad_data_prediction_folder.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e


    def move_bad_files_to_archive_bad(self):


        """
                                            Method Name: moveBadFilesToArchiveBad
                                            Description: This method deletes the directory made  to store the Bad Data
                                                          after moving the data in an archive folder. We archive the bad
                                                          files to send them back to the client for invalid data issue.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            path= "PredictionArchivedBadData"
            if not os.path.isdir(path):
                os.makedirs(path)
            source = 'Prediction_Raw_Files_Validated/Bad_Raw/'
            dest = 'PredictionArchivedBadData/BadData_' + str(date)+"_"+str(time)
            if not os.path.isdir(dest):
                os.makedirs(dest)
            files = os.listdir(source)
            for f in files:
                if f not in os.listdir(dest):
                    shutil.move(source + f, dest)
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Bad files moved to archive")
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
            self.logger.log(file,"Bad Raw Data Folder Deleted successfully!!")
            file.close()
        except OSError as e:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise OSError
        """
        try:
            now = datetime.now()
            date = now.date()
            time = now.strftime("%H%M%S")
            self.logger.log_collection_name = self.initializer.get_general_log_collection_name()
            self.logger.log("Archiving of bad prediction file begin.")
            source = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            destination = self.initializer.get_prediction_archive_bad_raw_data_file_path(self.project_id)
            is_source_path_exist = self.file_manager.is_directory_present(source)
            if not is_source_path_exist['status']:
                self.logger.log(f"Source {source} directory is not found ")
                return True
            destination = destination + '/BadData_' + str(date) + "_" + str(time)
            self.file_manager.create_directory(destination, over_write=True)
            response = self.file_manager.list_files(source)
            if not response['status']:
                self.logger.log(f"Files not found in bad prediction file path{source}.")
                return True
            files = None
            if 'files_list' in response:
                files = response['files_list']
            for f in files:
                self.file_manager.move_file(source, destination, f)
            self.logger.log("Bad files moved to archive")
            self.file_manager.remove_directory(source)
            self.logger.log("Bad Raw Data Folder Deleted successfully!!")
            self.logger.log("Archiving of bad prediction file done.")
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.move_bad_files_to_archive_bad.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e


    def validation_file_name_raw(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile):
        """
            Method Name: validationFileNameRaw
            Description: This function validates the name of the prediction csv file as per given name in the schema!
                         Regex pattern is used to do the validation.If name format do not match the file is moved
                         to Bad Raw Data folder else in Good raw data.
            Output: None
            On Failure: Exception

             Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

        """
        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        self.createDirectoryForGoodBadRawData()
        onlyfiles = [f for f in listdir(self.Batch_Directory)]
        try:
            f = open("Prediction_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Good_Raw")
                            self.logger.log(f,"Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                        else:
                            shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                            self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    else:
                        shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

            f.close()

        except Exception as e:
            f = open("Prediction_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, "Error occured while validating FileName %s" % e)
            f.close()
            raise e
    """
        try:
            self.logger.log_collection_name=self.initializer.get_name_validation_log_collection_name()
            self.delete_existing_bad_data_prediction_folder()
            self.delete_existing_good_data_prediction_folder()
            # create new directories
            self.create_directory_for_good_bad_raw_data()
            response=self.file_manager.list_files(self.Batch_Directory)
            if not response['status']:
                self.logger.log(response['message'])
                raise Exception(response['message'])
            onlyfiles=None
            if 'files_list' in response:
                onlyfiles=response['files_list']
            if onlyfiles is None:
                self.logger.log(response['message'])
                return True

            good_raw_data_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            bad_raw_data_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)

            #f = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            self.file_manager.copy_file(self.Batch_Directory,good_raw_data_file_path,filename,over_write=True)
                            #shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                            self.logger.log("Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                        else:
                            self.file_manager.copy_file(self.Batch_Directory,bad_raw_data_file_path,filename,over_write=True)
                            #shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                            self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    else:
                        self.file_manager.copy_file(self.Batch_Directory,bad_raw_data_file_path,filename,over_write=True)
                        #shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    self.file_manager.copy_file(self.Batch_Directory, bad_raw_data_file_path, filename, over_write=True)
                    #shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)


        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.validation_file_name_raw.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    def validate_column_length(self, number_of_columns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            self.logger.log_collection_name=self.initializer.get_column_validation_log_collection_name()
            self.logger.log("Column Length Validation Started!!")
            good_raw_file_path=self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            bad_raw_file_path=self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            response=self.file_manager.list_files(good_raw_file_path)
            if not response['status']:
                self.logger.log(response['message'])
                return True
            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                self.logger.log(response['message'])
                return True
            for file in files:
                #csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                response=self.file_manager.read_csv_file(good_raw_file_path,file)
                if not response['status']:
                    raise Exception(response)
                csv=None
                if 'data_frame' in response:
                    csv=response['data_frame']
                if csv is None:
                    raise Exception("dataframe not able to read from cloud storage storage")
                if csv.shape[1] == number_of_columns:
                    pass
                else:
                    self.file_manager.move_file(good_raw_file_path,bad_raw_file_path,file)
                    #shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log("Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log("Column Length Validation Completed!!")
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.validate_column_length.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    """
        try:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f,"Column Length Validation Started!!")
            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
                else:
                    shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)

            self.logger.log(f, "Column Length Validation Completed!!")
        except OSError:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Prediction_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e

        f.close()
        """


    def delete_prediction_file(self):
        """

        :return: nothing only remove the file if present at prediction location so that new file can
        be saved.
        """
        try:
            prediction_file_path=self.initializer.get_prediction_output_file_path(self.project_id)
            file_name=self.initializer.get_prediction_output_file_name()
            self.file_manager.remove_file(prediction_file_path,file_name)
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.delete_prediction_file().__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    """
        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')
        """
    def validate_missing_values_in_whole_column(self):
        """
                                  Method Name: validateMissingValuesInWholeColumn
                                  Description: This function validates if any column in the csv file has all values missing.
                                               If all the values are missing, the file is not suitable for processing.
                                               SUch files are moved to bad raw data.
                                  Output: None
                                  On Failure: Exception

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                              """
        try:
            self.logger.log_collection_name=self.initializer.get_missing_values_in_column_collection_name()
            self.logger.log("Missing Values Validation Started!!")
            good_raw_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            bad_raw_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            response = self.file_manager.list_files(good_raw_file_path)
            if not response['status']:
                self.logger.log(response['message'])
                return True
            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                self.logger.log(response['message'])
                return True
            for file in files:
               #csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                #csv=self.file_manager.read_csv_file(good_raw_file_path,file)
                response = self.file_manager.read_csv_file(good_raw_file_path, file)
                if not response['status']:
                    raise Exception(response)
                csv = None
                if 'data_frame' in response:
                    csv = response['data_frame']
                if csv is None:
                    raise Exception("dataframe not able to read from cloud storage storage")

                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        self.file_manager.move_file(good_raw_file_path,bad_raw_file_path,file,over_write=True)
                        #shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                        #            "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(
                                        "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.reset_index(drop=True,inplace=True)
                    self.file_manager.write_file_content(good_raw_file_path,file,csv,over_write=True)
                    #csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                        "Failed  in module [{0}] class [{1}] method [{2}]"
                            .format(self.__module__, PredictionDataValidation.__name__,
                                    self.validate_missing_values_in_whole_column.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e

    """
                try:
                    f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
                    self.logger.log(f, "Missing Values Validation Started!!")

                    for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                        csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                        count = 0
                        for columns in csv:
                            if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                                count+=1
                                shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                            "Prediction_Raw_Files_Validated/Bad_Raw")
                                self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                                break
                        if count==0:
                            csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                            csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
                except OSError:
                    f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
                    self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
                    f.close()
                    raise OSError
                except Exception as e:
                    f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
                    self.logger.log(f, "Error Occured:: %s" % e)
                    f.close()
                    raise e
                    f.close()
        """


    def validation_file_name_raw_start_with_index_two(self, regex, LengthOfDateStampInFile, LengthOfTimeStampInFile):
        """
        Method Name: validationFileNameRaw
        Description: This function validates the name of the training csv files as per given name in the schema!
        Regex pattern is used to do the validation.If name format do not match the file is moved
        to Bad Raw Data folder else in Good raw data.
        Output: None
        On Failure: Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

        # pattern = "['Wafer']+['\_'']+[\d_]+[\d]+\.csv"
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        try:
            self.logger.log_collection_name = self.initializer.get_name_validation_log_collection_name()
            self.delete_existing_bad_data_prediction_folder()
            self.delete_existing_good_data_prediction_folder()
            # create new directories
            self.create_directory_for_good_bad_raw_data()
            response = self.file_manager.list_files(self.Batch_Directory)
            if not response['status']:
                self.logger.log(response['message'])
                raise Exception(response['message'])
            onlyfiles = None
            if 'files_list' in response:
                onlyfiles = response['files_list']
            if onlyfiles is None:
                self.logger.log(response['message'])
                return True

            good_raw_data_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            bad_raw_data_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)

            # f = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[2]) == LengthOfDateStampInFile:
                        if len(splitAtDot[3]) == LengthOfTimeStampInFile:
                            self.file_manager.copy_file(self.Batch_Directory, good_raw_data_file_path, filename,
                                                        over_write=True)
                            # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                            self.logger.log("Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                        else:
                            self.file_manager.copy_file(self.Batch_Directory, bad_raw_data_file_path, filename,
                                                        over_write=True)
                            # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                            self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    else:
                        self.file_manager.copy_file(self.Batch_Directory, bad_raw_data_file_path, filename,
                                                    over_write=True)
                        # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    self.file_manager.copy_file(self.Batch_Directory, bad_raw_data_file_path, filename, over_write=True)
                    # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
        except Exception as e:
            raw_data_validation_exception = PredictionDataValidationException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictionDataValidation.__name__,
                            self.validation_file_name_raw_start_with_index_two.__name__))
            raise Exception(raw_data_validation_exception.error_message_detail(str(e), sys)) from e










