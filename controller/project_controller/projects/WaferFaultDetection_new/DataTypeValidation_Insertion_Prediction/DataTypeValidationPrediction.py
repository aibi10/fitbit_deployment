import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from exception_layer.generic_exception.generic_exception import GenericException as  DbOperationMongoDbException
from integration_layer.file_management.file_manager import FileManager
from logging_layer.logger.logger import AppLogger
from project_library_layer.initializer.initializer import Initializer


class DbOperationMongoDB:
    def __init__(self, project_id, executed_by, execution_id, cloud_storage,socket_io=None):
        try:
            self.mongodb = MongoDBOperation()
            self.file_manager = FileManager(cloud_storage)
            self.initializer = Initializer()
            self.project_id = project_id
            self.logger_db_writer = AppLogger(project_id=project_id, executed_by=executed_by, execution_id=execution_id,
                                              socket_io=socket_io)
            self.good_file_path = self.initializer.get_prediction_good_raw_data_file_path(self.project_id)
            self.bad_file_path = self.initializer.get_prediction_bad_raw_data_file_path(self.project_id)
            self.logger_db_writer.log_database = self.initializer.get_prediction_database_name()
        except Exception as e:
            db_operation_mongo_db_exception = DbOperationMongoDbException(
                "Failed during instantiation of object in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DbOperationMongoDB.__name__,
                            "__init__"))
            raise Exception(db_operation_mongo_db_exception.error_message_detail(str(e), sys)) from e

    def insert_into_table_good_data(self, column_name):
        """
        Description: Load all csv file into mongo db database "training_database" ,collection:"Good_Raw_Data"


        :return:
        """
        try:

            self.logger_db_writer.log_collection_name = self.initializer.get_db_insert_log_collection_name()

            prediction_database_name = self.initializer.get_prediction_database_name()
            self.logger_db_writer.log(
                "Droping existing collection if present in database {}".format(prediction_database_name))
            good_raw_data_collection_name = self.initializer.get_prediction_good_raw_data_collection_name(self.project_id)
            self.mongodb.drop_collection(prediction_database_name, good_raw_data_collection_name)

            self.logger_db_writer.log(
                "Starting loading of good files in database:{} and collection: {}"
                    .format(prediction_database_name,good_raw_data_collection_name))
            response = self.file_manager.list_files(self.good_file_path)
            if not response['status']:
                return True
            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                return True

            # files=self.az_blob_mgt.getAllFileNameFromDirectory(self.good_file_path)
            self.logger_db_writer.log("{} files found in {} ".format(len(files), self.good_file_path))
            for file in files:
                try:
                    self.logger_db_writer.log("Insertion of file " + file + " started...")
                    # df=self.az_blob_mgt.readCsvFileFromDirectory(self.good_file_path,file)
                    response = self.file_manager.read_file_content(self.good_file_path, file)
                    if not response['status']:
                        continue
                    df=None
                    if 'file_content' in response:
                        df=response['file_content']
                    if df is None:
                        continue
                    df.columns = column_name
                    self.mongodb.insert_dataframe_into_collection(prediction_database_name,
                                                                  good_raw_data_collection_name,
                                                                  df)
                    self.logger_db_writer.log("File: {0} loaded successfully".format(file))
                except Exception as e:
                    self.logger_db_writer.log(str(e))
                    self.file_manager.move_file(self.good_file_path, self.bad_file_path, file, over_write=True)
                    # self.az_blob_mgt.moveFileInDirectory(self.good_file_path,self.bad_file_path,file)
                    self.logger_db_writer.log(
                        "File " + file + " was not loaded successfully hence moved to dir:" + self.bad_file_path)


        except Exception as e:

            db_operation_mongo_db_exception = DbOperationMongoDbException(

                "Failed in module [{0}] class [{1}] method [{2}]"

                    .format(self.__module__, DbOperationMongoDB.__name__,

                            self.insert_into_table_good_data.__name__))

            raise Exception(db_operation_mongo_db_exception.error_message_detail(str(e), sys)) from e

    def insert_into_table_good_data_zomato(self, column_name):
        """
        Description: Load all csv file into mongo db database "training_database" ,collection:"Good_Raw_Data"
        :return:
        """
        try:

            self.logger_db_writer.log_collection_name = self.initializer.get_db_insert_log_collection_name()

            prediction_database_name = self.initializer.get_prediction_database_name()
            self.logger_db_writer.log(
                "Droping existing collection if present in database {}".format(prediction_database_name))
            good_raw_data_collection_name = self.initializer.get_prediction_good_raw_data_collection_name(self.project_id)
            self.mongodb.drop_collection(prediction_database_name, good_raw_data_collection_name)

            self.logger_db_writer.log(
                "Starting loading of good files in database:{} and collection: {}"
                    .format(prediction_database_name,good_raw_data_collection_name))
            response = self.file_manager.list_files(self.good_file_path)
            if not response['status']:
                return True
            files = None
            if 'files_list' in response:
                files = response['files_list']
            if files is None:
                return True

            # files=self.az_blob_mgt.getAllFileNameFromDirectory(self.good_file_path)
            self.logger_db_writer.log("{} files found in {} ".format(len(files), self.good_file_path))
            for file in files:
                try:
                    self.logger_db_writer.log("Insertion of file " + file + " started...")
                    # df=self.az_blob_mgt.readCsvFileFromDirectory(self.good_file_path,file)
                    response = self.file_manager.read_file_content(self.good_file_path, file)
                    if not response['status']:
                        continue
                    df=None
                    if 'file_content' in response:
                        df=response['file_content']
                    if df is None:
                        continue
                    df=df[column_name]
                    df.columns = column_name
                    self.mongodb.insert_dataframe_into_collection(prediction_database_name,
                                                                  good_raw_data_collection_name,
                                                                  df)
                    self.logger_db_writer.log("File: {0} loaded successfully".format(file))
                except Exception as e:
                    self.logger_db_writer.log(str(e))
                    self.file_manager.move_file(self.good_file_path, self.bad_file_path, file, over_write=True)
                    # self.az_blob_mgt.moveFileInDirectory(self.good_file_path,self.bad_file_path,file)
                    self.logger_db_writer.log(
                        "File " + file + " was not loaded successfully hence moved to dir:" + self.bad_file_path)


        except Exception as e:
            db_operation_mongo_db_exception = DbOperationMongoDbException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DbOperationMongoDB.__name__,
                            self.insert_into_table_good_data_zomato.__name__))
            raise Exception(db_operation_mongo_db_exception.error_message_detail(str(e), sys)) from e


    def selecting_data_from_table_into_csv(self, ):
        """

        :return:
        """
        try:
            directory_name = self.initializer.get_prediction_file_from_db_path(self.project_id)
            # directory_name="training-file-from-db"
            # file_name="InputFile.csv"
            file_name = self.initializer.get_prediction_input_file_name()  # directory name conatin project name hence only file name needed.
            database_name = self.initializer.get_prediction_database_name()
            collection_name = self.initializer.get_export_to_csv_log_collection_name()
            prediction_collection = self.initializer.get_prediction_good_raw_data_collection_name(self.project_id)
            self.logger_db_writer.log_collection_name = collection_name
            msg = "starting of loading of database:training_database,collection:Good_Raw_Data records into InputFile.csv"
            self.logger_db_writer.log(msg)
            df = self.mongodb.get_dataframe_of_collection(database_name, prediction_collection)
            msg = "Good_Raw_data has been loaded into pandas dataframe"
            self.logger_db_writer.log(msg)
            # self.az_blob_mgt.saveDataFrameTocsv(directory_name,file_name,df)
            df.reset_index(drop=True,inplace=True)
            self.file_manager.write_file_content(directory_name, file_name, df, over_write=True)
            msg = "InputFile.csv created successfully in directory" + directory_name
            self.logger_db_writer.log(msg)
        except Exception as e:
            db_operation_mongo_db_exception = DbOperationMongoDbException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DbOperationMongoDB.__name__,
                            self.selecting_data_from_table_into_csv.__name__))
            raise Exception(db_operation_mongo_db_exception.error_message_detail(str(e), sys)) from e

