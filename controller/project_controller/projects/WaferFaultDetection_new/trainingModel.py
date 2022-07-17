"""
This is the Entry point for Training the Machine Learning Model.

Written By: iNeuron Intelligence
Version: 1.0
Revisions: None

"""

# Doing the necessary imports
from sklearn.model_selection import train_test_split
from controller.project_controller.projects.WaferFaultDetection_new.data_ingestion import data_loader
from controller.project_controller.projects.WaferFaultDetection_new.data_preprocessing import preprocessing
from controller.project_controller.projects.WaferFaultDetection_new.data_preprocessing import clustering
from controller.project_controller.projects.WaferFaultDetection_new.best_model_finder import tuner
from controller.project_controller.projects.WaferFaultDetection_new.file_operations import file_methods
# from controller.project_controller.projects.WaferFaultDetection_new.application_logging import logger
from logging_layer.logger.logger import AppLogger
from project_library_layer.initializer.initializer import Initializer
from integration_layer.file_management.file_manager import FileManager
from exception_layer.generic_exception.generic_exception import GenericException as TrainModelException
import sys


# Creating the common Logging object


class TrainingModel:

    def __init__(self, project_id, executed_by, execution_id, cloud_storage, socket_io=None):
        try:
            self.log_writer = AppLogger(project_id=project_id, executed_by=executed_by, execution_id=execution_id,
                                        socket_io=socket_io)
            self.initializer = Initializer()
            self.log_writer.log_database = self.initializer.get_training_database_name()
            self.log_writer.log_collection_name = self.initializer.get_model_training_log_collection_name()
            self.file_manager = FileManager(cloud_storage)
            self.project_id = project_id
            self.socket_io = socket_io
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during object instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainingModel.__name__,
                            self.__init__.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e

    def training_model(self):

        try:
            self.log_writer.log('Start of Training')
            # Getting the data from the source
            data_getter = data_loader.DataGetter(project_id=self.project_id, file_object=self.file_manager,
                                                 logger_object=self.log_writer)
            data = data_getter.get_data()

            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor(file_object=self.file_manager, logger_object=self.log_writer,
                                                      project_id=self.project_id)

            target_column = None
            if self.project_id == 1:
                data = preprocessor.remove_columns(data, [
                    'Wafer'])  # remove the unnamed column as it doesn't contribute to prediction.
                target_column = 'Output'
            if self.project_id == 2:
                data = preprocessor.drop_unnecessary_columns(data, ['TSH_measured', 'T3_measured', 'TT4_measured',
                                                                    'T4U_measured', 'FTI_measured', 'TBG_measured',
                                                                    'TBG',
                                                                    'TSH'])


                data = preprocessor.replace_invalid_values_with_null(data)

                data = preprocessor.encode_categorical_values(data)
                target_column = "Class"

            # create separate features and labels
            if self.project_id==1:
                data=preprocessor.remove_null_string(data)
            X, Y = preprocessor.separate_label_feature(data, label_column_name=target_column)


            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                X = preprocessor.impute_missing_values(X)  # missing value imputation

            if self.project_id == 2:
                X, Y = preprocessor.handle_imbalance_dataset(X, Y)
            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            if self.project_id==1:
                cols_to_drop = preprocessor.get_columns_with_zero_std_deviation(X)

            # drop the columns obtained above
                X = preprocessor.remove_columns(X, cols_to_drop)

            """ Applying the clustering approach"""

            kmeans = clustering.KMeansClustering(project_id=self.project_id, file_object=self.file_manager,
                                                 logger_object=self.log_writer)  # object initialization.
            number_of_clusters = kmeans.elbow_plot(X)  # using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster'] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3,
                                                                    random_state=355)

                model_finder = tuner.ModelFinder(project_id=self.project_id, file_object=self.file_manager,
                                                 logger_object=self.log_writer)  # object initialization

                # getting the best model for each of the clusters
                if self.project_id==1:
                    best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test,
                                                                              cluster_no=str(i))
                if self.project_id==2:
                    best_model_name, best_model = model_finder.get_best_model_thyroid(x_train, y_train, x_test, y_test,
                                                                                      cluster_no=str(i))

                # saving the best model to the directory.
                file_op = file_methods.FileOperation(project_id=self.project_id, file_object=self.file_manager,
                                                     logger_object=self.log_writer)
                result = file_op.save_model(best_model, best_model_name + str(i))
                if result != 'success':
                    raise Exception("Model {} is failed to save".format(best_model_name + str(i)))
            self.log_writer.log('Successful End of Training')
        except Exception as e:
            train_model_exception = TrainModelException(
                "Failed during training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainingModel.__name__,
                            self.training_model.__name__))
            raise Exception(train_model_exception.error_message_detail(str(e), sys)) from e
