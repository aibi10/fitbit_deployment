import sys
from project_library_layer.initializer.initializer import Initializer
from exception_layer.generic_exception.generic_exception import GenericException as FileOperationException


class FileOperation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self, project_id, file_object, logger_object):
        try:
            self.file_object = file_object
            self.logger_object = logger_object
            self.initializer = Initializer()
            self.project_id = project_id
            self.model_directory = self.initializer.get_model_directory_path(project_id)
        except Exception as e:
            file_operation = FileOperationException(
                "Failed during training in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, FileOperation.__name__,
                            self.__init__.__name__))
            raise Exception(file_operation.error_message_detail(str(e), sys)) from e

    def save_model(self, model, filename):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None
"""

        try:
            self.logger_object.log('Entered the save_model method of the File_Operation class')

            # path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            path = "{}/{}".format(self.model_directory, filename)
            self.file_object.create_directory(path, over_write=True)
            self.file_object.write_file_content(path, filename + '.sav', model)
            self.logger_object.log('Model File ' + filename + 'saved. Exited the save_model method of the Model_Finder '
                                                              'class')

            return 'success'
        except Exception as e:
            file_operation = FileOperationException(
                "Model File " + filename + "could not be saved module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, FileOperation.__name__,
                        self.save_model.__name__))
            raise Exception(file_operation.error_message_detail(str(e), sys)) from e

    def load_model(self, filename):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """

        try:
            self.logger_object.log('Entered the load_model method of the File_Operation class')
            path = "{}/{}".format(self.model_directory, filename)
            filename = filename + '.sav'
            response = self.file_object.read_file_content(path, filename)
            """
            path=
            with open(self.model_directory + filename + '/' + filename + '.sav',
                      'rb') as f:
                self.logger_object.log(self.file_object,
                                       'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
            """
            if not response['status']:
                raise Exception(response)
            model = None
            if 'file_content' in response:
                model = response['file_content']
            if model is None:
                raise Exception("{} does not have model {}".format(path, filename))
            return model
        except Exception as e:
            file_operation = FileOperationException(
                "Model File " + filename + "could not be found in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, FileOperation.__name__,
                        self.load_model.__name__))
            raise Exception(file_operation.error_message_detail(str(e), sys)) from e

    def find_correct_model_file(self, cluster_number):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        try:
            self.logger_object.log('Entered the find_correct_model_file method of the File_Operation class')
            folder_name = self.model_directory
            model_folder_list = None
            response = self.file_object.list_directory(folder_name)
            if not response['status']:
                raise Exception(response)
            if 'directory_list' in response:
                model_folder_list = response['directory_list']
            if model_folder_list is None:
                raise Exception("Folder not found inside directory {}".format(folder_name))
            model_name = None
            for folder in model_folder_list:
                if folder.find(cluster_number) != -1:
                    model_name = folder

            if model_name is None:
                raise Exception("model not found for cluster {}".format(cluster_number))
            return model_name.replace("/", "")
        except Exception as e:
            file_operation = FileOperationException(
                "Exited the find_correct_model_file method of the Model_Finder class with Failure"
                " in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, FileOperation.__name__,
                            self.find_correct_model_file.__name__))
            raise Exception(file_operation.error_message_detail(str(e), sys)) from e

    def find_correct_model_without_cluster(self):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        try:
            self.logger_object.log('Entered the find_correct_model_file method of the File_Operation class')
            folder_name = self.model_directory
            model_folder_list = None
            response = self.file_object.list_directory(folder_name)
            if not response['status']:
                raise Exception(response)
            if 'directory_list' in response:
                model_folder_list = response['directory_list']
            if model_folder_list is None:
                raise Exception("Folder not found inside directory {}".format(folder_name))
            model_name = None
            for folder in model_folder_list:
                try:
                    model_name=folder
                except:
                    continue
                #if folder.find(cluster_number) != -1:
                    #model_name = folder

            if model_name is None:
                raise Exception("model not found for prediction")
            return model_name.replace("/", "")
        except Exception as e:
            file_operation = FileOperationException(
                "Exited the find_correct_model_file method of the Model_Finder class with Failure"
                " in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, FileOperation.__name__,
                            self.find_correct_model_without_cluster.__name__))
            raise Exception(file_operation.error_message_detail(str(e), sys)) from e
