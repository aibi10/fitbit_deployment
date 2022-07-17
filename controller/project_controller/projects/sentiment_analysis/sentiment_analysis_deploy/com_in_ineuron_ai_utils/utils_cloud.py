import json
import os
import shutil
import en_core_web_sm
import re
import string
import pandas
from integration_layer.file_management.file_manager import FileManager
from project_library_layer.initializer.initializer import Initializer
from entity_layer.project.project_configuration import ProjectConfiguration
from exception_layer.generic_exception.generic_exception import GenericException as UtilException
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.data.stop_words import get_stop_word_list
initializer = Initializer()
import sys


def get_File_manager_object(global_project_id):
    try:
        project_configuration = ProjectConfiguration()
        result = project_configuration.get_project_configuration_detail(project_id=global_project_id)
        if not result['status']:
            raise Exception(result['message'])
        project_config_detail = result.get('project_config_detail', None)
        if project_config_detail is None:
            raise Exception("Project configuration not found")
        cloud_storage = project_config_detail.get('cloud_storage', None)
        if cloud_storage is None:
            raise Exception("Cloud storage provider name missing")

        file_manager = FileManager(cloud_storage)
        return file_manager
    except Exception as e:
        util_excep = UtilException(
            "Failed during instantiation in module [{0}]  method [{1}]"
                .format("utils_cloud.py",
                        "get_File_manager_object"))
        raise Exception(util_excep.error_message_detail(str(e), sys)) from e


class UtilCloud:
    def __init__(self,global_project_id):
        try:
            self.global_project_id=global_project_id
            self.file_manager=get_File_manager_object(global_project_id)
        except Exception as e:
            util_exception = UtilException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.__init__.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e

    def get_training_file_path(self,userId,projectId):
        path = initializer.get_training_batch_file_path(self.global_project_id)
        path = path + "/{}".format(userId)
        path = path + "/{}".format(projectId)
        return path

    def createDirectoryForUser(self,userId,projectId):
        """
        path = os.path.join("trainingData/" + userId)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, projectId)
        if not os.path.isdir(path):
            os.mkdir(path)
            """
        try:
            path=initializer.get_training_batch_file_path(self.global_project_id)
            path=path+"/{}".format(userId)
            self.file_manager.create_directory(path)
            path=path+"/{}".format(projectId)
            self.file_manager.create_directory(path)
        except Exception as e:
            util_exception = UtilException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.createDirectoryForUser.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e

    def dataFromTextFile(self):
        try:
            return get_stop_word_list()
        except Exception as e:
            util_exception = UtilException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.dataFromTextFile.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e

    def data_preprocessing_predict(self,text_list,filepath=None):
        try:
            stop_words = self.dataFromTextFile()
            nlp = en_core_web_sm.load()
            pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

            clean_text = []
            for data in text_list:
                clean_data = []
                doc = nlp(data)
                for token in doc:
                    clean = re.sub(pattern, '', str(token.lemma_).lower())
                    if clean not in string.punctuation:
                        if clean not in stop_words:
                            clean_data.append(clean)
                clean_text.append(clean_data)
            return clean_text
        except Exception as e:
            util_exception = UtilException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.data_preprocessing_predict.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e

    def data_preprocessing_train(self,data_dict,filepath=None):
        try:
            stop_words = self.dataFromTextFile()
            nlp = en_core_web_sm.load()
            pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
            df = pandas.DataFrame(columns=['target', 'text'])
            for key in data_dict.keys():
                clean_text = []
                for line in data_dict[key]:
                    # clean_data = []
                    doc = nlp(line)
                    for token in doc:
                        clean = re.sub(pattern, '', str(token.lemma_).lower())
                        if clean not in string.punctuation:
                            if clean not in stop_words:
                                clean_text.append(clean)
                    # clean_text.append(clean_data)
                df = df.append({'target': key, 'text': clean_text}, ignore_index=True)
            return df
        except Exception as e:
            util_exception = UtilException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.data_preprocessing_train.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e


    def extractDataFromTrainingIntoDictionary(self,train_data):
        try:
            dict_train_data = {}
            # lNameList = []
            for dict in train_data:
                key_value = dict['lName']
                value = dict['lData']
                # lNameList.append(dict['lName'])
                if key_value not in dict_train_data.keys():
                    dict_train_data[key_value] = list([value])
                else:
                    (dict_train_data[key_value]).append(value)

            return dict_train_data
        except Exception as e:
            util_exception = UtilException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.extractDataFromTrainingIntoDictionary.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e


    def deleteExistingTrainingFolder(self,path):
        try:
            # if os.path.isdir("ids/" + userName):
            result=self.file_manager.is_directory_present(path)
            if result['status']:
                self.file_manager.remove_directory(path)
                return path + ".....deleted successfully.\n"
            else:
                print('File does not exists. ')
        except Exception as e:
                util_exception = UtilException(
                    "Failed  in module [{0}] class [{1}] method [{2}]"
                        .format(self.__module__, UtilCloud.__name__,
                                self.deleteExistingTrainingFolder.__name__))
                raise Exception(util_exception.error_message_detail(str(e), sys)) from e


    def preprocess_training_data(self,jsonFilePath,file_name, stop_words):
        try:

            result=self.file_manager.read_file_content(jsonFilePath,file_name=file_name)
            #with open(jsonFilePath, 'r') as f:
            #    data_dict = json.load(f)
            #### Data Cleaning
            if not result['status']:
                raise Exception("Failed while reading json file from dir {} path {}".format(jsonFilePath,file_name))
            data_dict=result.get('file_content',None)
            if data_dict is None:
                raise Exception("Failed while reading json file from dir {} path {}".format(jsonFilePath,file_name))
            clean_df = self.data_preprocessing_train(data_dict,stop_words)
            # converting preprocesed data from list to string to use in tfIdf
            clean_df['text'] = [" ".join(value) for value in clean_df['text'].values]
            return clean_df
        except Exception as e:
            util_exception = UtilException(
                "Failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, UtilCloud.__name__,
                            self.preprocess_training_data.__name__))
            raise Exception(util_exception.error_message_detail(str(e), sys)) from e

