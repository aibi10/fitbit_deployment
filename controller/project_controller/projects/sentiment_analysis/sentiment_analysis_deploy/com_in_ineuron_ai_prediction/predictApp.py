import pandas as pd
import en_core_web_sm
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_utils.utils import data_preprocessing_predict, preprocess_training_data
import pickle
import numpy as np
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_utils.utils_cloud import get_File_manager_object,UtilCloud
from exception_layer.generic_exception.generic_exception import GenericException as PredictApiException
import sys


class PredictApi:

    def __init__(self,global_project_id,stopWordsFilePath=None):
        try:
            self.nlp = en_core_web_sm.load()
            self.stop_words_path = stopWordsFilePath
            self.noOfClasses = ["Negative", "Positive"]
            self.file_manager=get_File_manager_object(global_project_id)
            self.util_cloud=UtilCloud(global_project_id=global_project_id)
        except Exception as e:
            predict_api_exception = PredictApiException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictApi.__name__,
                            self.__init__.__name__))
            raise Exception(predict_api_exception.error_message_detail(str(e), sys)) from e

    def executePreocessing(self, text,modelPath,model_file_name,vectorPath,vector_file_name):
        try:
            df_pred = pd.DataFrame([text],columns=['text'])
            df_pred['text'] = self.util_cloud.data_preprocessing_predict(df_pred['text'], self.stop_words_path)
            df_pred['text'] = [" ".join(value) for value in df_pred['text'].values]

            #with open(vectorPath,'rb') as f:
            #    vectorizer = pickle.load(f)
            result=self.file_manager.read_file_content(vectorPath,vector_file_name)
            if not result['status']:
                return Exception("Failed in reading vector file")
            vectorizer=result.get('file_content',None)
            if vectorizer is None:
                return Exception("Failed in reading vector file")

            result = self.file_manager.read_file_content(modelPath, model_file_name)
            if not result['status']:
                return Exception("Failed in reading model file")
            model = result.get('file_content', None)
            if model is None:
                return Exception("Failed in reading model file")

            #with open(modelPath, 'rb') as f:
            #    model = pickle.load(f)

            pred_vector_ = vectorizer.transform(df_pred['text'])
            prediction = model.predict(pred_vector_)
            predictedProbability = model.predict_proba(pred_vector_)
            if list(predictedProbability.flatten())[0] == list(predictedProbability.flatten())[1]:
                return "UNKNOWN"
            elif list(predictedProbability.flatten())[np.argmax(predictedProbability)] > .3:
                return prediction
            else:
                return "UNKNOWN"
        except Exception as e:
            predict_api_exception = PredictApiException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictApi.__name__,
                            self.executePreocessing.__name__))
            raise Exception(predict_api_exception.error_message_detail(str(e), sys)) from e










