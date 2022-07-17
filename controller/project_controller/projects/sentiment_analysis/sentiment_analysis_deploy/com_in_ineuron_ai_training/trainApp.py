import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_utils.utils import preprocess_training_data
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_utils.utils_cloud import get_File_manager_object,UtilCloud
from exception_layer.generic_exception.generic_exception import GenericException as TrainApiException

import sys

class TrainApi:

    def __init__(self, stopWordsFilePath,global_project_id):
        try:
            self.stop_words = stopWordsFilePath
            self.file_manager=get_File_manager_object(global_project_id)
            self.util_cloud=UtilCloud(global_project_id=global_project_id)
        except Exception as e:
            train_api_exception = TrainApiException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainApi.__name__,
                            self.__init__.__name__))
            raise Exception(train_api_exception.error_message_detail(str(e), sys)) from e

    def training_model(self, jsonFilePath,file_name, modelPath):
        try:
            data_df = self.util_cloud.preprocess_training_data(jsonFilePath=jsonFilePath,
                                                               file_name=file_name,
                                                               stop_words= self.stop_words)

            TfidfVect = TfidfVectorizer()
            TfidfVect.fit(data_df['text'])

            # saving vector for prediciton
            #with open(modelPath + '/vectorizer.pickle', 'wb') as f:
            #    pickle.dump(TfidfVect, f)
            self.file_manager.write_file_content(modelPath,"vectorizer.pickle",TfidfVect)
            #  train set and prediction set
            x = data_df['text']
            y = data_df['target']
            y = y.astype('int')
            # splitting training and test data
            # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state= 42)

            x_vector = TfidfVect.transform(x)
            # x_test_vector = TfidfVect.transform(x_test)

            # training data using SVM ##
            # model = svm.SVC(C=1.0, kernel='gaussian', degree=3, gamma='auto')
            model = naive_bayes.MultinomialNB()
            model.fit(x_vector, y)

            # y_pred = model.predict(x_test_vector)
            # score1 = metrics.accuracy_score(y_test, y_pred)

            # save the model to disk
            #with open(modelPath + '/modelForPrediction.sav', 'wb') as f:
            #    pickle.dump(model, f)
            self.file_manager.write_file_content(modelPath,"modelForPrediction.sav",model)
            return ("Success")
        except Exception as e:
            train_api_exception = TrainApiException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, TrainApi.__name__,
                            self.training_model.__name__))
            raise Exception(train_api_exception.error_message_detail(str(e), sys)) from e
