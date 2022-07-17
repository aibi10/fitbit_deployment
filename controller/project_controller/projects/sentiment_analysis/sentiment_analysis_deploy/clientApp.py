
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_prediction.predictApp import \
    PredictApi
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_training.trainApp import \
    TrainApi
from controller.project_controller.projects.sentiment_analysis.sentiment_analysis_deploy.com_in_ineuron_ai_utils.utils_cloud import \
    UtilCloud, get_File_manager_object
from exception_layer.generic_exception.generic_exception import GenericException as ClientApiException
from project_library_layer.initializer.initializer import Initializer
from logging_layer.logger.logger import AppLogger
import sys

class ClientApi:

    def __init__(self, global_project_id,execution_id,executed_by):
        try:
            stopWordsFilePath = "data/stopwords.txt"
            self.predictObj = PredictApi(global_project_id=global_project_id, stopWordsFilePath=None)

            self.trainObj = TrainApi(global_project_id=global_project_id, stopWordsFilePath=None)
            self.utils = UtilCloud(global_project_id=global_project_id)
            self.file_manager = get_File_manager_object(global_project_id=global_project_id)
            self.initializer = Initializer()

            self.log_writer = AppLogger(project_id=global_project_id, executed_by=executed_by, execution_id=execution_id,
                                        socket_io=None)


        except Exception as e:
            client_api_exception = ClientApiException(
                "Failed  in module [{0}] class [{1}] method [{2}]".format(self.__module__, ClientApi.__name__,
                                                                          "__init__"))
            raise Exception(client_api_exception.error_message_detail(str(e), sys)) from e

    def predictRoute(self, global_project_id, projectId, userId, text):
        try:
            self.log_writer.log_database = self.initializer.get_prediction_database_name()
            self.log_writer.log_collection_name = self.initializer.get_prediction_main_log_collection_name()
            self.log_writer.log(f"Training begin for project id {projectId} user id {userId}  ")
            self.log_writer.log("Prediction Started")
            training_file_name = self.initializer.get_sentiment_training_file_name()
            self.log_writer.log("Training file name fetched {}".format(training_file_name))
            path = self.utils.get_training_file_path(userId, projectId)
            self.log_writer.log("Training file path fetched {}".format(path))
            model_file_name = "modelForPrediction.sav"
            self.log_writer.log("Model file name {}".format(model_file_name))
            vector_file_name = "vectorizer.pickle"
            self.log_writer.log("Vector file name {}".format(vector_file_name))
            result = self.predictObj.executePreocessing(text, modelPath=path, model_file_name=model_file_name,
                                                        vectorPath=path, vector_file_name=vector_file_name)

            self.log_writer.log("Prediction completed  and predicted label {}".format(result))
            return result
        except Exception as e:
            client_api_exception = ClientApiException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ClientApi.__name__,
                            self.predictRoute.__name__))
            raise Exception(client_api_exception.error_message_detail(str(e), sys)) from e

    def trainModel(self, global_project_id, projectId, userId, data):
        try:
            self.log_writer.log_database = self.initializer.get_training_database_name()
            self.log_writer.log_collection_name = self.initializer.get_model_training_log_collection_name()
            self.log_writer.log(f"Training begin for project id {projectId} user id {userId}  ")
            training_file_name = self.initializer.get_sentiment_training_file_name()
            self.log_writer.log(f"sentiment file name extracted {training_file_name}")
            self.utils.createDirectoryForUser(userId, projectId)

            # path = trainingDataFolderPath + userId + "/" + projectId
            path = self.utils.get_training_file_path(userId, projectId)
            self.log_writer.log(f"Directory created {path}")
            trainingDataDict = self.utils.extractDataFromTrainingIntoDictionary(data)
            self.log_writer.log(f"Extarcted data from training into dictonary")
            # with open(path + '/trainingData.json', 'w', encoding='utf-8') as f:
            #    json.dump(trainingDataDict, f, ensure_ascii=False, indent=4)
            # dataFrame = pd.read_json(path + '/trainingData.json')
            self.file_manager.write_file_content(path, training_file_name, trainingDataDict)
            self.log_writer.log(f"file content has been written into path {path} file {training_file_name}   ")
            jsonpath = path
            modelPath = path
            self.log_writer.log(f"model and json file path {path}")
            modelscore = self.trainObj.training_model(jsonFilePath=jsonpath, file_name=training_file_name,
                                                      modelPath=modelPath
                                                      )
            self.log_writer.log(f"Training completed successfully")
            return True
        except Exception as e:
            client_api_exception = ClientApiException(
                "Failed  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ClientApi.__name__,
                            self.trainModel.__name__))
            raise Exception(client_api_exception.error_message_detail(str(e), sys)) from e

"""
data = [{
    "lName": 0,
    "lData": "@switchfoot http:/,witpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
},
    {
        "lName": 0,
        "lData": "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
    },
    {
        "lName": 0,
        "lData": "my whole body feels itchy and like its on fire "
    },
    {
        "lName": 4,
        "lData": "is now followinq @DAChesterFrench , you shud do tha same"
    },
    {
        "lName": 4,
        "lData": "Just added tweetie to my new iPhone "
    },
    {
        "lName": 4,
        "lData": "crazy day of school. there for 10 hours straiiight. about to watch the hills. @spencerpratt told me too! ha. happy birthday JB! "
    },
    {
        "lName": 4,
        "lData": "@ProductOfFear You can tell him that I just burst out laughing really loud because of that  Thanks for making me come out of my sulk!"
    },
    {
        "lName": 0,
        "lData": "spring break in plain city... it's snowing "
    },
    {
        "lName": 0,
        "lData": "Hollis' death scene will hurt me severely to watch on film  wry is directors cut not out now?"
    },
    {
        "lName": 0,
        "lData": "about to file taxes "
    },

    {
        "lName": 0,
        "lData": "Need a hug "
    },
    {
        "lName": 4,
        "lData": "@LutheranLucciol Make sure you DM me if you post a link to that video! &lt;LOL&gt;So I don't miss it   Better get permission and blessing first?"
    },
    {
        "lName": 4,
        "lData": "Going to bed so goodnight everyone  and sweet dreams  http://twitpic.com/2y2e0"
    },
    {
        "lName": 4,
        "lData": "@nileyjileyluver Haha, don't worry! You'll get the hang of it! "
    }
]
try:
	capi = ClientApi(global_project_id=16)
	res=capi.trainModel(global_project_id=16, projectId=1, userId=1, data=data)
	print(res)
	res=capi.predictRoute(global_project_id=16, projectId=1, userId=1, text=["Just added tweetie to my new iPhone"])
	print(res)

	#
except Exception as e:
	print(e)
	
	"""
