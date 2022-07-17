
import yaml

config = yaml.safe_load(open("project_credentials.yaml"))
root_folder = config["root_folder"]
root_file_training_path="ineuron/training/data/project"
root_file_prediction_path="ineuron/prediction/data/project"
root_archive_training_path="ineuron/training/archive/project"
root_archive_prediction_path="ineuron/prediction/archive/project"
root_graph_path="ineuron/report/graph/project"
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation as MongoDB
mongodb = MongoDB()

def get_watcher_input_file_path(project_id):
    return "ineuron/training/data/project/project_id_{}".format(project_id)


def get_project_id(file_path):
    try:
        reverse_file_path=file_path[-1::-1]
        end_index=reverse_file_path.find("_")
        project_id=reverse_file_path[:end_index]
        if project_id.isdigit():
            project_id=project_id[-1::-1]
            return int(project_id)
        else:
            return False
    except Exception as e:
        raise e




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


class Initializer():
    def __init__(self):
        self.data=data

    def get_sentiment_training_file_name(self):
        return "training_data.json"

    def get_aws_bucket_name(self):
        return root_folder

    def get_azure_container_name(self):
        return root_folder

    def get_google_bucket_name(self):
        return root_folder

    def get_session_secret_key(selfs):

        data = mongodb.get_record("session", "secretKey")
        return data['secret-key']

    def get_training_database_name(self):
        return "training_system_log"

    def get_prediction_database_name(self):
        return "prediction_system_log"

    def get_project_system_database_name(self):
        return "project_system"

    def get_schema_training_collection_name(self):
        return "schema_training"

    def get_schema_prediction_collection_name(self):
        return "schema_prediction"

    def get_training_data_collection_name(self):
        return "good_raw_data"

    def get_prediction_data_collection_name(self):
        return "good_raw_data"

    def get_column_validation_log_collection_name(self):
        return "column_validation_log"

    def get_data_transform_log_collection_name(self):
        return "data_transform_log"
    def get_db_insert_log_collection_name(self):
        return "db_insert_log"

    def get_export_to_csv_log_collection_name(self):
        return "export_to_csv"

    def get_general_log_collection_name(self):
        return "general_log"

    def get_missing_values_in_column_collection_name(self):
        return "missing_values_in_column"

    def get_model_training_log_collection_name(self):
        return "model_training"

    def get_name_validation_log_collection_name(self):
        return "name_validation_log"

    def get_training_main_log_collection_name(self):
        return "training_main_log"

    def get_prediction_main_log_collection_name(self):
        return "prediction_main_log"

    def get_values_from_schema_validation_collection_name(self):
        return "values_from_schema_validation"

    def get_project_collection_name(self):
        return "project"

    def get_project_configuration_collection_name(self):
        return "project_configuration"

    def get_training_batch_file_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),
                                            {'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"training_batch_files")
            return path
        except Exception as e:
            raise e

    def get_project_report_graph_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}".format(root_graph_path, project_name)
            return path
        except Exception as e:
            raise e

    def get_project_report_graph_file_path(self,project_id,execution_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_graph_path, project_name,self.get_time_stamp_as_file_name_of_execution_id(execution_id))
            return path
        except Exception as e:
            raise e

    def get_prediction_batch_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_batch_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_batch_files")
            return path
        except Exception as e:
            raise e

    def get_training_good_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "good_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_good_raw_data_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "good_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_bad_raw_data_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_archive_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_archive_training_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_prediction_archive_bad_raw_data_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_archive_bad_raw_data_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_archive_prediction_path, project_name, "bad_raw_data_files")
            return path
        except Exception as e:
            raise e

    def get_training_good_raw_data_collection_name(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_collection_name.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_good_raw_data_collection_name.__name__)
                raise Exception(message)

            collection_name="{}_{}_{}".format(project_name,"good_raw_data",project_id)
            return collection_name
        except Exception as e:
            raise e

    def get_prediction_good_raw_data_collection_name(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_collection_name.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_good_raw_data_collection_name.__name__)
                raise Exception(message)

            collection_name="{}_{}_{}".format(project_name,"good_raw_data",project_id)
            return collection_name
        except Exception as e:
            raise e

    def get_training_file_from_db_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message='Project not found. failed in initializer.py method {}'.format(
                    self.get_training_file_from_db_path.__name__)
                raise Exception(message)
            project_name=None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message='Project name not found failed in initializer.py method {}'.format(
                    self.get_training_file_from_db_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "training_file_from_db")
            return path
        except Exception as e:
            raise e

    def get_prediction_file_from_db_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(), self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message='Project not found. failed in initializer.py method {}'.format(
                    self.get_prediction_file_from_db_path.__name__)
                raise Exception(message)
            project_name=None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message='Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_file_from_db_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_file_from_db")
            return path
        except Exception as e:
            raise e


    def get_training_input_file_name(self):
        try:
            return "InputFile.csv"
        except Exception as e:
            raise e

    def get_prediction_input_file_name(self):
        try:
            return "InputFile.csv"
        except Exception as e:
            raise e
    def get_encoder_pickle_file_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"EncoderPickle")
            return path
        except Exception as e:
            raise e


    def get_encoder_pickle_file_name(self):
        return "encoder.pickle"

    def get_training_preprocessing_data_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"preprocessing_data")
            return path
        except Exception as e:
            raise e

    def get_null_value_csv_file_name(self):
        return "null_values.csv"

    def get_model_directory_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_file_training_path,project_name,"model")
            return path
        except Exception as e:
            raise e

    def get_model_directory_archive_path(self,project_id):
        try:
            project_data=mongodb.get_record(self.get_project_system_database_name(),self.get_project_collection_name(),{'project_id':project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_training_batch_file_path.__name__)
                raise Exception(message)
            path="{}/{}/{}".format(root_archive_training_path,project_name,"model")
            return path
        except Exception as e:
            raise e

    def get_kmean_folder_name(self):
        try:
            return "KMeans"
        except Exception as e:
            raise e

    def get_prediction_output_file_path(self,project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(),
                                              self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_prediction_path, project_name, "prediction_output_file")
            return path
        except Exception as e:
            raise e

    def get_prediction_output_file_name(self):
        return "Output.csv"

    def get_training_thread_database_name(self):
        return "training_prediction_thread"

    def get_prediction_thread_database_name(self):
        return "training_prediction_thread"

    def get_thread_status_collection_name(self):
        return "thread_status"

    def get_add_quotes_to_string_values_in_column_collection_name(self):
        return "add_quotes_to_string_values_in_column"

    def get_encoded_column_name_file_path(self, project_id):
        try:
            project_data = mongodb.get_record(self.get_project_system_database_name(),
                                              self.get_project_collection_name(),
                                              {'project_id': project_id})
            if project_data is None:
                message = 'Project not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            project_name = None
            if 'project_name' in project_data:
                project_name = project_data['project_name']
            if project_name is None:
                message = 'Project name not found failed in initializer.py method {}'.format(
                    self.get_prediction_output_file_path.__name__)
                raise Exception(message)
            path = "{}/{}/{}".format(root_file_training_path, project_name, "encoder_column_name")
            return path
        except Exception as e:
            raise e

    def get_encoded_column_file_name(self):
        return "data_input_column_name.csv"



    def get_accuracy_metric_database_name(self):
        return "accuracy_metric"

    def get_accuracy_metric_collection_name(self):
        return "accuracy_metric_model_collection"

    def get_time_stamp_as_file_name_of_execution_id(self,execution_id):
        try:
            result=mongodb.get_record(self.get_training_thread_database_name(),
                                     self.get_thread_status_collection_name(),
                                     {'execution_id':execution_id})
            if result is not None:
                start_date=result.get('start_date',None)
                start_date='' if start_date is None else start_date.__str__().replace("-","_")
                start_time=result.get('start_time',None)
                start_time='' if start_date is None else start_time.__str__().replace(":","_")
                file_name="{}_{}".format(start_date,start_time)
                if len(file_name)==1:
                    file_name=execution_id
                return file_name
            return execution_id
        except Exception as e:
            raise e

    def get_scheduler_database_name(self):
        return "schedulers"

    def get_scheduler_collection_name(self):
        return "schedulers_job"






