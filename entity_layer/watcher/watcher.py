import asyncio
import datetime
import json
import os
import re
import sys

import nest_asyncio
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore
from cloud_storage_layer.microsoft_azure.azure_blob_storage import MicrosoftAzureBlobStorage
from project_library_layer.credentials.credential_data import get_azure_input_file_storage_connection_str
from project_library_layer.initializer.initializer import get_watcher_input_file_path, get_project_id
from project_library_layer.initializer.initializer import Initializer
from entity_layer.project.project_configuration import ProjectConfiguration
from integration_layer.file_management.file_manager import FileManager
from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from project_library_layer.credentials import credential_data

watcher_database_name = "watcher_db"
watcher_collection_name = "watcher_events"


def insert_watcher_event_into_db(watcher_data):
    try:
        is_present = MongoDBOperation().get_record(watcher_database_name, watcher_collection_name,
                                                   watcher_data)
        if is_present is None:
            MongoDBOperation().insert_record_in_collection(watcher_database_name, watcher_collection_name, watcher_data)

    except Exception as e:
        print(str(e))
        if "duplicate key error collection" in str(e):
            pass
        else:
            raise e


def create_directory_input_file_storage(project_id):
    azm = MicrosoftAzureBlobStorage(container_name="machine-learning-327030",
                                    connection_string=get_azure_input_file_storage_connection_str())
    watcher_input_file_path = get_watcher_input_file_path(project_id)
    if not azm.is_directory_present(watcher_input_file_path)['status']:
        azm.create_directory(watcher_input_file_path)
        print(f"Directory: {watcher_input_file_path} created.")


blob_key_path = "company_name/training/data/project/project_id_"
blob_end_path = "project_id_"


def getEventAndSubject(data):
    try:

        event_type = None
        container = None
        if 'eventType' in data.keys():
            event_type = data['eventType']
        if 'subject' in data.keys():
            is_event_from_desired_location = data['subject'].find(blob_key_path)
            if is_event_from_desired_location < 0:
                print("Not from appropriate location")

                return False
            start_index = data['subject'].find(blob_key_path)

            watcher_input_file_path = data['subject'][start_index:]
            reverse_path = watcher_input_file_path[-1::-1]
            idx = reverse_path.find('/')
            reverse_path = reverse_path[:idx + 1]
            reverse_path_revert = reverse_path[-1::-1]
            watcher_input_file_path = watcher_input_file_path.replace(reverse_path_revert, "")
            start_index = data['subject'].index('containers') + len('containers') + 1
            stop_index = data['subject'].index('/blobs/', start_index, )
            container = data['subject'][start_index:stop_index]

        if container == 'machine-learning-327030':
            watcher_data = data.copy()
            if 'data' in watcher_data:
                watcher_data.pop('data')
            watcher_data.update(data['data'])

            if event_type == 'Microsoft.Storage.BlobCreated':
                azm = MicrosoftAzureBlobStorage(container_name="machine-learning-327030",
                                                connection_string=get_azure_input_file_storage_connection_str())

                if not azm.is_directory_present(watcher_input_file_path)['status']:
                    azm.create_directory(watcher_input_file_path)

                result = azm.list_files(watcher_input_file_path)
                file_names = result.get('files_list', None)
                file_names = list(filter(lambda filename: filename.split(".")[-1] == 'csv', file_names))
                if file_names is None:
                    watcher_data.update({result})
                    insert_watcher_event_into_db(watcher_data)
                    return False

                if len(file_names) > 0:
                    project_id = get_project_id(watcher_input_file_path)
                    if not project_id:
                        watcher_data.update(
                            {'status': 'False', 'message': "Project id {} is not proper".format(project_id)})
                        insert_watcher_event_into_db(watcher_data)
                        return False
                    result = ProjectConfiguration().get_project_configuration_detail(int(project_id))
                    project_config_detail = result.get('project_config_detail', None)
                    if result is None:
                        watcher_data.update(
                            {'status': 'False', 'message': "Project Configuration not found"})
                        insert_watcher_event_into_db(watcher_data)
                        return False
                    cloud_name = project_config_detail.get('cloud_storage', None)
                    if cloud_name is None:
                        watcher_data.update(
                            {'status': 'False', 'message': "Cloud {} provider does not support ".format(cloud_name)})
                        insert_watcher_event_into_db(watcher_data)
                        return False
                    file_manager = FileManager(cloud_name)
                    file_path_location = Initializer().get_training_batch_file_path(project_id)
                    result = file_manager.is_directory_present(file_path_location)
                    if not result:
                        watcher_data.update(
                            {'status': 'False', 'message': "Destination directory missing"})
                        insert_watcher_event_into_db(watcher_data)
                        return False
                    for file in file_names:
                        result = azm.read_csv_file(watcher_input_file_path, file)
                        if not result['status']:
                            watcher_data.update(
                                {'status': 'False', 'message': "Failed to read csv file"})
                            insert_watcher_event_into_db(watcher_data)
                            continue
                        df = result.get('data_frame', None)
                        if df is None:
                            watcher_data.update(
                                {'status': 'False', 'message': "Failed to load dataframe"})
                            insert_watcher_event_into_db(watcher_data)
                            continue
                        data = df.to_csv(encoding="utf-8", header=True, index=None)

                        result = file_manager.upload_file(directory_full_path=file_path_location, file_name=file
                                                          , stream_data=data, over_write=False)
                        print(result)
                        if not result['status']:
                            watcher_data.update(result)
                            insert_watcher_event_into_db(watcher_data)
                            continue
                        watcher_data.update({'status': True,
                                             'message': 'Recevied file {} is copied into directory {} '.format(file,
                                                                                                               file_path_location)})
                        azm.remove_file(watcher_input_file_path, file)
                        insert_watcher_event_into_db(watcher_data)
                    print(event_type, container)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        exception_type = e.__repr__()
        exception_detail = {'exception_type': exception_type,
                            'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                            'detail': sys.exc_info().__str__()}
        print(exception_detail)


def updateSingleQuote(text):
    p = re.compile('(?<!\\\\)\'')
    text = p.sub('\"', text)
    return text


def messgae(data):
    data = list(json.loads(updateSingleQuote(data)))
    print("--------------------------------------")
    if len(data) > 0:
        getEventAndSubject(data[0])


async def on_event(partition_context, event):
    # Print the event data.
    event_data = event.body_as_str(encoding='UTF-8')
    print("Received the event: \"{}\" from the partition with ID: \"{}\"".format(event_data,
                                                                                 partition_context.partition_id))
    # Update the checkpoint so that the program doesn't read the events
    # that it has already read when you run it next time.
    messgae(event_data)
    await partition_context.update_checkpoint(event)


async def start_watcher():
    connection_str = credential_data.get_azure_event_hub_namespace_connection_str()
    event_hubname = "input-file-storage"

    # Create an Azure blob checkpoint store to store the checkpoints.
    checkpoint_store = BlobCheckpointStore.from_connection_string(
        conn_str=credential_data.get_watcher_checkpoint_storage_account_connection_str(),
        container_name="check-point")

    # Create a consumer client for the event hub.
    client = EventHubConsumerClient.from_connection_string(
        connection_str, consumer_group="$Default",
        eventhub_name=event_hubname,
        checkpoint_store=checkpoint_store)

    async with client:
        # Call the receive method. Read from the beginning of the partition (starting_position: "-1")
        await client.receive(on_event=on_event, starting_position=datetime.datetime.now())


def start_call():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_watcher())
    loop.close()


def between_callback(message):
    print("Between callback is called")
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_watcher())
    loop.close()


"""
if __name__=="__main__":
    for i in range(0,16):
        create_directory_input_file_storage(str(i))
        """
