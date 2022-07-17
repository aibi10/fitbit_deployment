from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from entity_layer.encryption.encrypt_confidential_data import EncryptData
import yaml
mgdb = MongoDBOperation()

database_name = "Credentials"


def get_azure_event_hub_namespace_connection_str():
    database_name = "Credentials"
    collection_name = "event_hub_name_space"
    data = mgdb.get_record(database_name, collection_name, {})
    if data is None:
        return False
    credentials = data['connection_str']
    return credentials
    # return "Endpoint=sb://avnish327030.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=BwK0zozJe+Oz46F5BgpHUtq5zkzqMKCzPdc2kOg2WW8="


def save_azure_event_hub_namespace_connection_str(connection_str):
    database_name = "Credentials"
    collection_name = "event_hub_name_space"
    record = {'connection_str': connection_str}
    mgdb.drop_collection(database_name, collection_name)
    data = mgdb.insert_record_in_collection(database_name, collection_name, record)
    if data > 0:
        print("aws_event_hub_namespace_connection_str has been saved")
    else:
        print("Error occured")


def get_aws_credentials():
    try:

        collection_name = "aws"
        data = mgdb.get_record(database_name, collection_name, {'name': 'aws_access_key'})
        credentials = {'access_key_id': data['Access Key ID'],
                       'secret_access_key': data['Secret Access Key']}
        return credentials
    except Exception as e:
        raise e


def save_aws_credentials(data: dict):
    try:

        collection_name = "aws"
        mgdb.drop_collection(database_name, collection_name)
        result = mgdb.insert_record_in_collection(database_name, collection_name, data)
    except Exception as e:
        raise e


def get_azure_blob_storage_connection_str():
    collection_name = "azure_blob_storage_connection_str"

    data = mgdb.get_record(database_name, collection_name, {})
    if data is None:
        return False
    credentials = data['connection_str']
    return credentials
    # return "DefaultEndpointsProtocol=https;AccountName=storageaccountrgai197de;AccountKey=o56dC8V4h6GmP1bin7a9NO0e6pUWTJrf/lzO3ogX1vFOTnodTCKbZp5VsRHfCSSHDLF8XqMKP0wDVi82eiaT7Q==;EndpointSuffix=core.windows.net"


def save_azure_blob_storage_connection_str(connection_str):
    try:
        collection_name = "azure_blob_storage_connection_str"
        record = {'connection_str': connection_str}
        mgdb.drop_collection(database_name, collection_name)
        data = mgdb.insert_record_in_collection(database_name, collection_name, record)
        if data > 0:
            print("azure_blob_storage_connection_str has been saved")
        else:
            print("Error occured")
    except Exception as e:
        raise e


def get_google_cloud_storage_credentials():
    try:
        data = mgdb.get_record('Credentials', 'gcp', {})
        return data
    except Exception as e:
        raise e


def save_google_cloud_storage_credentials(data: dict):
    try:
        collection_name = "gcp"
        mgdb.drop_collection(database_name, collection_name)
        mgdb.insert_record_in_collection(database_name, collection_name, data)
    except Exception as e:
        raise e


def get_azure_input_file_storage_connection_str():
    collection_name = "azure_input_file_storage_connection_str"
    data = mgdb.get_record(database_name, collection_name, {})
    if data is None:
        return False
    credentials = data['connection_str']
    return credentials

    # return "DefaultEndpointsProtocol=https;AccountName=inputfilestorage;AccountKey=u2jUt3DLgsPiZO8S7kZ1yk4B7FJ6gCtdTeckaige1AXTnyPA1zby1w0n6HUi/2cdOAoo5E1jXrEeofyI6UDtFw==;EndpointSuffix=core.windows.net"


def save_azure_input_file_storage_connection_str(connection_str):
    collection_name = "azure_input_file_storage_connection_str"
    record = {'connection_str': connection_str}
    mgdb.drop_collection(database_name, collection_name)
    data = mgdb.insert_record_in_collection(database_name, collection_name, record)
    if data > 0:
        print("azure_input_file_storage_connection_str has been saved")
    else:
        print("Error occured")


def get_watcher_checkpoint_storage_account_connection_str():
    collection_name = "watcher_checkpoint_storage_account_connection_str"
    data = mgdb.get_record(database_name, collection_name, {})
    if data is None:
        return False
    credentials = data['connection_str']
    return credentials


def save_watcher_checkpoint_storage_account_connection_str(connection_str):
    collection_name = "watcher_checkpoint_storage_account_connection_str"
    record = {'connection_str': connection_str}
    mgdb.drop_collection(database_name, collection_name)
    data = mgdb.insert_record_in_collection(database_name, collection_name, record)
    if data > 0:
        print("watcher_checkpoint_storage_account_connection_str has been saved")
    else:
        print("Error occured")


def save_email_configuration(data: dict):
    try:
        collection_name = "email_config"
        mgdb.drop_collection(database_name, collection_name)
        mgdb.insert_record_in_collection(database_name, collection_name, data)
    except Exception as e:
        raise e


def get_sender_email_id_credentials():
    encrypt_data = EncryptData()
    collection_name = "email_config"
    email_config = mgdb.get_record(database_name, collection_name, {})
    config=yaml.safe_load(open("project_credentials.yaml"))
    
    key =config['key']
    result = {
        'email_address': email_config["sender_email_id"],
        'passkey': encrypt_data.decrypt_message(email_config["passkey"],key).decode('utf-8'),
    }
    return result


def get_receiver_email_id_credentials():
    collection_name = "email_config"
    result = mgdb.get_record(database_name=database_name, collection_name=collection_name, query={})
    email_list =  result.get('receiver_email_id', None)
    receiver_email_ids=";".join(email_list)
    return receiver_email_ids


def save_user_detail(email, role_id):
    database_name = "registration"
    collection_name = "user_allowed"
    mgdb.drop_collection(database_name, collection_name)
    mgdb.insert_record_in_collection(database_name, collection_name, {
        "email_address": email,
        "user_role_id": role_id
    })
    role_id = int(role_id)
    collection_name = "user_role"
    user_role = None
    if role_id == 1:
        user_role = "admin"
    if role_id == 2:
        user_role = "viewer"
    mgdb.drop_collection(database_name, collection_name)
    mgdb.insert_record_in_collection(database_name, collection_name, {
        "user_role_id": role_id,
        "user_role": user_role
    })


def save_flask_session_key(secret_key):
    database = "session"
    collection_name = "secretKey"
    mgdb.insert_record_in_collection(database, collection_name, {"secret-key": secret_key})
