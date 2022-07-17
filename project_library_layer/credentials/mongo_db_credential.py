#from encryption.encrypt_confidential_data import EncryptData
from entity_layer.encryption.encrypt_confidential_data import EncryptData
import yaml
def get_mongo_db_credentials():
    config=yaml.safe_load(open("project_credentials.yaml"))
    encrypt_data=EncryptData()
    key, user_name, password, url =config['key'],\
                                   config["mongodb"]["user_name"],\
                                   config["mongodb"]["password"],\
                                   config["mongodb"]["url"]
    user_name=encrypt_data.decrypt_message(user_name,key).decode("utf-8")
    password = encrypt_data.decrypt_message(password,key).decode("utf-8")
    url=encrypt_data.decrypt_message(url,key).decode("utf-8")
    is_cloud=config["mongodb"]["is_cloud"]

    return {'user_name': user_name, 'password': password,"url":url,
            "is_cloud":int(is_cloud)
            }



