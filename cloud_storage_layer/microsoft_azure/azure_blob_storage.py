"""
AWS SDK for Python (Boto3) to create, configure, and manage AWS services,
such as Amazon Elastic Compute Cloud (Amazon EC2) and Amazon Simple Storage Service (Amazon S3)
"""
import json

from project_library_layer.credentials.credential_data import get_azure_blob_storage_connection_str
from exception_layer.generic_exception.generic_exception import GenericException as MicrosoftAzureException
import sys
from project_library_layer.datetime_libray.date_time import get_time, get_date
import dill
from azure.storage.blob import BlobServiceClient
import pandas as pd
from project_library_layer.initializer.initializer import Initializer
from io import StringIO


class MicrosoftAzureBlobStorage:

    def __init__(self, container_name=None, connection_string=None):
        """

        :param container_name:specify container name
        :param connection_string: specify connection_string name
        """
        try:
            if connection_string is None:
                connection_string = get_azure_blob_storage_connection_str()
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            if container_name is None:
                initial = Initializer()
                self.container_name = initial.get_azure_container_name()
            else:
                self.container_name = container_name
            container_name = self.container_name
            response = self.get_container(container_name)
            if response['status']:
                self.container = response['container']
            else:
                self.create_container(container_name)
                response = self.get_container(container_name)
                if response['status']:
                    self.container = response['container']
                else:
                    raise Exception("Unable to created container [{0}]".format(container_name))
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create object of MicrosoftAzureBlobStorage"
                " in module [{0}] class [{1}] method [{2}]".format(MicrosoftAzureBlobStorage.__module__.__str__(),
                                                                   MicrosoftAzureBlobStorage.__name__,
                                                                   "__init__"))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def create_container(self, container_name, over_write=False):
        """
        :param container_name: container_name in azure storage account
        :param over_write: If true then existing container content will be removed
        :return: True if created else False
        """
        try:
            container_list = self.list_container()
            if container_name not in container_list:
                self.blob_service_client.create_container(container_name)
                return {'status': True, 'message': 'Container [{0}] created successfully'.format(container_name)}
            elif over_write and container_name in container_list:
                # clean the existing container
                return {'status': True, 'message': 'Container [{0}] created successfully'.format(container_name)}
            else:
                return {'status': False,
                        'message': 'Container [{0}] is present try with over write option'.format(container_name)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create container in object in"
                " module [{0}] class [{1}] method [{2}]".format(MicrosoftAzureBlobStorage.__module__.__str__(),
                                                                MicrosoftAzureBlobStorage.__name__,
                                                                self.create_container.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def list_container(self):
        try:
            container_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
            return container_list
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to list container in object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.list_container.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def add_param(self, acceptable_param, additional_param):
        """

        :param acceptable_param: specify param list can be added
        :param additional_param: accepts a dictionary object
        :return: list of param added to current instance of class
        """
        try:
            self.__dict__.update((k, v) for k, v in additional_param.items() if k in acceptable_param)
            return [k for k in additional_param.keys() if k in acceptable_param]
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to add parameter in object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.add_param.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def filter_param(self, acceptable_param, additional_param):
        """

        :param acceptable_param: specify param list can be added
        :param additional_param: accepts a dictionary object
        :return: dict of param after filter
        """
        try:
            accepted_param = {}
            accepted_param.update((k, v) for k, v in additional_param.items() if k in acceptable_param)
            return accepted_param
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to filter parameter in object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.filter_param.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def remove_param(self, param):
        """

        :param param: list of param argument need to deleted from instance object
        :return True if deleted successfully else false:
        """
        try:
            for key in param:
                self.__dict__.pop(key)
            return True

        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to remove parameter in object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.remove_param.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def get_container(self, container_name=None):
        """

        :param container_name: container name
        :return: {'status':True/False,'message':'message_detail,'container':container_object}
        """
        try:
            if container_name is None:
                container_name = self.container_name
            container_list = self.list_container()
            if container_name in container_list:
                return {'status': True, 'message': 'Container [{0}] is present'.format(container_name),
                        'container': self.blob_service_client.get_container_client(container_name)}
            return {'status': False, 'message': 'Container [{0}] is not present'.format(container_name),
                    }
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to fetch container object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.remove_param.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def list_directory(self, directory_full_path=None,is_delete_request=None):
        """
        :param is_delete_request: Specify whether request is coming from delete method or from somewhere else
        :param directory_full_path:directory path
        :return {'status': True/False, 'message': 'message_detail',
                    , 'directory_list': directory_list}
        """
        try:

            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"

            is_directory_exist = False
            directory_list = []
            for directories in self.container.list_blobs():
                dir_name=str(directories.name)
                if dir_name.startswith(directory_full_path):
                    is_directory_exist = True
                    dir_name=dir_name.replace(directory_full_path,"")

                    slash_index=dir_name.find("/")
                    if slash_index>=0:
                        if is_delete_request:
                            directory_list.append(directories.name)
                            continue
                        name_after_slash=dir_name[slash_index:]
                        if len(name_after_slash)<=0:
                            directory_list.append(dir_name)
                        elif name_after_slash=="/initial.txt.dat":
                            directory_list.append(dir_name[:slash_index])

                    else:
                        if "initial.txt.dat"==dir_name and is_delete_request is None:
                            continue
                        directory_list.append(dir_name)

            if is_directory_exist:
                return {'status': True, 'message': 'Directory [{0}]  exist'.format(directory_full_path)
                    , 'directory_list': directory_list}
            else:
                return {'status': False, 'message': 'Directory [{0}] does not exist'.format(directory_full_path)}

        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to list directory in object in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.list_directory.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def list_files(self, directory_full_path):
        """

        :param directory_full_path: directory
        :return:{'status': True/False, 'message': 'Directory [{0}]  present'.format(directory_full_path)
                    , 'files_list': File list will be available only if status is True}
        """
        try:
            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"
            is_directory_exist = False
            list_files = []
            response = self.list_directory(directory_full_path)
            if not response['status']:
                return response
            directories = response['directory_list']
            for file_name in directories:
                is_directory_exist = True
                if "/" not in file_name and file_name != "":
                    #if "initial.txt.dat" in file_name:
                    #    continue
                    list_files.append(file_name)
            if is_directory_exist:
                return {'status': True, 'message': 'Directory [{0}]  present'.format(directory_full_path)
                    , 'files_list': list_files}
            else:
                return {'status': False, 'message': 'Directory [{0}] is not present'.format(directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to list files in object in module [{0}] "
                "class [{1}] method [{2}]".format(MicrosoftAzureBlobStorage.__module__.__str__(),
                                                  MicrosoftAzureBlobStorage.__name__,
                                                  self.list_files.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def create_directory(self, directory_full_path, over_write=False, **kwargs):
        """

        :param directory_full_path: provide full directory path along with name
        :param over_write: default False if accept True then overwrite existing directory if exist
        :return True if created else false
        """
        try:
            if directory_full_path == "" or directory_full_path == "/" or directory_full_path is None:
                return {'status': False, 'message': 'Provide directory name'}

            if directory_full_path[-1] != "/":
                directory_full_path += "/"
            response = self.list_directory(directory_full_path)

            if over_write and response['status']:
                self.remove_directory(directory_full_path)
            if not over_write:
                if response['status']:
                    return {'status': False, 'message': 'Directory is already present. try with overwrite option.'}

            possible_directory = directory_full_path[:-1].split("/")

            directory_name = ""
            for dir_name in possible_directory:
                directory_name += dir_name + "/"
                response = self.is_directory_present(directory_name)
                if not response['status']:
                    content = "This directory is created on [{}] [{}] directory path [{}] ".format(get_date(),
                                                                                                   get_time(),
                                                                                                   directory_name)
                    self.blob_service_client.get_blob_client(container=self.container_name,
                                                             blob=directory_name + "initial.txt.dat").upload_blob(
                        content.encode())
            return {'status': True, 'message': 'Directory [{0}] created successfully '.format(directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create directory in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.create_directory.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def remove_directory(self, directory_full_path):
        """

        :param directory_full_path:provide full directory path along with name
        kindly provide "" or "/" to remove all directory and file from bucket.
        :return: True if removed else false
        """
        try:
            # updating directory full path to ""
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            is_directory_found = False
            response = self.list_directory(directory_full_path,is_delete_request=True)
            if not response['status']:
                return response
            directories = response['directory_list']
            for directory_name in directories:
                self.blob_service_client.get_blob_client(container=self.container_name,
                                                         blob=directory_full_path + directory_name).delete_blob()
            if response['status']:
                return {'status': True, 'message': 'Directory [{0}] removed.'.format(directory_full_path)}
            else:
                return {'status': False, 'message': 'Directory [{0}] is not present.'.format(directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.remove_directory.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def is_file_present(self, directory_full_path, file_name):
        """

        :param directory_full_path: directory_full_path
        :param file_name: file_name
        :return:  return {'status': True/False, 'message': 'message'}
        """
        try:
            response = self.list_files(directory_full_path)
            if response['status']:
                if file_name in response['files_list']:
                    return {'status': True, 'message': 'File [{0}] is present.'.format(directory_full_path + file_name)}
            else:
                return response
            return {'status': False, 'message': 'File [{0}] is not present.'.format(directory_full_path + file_name)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.is_file_present.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def is_directory_present(self, directory_full_path,is_delete_request=None):
        try:
            response = self.list_directory(directory_full_path,is_delete_request=None)
            if response['status']:
                return {'status': True, 'message': 'Directory [{0}] is present'.format(directory_full_path)}
            else:
                return {'status': False, 'message': 'Directory [{0}] is not present'.format(directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to delete directory in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.is_file_present.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def upload_file(self, directory_full_path, file_name, stream_data,local_file_path=False, over_write=False):
        """

        :param stream_data: File stream which you want to upload
        :param directory_full_path: s3 bucket directory
        :param file_name: name you want to specify for file in s3 bucket
        :param local_file_path: your local system file path of file needs to be uploaded
        :param over_write:
        :return:
        """
        try:
            if directory_full_path == "" or directory_full_path == "/":
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path += "/"
            response = self.is_directory_present(directory_full_path)
            if not response['status']:
                self.create_directory(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        'message': "File [{0}] already present in directory [{1}]. "
                                   "try with overwrite option".format(file_name, directory_full_path)}

            if response['status'] and over_write:
                response = self.remove_file(directory_full_path, file_name)
                if not response['status']:
                    return response
            if local_file_path:
                with open(local_file_path, 'rb') as f:
                    self.blob_service_client.get_blob_client(
                        container=self.container_name,
                        blob=directory_full_path + file_name).upload_blob(f)
            else:
                self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=directory_full_path + file_name).upload_blob(stream_data)


            return {'status': True,
                    'message': 'File [{0}] uploaded to directory [{1}]'.format(file_name, directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to upload file in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.upload_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def download_file(self, directory_full_path, file_name, local_system_directory=""):
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path=directory_full_path, file_name=file_name)
            local_system_directory = self.update_directory_full_path_string(local_system_directory)
            if not response['status']:
                return response
            with open(local_system_directory + file_name, "wb") as blob_obj:
                data = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=directory_full_path + file_name).download_blob()
                blob_obj.write(data.readall())
            return {'status': True,
                    'message': 'file [{0}] is downloaded in your system at location [{1}] '
                        .format(file_name, local_system_directory)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to upload file in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.download_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def remove_file(self, directory_full_path, file_name):
        """
        :param directory_full_path: provide full directory path along with name
        :param file_name: file name with extension if possible
        :return: True if removed else false
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if response['status']:
                if file_name=="initial.txt.dat":
                    return {'status':False,'message':'This file [{0}] is not deletable'.format(file_name)}

                remove_blob = self.blob_service_client.get_blob_client(container=self.container_name,
                                                                       blob=directory_full_path + file_name)
                remove_blob.delete_blob()
                return {'status': True,
                        'message': 'File [{}] deleted from directory [{}]'.format(file_name, directory_full_path)}
            return {'status': False, 'message': response['message']}

        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to remove file in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.remove_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def write_file_content(self, directory_full_path, file_name, content, over_write=False):
        """

        :param directory_full_path:  provide full directory path along with name
        :param file_name: file name with extension if possible
        :param content: content need to store in file
        :param over_write:  default False if accept True then overwrite file in directory if exist
        :return: True if created with content else false
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_directory_present(directory_full_path)
            if not response['status']:
                response = self.create_directory(directory_full_path)
                if not response['status']:
                    return {'status': False,
                            'message': 'Failed to created directory [{0}] [{1}]'.format(directory_full_path,
                                                                                        response['message'])}
            response = self.is_file_present(directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        "message": "File [{0}] is already present in directory [{1}]. try with over write option".format(
                            file_name, directory_full_path)}
            if response['status'] and over_write:
                response = self.remove_file(directory_full_path, file_name)
                if not response['status']:
                    return response

            blob_client = self.blob_service_client.get_blob_client(container=self.container_name,
                                                                   blob=directory_full_path + file_name)
            blob_client.upload_blob(dill.dumps(content))
            return {'status': True,
                    'message': 'File [{0}] is created in directory [{1}]'.format(file_name, directory_full_path)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.write_file_content.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def update_directory_full_path_string(self, directory_full_path):
        try:
            if directory_full_path == "" or directory_full_path == "/":
                directory_full_path = ""
            else:
                if directory_full_path[-1] != "/":
                    directory_full_path = directory_full_path + "/"
            return directory_full_path
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.update_directory_full_path_string.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def read_csv_file(self, directory_full_path, file_name):
        """

        :param directory_full_path:
        :param file_name:
        :return: {'status': True,
                    'message': 'File [{0}] has been read into data frame'.format(directory_full_path + file_name),
                    'data_frame': df}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name,
                                                                   blob=directory_full_path + file_name)
            df = pd.read_csv(StringIO(blob_client.download_blob().readall().decode()))
            return {'status': True,
                    'message': 'File [{0}] has been read into data frame'.format(directory_full_path + file_name),
                    'data_frame': df}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.update_directory_full_path_string.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e


    def read_json_file(self, directory_full_path, file_name):
        """

        :param directory_full_path:
        :param file_name:
        :return:  {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': content}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name,
                                                                   blob=directory_full_path + file_name)
            content = json.loads(blob_client.download_blob().readall())
            return {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': content}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.read_json_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e


    def read_file_content(self, directory_full_path, file_name):
        """

        :param directory_full_path:
        :param file_name:
        :return:  {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': content}
        """
        try:
            directory_full_path = self.update_directory_full_path_string(directory_full_path)
            response = self.is_file_present(directory_full_path, file_name)
            if not response['status']:
                return response
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name,
                                                                   blob=directory_full_path + file_name)
            content = dill.loads(blob_client.download_blob().readall())
            return {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': content}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.read_file_content.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def move_file(self, source_directory_full_path, target_directory_full_path, file_name, over_write=False,
                  container_name=None):
        """

        :param source_directory_full_path: provide source directory path along with name
        :param target_directory_full_path: provide target directory path along with name
        :param file_name: file need to move
        :param over_write:  default False if accept True then overwrite file in target directory if exist
        :return: True if file moved else false
        """
        try:
            response = self.copy_file(source_directory_full_path, target_directory_full_path, file_name, over_write,
                                      container_name)
            if not response['status']:
                return {'status': False, 'message': 'Failed to move file due to [{}]'.format(response['message'])}
            else:
                if container_name is None:
                    container_name = self.container_name
                self.remove_file(source_directory_full_path, file_name)
                return {'status': True,
                        'message': 'File moved successfully from container: [{0}] directory [{1}] to container:[{2}] '
                                   'directory[{3}]'.format(self.container_name,
                                                           source_directory_full_path + file_name, container_name,
                                                           target_directory_full_path + file_name)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.move_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e

    def copy_file(self, source_directory_full_path, target_directory_full_path, file_name, over_write=False,
                  container_name=None):
        """

        :param container_name: specify container name if you want to choose other container.
        :param source_directory_full_path: provide source directory path along with name
        :param target_directory_full_path: provide target directory path along with name
        :param file_name: file need to copy
        :param over_write: default False if accept True then overwrite file in target directory if exist
        :return: True if file copied else false
        """
        try:
            target_directory_full_path = self.update_directory_full_path_string(target_directory_full_path)
            source_directory_full_path = self.update_directory_full_path_string(source_directory_full_path)
            response = self.is_file_present(source_directory_full_path, file_name)
            if not response['status']:
                return {'status': False,
                        'message': 'Source file [{0}] is not present'.format(source_directory_full_path + file_name)}
            if container_name is None:
                container_name = self.container_name
                container_obj = self
            else:
                container_name = container_name
                container_obj = MicrosoftAzureBlobStorage(container_name=container_name)

            response = container_obj.is_directory_present(target_directory_full_path)

            if not response['status']:
                response = container_obj.create_directory(target_directory_full_path)
                if not response['status']:
                    return {'status': False,
                            'message': 'Failed to create'
                                       ' target directory [{}] in container:[{}]'.format(
                                target_directory_full_path,
                                container_name
                            )}
            response = container_obj.is_file_present(target_directory_full_path, file_name)
            if response['status'] and not over_write:
                return {'status': False,
                        'message': 'Container:[{0}] target directory '
                                   '[{1}] contains file [{2}] please'
                                   ' try with over write option.'.format(container_name,
                                                                         target_directory_full_path,
                                                                         file_name
                                                                         )}

            account_name = self.blob_service_client.account_name
            source_blob = r"https://{}.blob.core.windows.net/{}/{}{}" \
                .format(account_name, self.container_name, source_directory_full_path, file_name)

            copied_blob = container_obj.blob_service_client.get_blob_client(container_name,
                                                                            target_directory_full_path + file_name)
            copied_blob.start_copy_from_url(source_blob)

            return {'status': True,
                    'message': 'File copied successfully from bucket: [{0}] directory [{1}] to bucket:[{2}] '
                               'directory[{3}]'.format(self.container_name,
                                                       source_directory_full_path + file_name, container_name,
                                                       target_directory_full_path + file_name)}
        except Exception as e:
            azure_exception = MicrosoftAzureException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(MicrosoftAzureBlobStorage.__module__.__str__(), MicrosoftAzureBlobStorage.__name__,
                            self.copy_file.__name__))
            raise Exception(azure_exception.error_message_detail(str(e), sys)) from e
