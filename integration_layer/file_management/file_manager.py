from exception_layer.generic_exception.generic_exception import GenericException as FileManagerException
from cloud_storage_layer.microsoft_azure.azure_blob_storage import MicrosoftAzureBlobStorage
from cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from cloud_storage_layer.google.google_cloud_storage import GoogleCloudStorage
import sys, os


class FileManager:
    def __init__(self, cloud_provider):
        try:
            if cloud_provider == "google":
                self.manager_object = GoogleCloudStorage()
            if cloud_provider == "amazon":
                self.manager_object = AmazonSimpleStorageService()
            if cloud_provider == "microsoft":
                self.manager_object = MicrosoftAzureBlobStorage()
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failure occured during object creation in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            "__init__"))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def list_directory(self, directory_full_path=None):
        """
        :param directory_full_path:directory path
        :return {'status': True/False, 'message': 'message_detail',
                    , 'directory_list': directory_list}
        """
        try:
            return self.manager_object.list_directory(directory_full_path)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to list directory in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.list_directory.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def list_files(self, directory_full_path):
        """

        :param directory_full_path: directory
        :return:{'status': True/False, 'message': 'Directory [{0}]  present'.format(directory_full_path)
                    , 'files_list': File list will be available only if status is True}
        """
        try:
            return self.manager_object.list_files(directory_full_path)
        except Exception as e:
            file_exception = FileManagerException(
                "Failed to list files in object in module [{0}] "
                "class [{1}] method [{2}]".format(FileManager.__module__.__str__(),
                                                  FileManager.__name__,
                                                  self.list_files.__name__))
            raise Exception(file_exception.error_message_detail(str(e), sys)) from e

    def create_directory(self, directory_full_path, over_write=False):
        """

        :param directory_full_path: provide full directory path along with name
        :param over_write: default False if accept True then overwrite existing directory if exist
        :return True if created else false
        """
        try:
            return self.manager_object.create_directory(directory_full_path,over_write)
        except Exception as e:
            file_exception = FileManagerException(
                "Failed during directory creation in object in module [{0}] "
                "class [{1}] method [{2}]".format(FileManager.__module__.__str__(),
                                                  FileManager.__name__,
                                                  self.create_directory().__name__))
            raise Exception(file_exception.error_message_detail(str(e), sys)) from e

    def remove_directory(self, directory_full_path):
        """

        :param directory_full_path:provide full directory path along with name
        kindly provide "" or "/" to remove all directory and file from bucket.
        :return: True if removed else false
        """
        try:
            return self.manager_object.remove_directory(directory_full_path)
        except Exception as e:
            file_exception = FileManagerException(
                "Failed to remove directory in object in module [{0}] "
                "class [{1}] method [{2}]".format(FileManager.__module__.__str__(),
                                                  FileManager.__name__,
                                                  self.remove_directory.__name__))
            raise Exception(file_exception.error_message_detail(str(e), sys)) from e

    def is_file_present(self, directory_full_path, file_name):
        """

        :param directory_full_path: directory_full_path
        :param file_name: file_name
        :return {'status': True/False, 'message': 'message'}
        """
        try:
            return self.manager_object.is_file_present(directory_full_path,file_name)
        except Exception as e:
            file_exception = FileManagerException(
                "Failed in checking file in module [{0}] "
                "class [{1}] method [{2}]".format(FileManager.__module__.__str__(),
                                                  FileManager.__name__,
                                                  self.is_file_present().__name__))
            raise Exception(file_exception.error_message_detail(str(e), sys)) from e

    def is_directory_present(self, directory_full_path):
        try:
            return self.manager_object.is_directory_present(directory_full_path)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed during directory checking in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.is_file_present.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e


    def read_json_file(self,directory_full_path, file_name):
        try:
            return self.manager_object.read_json_file(directory_full_path,file_name)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed during directory checking in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.read_json_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

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
            return self.manager_object.upload_file(directory_full_path, file_name, stream_data,local_file_path, over_write)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to upload file in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.upload_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def download_file(self, directory_full_path, file_name, local_system_directory=""):
        try:
            return self.manager_object.download_file(directory_full_path, file_name, local_system_directory)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to download file in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.download_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def remove_file(self, directory_full_path, file_name):
        """
        :param directory_full_path: provide full directory path along with name
        :param file_name: file name with extension if possible
        :return: True if removed else false
        """
        try:
            return self.manager_object.remove_file(directory_full_path, file_name)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to remove file in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.remove_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def write_file_content(self, directory_full_path, file_name, content, over_write=False):
        """

        :param directory_full_path:  provide full directory path along with name
        :param file_name: file name with extension if possible
        :param content: content need to store in file
        :param over_write:  default False if accept True then overwrite file in directory if exist
        :return: True if created with content else false
        """
        try:
            return self.manager_object.write_file_content(directory_full_path, file_name, content, over_write)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.write_file_content.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e


    def read_csv_file(self, directory_full_path, file_name):
        try:
            return self.manager_object.read_csv_file(directory_full_path, file_name)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.read_csv_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

    def read_file_content(self, directory_full_path, file_name):
        """

        :param directory_full_path:
        :param file_name:
        :return:  {'status': True, 'message': 'File [{0}] has been read'.format(directory_full_path + file_name),
                    'file_content': content}
        """
        try:
            return self.manager_object.read_file_content(directory_full_path,file_name)
        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.read_file_content.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

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
            return self.manager_object.move_file( source_directory_full_path, target_directory_full_path, file_name, over_write,
                  container_name)

        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.move_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

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
            return self.manager_object.copy_file(source_directory_full_path, target_directory_full_path, file_name, over_write,
                  container_name)

        except Exception as e:
            file_manager_exception = FileManagerException(
                "Failed to create file with content in module [{0}] class [{1}] method [{2}]"
                    .format(FileManager.__module__.__str__(), FileManager.__name__,
                            self.copy_file.__name__))
            raise Exception(file_manager_exception.error_message_detail(str(e), sys)) from e

