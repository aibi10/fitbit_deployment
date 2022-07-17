import io

from passlib.hash import pbkdf2_sha256
from exception_layer.generic_exception.generic_exception import GenericException as EncryptionException
import os,sys
from project_library_layer.datetime_libray import date_time

from cryptography.fernet import Fernet
import uuid
class EncryptData:
    def __init__(self):
        pass

    def get_encrypted_text(self,text):
        """
        This function will return hash calcualted on your data
        :param data:
        :return encrypted hash:
        """
        try:
            start_date=date_time.get_date()
            start_time=date_time.get_time()
            if text is not None:
                hash = pbkdf2_sha256.hash(text)
                return hash
            else:
                raise EncryptionException("To encrypt text. you must provide some text")
        except EncryptionException as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'start_date': start_date, 'start_time': start_time, 'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}

            raise e


    def verify_encrypted_text(self,text,encrypted_text):
        try:
            return pbkdf2_sha256.verify(text, encrypted_text)
        except Exception as e:
            pass



    def generate_key(self,):
        """
        Generates a key and save it into a file
        """
        key = Fernet.generate_key()
        #with open("secret.key", "wb") as key_file:
            #key_file.write(key)
        key=key.decode('utf-8')
        return key

    def load_key(self):
        """

        :return:
        """
        key = os.environ.get('SECRET_KEY_MONGO_DB', None)
        key=key.encode('utf-8')
        return key


    def encrypt_message(self,message,key=None):
        """
        Encrypts a message
        """

        encoded_message = message.encode()
        if key is None:
            key=self.load_key()
        #print(key)
        f = Fernet(key)
        encrypted_message = f.encrypt(encoded_message)

        #print(encrypted_message)
        return encrypted_message

    def decrypt_message(self,encrypted_message,key=None):
        """
        Decrypts an encrypted message
        """
        if key is None:
            key=self.load_key()
        f = Fernet(key)
        decrypted_message = f.decrypt(encrypted_message)
        return decrypted_message



