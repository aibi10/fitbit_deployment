from data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from exception_layer.generic_exception.generic_exception import GenericException as RegistrationException
from entity_layer.encryption.encrypt_confidential_data import EncryptData
from project_library_layer.datetime_libray.date_time import get_time, get_date
import sys
import re


class Register:
    def __init__(self):
        self.mongo_db = MongoDBOperation()
        self.database_name = "registration"
        self.collection_name_user = "user"
        self.collection_name_user_allow = "user_allowed"
        self.admin_email_id = "yadav.tara.avnish@gmail.com"
        self.collection_name_user_role = "user_role"
        self.n_attempt = 5

    def is_email_address_allowed(self, email_address):
        try:
            record = self.mongo_db.get_record(self.database_name, self.collection_name_user_allow,
                                              {'email_address': email_address})
            if record is None:
                return {'status': False, 'message': "Email address [{0}] is not allow !! please contact admin on email "
                                                    "id [{1}] ".format(email_address, self.admin_email_id)}
            return {'status': True, 'message': 'Email address can be used for registration.'}
        except Exception as e:
            registration_exception = RegistrationException("Failed email address validation in class [{0}] method [{1}]"
                                                           .format(Register.__name__,
                                                                   self.is_email_address_allowed.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def is_valid_email(self, email_address):
        try:
            regex = '^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$'
            if re.search(regex, email_address):
                return {'status': True, 'message': "Valid email address"}

            else:
                return {'status': False, 'message': "Invalid email address [{0}]".format(email_address)}

        except Exception as e:
            registration_exception = RegistrationException("Failed email address validation in class [{0}] method [{1}]"
                                                           .format(Register.__name__,
                                                                   self.is_valid_email.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def validate_user_detail(self, user_name, email_address, password, confirm_password):
        try:
            error_message = ""
            if password != confirm_password:
                error_message = "Password  and confirm password didn't matched"
            response = self.is_valid_email(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])
            response = self.is_email_address_allowed(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])
            response = self.is_email_address_used(email_address)
            if response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])
            if error_message.__len__() == 0:
                return {'status': True, 'message': "user detail validated successfully."}
            return {'status': False, 'message': error_message}
        except Exception as e:
            registration_exception = RegistrationException(
                "Failed user detail validation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.validate_user_detail.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def register_user(self, user_name, email_address, password, confirm_password):
        try:
            response = self.validate_user_detail(user_name, email_address, password, confirm_password)
            if not response['status']:
                return {'status': False, 'message': response['message']}
            encryptor = EncryptData()
            encrypted_password = encryptor.get_encrypted_text(password)
            self.mongo_db.insert_record_in_collection(self.database_name, self.collection_name_user,
                                                      {'user_name': user_name,
                                                       'email_address': email_address,
                                                       'password': encrypted_password,
                                                       'register_date': get_date(),
                                                       'register_time': get_time(),
                                                       'updated_time': get_time(),
                                                       'updated_date': get_date(),
                                                       'n_attempt': 0,
                                                       'is_locked': False
                                                       })
            return {'status': True, 'message': "Registration successful."}
        except Exception as e:
            registration_exception = RegistrationException(
                "Failed to save user detail database in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.register_user.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def is_email_address_used(self, email_address):
        try:
            user = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                            {'email_address': email_address})
            if user is None:
                return {'status': False, 'message': "Email address is not used {0}".format(email_address)}
            else:
                return {'status': True, 'message': "Email address is used {0}".format(email_address)}
        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.is_email_address_used.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def verify_user(self, email_address, password):
        try:
            user = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                            {'email_address': email_address})
            if user is None:
                return {'status': False, 'message': "Invalid email_address {0}".format(email_address)}
            else:
                n_attempt = int(user['n_attempt'])
                is_locked = bool(user['is_locked'])
                print(is_locked)
                if is_locked:
                    return {'status': False, 'message': 'Account locked contact admin emaild id:' + self.admin_email_id}

                encryptor = EncryptData()
                response = encryptor.verify_encrypted_text(password, user['password'])
                if response:
                    self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                              {'email_address': email_address},
                                                              {"n_attempt": 0, 'is_locked': False,
                                                               'updated_time': get_time(),
                                                               'updated_date': get_date()})
                    return {'status': response, 'message': 'Login successfully'}
                else:
                    n_attempt += 1
                    is_locked = False
                    if n_attempt == self.n_attempt:
                        is_locked = True
                    self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                              {'email_address': email_address},
                                                              {"n_attempt": n_attempt, 'is_locked': is_locked,
                                                               'updated_time': get_time(),
                                                               'updated_date': get_date()})
                    return {'status': False, 'message': 'Invalid password'}
        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.verify_user.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def reset_password(self, email_address, password, confirm_password):
        try:
            error_message = ""
            response = self.is_valid_email(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])
            response = self.mongo_db.get_record(self.database_name, self.collection_name_user,
                                                {'email_address': email_address})
            if response is None:
                error_message = "{0} {1}".format(error_message,
                                                 'No account exist with email address [{}]'.format(email_address))
            if password != confirm_password:
                error_message = "{0} {1}".format(error_message, "Password  and confirm password didn't matched")
            response = self.is_email_address_allowed(email_address)
            if not response['status']:
                error_message = "{0} {1}".format(error_message, response['message'])
            if error_message.__len__() == 0:
                encryptor = EncryptData()
                encrypted_password = encryptor.get_encrypted_text(password)
                self.mongo_db.update_record_in_collection(self.database_name, self.collection_name_user,
                                                          {'email_address': email_address},
                                                          {"password": encrypted_password, 'updated_time': get_time(),
                                                           'updated_date': get_date()})
                return {'status': True, 'message': "password updated successfully."}
            return {'status': False, 'message': error_message}
        except Exception as e:
            registration_exception = RegistrationException(
                "Login failed in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def add_user_role(self, role_name):
        try:
            records = self.mongo_db.get_record(self.database_name, self.collection_name_user_role,
                                               {'user_role': role_name})
            if records is None:
                user_role_id = self.mongo_db.get_max_value_of_column(self.database_name,
                                                                     self.collection_name_user_role,
                                                                     query={},
                                                                     column='user_role_id'
                                                                     )
                if user_role_id is None:
                    user_role_id = 1
                else:
                    user_role_id = user_role_id + 1
                record = {'user_role_id': user_role_id, 'user_role': role_name}
                result = self.mongo_db.insert_record_in_collection(
                    self.database_name,
                    self.collection_name_user_role,
                    record
                )
                if result > 0:
                    return {'status': True, 'message': 'User {} role added. '.format(role_name)}
            else:
                return {'status': False, 'message': 'User {} already present. '.format(role_name)}
        except Exception as e:
            registration_exception = RegistrationException(
                "Add user role in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e

    def validate_access(self, email_address, operation_type="WRITE"):
        try:
            admin = 'admin'
            viewer = 'viewer'
            RW = ['READ', 'WRITE']
            R = ['READ']
            return {'status': True, 'message': 'You  have all access for '.format(RW)}
            record = self.mongo_db.get_record(self.database_name, self.collection_name_user_allow,
                                              {'email_address': email_address})
            role_id = record['user_role_id']
            role = self.mongo_db.get_record(self.database_name, self.collection_name_user_role,
                                            {'user_role_id': role_id})
            role_name = role['user_role']
            if operation_type in RW and role_name == admin:
                return {'status': True, 'message': 'You  have all access for '.format(RW)}
            if operation_type in ['READ'] and role_name == viewer:
                return {'status': True, 'message': 'You have all access for'.format(R)}
            else:
                return {'status': False, 'message': 'You can not perform this action due to insufficient privilege '}
        except Exception as e:
            registration_exception = RegistrationException(
                "validating access in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Register.__name__,
                            self.reset_password.__name__))
            raise Exception(registration_exception.error_message_detail(str(e), sys)) from e
