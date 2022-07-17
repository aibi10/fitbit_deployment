from exception_layer.generic_exception.generic_exception import GenericException as EmailSenderException
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
from project_library_layer.credentials import credential_data

import os
from email.message import EmailMessage
from exception_layer.generic_exception.generic_exception import GenericException  as EmailSenderException
import smtplib
from email.mime.text import MIMEText
import sys
from project_library_layer.credentials import credential_data


class EmailSender:

    def __init__(self):
        try:
            sender_credentials = credential_data.get_sender_email_id_credentials()
            self.__sender_email_id = sender_credentials.get('email_address', None)
            self.__passkey = sender_credentials.get('passkey', None)
            self.__receiver_email_id = credential_data.get_receiver_email_id_credentials()
        except Exception as e:
            email_sender_excep = EmailSenderException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, EmailSender.__name__,
                            self.__init__.__name__))
            raise Exception(email_sender_excep.error_message_detail(str(e), sys)) from e

    def send_email(self, mail_text, subject):
        """
        message: Message string in html format
        subject: subject of email
        """
        try:
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = self.__sender_email_id
            message["To"] = self.__receiver_email_id
            text = f"Hi recipient,\n\n This is notification email from Machine Learning Application.\n\n" \
                   f"Description: \n\n{mail_text} \n\n Thanks & Regards," \
                   f"\nAvnish Yadav"
            message.set_content(text)
            # Create secure connection with server and send email
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.__sender_email_id, self.__passkey)
                smtp.send_message(message)
        except Exception as e:
            email_sender_excep = EmailSenderException(
                "Failed during sending email module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, EmailSender.__name__,
                            self.send_email.__name__))
            raise Exception(email_sender_excep.error_message_detail(str(e), sys)) from e
