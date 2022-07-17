import os
import sys
from os import abort
from flask import render_template, redirect, url_for, jsonify, session

from flask_cors import cross_origin


class HomeController:
    def __init__(self):
        pass

    @cross_origin()
    def index(self):
        try:

            if 'email_address' not in session:
                return redirect(url_for('login'))

            return render_template("index.html",
                                   context={'message': None, 'process_value': None, 'message_status': 'info'})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            #print(exception_detail)
            return render_template('error.html',
                                   context={'message': None,'status ':False,'message_status': 'info', 'error_message': exception_detail.__str__()})