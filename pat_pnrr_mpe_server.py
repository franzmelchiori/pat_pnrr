"""
    PAT MPE Server
    Francesco Melchiori, 2025
"""


import os
from flask import Flask

TEMPLATE_FOLDER = 'pat_pnrr_mpe_web_app/pages'
STATIC_FOLDER = 'pat_pnrr_mpe_web_app/static'
CERT_FOLDER = 'pat_pnrr_mpe_web_app/certs'

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
from pat_pnrr_mpe_web_app import *


if __name__ == '__main__':
    context = (CERT_FOLDER + '/' + 'localserver.crt',
               CERT_FOLDER + '/' + 'localserver.key')
    app.run(host='127.0.0.1', ssl_context=context)
