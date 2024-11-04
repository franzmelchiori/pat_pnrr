import os
from flask import Flask

TEMPLATE_FOLDER = 'pat_pnrr_mpe_web_app/pages'

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
from pat_pnrr_mpe_web_app import *


if __name__ == '__main__':
    app.run()
