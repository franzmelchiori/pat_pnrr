from flask import render_template, request

from pat_pnrr.pat_pnrr_mpe_server import app


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/progetto')
# def project():
#     return '<h1>project</h1>'

# @app.route('/mpe_comunale/<name_comune>')
# def mpe_comunale(name_comune):
#     return f'<h1>mpe del comune di {name_comune}<h1>'
