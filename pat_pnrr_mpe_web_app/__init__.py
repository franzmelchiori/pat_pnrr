from flask import render_template, url_for, request
from markupsafe import escape, Markup

from pat_pnrr.pat_pnrr_mpe_server import app


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/progetto')
# def project():
#     return '<h1>project</h1>'

# @app.route('/mpe_comunale/<name_comune>', methods=['GET', 'POST'])
# def mpe_comunale(name_comune):
#     arg = request.args.get('arg')
#     if request.method == 'POST':
#         pass
#     return render_template(f'<h1>mpe del comune di {escape(name_comune)}<h1>', name_comune=name_comune, arg=arg)

# with app.test_request_context():
#     print(url_for('index'))
#     print(url_for('project'))
#     print(url_for('mpe_comunale', name_comune='trento'))
