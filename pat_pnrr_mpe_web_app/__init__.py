import numpy as np
from flask import render_template, url_for, request
from markupsafe import escape, Markup

from pat_pnrr.pat_pnrr_mpe_server import app
from pat_pnrr.pat_pnrr_mpe import pat_pnrr_5a_misurazione
from pat_pnrr.pat_pnrr_mpe import pat_pnrr_6a_misurazione


@app.route('/')
def index():
    target = {
        'pdc_ov_durata': 109,
        'pdc_ov_arretrati': 324,
        'pds_durata': 130,
        'pds_arretrati': 300}
    btnradio_mpe = request.args.get('btnradio_mpe')
    if btnradio_mpe == 'btnradio_mpe_2024Q1_2':
        comuni_pdc_ov_measure, comuni_monitored = pat_pnrr_6a_misurazione.get_comuni_measure(
            pat_pnrr_6a_misurazione.comuni_excel_map, 'Permessi di Costruire',
            'pat_pnrr_6a_misurazione_tabelle_comunali\\')
        comuni_pds_measure, comuni_monitored = pat_pnrr_6a_misurazione.get_comuni_measure(
            pat_pnrr_6a_misurazione.comuni_excel_map, 'Prov di sanatoria',
            'pat_pnrr_6a_misurazione_tabelle_comunali\\')
        comuni_pdc_measure, comuni_monitored = pat_pnrr_6a_misurazione.get_comuni_measure(
            pat_pnrr_6a_misurazione.comuni_excel_map, 'Permessi di Costruire',
            'pat_pnrr_6a_misurazione_tabelle_comunali\\', type_pdc_ov=False)
        comuni_cila_measure, comuni_monitored = pat_pnrr_6a_misurazione.get_comuni_measure(
            pat_pnrr_6a_misurazione.comuni_excel_map, 'Controllo CILA',
            'pat_pnrr_6a_misurazione_tabelle_comunali\\')
        return render_template('index.html',
            pdc_ov = np.ceil(comuni_pdc_ov_measure.values).astype(int),
            pds = np.ceil(comuni_pds_measure.values).astype(int),
            pdc = np.ceil(comuni_pdc_measure.values).astype(int),
            cila = np.ceil(comuni_cila_measure.values).astype(int),
            comuni = comuni_monitored,
            target = target,
            btnradio_mpe = btnradio_mpe)
    elif btnradio_mpe == 'btnradio_mpe_2023Q3_4':
        comuni_pdc_ov_measure, comuni_monitored = pat_pnrr_5a_misurazione.get_comuni_measure(
            pat_pnrr_5a_misurazione.comuni_excel_map, 'Permessi di Costruire',
            'pat_pnrr_5a_misurazione_tabelle_comunali\\')
        comuni_pds_measure, comuni_monitored = pat_pnrr_5a_misurazione.get_comuni_measure(
            pat_pnrr_5a_misurazione.comuni_excel_map, 'Prov di sanatoria',
            'pat_pnrr_5a_misurazione_tabelle_comunali\\')
        comuni_pdc_measure, comuni_monitored = pat_pnrr_5a_misurazione.get_comuni_measure(
            pat_pnrr_5a_misurazione.comuni_excel_map, 'Permessi di Costruire',
            'pat_pnrr_5a_misurazione_tabelle_comunali\\', type_pdc_ov=False)
        comuni_cila_measure, comuni_monitored = pat_pnrr_5a_misurazione.get_comuni_measure(
            pat_pnrr_5a_misurazione.comuni_excel_map, 'Controllo CILA',
            'pat_pnrr_5a_misurazione_tabelle_comunali\\')
        return render_template('index.html',
            pdc_ov = np.ceil(comuni_pdc_ov_measure.values).astype(int),
            pds = np.ceil(comuni_pds_measure.values).astype(int),
            pdc = np.ceil(comuni_pdc_measure.values).astype(int),
            cila = np.ceil(comuni_cila_measure.values).astype(int),
            comuni = comuni_monitored,
            target = target,
            btnradio_mpe = btnradio_mpe)

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
