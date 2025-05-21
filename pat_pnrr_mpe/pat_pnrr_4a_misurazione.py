"""
    PAT-PNRR 4a misurazione (December 2023)
    Monitoraggio Procedimenti Edilizi v2.x
    Francesco Melchiori, 2023
"""


import os
import shelve
# import datetime
import difflib

import numpy as np
import pandas as pd

from .pat_pnrr_comuni_excel_mapping import *
from . import pat_pnrr_3a_misurazione as pat_pnrr_3a


def get_dataframe_excel(path_file_excel, sheet_name, names, usecols, skiprows, droprows,
                        nrows=None, dtype=None, parse_dates=False):

    dataframe_excel = pd.read_excel(path_file_excel, sheet_name=sheet_name, names=names,
                                    usecols=usecols, skiprows=skiprows, nrows=nrows, dtype=dtype,
                                    parse_dates=parse_dates, header=None)

    for row in droprows:
        dataframe_excel.drop(dataframe_excel.index[dataframe_excel[row].isna()], inplace=True)
    dataframe_excel.drop_duplicates(inplace=True)
    dataframe_excel.reset_index(drop=True, inplace=True)

    return dataframe_excel


def get_list_excel(path_base=None, missing=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'

    list_xls = []
    for file in os.listdir(path_base + 'pat_pnrr_mpe\\pat_pnrr_4a_misurazione_tabelle_comunali\\'):
        if file.find('xls') != -1:
            list_xls.append(file)

    if missing:
        list_excel = [comune[0] for comune in comuni_excel_map
                      if comune[2] is None]
    else:
        list_excel = [(comune[2], comune[0]) for comune in comuni_excel_map
                      if comune[2] is not None]

    return list_excel, list_xls


def check_comuni_excel():
    list_excel, list_xls = get_list_excel()
    for file, comune_name in list_excel:
        print('controllo il file excel del comune di {0}'.format(comune_name))
        comune_excel = ComuneExcel(file, comune_name)
        comune_excel.check_headers_excel()
        comune_excel.check_dataframes_excel()

    return True


def get_comuni_dataframe(comuni_excel_map, sheet_name, load=True, path_base=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'
    path_shelve = path_base + 'pat_pnrr_mpe\\pat_pnrr_4a_misurazione_tabelle_comunali\\'

    sheet_suffix = ''
    if sheet_name == 'Permessi di Costruire':
        sheet_suffix += '_pdc'
    if sheet_name == 'Prov di sanatoria':
        sheet_suffix += '_pds'
    if sheet_name == 'Controllo CILA':
        sheet_suffix += '_cila'

    if load:
        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + sheet_suffix)
        comuni_dataframe = comuni_dataframe_shelve['comuni_dataframe']
        comuni_dataframe_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[2])
                            for comune in comuni_excel_map if comune[2] is not None]
        comuni_dataframe = []
        for comune_name, path_file in comuni_excel_map:
            print(comune_name + ' | ' + sheet_name)
            comune = ComuneExcel(path_file, comune_name)
            comune_dataframe = comune.get_comune_dataframe(sheet_name)
            comuni_dataframe.append(comune_dataframe)
        comuni_dataframe = pd.concat(comuni_dataframe, axis='rows', join='outer')
        comuni_dataframe.reset_index(drop=True, inplace=True)

        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + sheet_suffix)
        comuni_dataframe_shelve['comuni_dataframe'] = comuni_dataframe
        comuni_dataframe_shelve.close()

    return comuni_dataframe


def get_comuni_organico(comuni_excel_map, load=True, path_base=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'
    path_shelve = path_base + 'pat_pnrr_mpe\\pat_pnrr_4a_misurazione_tabelle_comunali\\'

    sheet_name = 'ISTRUZIONI'
    sheet_suffix = '_ist'

    if load:
        comuni_organico_shelve = shelve.open(path_shelve + 'comuni_organico' + sheet_suffix)
        comuni_organico = comuni_organico_shelve['comuni_organico']
        comuni_organico_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[2])
                            for comune in comuni_excel_map if comune[2] is not None]
        comuni_organico = []
        comuni_name = []
        for comune_name, path_file in comuni_excel_map:
            print(comune_name + ' | ' + sheet_name)
            comune = ComuneExcel(path_file, comune_name)

            names = comune.excel_structure[sheet_name]['column_labels']
            usecols = comune.excel_structure[sheet_name]['column_indexes']
            skiprows = comune.excel_structure[sheet_name]['row_skips']
            droprows = comune.excel_structure[sheet_name]['column_mandatory']
            nrows = 4

            comune_organico = get_dataframe_excel(
                comune.excel_path, sheet_name, names, usecols, skiprows, droprows, nrows)

            comuni_organico.append(comune_organico.T)
            comuni_name.append(comune_name)
        comuni_organico = pd.concat(comuni_organico, axis=0, join='outer')
        comuni_organico.reset_index(drop=True, inplace=True)

        comuni_organico.columns = [
            'nome_responsabile',
            'numero_tecnici',
            'numero_amministrativi',
            'gestione_associata'
        ]
        comuni_organico.index = comuni_name

        comuni_organico_shelve = shelve.open(path_shelve + 'comuni_organico' + sheet_suffix)
        comuni_organico_shelve['comuni_organico'] = comuni_organico
        comuni_organico_shelve.close()

    return comuni_organico


def get_comuni_measure_dataframe(comuni_excel_map, sheet_name, type_name=False, type_pdc_ov=True,
                                 load=True, path_base=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'
    path_shelve = path_base + 'pat_pnrr_mpe\\pat_pnrr_4a_misurazione_tabelle_comunali\\'

    sheet_suffix = ''
    if sheet_name == 'Permessi di Costruire':
        if type_pdc_ov:
            sheet_suffix += '_pdc_ov'
        else:
            sheet_suffix += '_pdc'
    if sheet_name == 'Prov di sanatoria':
        sheet_suffix += '_pds'
    if sheet_name == 'Controllo CILA':
        sheet_suffix += '_cila'

    if load:
        comuni_measure_dataframe_shelve = shelve.open(
            path_shelve + 'comuni_measure_dataframe' + sheet_suffix)
        comuni_measure_dataframe = comuni_measure_dataframe_shelve['comuni_measure_dataframe']
        comuni_measure_dataframe_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[2])
                            for comune in comuni_excel_map if comune[2] is not None]
        comuni_names = [comune[0] for comune in comuni_excel_map]
        comuni_measure_dataframe = []
        for comune_name, path_file in comuni_excel_map:
            message = comune_name + ' | ' + sheet_name
            if type_name:
                message += ' | ' + type_name
            if sheet_name == 'Permessi di Costruire':
                if type_pdc_ov:
                    message += ' | ' + 'Ordinari e in Variante'
            print(message)
            comune = ComuneExcel(path_file, comune_name)
            comune_measure_series = comune.get_comune_measure_series(sheet_name, type_name,
                                                                     type_pdc_ov)
            comuni_measure_dataframe.append(comune_measure_series)
        comuni_measure_dataframe = pd.DataFrame(comuni_measure_dataframe, index=comuni_names)

        comuni_measure_dataframe_shelve = shelve.open(
            path_shelve + 'comuni_measure_dataframe' + sheet_suffix)
        comuni_measure_dataframe_shelve['comuni_measure_dataframe'] = comuni_measure_dataframe
        comuni_measure_dataframe_shelve.close()

    return comuni_measure_dataframe


def get_comuni_dataframes(comuni_excel_map, load=True):

    get_comuni_dataframe(comuni_excel_map, 'Permessi di Costruire', load=load)
    get_comuni_dataframe(comuni_excel_map, 'Prov di sanatoria', load=load)
    get_comuni_dataframe(comuni_excel_map, 'Controllo CILA', load=load)

    return True


def get_comuni_measures_dataframe(comuni_excel_map, load=True):

    comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(
        comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=True, load=load)
    comuni_measure_dataframe_pdc = get_comuni_measure_dataframe(
        comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=False, load=load)
    comuni_measure_dataframe_pds = get_comuni_measure_dataframe(
        comuni_excel_map, 'Prov di sanatoria', load=load)
    comuni_measure_dataframe_cila = get_comuni_measure_dataframe(
        comuni_excel_map, 'Controllo CILA', load=load)

    comuni_measures_dataframe = pd.concat(
        [comuni_measure_dataframe_pdc_ov,
         comuni_measure_dataframe_pdc,
         comuni_measure_dataframe_pds,
         comuni_measure_dataframe_cila],
        axis='columns', join='outer')

    return comuni_measures_dataframe


def get_comuni_measure(comuni_excel_map, sheet_name, type_name=False, type_pdc_ov=True,
                       measure_period='2023q1-2', load=True):

    comuni_measure_dataframe = get_comuni_measure_dataframe(
        comuni_excel_map=comuni_excel_map, sheet_name=sheet_name, type_name=type_name,
        type_pdc_ov=type_pdc_ov, load=load)

    if sheet_name == 'Permessi di Costruire':
        if type_pdc_ov:
            measure_labels = [
                'numero_permessi_costruire_ov_conclusi_con_silenzio-assenso',
                'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso',
                'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_con_sospensioni',
                'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
                'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso',
                'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati',
                'numero_permessi_costruire_ov_avviati',
                'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo']
        else:
            measure_labels = [
                'numero_permessi_costruire_conclusi_con_silenzio-assenso',
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso',
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni',
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
                'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso',
                'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati',
                'numero_permessi_costruire_avviati',
                'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo']
    elif sheet_name == 'Prov di sanatoria':
        measure_labels = [
            'numero_sanatorie_concluse_con_silenzio-assenso',
            'numero_sanatorie_concluse_con_provvedimento_espresso',
            'numero_sanatorie_concluse_con_provvedimento_espresso_con_sospensioni',
            'numero_sanatorie_concluse_con_provvedimento_espresso_con_conferenza_servizi',
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso',
            'giornate_durata_mediana_termine_massimo_sanatorie_avviate',
            'numero_sanatorie_avviate',
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo']
    elif sheet_name == 'Controllo CILA':
        measure_labels = [
            'numero_controlli_cila_conclusi_con_silenzio-assenso',
            'numero_controlli_cila_conclusi_con_provvedimento_espresso',
            'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_sospensioni',
            'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
            'giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso',
            'giornate_durata_mediana_termine_massimo_controlli_cila_avviati',
            'numero_controlli_cila_avviati',
            'numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo']
    else:
        measure_labels = [
            'numero_pratiche_concluse_con_silenzio-assenso',
            'numero_pratiche_concluse_con_provvedimento_espresso',
            'numero_pratiche_concluse_con_provvedimento_espresso_con_sospensioni',
            'numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi',
            'giornate_durata_media_pratiche_concluse_con_provvedimento_espresso',
            'giornate_durata_mediana_termine_massimo_pratiche_avviate',
            'numero_pratiche_avviate',
            'numero_pratiche_arretrate_non_concluse_scaduto_termine_massimo']
    measure_labels = [label + '_' + measure_period for label in measure_labels]

    comuni_measure = []
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[0]].sum())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[1]].sum())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[2]].sum())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[3]].sum())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[4]].mean())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[5]].median())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[6]].sum())
    comuni_measure.append(comuni_measure_dataframe.loc[:, measure_labels[7]].sum())

    comuni_measure = pd.Series(comuni_measure, index=measure_labels)
    comuni_monitored = comuni_measure_dataframe.__len__()
    message = 'Misurazione | ' + sheet_name
    if type_name:
        message += ' | ' + type_name
    if sheet_name == 'Permessi di Costruire':
        if type_pdc_ov:
            message += ' | ' + 'Ordinari e in Variante'
    print()
    print(message)
    print(comuni_measure)
    print('{0}/166 comuni'.format(comuni_measure_dataframe.__len__()))
    print()

    return comuni_measure, comuni_monitored


def get_comuni_measures(comuni_excel_map, save_tex=False, temp_tex=False):

    comuni_pdc_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=False)
    comuni_pdc_ov_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire')
    comuni_pds_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Prov di sanatoria')
    comuni_cila_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Controllo CILA')

    if save_tex:
        measurement_04_pratiche_header = [
            'Concluse con SA',
            'Concluse',
            'con sospensioni',
            'con CdS',
            'Durata [gg]',
            'Termine [gg]',
            'Avviate',
            'Arretrate']
        tex_file_header = measurement_04_pratiche_header

        comuni_pdc_measure.index = measurement_04_pratiche_header
        comuni_pdc_ov_measure.index = measurement_04_pratiche_header
        comuni_pds_measure.index = measurement_04_pratiche_header
        comuni_cila_measure.index = measurement_04_pratiche_header

        measurement_04_series = {
            'Permesso di Costruire OV': comuni_pdc_ov_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int)}
        measurement_04 = pd.DataFrame(measurement_04_series).T

        measurement_04b_series = {
            'Permesso di Costruire': comuni_pdc_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int),
            'Controllo della CILA': comuni_cila_measure.apply(np.rint).astype(int)}
        measurement_04b = pd.DataFrame(measurement_04b_series).T

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_2023q1-2.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_04.columns = tex_file_header
            baseline_styler = measurement_04.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            caption = 'PAT-PNRR | Procedimenti Edilizi | Misurazione 2023Q1-2'
            if temp_tex:
                caption += ' | PARZIALE'
            table_tex_content = baseline_styler.to_latex(
                caption=caption, label='pat-pnrr_mpe_2023q1-2', position='!htbp',
                position_float="centering", hrules=True)
            table_tex_file.write(table_tex_content)

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_2023q1-2b.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_04b.columns = tex_file_header
            baseline_styler = measurement_04b.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            caption='PAT-PNRR | Procedimenti Edilizi | Misurazione 2023Q1-2b'
            if temp_tex:
                caption += ' | PARZIALE'
            table_tex_content = baseline_styler.to_latex(
                caption=caption, label='pat-pnrr_mpe_2023q1-2b', position='!htbp',
                position_float="centering", hrules=True)
            table_tex_file.write(table_tex_content)

    return True


class ComuneExcel:

    def __init__(self, path_file, comune_name='Test', path_base=''):
        if path_base == '':
            self.path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\' \
                             'pat_pnrr_mpe\\pat_pnrr_4a_misurazione_tabelle_comunali\\'
        else:
            self.path_base = path_base
        self.path_file = path_file
        self.excel_path = self.path_base + self.path_file
        self.comune_name = comune_name
        self.excel_structure = {
            'ISTRUZIONI': {
                'column_labels': [
                    'organico'  # string | object
                ],
                'column_indexes': [
                    2
                ],
                'row_skips': 4,
                'column_mandatory': [
                ],
                'health_header_checks': [
                ],
                'health_na_content_checks': [
                ]
            },
            'Permessi di Costruire': {
                'column_labels': [
                    'tipologia_pratica',  # string | object
                    'id_pratica',  # string | object
                    'data_inizio_pratica',  # date dd/mm/yyyy | timestamp
                    'giorni_termine_normativo',  # integer | float
                    'data_fine_pratica',  # date dd/mm/yyyy
                    'data_fine_pratica_silenzio-assenso',  # date dd/mm/yyyy | timestamp
                    'conferenza_servizi',  # string | object
                    'tipologia_massima_sospensione',  # string | object
                    'giorni_sospensioni'  # integer | float
                ],
                'column_indexes': [
                    1, 2, 3, 4, 5, 6, 7, 8, 9
                ],
                'row_skips': 5,
                'column_mandatory': [
                    'id_pratica',
                    'data_inizio_pratica'
                ],
                'health_header_checks': [
                    'tipologia',
                    'id',
                    'presentazione',
                    'termine',
                    'conclusione',
                    'silenzio',
                    'conferenza',
                    'sospensioni'
                ],
                'health_na_content_checks': [
                    'tipologia_pratica'  # ,
                    # 'giorni_termine_normativo'
                ]
            },
            'Prov di sanatoria': {
                'column_labels': [
                    'tipologia_pratica',
                    'id_pratica',
                    'data_inizio_pratica',
                    'giorni_termine_normativo',
                    'data_fine_pratica',
                    'conferenza_servizi',
                    'tipologia_massima_sospensione',
                    'giorni_sospensioni'
                ],
                'column_indexes': [
                    1, 2, 3, 4, 5, 6, 7, 8
                ],
                'row_skips': 5,
                'column_mandatory': [
                    'id_pratica',
                    'data_inizio_pratica'
                ],
                'health_header_checks': [
                    'tipologia',
                    'id',
                    'presentazione',
                    'termine',
                    'conclusione',
                    'conferenza',
                    'sospensioni'
                ],
                'health_na_content_checks': [
                    'tipologia_pratica'  # ,
                    # 'giorni_termine_normativo'
                ]
            },
            'Controllo CILA': {
                'column_labels': [
                    'tipologia_pratica',
                    'id_pratica',
                    'data_inizio_pratica',
                    'giorni_termine_normativo',
                    'data_fine_pratica',
                    'tipologia_massima_sospensione',
                    'giorni_sospensioni'
                ],
                'column_indexes': [
                    1, 2, 3, 4, 5, 6, 7
                ],
                'row_skips': 5,
                'column_mandatory': [
                    'id_pratica',
                    'data_inizio_pratica'
                ],
                'health_header_checks': [
                    'tipologia',
                    'id',
                    'inizio',
                    'termine',
                    'conclusione',
                    'sospensione'
                ],
                'health_na_content_checks': [
                    'tipologia_pratica'  # ,
                    # 'giorni_termine_normativo'
                ]
            }
        }

    def check_headers_excel(self):
        sheet_names = list(self.excel_structure.keys())

        for sheet_name in sheet_names:
            names = self.excel_structure[sheet_name]['column_labels']
            usecols = self.excel_structure[sheet_name]['column_indexes']
            skiprows = self.excel_structure[sheet_name]['row_skips']-2
            droprows = self.excel_structure[sheet_name]['column_mandatory']
            nrows = 1
            health_header_checks = self.excel_structure[sheet_name]['health_header_checks']

            header_excel = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                               skiprows, droprows, nrows)

            for (column_index, health_header_check) in enumerate(health_header_checks):
                if not header_excel.iloc[0].values[column_index].casefold().find(
                        health_header_check) >= 0:
                    print('[!] excel header health check')
                    print('    ' + 'in the file [' + self.path_file + ']')
                    print('    ' + '    ' + 'in the sheet [' + sheet_name + ']')
                    print('    ' + '    ' + '    ' + 'the header column [' + health_header_check +
                          '] was NOT FOUND')

        return True

    def check_dataframes_excel(self):
        sheet_names = list(self.excel_structure.keys())

        for sheet_name in sheet_names:
            names = self.excel_structure[sheet_name]['column_labels']
            usecols = self.excel_structure[sheet_name]['column_indexes']
            skiprows = self.excel_structure[sheet_name]['row_skips']
            droprows = self.excel_structure[sheet_name]['column_mandatory']
            health_na_content_checks = self.excel_structure[sheet_name]['health_na_content_checks']

            dataframe_excel = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                                  skiprows, droprows, dtype=None,
                                                  parse_dates=False)

            for health_na_content_check in health_na_content_checks:
                if sum(dataframe_excel[health_na_content_check].isna()) > 0:
                    print('{0} assente nel foglio {1} del .csv di {2}'.format(
                        health_na_content_check, sheet_name, self.comune_name))
                    # print('[!] excel na content health check')
                    # print('    ' + 'in the file [' + self.path_file + ']')
                    # print('    ' + '    ' + 'in the sheet [' + sheet_name + ']')
                    # print('    ' + '    ' + '    ' + 'the column [' + health_na_content_check +
                    #       '] has some NA')

        return True

    def get_comune_dataframe(self, sheet_name):
        names = self.excel_structure[sheet_name]['column_labels']
        usecols = self.excel_structure[sheet_name]['column_indexes']
        skiprows = self.excel_structure[sheet_name]['row_skips']
        droprows = self.excel_structure[sheet_name]['column_mandatory']

        comune_dataframe = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                              skiprows, droprows, dtype=None)

        comune_dataframe.insert(0, 'comune', self.comune_name)
        comune_dataframe.dropna(axis=0, subset='id_pratica', inplace=True, ignore_index=True)
        comune_dataframe.dropna(axis=0, subset='data_inizio_pratica', inplace=True,
                                ignore_index=True)

        if sheet_name == 'Permessi di Costruire':
            comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_pdc[0], inplace=True)
            comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(60, inplace=True)
            comune_dataframe.loc[:, 'conferenza_servizi'].fillna('NO', inplace=True)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            if comune_dataframe.loc[:, 'data_inizio_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('06/04/022', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '06/04/2022'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('23/0523', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '23/05/2023'
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)

            if comune_dataframe.loc[:, 'data_fine_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('in attesa', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('In lavorazione', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] == ' '
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('non concluso', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('20/07/203', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '20/07/2023'
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-12-31 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.loc[change_mask, 'data_fine_pratica'] = pd.NaT

            if comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].dtype.str[1] in \
                    ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].astype(
                    'string').str.contains('NO', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica_silenzio-assenso'] = None
            try:
                comune_dataframe['data_fine_pratica_silenzio-assenso'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica_silenzio-assenso'],
                    errors='raise')
            except:
                print('data_fine_pratica_silenzio-assenso is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'])

            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60-90', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60 gg', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('-', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['i', 'f']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'] == 0
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
            try:
                comune_dataframe['giorni_termine_normativo'] = pd.to_numeric(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_termine_normativo'] = pd.to_timedelta(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='coerce', unit='D')
            except:
                print('giorni_termine_normativo is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_termine_normativo'])

            try:
                comune_dataframe['giorni_sospensioni'] = pd.to_numeric(
                    comune_dataframe['giorni_sospensioni'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_sospensioni'] = pd.to_timedelta(
                    comune_dataframe['giorni_sospensioni'],
                    errors='coerce', unit='D')
            except:
                print('giorni_sospensioni is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_sospensioni'])

            for index in comune_dataframe.index:
                tipologia_pratica_originale = comune_dataframe.loc[index, 'tipologia_pratica']
                tipologia_pratica = tipologia_pratica_originale.lower()
                tipologia_pratica = tipologia_pratica.replace('permesso di costruire', '')
                tipologia_pratica = tipologia_pratica.lstrip(' ').rstrip(' ')
                if tipologia_pratica == '':
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_pdc[0]
                else:
                    close_match = difflib.get_close_matches(tipologia_pratica, types_pdc)
                    if len(close_match) > 0:
                        comune_dataframe.loc[index, 'tipologia_pratica'] = close_match[0]
                    elif 'costruire' in tipologia_pratica_originale.lower():
                        comune_dataframe.loc[index, 'tipologia_pratica'] = types_pdc[0]
                    elif 'sanatoria' in tipologia_pratica_originale.lower():
                        comune_dataframe.drop(index, inplace=True)
                    elif 'parere' in tipologia_pratica_originale.lower():
                        comune_dataframe.drop(index, inplace=True)
                    else:
                        print('tipologia_pratica is UNKNOWN: ')
                        print(comune_dataframe.loc[index, 'tipologia_pratica'])

            for index in comune_dataframe.index:
                conferenza_servizi_originale = comune_dataframe.loc[index, 'conferenza_servizi']
                conferenza_servizi = conferenza_servizi_originale.lower()
                conferenza_servizi = conferenza_servizi.replace("'", '')
                conferenza_servizi = conferenza_servizi.lstrip(' ').rstrip(' ')
                if conferenza_servizi == 'no':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = False
                elif conferenza_servizi == 'si':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = True
                elif conferenza_servizi == 'sino':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = False
                else:
                    print('conferenza_servizi is UNKNOWN: ')
                    print(comune_dataframe.loc[index, 'conferenza_servizi'])

        if sheet_name == 'Prov di sanatoria':
            comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_pds[0], inplace=True)
            comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(60, inplace=True)
            comune_dataframe.loc[:, 'conferenza_servizi'].fillna('NO', inplace=True)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            if comune_dataframe.loc[:, 'data_inizio_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('26/05/23', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '26/05/2023'
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)

            if comune_dataframe.loc[:, 'data_fine_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('In lavorazione', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] == ' '
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('non concluso', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('02/02/223', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '02/02/2023'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('20/07/203', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '20/07/2023'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('09/10(2023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '09/10/2023'
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-12-31 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.loc[change_mask, 'data_fine_pratica'] = pd.NaT

            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60-90', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60 gg', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('senza termine', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 0
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('-', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 0
            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['i', 'f']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'] == 0
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
            try:
                comune_dataframe['giorni_termine_normativo'] = pd.to_numeric(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_termine_normativo'] = pd.to_timedelta(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='coerce', unit='D')
            except:
                print('giorni_termine_normativo is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_termine_normativo'])
            change_mask = comune_dataframe.loc[:, 'tipologia_pratica'] == 'Regolarizzazione'
            comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = pd.NaT

            try:
                comune_dataframe['giorni_sospensioni'] = pd.to_numeric(
                    comune_dataframe['giorni_sospensioni'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_sospensioni'] = pd.to_timedelta(
                    comune_dataframe['giorni_sospensioni'],
                    unit='D')
            except:
                print('giorni_sospensioni is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_sospensioni'])

            for index in comune_dataframe.index:
                tipologia_pratica_originale = comune_dataframe.loc[index, 'tipologia_pratica']
                tipologia_pratica = tipologia_pratica_originale.lower()
                tipologia_pratica = tipologia_pratica.lstrip(' ').rstrip(' ')
                tipologia_pratica = difflib.get_close_matches(tipologia_pratica, types_pds)
                if len(tipologia_pratica) > 0:
                    comune_dataframe.loc[index, 'tipologia_pratica'] = tipologia_pratica[0]
                elif 'sanatoria' in tipologia_pratica_originale.lower():
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_pds[0]
                elif 'santoria' in tipologia_pratica_originale.lower():
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_pds[0]
                else:
                    print('    ' + 'tipologia_pratica is UNKNOWN: ')
                    print(comune_dataframe.loc[index, 'tipologia_pratica'])

            for index in comune_dataframe.index:
                conferenza_servizi_originale = comune_dataframe.loc[index, 'conferenza_servizi']
                conferenza_servizi = conferenza_servizi_originale.lower()
                conferenza_servizi = conferenza_servizi.replace("'", '')
                conferenza_servizi = conferenza_servizi.lstrip(' ').rstrip(' ')
                if conferenza_servizi == 'no':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = False
                elif conferenza_servizi == 'si':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = True
                elif conferenza_servizi == 'sino':
                    comune_dataframe.loc[index, 'conferenza_servizi'] = False
                else:
                    print('conferenza_servizi is UNKNOWN: ')
                    print(comune_dataframe.loc[index, 'conferenza_servizi'])

        if sheet_name == 'Controllo CILA':
            comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_cila[0], inplace=True)
            comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(0, inplace=True)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            if comune_dataframe.loc[:, 'data_inizio_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('17/03/203', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '17/03/2023'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('30/03/203', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '30/03/2023'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('18/5//2023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '18/05/2023'
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)

            if comune_dataframe.loc[:, 'data_fine_pratica'].dtype == 'O':
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] == ' '
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('non concluso', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('08/092023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '08/09/2023'
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-12-31 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2023-06-30 23:59:59.999')
            comune_dataframe.loc[change_mask, 'data_fine_pratica'] = pd.NaT

            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.match('0', case=False, na=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = None
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('nessuno', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = None
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('00:00:00', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = None
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.match('-', case=False, na=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = None
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('\d{1,4}(\/|\-)\d{1,4}((\/|\-)\d{1,4})?',
                                           case=False, na=False, regex=True)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 30
            if comune_dataframe.loc[:, 'giorni_termine_normativo'].dtype.str[1] in ['i', 'f']:
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'] == 0
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = None
            try:
                comune_dataframe['giorni_termine_normativo'] = pd.to_numeric(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_termine_normativo'] = pd.to_timedelta(
                    comune_dataframe['giorni_termine_normativo'],
                    errors='coerce', unit='D')
            except:
                print('giorni_termine_normativo is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_termine_normativo'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] < \
                          pd.Timestamp('2022-09-01 00:00:00.000')
            comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = pd.NaT

            try:
                comune_dataframe['giorni_sospensioni'] = pd.to_numeric(
                    comune_dataframe['giorni_sospensioni'],
                    errors='raise', downcast='integer')
                comune_dataframe['giorni_sospensioni'] = pd.to_timedelta(
                    comune_dataframe['giorni_sospensioni'],
                    unit='D')
            except:
                print('giorni_sospensioni is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'giorni_sospensioni'])

            for index in comune_dataframe.index:
                tipologia_pratica_originale = comune_dataframe.loc[index, 'tipologia_pratica']
                tipologia_pratica = tipologia_pratica_originale.lower()
                tipologia_pratica = tipologia_pratica.lstrip(' ').rstrip(' ')
                tipologia_pratica = difflib.get_close_matches(tipologia_pratica, types_cila)
                if len(tipologia_pratica) > 0:
                    comune_dataframe.loc[index, 'tipologia_pratica'] = tipologia_pratica[0]
                elif 'cila' in tipologia_pratica_originale.lower():
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_cila[0]
                elif 'variante' in tipologia_pratica_originale.lower():
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_cila[0]
                elif 'comunicazione' in tipologia_pratica_originale.lower():
                    comune_dataframe.loc[index, 'tipologia_pratica'] = types_cila[0]
                else:
                    print('    ' + 'tipologia_pratica is UNKNOWN: ')
                    print(comune_dataframe.loc[index, 'tipologia_pratica'])

        return comune_dataframe

    def get_comune_measure_series(self, sheet_name, type_name=False, type_pdc_ov=True,
                                  measure_period='2023q1-2'):
        comune_dataframe = self.get_comune_dataframe(sheet_name=sheet_name)

        if sheet_name == 'Permessi di Costruire':
            if type_pdc_ov:
                measure_labels = [
                    'numero_permessi_costruire_ov_conclusi_con_silenzio-assenso',
                    'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso',
                    'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_con_sospensioni',
                    'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
                    'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso',
                    'giornate_durata_media_netta_permessi_costruire_ov_conclusi_con_provvedimento_espresso',
                    'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati',
                    'numero_permessi_costruire_ov_avviati',
                    'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo']
            else:
                measure_labels = [
                    'numero_permessi_costruire_conclusi_con_silenzio-assenso',
                    'numero_permessi_costruire_conclusi_con_provvedimento_espresso',
                    'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni',
                    'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
                    'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso',
                    'giornate_durata_media_netta_permessi_costruire_conclusi_con_provvedimento_espresso',
                    'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati',
                    'numero_permessi_costruire_avviati',
                    'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo']
        elif sheet_name == 'Prov di sanatoria':
            measure_labels = [
                'numero_sanatorie_concluse_con_silenzio-assenso',
                'numero_sanatorie_concluse_con_provvedimento_espresso',
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_sospensioni',
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_conferenza_servizi',
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso',
                'giornate_durata_media_netta_sanatorie_concluse_con_provvedimento_espresso',
                'giornate_durata_mediana_termine_massimo_sanatorie_avviate',
                'numero_sanatorie_avviate',
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo']
        elif sheet_name == 'Controllo CILA':
            measure_labels = [
                'numero_controlli_cila_conclusi_con_silenzio-assenso',
                'numero_controlli_cila_conclusi_con_provvedimento_espresso',
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_sospensioni',
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_conferenza_servizi',
                'giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso',
                'giornate_durata_media_netta_controlli_cila_conclusi_con_provvedimento_espresso',
                'giornate_durata_mediana_termine_massimo_controlli_cila_avviati',
                'numero_controlli_cila_avviati',
                'numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo']
        else:
            measure_labels = [
                'numero_pratiche_concluse_con_silenzio-assenso',
                'numero_pratiche_concluse_con_provvedimento_espresso',
                'numero_pratiche_concluse_con_provvedimento_espresso_con_sospensioni',
                'numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi',
                'giornate_durata_media_pratiche_concluse_con_provvedimento_espresso',
                'giornate_durata_media_netta_pratiche_concluse_con_provvedimento_espresso',
                'giornate_durata_mediana_termine_massimo_pratiche_avviate',
                'numero_pratiche_avviate',
                'numero_pratiche_arretrate_non_concluse_scaduto_termine_massimo']
        measure_labels = [label + '_' + measure_period for label in measure_labels]

        measure_year, measure_quarters = measure_period.split('q')
        if measure_quarters == '1-2':
            measure_end_date = pd.Timestamp('{}-06-30'.format(measure_year))
        elif measure_quarters == '3-4':
            measure_end_date = pd.Timestamp('{}-12-31'.format(measure_year))
        else:
            raise

        if type_name:
            filter_type = comune_dataframe.loc[:, 'tipologia_pratica'] == type_name
        else:
            if sheet_name == 'Permessi di Costruire' and type_pdc_ov:
                filter_type = (comune_dataframe.loc[:, 'tipologia_pratica'] ==
                               'PdC ordinario') ^ \
                              (comune_dataframe.loc[:, 'tipologia_pratica'] ==
                               'PdC in variante')
            else:
                filter_type = comune_dataframe.loc[:, 'tipologia_pratica'] != ''

        if sheet_name == 'Permessi di Costruire':
            filter_mask = \
                comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False
            filter_mask = filter_mask & filter_type
            numero_pratiche_concluse_con_silenzio_assenso = \
                comune_dataframe[filter_mask].__len__()
            filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna() == False
            filter_mask = filter_mask & (comune_dataframe.loc[:, 'conferenza_servizi'] == True)
            filter_mask = filter_mask & filter_type
            numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi = \
                comune_dataframe[filter_mask].__len__()
        elif sheet_name == 'Prov di sanatoria':
            numero_pratiche_concluse_con_silenzio_assenso = 0
            filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna() == False
            filter_mask = filter_mask & (comune_dataframe.loc[:, 'conferenza_servizi'] == True)
            filter_mask = filter_mask & filter_type
            numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi = \
                comune_dataframe[filter_mask].__len__()
        else:
            numero_pratiche_concluse_con_silenzio_assenso = 0
            numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi = 0

        filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna() == False
        filter_mask = filter_mask & filter_type
        numero_pratiche_concluse_con_provvedimento_espresso = \
            comune_dataframe[filter_mask].__len__()
        giornate_durata_pratiche_concluse_con_provvedimento_espresso = \
            comune_dataframe.loc[filter_mask, 'data_fine_pratica'] - \
            comune_dataframe.loc[filter_mask, 'data_inizio_pratica']
        pratiche_concluse_con_provvedimento_espresso_meno_di_una_giornata = \
            giornate_durata_pratiche_concluse_con_provvedimento_espresso < \
            pd.Timedelta(1, unit='D')
        giornate_durata_pratiche_concluse_con_provvedimento_espresso.loc[
            pratiche_concluse_con_provvedimento_espresso_meno_di_una_giornata] = \
            pd.Timedelta(1, unit='D')
        giornate_durata_media_pratiche_concluse_con_provvedimento_espresso = (
                comune_dataframe.loc[filter_mask, 'data_fine_pratica'] -
                comune_dataframe.loc[filter_mask, 'data_inizio_pratica']).mean().days
        giornate_durata_media_netta_pratiche_concluse_con_provvedimento_espresso = (
                comune_dataframe.loc[filter_mask, 'data_fine_pratica'] -
                comune_dataframe.loc[filter_mask, 'data_inizio_pratica'] -
                comune_dataframe.loc[filter_mask, 'giorni_sospensioni']).mean().days

        filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna() == False
        filter_mask = filter_mask & (comune_dataframe.loc[:, 'giorni_sospensioni'] >
                                     pd.Timedelta(0, unit='D'))
        filter_mask = filter_mask & filter_type
        numero_pratiche_concluse_con_provvedimento_espresso_con_sospensioni = \
            comune_dataframe[filter_mask].__len__()

        filter_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].isna() == False
        filter_mask = filter_mask & filter_type
        numero_pratiche_avviate = \
            comune_dataframe[filter_mask].__len__()
        giornate_durata_mediana_termine_massimo_pratiche_avviate = \
            comune_dataframe.loc[filter_mask, 'giorni_termine_normativo'].median().days

        filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna()
        if sheet_name == 'Permessi di Costruire':
            filter_mask = filter_mask ^ (
                    comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False)
        filter_mask = filter_mask & filter_type
        numero_pratiche_arretrate_non_concluse_scaduto_termine_massimo = ((
            measure_end_date -
            comune_dataframe.loc[filter_mask, 'data_inizio_pratica'] -
            comune_dataframe.loc[filter_mask, 'giorni_sospensioni']) >
            comune_dataframe.loc[filter_mask, 'giorni_termine_normativo']).sum()

        measure = [
            numero_pratiche_concluse_con_silenzio_assenso,
            numero_pratiche_concluse_con_provvedimento_espresso,
            numero_pratiche_concluse_con_provvedimento_espresso_con_sospensioni,
            numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi,
            giornate_durata_media_pratiche_concluse_con_provvedimento_espresso,
            giornate_durata_media_netta_pratiche_concluse_con_provvedimento_espresso,
            giornate_durata_mediana_termine_massimo_pratiche_avviate,
            numero_pratiche_avviate,
            numero_pratiche_arretrate_non_concluse_scaduto_termine_massimo]

        comune_measure_series = pd.Series(measure, index=measure_labels)

        return comune_measure_series


def check_transit_03_04(still_to_check=True, already_fine=False):
    ''' controllo se, nella 4a misurazione, sono state riportate le pratiche non concluse
        della 3a misurazione (controllare id pratica e data presentazione)
    '''
    list_excel, list_xls = get_list_excel()

    load = True
    comuni_dataframe_pdc_03 = pat_pnrr_3a.get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire', load=load)
    comuni_dataframe_pds_03 = pat_pnrr_3a.get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria', load=load)

    comuni_dataframe_pdc_04 = get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire', load=load)
    comuni_dataframe_pds_04 = get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria', load=load)

    comuni_received_04 = [comune_excel[1] for comune_excel in list_excel]
    comuni_checked_04 = [comune[0] for comune in comuni_4a_misurazione_excel_check
                         if comune[1] is True]
    comuni_dataframe_couples = [['pdc', comuni_dataframe_pdc_03, comuni_dataframe_pdc_04],
                                ['pds', comuni_dataframe_pds_03, comuni_dataframe_pds_04]]
    comuni_to_revise = []
    comuni_just_fine = []

    for comune in comuni_received_04:
        # nuovo controllo considerando i comuni gia' revisionati a vista
        # (secondo la tabella Verifica transito da 3a a 4a rilevazione) da non ricomunicare
        if still_to_check:
            if comune in comuni_checked_04:
                continue
        issues = 0
        for comuni_dataframe_couple in comuni_dataframe_couples:
            comuni_tipo_pratica = comuni_dataframe_couple[0]
            comuni_dataframe_03 = comuni_dataframe_couple[1]
            comuni_dataframe_04 = comuni_dataframe_couple[2]

            filter_mask_03 = comuni_dataframe_03.loc[:, 'comune'] == comune
            filter_nonconclusa_03 = comuni_dataframe_03.loc[:, 'data_fine_pratica'].isna()
            filter_mask_03 = filter_mask_03 & filter_nonconclusa_03
            if 'data_fine_pratica_silenzio-assenso' in comuni_dataframe_03.columns:
                filter_silenzioassenso_03 = (
                    comuni_dataframe_03.loc[:, 'data_fine_pratica_silenzio-assenso'].isna())
                filter_mask_03 = filter_mask_03 & filter_silenzioassenso_03
            filter_mask_04 = comuni_dataframe_04.loc[:, 'comune'] == comune

            issue = False
            if filter_mask_03.sum() > 0:
                id_pratiche_04 = []
                for id_pratica_04 in comuni_dataframe_04[filter_mask_04].id_pratica.values:
                    id_pratiche_04.append(id_pratica_04.__str__().lower().rstrip('.0').split(' ')[-1])
                date_inizio_pratiche_04 = []
                for data_inizio_pratica_04 in comuni_dataframe_04[filter_mask_04].data_inizio_pratica.values:
                    date_inizio_pratiche_04.append(data_inizio_pratica_04)

                for index_pratica_03 in comuni_dataframe_03[filter_mask_03].index:
                    pratica_03 = comuni_dataframe_03.iloc[index_pratica_03]
                    id_pratica_03 = pratica_03.id_pratica.__str__().lower().rstrip('.0').split(' ')[-1]
                    data_inizio_pratica_03 = pratica_03.data_inizio_pratica
                    if id_pratica_03 not in id_pratiche_04:
                        if data_inizio_pratica_03 not in date_inizio_pratiche_04:
                            print('comune di {0}: pratica {1} {2} non transitata '
                                  'dalla 3a alla 4a misurazione'.format(
                                comune, comuni_tipo_pratica, pratica_03.id_pratica))
                            issues += 1
                            issue = True

                # for id_pratica_03_orig in comuni_dataframe_03[filter_mask_03].id_pratica.values:
                #     id_pratica_03 = id_pratica_03_orig.__str__().lower().rstrip('.0').split(' ')[-1]
                #     if id_pratica_03 not in id_pratiche_04:
                #         print('comune di {0}: pratica {1} {2} non transitata '
                #               'dalla 3a alla 4a misurazione'.format(
                #             comune, comuni_tipo_pratica, id_pratica_03_orig))
                #         issues += 1
                #         issue = True

                if issue:
                    # print(comuni_dataframe_03[filter_mask_03])
                    # print(comuni_dataframe_04[filter_mask_04])
                    print('')

        if issues > 0:
            comuni_to_revise.append([issues, comune])
            print('')
        else:
            comuni_just_fine.append([issues, comune])
            print('')

    print('{0} pdc e/o pds in {1} comuni probabilmente da revisionare:'.format(
        sum([issues for [issues, comune] in comuni_to_revise]), comuni_to_revise.__len__()))
    comuni_to_revise.sort(reverse=True)
    for [issues, comune] in comuni_to_revise:
        print('{0} pdc e/o pds del comune di {1} probabilmente da revisionare'.format(
            issues, comune))
    print('')

    if already_fine:
        print('{0} comuni probabilmente già a posto '
              '(eventuali pratiche non concluse nella 3a misurazione '
              'risultano transitate nella 4a):'.format(comuni_just_fine.__len__()))
        for issues, comune in comuni_just_fine:
            print('{0}'.format(comune))
        print('')


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


    # list_excel, list_xls = get_list_excel()
    # check_comuni_excel()
    # check_transit_03_04(still_to_check=True, already_fine=False)


    # comune_name = 'Canazei'
    # file = '039_Canazei_Edilizia.xls'
    # comune = ComuneExcel(file, comune_name)
    # # print('controllo il file excel del comune di {0}'.format(comune_name))
    # # comune.check_headers_excel()
    # # comune.check_dataframes_excel()
    
    # # comune_dataframe_pdc = comune.get_comune_dataframe('Permessi di Costruire')
    # comune_dataframe_pds = comune.get_comune_dataframe('Prov di sanatoria')
    # # comune_dataframe_cila = comune.get_comune_dataframe('Controllo CILA')
    
    # # comune_measure_series_pdc = comune.get_comune_measure_series('Permessi di Costruire')
    # comune_measure_series_pds = comune.get_comune_measure_series('Prov di sanatoria')
    # # comune_measure_series_cila = comune.get_comune_measure_series('Controllo CILA')


    # load = True
    # comuni_dataframe_pdc_04 = get_comuni_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', load=load)
    # comuni_dataframe_pds_04 = get_comuni_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', load=load)
    # comuni_dataframe_cila_04 = get_comuni_dataframe(
    #     comuni_excel_map, 'Controllo CILA', load=load)

    # comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=True, load=load)
    # comuni_measure_dataframe_pdc = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=False, load=load)
    # comuni_measure_dataframe_pds = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', load=load)
    # comuni_measure_dataframe_cila = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Controllo CILA', load=load)


    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    type_name='PdC ordinario', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    type_name='PdC in variante', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    type_name='PdC in deroga', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    type_name='PdC convenzionato', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    type_name='PdC asseverato', type_pdc_ov=False, load=False)
    #
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    type_name='PdC in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    type_name='Provvedimenti in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    type_name='Regolarizzazione', load=False)


    # get_comuni_dataframes(comuni_excel_map, load=False)
    # get_comuni_measures_dataframe(comuni_excel_map, load=True)
    # get_comuni_measures(comuni_excel_map)
    #
    # get_comuni_organico(comuni_excel_map, load=True)


    # REQUEST 20240513_01 | pdc-ov non conclusi durata netta > 120 gg
    # - pdc-ov, non conclusi, mpe 04
    #   - data fine semestre (30/06/2022) - data inizio pratica - sospensione (se c'e')
    #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
    # measure_end_date = pd.Timestamp('2023-06-30')
    # filter_type = (comuni_dataframe_pdc_04.loc[:, 'tipologia_pratica'] ==
    #                   'PdC ordinario') ^ \
    #               (comuni_dataframe_pdc_04.loc[:, 'tipologia_pratica'] ==
    #                   'PdC in variante')
    # filter_mask = filter_type & (
    #     comuni_dataframe_pdc_04.loc[:, 'data_fine_pratica_silenzio-assenso'].isna())
    # filter_mask = filter_mask & (
    #     comuni_dataframe_pdc_04.loc[:, 'data_fine_pratica'].isna())
    # filtro_pratiche_non_concluse = (\
    #     measure_end_date - \
    #     comuni_dataframe_pdc_04.loc[filter_mask, 'data_inizio_pratica'] - \
    #     comuni_dataframe_pdc_04.loc[filter_mask, 'giorni_sospensioni']) > \
    #         pd.to_timedelta(120, errors='coerce', unit='D')
    # filtro_no_trento = comuni_dataframe_pdc_04.loc[:, 'comune'] != 'Trento'
    
    # pratiche_non_concluse = comuni_dataframe_pdc_04[
    #     filter_mask & filtro_pratiche_non_concluse]
    # filtro_non_concluse_giorni_sospensioni_nulli = \
    #     pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
    #         pd.to_timedelta(0, errors='coerce', unit='D')
    # pratiche_non_concluse_giorni_sospensioni_nulli = pratiche_non_concluse[
    #     filtro_non_concluse_giorni_sospensioni_nulli]
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
    #     filtro_non_concluse_giorni_sospensioni_nulli.sum()
    # print('numero pdc-ov mpe-04 non conclusi con giorni sospensioni nulli = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
    # comuni_dataframe_pdc_04[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].to_csv(
    #     'pat-pnrr_edilizia_pdc_ov_mpe_04_non_concluse_no_sospensioni_request_20240513_01_02.csv')
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento = \
    #     (filtro_non_concluse_giorni_sospensioni_nulli & \
    #      filtro_no_trento).sum()
    # print('numero pdc-ov mpe-04 non conclusi con giorni sospensioni nulli senza trento = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento))


    # REQUEST 20240515_01 | pds non conclusi durata netta > 120 gg
    # - pds, non conclusi, mpe 04
    #   - data fine semestre (30/06/2022) - data inizio pratica - sospensione (se c'e')
    #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
    # measure_end_date = pd.Timestamp('2023-06-30')
    # filter_mask = (
    #     comuni_dataframe_pds_04.loc[:, 'data_fine_pratica'].isna())
    # filtro_pratiche_non_concluse = (\
    #     measure_end_date - \
    #     comuni_dataframe_pds_04.loc[filter_mask, 'data_inizio_pratica'] - \
    #     comuni_dataframe_pds_04.loc[filter_mask, 'giorni_sospensioni']) > \
    #         pd.to_timedelta(120, errors='coerce', unit='D')
    # filtro_no_trento = comuni_dataframe_pds_04.loc[:, 'comune'] != 'Trento'
    
    # pratiche_non_concluse = comuni_dataframe_pds_04[
    #     filter_mask & filtro_pratiche_non_concluse]
    # filtro_non_concluse_giorni_sospensioni_nulli = \
    #     pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
    #         pd.to_timedelta(0, errors='coerce', unit='D')
    # pratiche_non_concluse_giorni_sospensioni_nulli = pratiche_non_concluse[
    #     filtro_non_concluse_giorni_sospensioni_nulli]
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
    #     filtro_non_concluse_giorni_sospensioni_nulli.sum()
    # print('numero pds mpe-04 non conclusi con giorni sospensioni nulli = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
    # comuni_dataframe_pds_04[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].to_csv(
    #     'pat-pnrr_edilizia_pds_mpe_04_non_concluse_no_sospensioni_request_20240515_01_02.csv')
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento = \
    #     (filtro_non_concluse_giorni_sospensioni_nulli & \
    #      filtro_no_trento).sum()
    # print('numero pds mpe-04 non conclusi con giorni sospensioni nulli senza trento = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento))
