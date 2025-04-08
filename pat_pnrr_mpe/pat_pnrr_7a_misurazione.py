"""
    PAT-PNRR 7a misurazione (June 2025)
    Monitoraggio Procedimenti Edilizi v2.x
    Francesco Melchiori, 2025
"""


import os
import shelve
import difflib

import numpy as np
import pandas as pd

import warnings

from .pat_pnrr_comuni_excel_mapping import *


DATA_INIZIO_MONITORAGGIO = '2024' + '-07-01'  # '-01-01'
DATA_FINE_MONITORAGGIO = '2024' + '-12-31'  # '-06-30'
PERIODO_MONITORAGGIO = '2024q3-4'
FOLDER_COMUNI_EXCEL = 'pat_pnrr_7a_misurazione_tabelle_comunali\\'  # drive-download-..\\'
INDEX_COMUNI_EXCEL_MAP = 5


def get_dataframe_excel(path_file_excel, sheet_name, names, usecols, skiprows, droprows,
                        nrows=None, dtype=None, parse_dates=False):

    # warnings.simplefilter(action='ignore', category=UserWarning)
    dataframe_excel = pd.read_excel(path_file_excel, sheet_name=sheet_name, names=names,
                                    usecols=usecols, skiprows=skiprows, nrows=nrows, dtype=dtype,
                                    parse_dates=parse_dates, header=None)
    # warnings.resetwarnings()

    for row in droprows:
        dataframe_excel.drop(dataframe_excel.index[dataframe_excel[row].isna()], inplace=True)
    
    # if sheet_name == 'ORGANICO':
    #     pass
    # else:
    dataframe_excel.drop_duplicates(inplace=True)
    
    dataframe_excel.reset_index(drop=True, inplace=True)

    return dataframe_excel


def get_list_excel(path_to_excel_files, path_to_mpe=None, missing=False):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'

    list_xls = []
    for file in os.listdir(path_to_mpe + path_to_excel_files):
        if file.find('xls') != -1:
            list_xls.append(file)

    if missing:
        list_excel = [comune[0] for comune in comuni_excel_map
                      if comune[INDEX_COMUNI_EXCEL_MAP] is None]
    else:
        list_excel = [(comune[INDEX_COMUNI_EXCEL_MAP], comune[0]) for comune in comuni_excel_map
                      if comune[INDEX_COMUNI_EXCEL_MAP] is not None]

    return list_excel, list_xls


class ComuneExcel:

    def __init__(self, name_excel_file, path_to_excel_files, comune_name='Test', path_to_mpe=None):
        if not path_to_mpe:
            path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'    
        self.path_base = path_to_mpe + path_to_excel_files
        self.path_file = name_excel_file
        self.excel_path = self.path_base + self.path_file
        self.comune_name = comune_name
        self.excel_structure = {
            'ORGANICO': {
                'column_labels': [
                    'numero_dipendente',  # string | object
                    'tipo_dipendente',  # string | object
                    'ore_settimana',  # integer | float
                    'percentuale_ore_edilizia_privata',  # integer | float
                    'percentuale_ore_comune_considerato'  # integer | float
                ],
                'column_dtype': {
                    'numero_dipendente': str,
                    'tipo_dipendente': str,
                    'ore_settimana': str,  # float,
                    'percentuale_ore_edilizia_privata': str,  # float,
                    'percentuale_ore_comune_considerato': str  # float
                },
                'column_indexes': [
                    1, 2, 3, 9, 11
                ],
                'row_skips': 9,
                'column_mandatory': [
                    'numero_dipendente',
                    'tipo_dipendente',
                    'ore_settimana'
                ],
                'health_header_checks': [
                    'dipendente',
                    'tecnico',
                    'ore',
                    'privata',
                    'a questo comune'
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
                'column_dtype': {
                    'tipologia_pratica': str,
                    'id_pratica': str,
                    'data_inizio_pratica': str,
                    'giorni_termine_normativo': str,  # float,
                    'data_fine_pratica': str,
                    'data_fine_pratica_silenzio-assenso': str,
                    'conferenza_servizi': str,
                    'tipologia_massima_sospensione': str,
                    'giorni_sospensioni': str  # float
                },
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
                'column_dtype': {
                    'tipologia_pratica': str,
                    'id_pratica': str,
                    'data_inizio_pratica': str,
                    'giorni_termine_normativo': str,  # float,
                    'data_fine_pratica': str,
                    'conferenza_servizi': str,
                    'tipologia_massima_sospensione': str,
                    'giorni_sospensioni': str,  # float
                },
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
                'column_dtype': {
                    'tipologia_pratica': str,
                    'id_pratica': str,
                    'data_inizio_pratica': str,
                    'giorni_termine_normativo': str,  # float,
                    'data_fine_pratica': str,
                    'tipologia_massima_sospensione': str,
                    'giorni_sospensioni': str  # float
                },
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
            dtypecols = None  # self.excel_structure[sheet_name]['column_dtype']
            usecols = self.excel_structure[sheet_name]['column_indexes']
            skiprows = self.excel_structure[sheet_name]['row_skips']-2
            droprows = self.excel_structure[sheet_name]['column_mandatory']
            nrows = 1
            health_header_checks = self.excel_structure[sheet_name]['health_header_checks']

            header_excel = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                               skiprows, droprows, nrows, dtypecols)

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
            dtypecols = None  # self.excel_structure[sheet_name]['column_dtype']
            usecols = self.excel_structure[sheet_name]['column_indexes']
            skiprows = self.excel_structure[sheet_name]['row_skips']
            droprows = self.excel_structure[sheet_name]['column_mandatory']
            health_na_content_checks = self.excel_structure[sheet_name]['health_na_content_checks']

            dataframe_excel = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                                  skiprows, droprows, dtype=dtypecols,
                                                  parse_dates=False)

            for health_na_content_check in health_na_content_checks:
                if sum(dataframe_excel[health_na_content_check].isna()) > 0:
                    print('{0} assente nel foglio {1} del file Excel di {2}'.format(
                        health_na_content_check, sheet_name, self.comune_name))
                    # print('[!] excel na content health check')
                    # print('    ' + 'in the file [' + self.path_file + ']')
                    # print('    ' + '    ' + 'in the sheet [' + sheet_name + ']')
                    # print('    ' + '    ' + '    ' + 'the column [' + health_na_content_check +
                    #       '] has some NA')

        return True

    def get_comune_dataframe(self, sheet_name):
        names = self.excel_structure[sheet_name]['column_labels']
        dtypecols = None  # self.excel_structure[sheet_name]['column_dtype']
        usecols = self.excel_structure[sheet_name]['column_indexes']
        skiprows = self.excel_structure[sheet_name]['row_skips']
        droprows = self.excel_structure[sheet_name]['column_mandatory']

        comune_dataframe = get_dataframe_excel(self.excel_path, sheet_name, names, usecols,
                                              skiprows, droprows, dtype=dtypecols)
        
        if sheet_name == 'ORGANICO':
            comune_dataframe.insert(0, 'comune', self.comune_name)
            comune_dataframe.loc[:, 'percentuale_ore_edilizia_privata'] = \
                comune_dataframe.loc[:, 'percentuale_ore_edilizia_privata'].fillna(1)
            comune_dataframe.loc[:, 'percentuale_ore_comune_considerato'] = \
                comune_dataframe.loc[:, 'percentuale_ore_comune_considerato'].fillna(1)
            if comune_dataframe.loc[:, 'ore_settimana'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'ore_settimana'].astype(
                    'string').str.contains('36 ore', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'ore_settimana'] = 36
            try:
                comune_dataframe['ore_settimana'] = pd.to_numeric(
                    comune_dataframe['ore_settimana'],
                    errors='raise', downcast='integer')
            except:
                print('ore_settimana is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'ore_settimana'])

            try:
                comune_dataframe['percentuale_ore_edilizia_privata'] = pd.to_numeric(
                    comune_dataframe['percentuale_ore_edilizia_privata'],
                    errors='raise', downcast='integer')
            except:
                print('percentuale_ore_edilizia_privata is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'percentuale_ore_edilizia_privata'])

            try:
                comune_dataframe['percentuale_ore_comune_considerato'] = pd.to_numeric(
                    comune_dataframe['percentuale_ore_comune_considerato'],
                    errors='raise', downcast='integer')
            except:
                print('percentuale_ore_comune_considerato is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'percentuale_ore_comune_considerato'])

            # TODO: se il comune ha indicato "numero" ore e "numero" percentuale ore edilizia privata e "0" nella percentuale ore comune, io sostituirei lo "0" con "1"
            for index_dipendente in comune_dataframe.index:
                if (comune_dataframe.loc[index_dipendente, 'ore_settimana'] > 0) & \
                   (comune_dataframe.loc[index_dipendente, 'percentuale_ore_edilizia_privata'] > 0) & \
                   (comune_dataframe.loc[index_dipendente, 'percentuale_ore_comune_considerato'] == 0):
                    comune_dataframe.loc[index_dipendente, 'percentuale_ore_comune_considerato'] = 1
        else:
            comune_dataframe.insert(0, 'comune', self.comune_name)
            comune_dataframe.dropna(axis=0, subset='id_pratica', inplace=True, ignore_index=True)
            comune_dataframe.dropna(axis=0, subset='data_inizio_pratica', inplace=True,
                                    ignore_index=True)

        if sheet_name == 'Permessi di Costruire':
            comune_dataframe.loc[:, 'tipologia_pratica'] = \
                comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_pdc[0])
            comune_dataframe.loc[:, 'giorni_termine_normativo'] = \
                comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(60)
            comune_dataframe.loc[:, 'conferenza_servizi'] = \
                comune_dataframe.loc[:, 'conferenza_servizi'].fillna('NO')
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'] = \
                comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('')
            comune_dataframe.loc[:, 'giorni_sospensioni'] = \
                comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0)

            if comune_dataframe.loc[:, 'data_inizio_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('06/04/022', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '06/04/2022'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('23/0523', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '23/05/2023'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('8/5/204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '08/05/2024'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('10/6/0204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '10/06/2024'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('12/112024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '12/11/2024'
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
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
                    'string').str.contains('SOSPESO', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('20/07/203', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '20/07/2023'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('05/012024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '05/01/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('01/03/24 - DINIEGO', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '01/03/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('1575/2024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '15/05/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('42/2025', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '04/02/2025'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('12/128/2024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '12/12/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('09/10/204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '09/10/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARCHIVIATA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARCHIVATA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('archiviato', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('superata da', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('rigettata', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ANOMALA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('da archiviare', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp(DATA_INIZIO_MONITORAGGIO + ' 00:00:00.000')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
            comune_dataframe.loc[change_mask, 'data_fine_pratica'] = pd.NaT

            if comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].dtype.str[1] in \
                    ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].astype(
                    'string').str.contains('NO', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica_silenzio-assenso'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].astype(
                    'string').str.contains('Ritirata', case=False, na=False, regex=False)
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
            comune_dataframe.loc[:, 'tipologia_pratica'] = \
                comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_pds[0])
            comune_dataframe.loc[:, 'giorni_termine_normativo'] = \
                comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(60)
            comune_dataframe.loc[:, 'conferenza_servizi'] = \
                comune_dataframe.loc[:, 'conferenza_servizi'].fillna('NO')
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'] = \
                comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('')
            comune_dataframe.loc[:, 'giorni_sospensioni'] = \
                comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0)

            if comune_dataframe.loc[:, 'data_inizio_pratica'].dtype.str[1] in ['O', 'M']:
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('26/05/23', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '26/05/2023'
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('ARCHIVIATA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
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
                    'string').str.contains('SOSPESO', case=False, na=False, regex=False)
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
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('08/01/20241', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '08/01/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('08/01/20241', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '08/01/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('9/10/204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '09/10/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARCHIVIATA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARCHIVIATO', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ORDINANZA DI MESSA IN PRISTINO', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ANOMALA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('archiviazione', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ritirata 28/10/2024', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp(DATA_INIZIO_MONITORAGGIO + ' 00:00:00.000')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
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
            comune_dataframe.loc[:, 'tipologia_pratica'] = \
                comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_cila[0])
            comune_dataframe.loc[:, 'giorni_termine_normativo'] = \
                comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(0)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'] = \
                comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('')
            comune_dataframe.loc[:, 'giorni_sospensioni'] = \
                comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0)

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
                change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'].astype(
                    'string').str.contains('19/072024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_inizio_pratica'] = '19/07/2024'
            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)

            if comune_dataframe.loc[:, 'data_fine_pratica'].dtype == 'O':
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] == ' '
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('non concluso', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('-', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('08/092023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '08/09/2023'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('03/102023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '03/10/2023'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('23/082024', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '23/08/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('07/02/204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '07/02/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('07/08/204', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '07/08/2024'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARCHIVIATA', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('ARICHIVIATA 6/11/2024', case=False, na=False, regex=False)
                comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp(DATA_INIZIO_MONITORAGGIO + ' 00:00:00.000')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp(DATA_FINE_MONITORAGGIO + ' 23:59:59.999')
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
                                  measure_period=PERIODO_MONITORAGGIO, lpf=False):
        if lpf:
            path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
            path_to_excel_files = FOLDER_COMUNI_EXCEL
            path_shelve = path_to_mpe + path_to_excel_files

            sheet_suffix = ''
            if sheet_name == 'Permessi di Costruire':
                sheet_suffix += '_pdc_ov'
            if sheet_name == 'Prov di sanatoria':
                sheet_suffix += '_pds'

            comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + \
                                                  sheet_suffix + '_lpf')
            comuni_dataframe = comuni_dataframe_shelve['comuni_dataframe']
            comuni_dataframe_shelve.close()

            comune_dataframe = comuni_dataframe[comuni_dataframe.comune == self.comune_name]
        else:
            comune_dataframe = self.get_comune_dataframe(sheet_name=sheet_name)

        if sheet_name == 'ORGANICO':
            measure_labels = [
                'ore_tecnici_settimana']
        elif sheet_name == 'Permessi di Costruire':
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
            if sheet_name == 'ORGANICO':
                filter_type = comune_dataframe.loc[:, 'tipo_dipendente'] == 'tecnico'
            elif sheet_name == 'Permessi di Costruire' and type_pdc_ov:
                filter_type = (comune_dataframe.loc[:, 'tipologia_pratica'] ==
                               'PdC ordinario') ^ \
                              (comune_dataframe.loc[:, 'tipologia_pratica'] ==
                               'PdC in variante')
            else:
                filter_type = comune_dataframe.loc[:, 'tipologia_pratica'] != ''

        if sheet_name == 'ORGANICO':
            filter_mask = filter_type
        elif sheet_name == 'Permessi di Costruire':
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

        if sheet_name == 'ORGANICO':
            ore_tecnici_settimana = (
                comune_dataframe.loc[filter_mask, 'ore_settimana'] * \
                comune_dataframe.loc[filter_mask, 'percentuale_ore_edilizia_privata'] * \
                comune_dataframe.loc[filter_mask, 'percentuale_ore_comune_considerato']).sum()

            measure = [
                ore_tecnici_settimana]

            comune_measure_series = pd.Series(measure, index=measure_labels)
        else:
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
                    comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == 
                    False)
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


def check_comuni_excel(path_to_excel_files, path_to_mpe=None):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
    
    list_excel, list_xls = get_list_excel(path_to_excel_files, path_to_mpe)
    for name_excel_file, name_comune in list_excel:
        print('controllo il file excel del comune di {0}'.format(name_comune))
        comune_excel = ComuneExcel(name_excel_file, path_to_excel_files, name_comune, path_to_mpe)
        comune_excel.check_headers_excel()
        comune_excel.check_dataframes_excel()

    return True


def get_comuni_dataframe(comuni_excel_map, sheet_name, path_to_excel_files, load=True,
                         path_to_mpe=None, pf=''):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
    path_shelve = path_to_mpe + path_to_excel_files

    sheet_suffix = ''
    if sheet_name == 'ORGANICO':
        sheet_suffix += '_org'
        row_type_name = '_addetti'
    if sheet_name == 'Permessi di Costruire':
        sheet_suffix += '_pdc'
        row_type_name = '_pratiche'
    if sheet_name == 'Prov di sanatoria':
        sheet_suffix += '_pds'
        row_type_name = '_pratiche'
    if sheet_name == 'Controllo CILA':
        sheet_suffix += '_cila'
        row_type_name = '_pratiche'

    if load:
        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + sheet_suffix)
        comuni_dataframe = comuni_dataframe_shelve['comuni_dataframe']
        comuni_dataframe_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[INDEX_COMUNI_EXCEL_MAP])
                            for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
        comuni_dataframe = []
        for name_comune, name_excel_file in comuni_excel_map:
            print(name_comune + ' | ' + sheet_name)
            comune_excel = ComuneExcel(name_excel_file, path_to_excel_files, name_comune,
                                       path_to_mpe)
            comune_dataframe = comune_excel.get_comune_dataframe(sheet_name)
            comuni_dataframe.append(comune_dataframe)
        comuni_dataframe = pd.concat(comuni_dataframe, axis='rows', join='outer')
        comuni_dataframe.reset_index(drop=True, inplace=True)

        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + sheet_suffix)
        comuni_dataframe_shelve['comuni_dataframe'] = comuni_dataframe
        comuni_dataframe_shelve.close()

        comuni_dataframe.to_csv(path_shelve + 'pat-pnrr_edilizia' + row_type_name + \
                                sheet_suffix + '_' + PERIODO_MONITORAGGIO + '.csv')
    
    if pf == 'l_01':
        if sheet_name=='Permessi di Costruire':
            # REQUEST 20240513_01 | pdc-ov non conclusi durata netta > 120 gg
            # - pdc-ov, non conclusi, mpe 05
            #   - data fine semestre (31/12/2023) - data inizio pratica - sospensione (se c'e')
            #     - quante pratiche risultanti > 120 gg? quante di queste con sospensioni nulle?
            #     - conteggi con Trento e senza
            measure_end_date = pd.Timestamp(DATA_FINE_MONITORAGGIO)
            filter_type = (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC ordinario') ^ \
                          (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC in variante')
            filter_mask = filter_type & (
                comuni_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna())
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna())
            filtro_pratiche_non_concluse = (\
                measure_end_date - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica'] - \
                comuni_dataframe.loc[filter_mask, 'giorni_sospensioni']) > \
                    pd.to_timedelta(120, errors='coerce', unit='D')
            
            pratiche_non_concluse = comuni_dataframe[
                filter_mask & filtro_pratiche_non_concluse]
            filtro_non_concluse_giorni_sospensioni_nulli = \
                pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D')
            
            numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
                filtro_non_concluse_giorni_sospensioni_nulli.sum()
            print('numero pdc-ov mpe-05 non conclusi con giorni sospensioni nulli = ' + \
                str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
            
            sheet_suffix += '_ov'

        if sheet_name=='Prov di sanatoria':
            # REQUEST 20240515_01 | pds non conclusi durata netta > 120 gg
            # - pdc-ov, non conclusi, mpe 05
            #   - data fine semestre (31/12/2023) - data inizio pratica - sospensione (se c'e')
            #     - quante pratiche risultanti > 120 gg? quante di queste con sospensioni nulle?
            #     - conteggi con Trento e senza
            measure_end_date = pd.Timestamp(DATA_FINE_MONITORAGGIO)
            filter_mask = (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna())
            filtro_pratiche_non_concluse = (\
                measure_end_date - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica'] - \
                comuni_dataframe.loc[filter_mask, 'giorni_sospensioni']) > \
                    pd.to_timedelta(120, errors='coerce', unit='D')
            
            pratiche_non_concluse = comuni_dataframe[
                filter_mask & filtro_pratiche_non_concluse]
            filtro_non_concluse_giorni_sospensioni_nulli = \
                pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D')
            
            numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
                filtro_non_concluse_giorni_sospensioni_nulli.sum()
            print('numero pds mpe-05 non conclusi con giorni sospensioni nulli = ' + \
                str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
        
        comuni_dataframe.drop(
            comuni_dataframe[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].index,
            inplace=True)

        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + \
                                              sheet_suffix + '_lpf')
        comuni_dataframe_shelve['comuni_dataframe'] = comuni_dataframe
        comuni_dataframe_shelve.close()

    elif pf == 'l_02':
        if sheet_name=='Permessi di Costruire':
            # REQUEST 20240527_01 | pdc-ov avviati, fuori norma, senza sospensioni
            # - lista ordinata di comuni per pratiche del tipo seguente
            #   - pdc-ov avviati, mpe-05
            #     - fuori norma (durata netta > termine normativo, pratica per pratica)
            #     - senza sospensioni (0gg)
            #       - valore % sulle pratiche fuori norma
            # - lista pratiche concluse e non concluse
            measure_end_date = pd.Timestamp(DATA_FINE_MONITORAGGIO)
            filter_type = (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC ordinario') ^ \
                          (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC in variante')
            sheet_suffix += '_ov'
            filter_mask = filter_type & (
                comuni_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna())
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna())
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D'))
            
            filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma = (\
                measure_end_date - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica']) > \
                    comuni_dataframe.loc[filter_mask, 'giorni_termine_normativo']
            comuni_dataframe.loc[
                (filter_mask & \
                 filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma)].to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_pratiche_non_concluse_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_01' + '.csv')

            nomi_comuni = [comune[0] for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
            totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma = []
            for nome_comune in nomi_comuni:
                filter_comune = comuni_dataframe.loc[:, 'comune'] == nome_comune
                totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma.append(
                    comuni_dataframe.loc[
                        (filter_comune & \
                         filter_mask & \
                         filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma)].__len__())
            totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma = \
                pd.Series(totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma, \
                          index=nomi_comuni,
                          name='pdc_non_conclusi_sospensioni_nulle_fuori_norma')
            
            filter_type = (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC ordinario') ^ \
                          (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC in variante')
            filter_mask = filter_type & (
                comuni_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False)
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D'))
            
            filtro_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma = (\
                comuni_dataframe.loc[filter_mask, 'data_fine_pratica_silenzio-assenso'] - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica']) > \
                    comuni_dataframe.loc[filter_mask, 'giorni_termine_normativo']
            comuni_dataframe.loc[
                (filter_mask & \
                 filtro_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma)].to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_01' + '.csv')

            nomi_comuni = [comune[0] for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
            totali_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma = []
            for nome_comune in nomi_comuni:
                filter_comune = comuni_dataframe.loc[:, 'comune'] == nome_comune
                totali_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma.append(
                    comuni_dataframe.loc[
                        (filter_comune & \
                         filter_mask & \
                         filtro_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma)].__len__())
            totali_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma = \
                pd.Series(totali_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma, \
                          index=nomi_comuni,
                          name='pdc_conclusi_con_silenzio_assenso_sospensioni_nulle_fuori_norma')
            
            filter_type = (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC ordinario') ^ \
                          (comuni_dataframe.loc[:, 'tipologia_pratica'] ==
                            'PdC in variante')
            filter_mask = filter_type & (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna() == False)
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D'))
            
            filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = (\
                comuni_dataframe.loc[filter_mask, 'data_fine_pratica'] - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica']) > \
                    comuni_dataframe.loc[filter_mask, 'giorni_termine_normativo']
            comuni_dataframe.loc[
                (filter_mask & \
                 filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma)].to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_01' + '.csv')

            nomi_comuni = [comune[0] for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
            totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = []
            for nome_comune in nomi_comuni:
                filter_comune = comuni_dataframe.loc[:, 'comune'] == nome_comune
                totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma.append(
                    comuni_dataframe.loc[
                        (filter_comune & \
                         filter_mask & \
                         filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma)].__len__())
            totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = \
                pd.Series(totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma, \
                          index=nomi_comuni,
                          name='pdc_conclusi_con_espressione_sospensioni_nulle_fuori_norma')
            
            totali_pratiche_avviate_sospensioni_nulle_fuori_norma = pd.concat([
                totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma,
                totali_pratiche_concluse_con_silenzio_assenso_sospensioni_nulle_fuori_norma,
                totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma],
                axis='columns', join='outer')
            totali_pratiche_avviate_sospensioni_nulle_fuori_norma\
                .sum(axis=1).sort_values(ascending=False)\
                .to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_comuni_per_pdc_avviati_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_01' + '.csv')

        if sheet_name=='Prov di sanatoria':
            # REQUEST 20240527_02 | pds avviati, fuori norma, senza sospensioni
            # - lista ordinata di comuni per pratiche del tipo seguente
            #   - pds avviati, mpe-05
            #     - fuori norma (durata netta > termine normativo, pratica per pratica)
            #     - senza sospensioni (0gg)
            #       - valore % sulle pratiche fuori norma
            # - lista pratiche concluse e non concluse
            measure_end_date = pd.Timestamp(DATA_FINE_MONITORAGGIO)
            filter_mask = (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna())
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D'))
            
            filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma = (\
                measure_end_date - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica']) > \
                    comuni_dataframe.loc[filter_mask, 'giorni_termine_normativo']
            comuni_dataframe.loc[
                (filter_mask & \
                 filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma)].to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_pratiche_non_concluse_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_02' + '.csv')

            nomi_comuni = [comune[0] for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
            totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma = []
            for nome_comune in nomi_comuni:
                filter_comune = comuni_dataframe.loc[:, 'comune'] == nome_comune
                totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma.append(
                    comuni_dataframe.loc[
                        (filter_comune & \
                         filter_mask & \
                         filtro_pratiche_non_concluse_sospensioni_nulle_fuori_norma)].__len__())
            totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma = \
                pd.Series(totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma, \
                          index=nomi_comuni,
                          name='pds_non_conclusi_sospensioni_nulle_fuori_norma')
            
            filter_mask = (
                comuni_dataframe.loc[:, 'data_fine_pratica'].isna() == False)
            filter_mask = filter_mask & (
                comuni_dataframe.loc[:, 'giorni_sospensioni'] == \
                    pd.to_timedelta(0, errors='coerce', unit='D'))
            
            filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = (\
                comuni_dataframe.loc[filter_mask, 'data_fine_pratica'] - \
                comuni_dataframe.loc[filter_mask, 'data_inizio_pratica']) > \
                    comuni_dataframe.loc[filter_mask, 'giorni_termine_normativo']
            comuni_dataframe.loc[
                (filter_mask & \
                 filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma)].to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_02' + '.csv')

            nomi_comuni = [comune[0] for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
            totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = []
            for nome_comune in nomi_comuni:
                filter_comune = comuni_dataframe.loc[:, 'comune'] == nome_comune
                totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma.append(
                    comuni_dataframe.loc[
                        (filter_comune & \
                         filter_mask & \
                         filtro_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma)].__len__())
            totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma = \
                pd.Series(totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma, \
                          index=nomi_comuni,
                          name='pds_conclusi_con_espressione_sospensioni_nulle_fuori_norma')
            
            totali_pratiche_avviate_sospensioni_nulle_fuori_norma = pd.concat([
                totali_pratiche_non_concluse_sospensioni_nulle_fuori_norma,
                totali_pratiche_concluse_con_espressione_sospensioni_nulle_fuori_norma],
                axis='columns', join='outer')
            totali_pratiche_avviate_sospensioni_nulle_fuori_norma\
                .sum(axis=1).sort_values(ascending=False)\
                .to_csv(
                    path_shelve + 'pat-pnrr_edilizia_misure' + sheet_suffix + '_mpe_05' + \
                    '_comuni_per_pds_avviati_sospensioni_nulle_fuori_norma' + \
                    '_request_20240527_02' + '.csv')
    return comuni_dataframe


def check_comuni_dataframe(comuni_excel_map, sheet_name, path_to_excel_files, path_to_mpe=None):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
    path_shelve = path_to_mpe + path_to_excel_files

    sheet_suffix = ''
    if sheet_name == 'ORGANICO':
        sheet_suffix += '_org'
    if sheet_name == 'Permessi di Costruire':
        sheet_suffix += '_pdc'
    if sheet_name == 'Prov di sanatoria':
        sheet_suffix += '_pds'
    if sheet_name == 'Controllo CILA':
        sheet_suffix += '_cila'

    comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + sheet_suffix)
    comuni_dataframe = comuni_dataframe_shelve['comuni_dataframe']
    comuni_dataframe_shelve.close()

    if sheet_name == 'Prov di sanatoria':
        # alcuni comuni riportano pds senza termine normativo e non sono regolarizzazioni
        filter_mask = comuni_dataframe.loc[:, 'tipologia_pratica'] == 'PdC in Sanatoria'
        filter_mask = filter_mask | (
            comuni_dataframe.loc[:, 'tipologia_pratica'] == 'Provvedimenti in Sanatoria')
        filter_mask = filter_mask | (
            comuni_dataframe.loc[:, 'tipologia_pratica'] == 'c.d. SCIA in sanatoria')
        filter_mask = filter_mask & (
            comuni_dataframe.loc[:, 'giorni_termine_normativo'] != pd.Timedelta(days=60))
        comuni_to_revise = set(comuni_dataframe[filter_mask].loc[:, 'comune'])
        for comune in comuni_to_revise:
            print('controllare i termini normativi dei pds di ' + comune)
    else:
        pass
    
    return


def get_comuni_measure_dataframe(comuni_excel_map, sheet_name, path_to_excel_files,
                                 type_name=False, type_pdc_ov=True, load=True, path_to_mpe=None,
                                 lpf=False):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
    path_shelve = path_to_mpe + path_to_excel_files

    sheet_suffix = ''
    if sheet_name == 'ORGANICO':
        sheet_suffix += '_org'
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
        path_shelve_file = path_shelve + 'comuni_measure_dataframe' + sheet_suffix
        if lpf:
            path_shelve_file += '_lpf'
        comuni_measure_dataframe_shelve = shelve.open(path_shelve_file)
        comuni_measure_dataframe = comuni_measure_dataframe_shelve['comuni_measure_dataframe']
        comuni_measure_dataframe_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[INDEX_COMUNI_EXCEL_MAP])
                            for comune in comuni_excel_map if comune[INDEX_COMUNI_EXCEL_MAP] is not None]
        comuni_names = [comune[0] for comune in comuni_excel_map]
        comuni_measure_dataframe = []
        for name_comune, name_excel_file in comuni_excel_map:
            message = name_comune + ' | ' + sheet_name
            if type_name:
                message += ' | ' + type_name
            if sheet_name == 'Permessi di Costruire':
                if type_pdc_ov:
                    message += ' | ' + 'Ordinari e in Variante'
            print(message)
            comune_excel = ComuneExcel(name_excel_file, path_to_excel_files, name_comune,
                                       path_to_mpe)
            comune_measure_series = comune_excel.get_comune_measure_series(sheet_name, type_name,
                                                                           type_pdc_ov, lpf=lpf)
            comuni_measure_dataframe.append(comune_measure_series)
        comuni_measure_dataframe = pd.DataFrame(comuni_measure_dataframe, index=comuni_names)

        path_shelve_file = path_shelve + 'comuni_measure_dataframe' + sheet_suffix
        if lpf:
            path_shelve_file += '_lpf'
        comuni_measure_dataframe_shelve = shelve.open(path_shelve_file)
        comuni_measure_dataframe_shelve['comuni_measure_dataframe'] = comuni_measure_dataframe
        comuni_measure_dataframe_shelve.close()

        comuni_measure_dataframe.to_csv(path_shelve + 'pat-pnrr_edilizia_misure' + \
                                        sheet_suffix + '_' + PERIODO_MONITORAGGIO + '.csv')

    return comuni_measure_dataframe


def get_comuni_dataframes(comuni_excel_map, load=True):

    get_comuni_dataframe(comuni_excel_map, 'ORGANICO',
                         FOLDER_COMUNI_EXCEL, load=load)
    get_comuni_dataframe(comuni_excel_map, 'Permessi di Costruire',
                         FOLDER_COMUNI_EXCEL, load=load)
    get_comuni_dataframe(comuni_excel_map, 'Prov di sanatoria',
                         FOLDER_COMUNI_EXCEL, load=load)
    get_comuni_dataframe(comuni_excel_map, 'Controllo CILA',
                         FOLDER_COMUNI_EXCEL, load=load)

    return True


def check_comuni_dataframes(comuni_excel_map):

    check_comuni_dataframe(comuni_excel_map, 'ORGANICO',
                           FOLDER_COMUNI_EXCEL)
    check_comuni_dataframe(comuni_excel_map, 'Permessi di Costruire',
                           FOLDER_COMUNI_EXCEL)
    check_comuni_dataframe(comuni_excel_map, 'Prov di sanatoria',
                           FOLDER_COMUNI_EXCEL)
    check_comuni_dataframe(comuni_excel_map, 'Controllo CILA',
                           FOLDER_COMUNI_EXCEL)

    return True


def get_comuni_measures_dataframe(comuni_excel_map, load=True):

    comuni_measure_dataframe_org = get_comuni_measure_dataframe(comuni_excel_map,
        'ORGANICO', FOLDER_COMUNI_EXCEL,
        load=load)
    comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(comuni_excel_map,
        'Permessi di Costruire', FOLDER_COMUNI_EXCEL,
        type_pdc_ov=True, load=load)
    comuni_measure_dataframe_pdc = get_comuni_measure_dataframe(comuni_excel_map,
        'Permessi di Costruire', FOLDER_COMUNI_EXCEL,
        type_pdc_ov=False, load=load)
    comuni_measure_dataframe_pds = get_comuni_measure_dataframe(comuni_excel_map,
        'Prov di sanatoria', FOLDER_COMUNI_EXCEL,
        load=load)
    comuni_measure_dataframe_cila = get_comuni_measure_dataframe(comuni_excel_map,
        'Controllo CILA', FOLDER_COMUNI_EXCEL,
        load=load)

    comuni_measures_dataframe = pd.concat(
        [comuni_measure_dataframe_org,
         comuni_measure_dataframe_pdc_ov,
         comuni_measure_dataframe_pdc,
         comuni_measure_dataframe_pds,
         comuni_measure_dataframe_cila],
        axis='columns', join='outer')

    return comuni_measures_dataframe


def get_comuni_measure(comuni_excel_map, sheet_name, path_to_excel_files, type_name=False,
                       type_pdc_ov=True, measure_period=PERIODO_MONITORAGGIO, load=True, path_to_mpe=None,
                       lpf=False):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'

    comuni_measure_dataframe = get_comuni_measure_dataframe(
        comuni_excel_map=comuni_excel_map, sheet_name=sheet_name,
        path_to_excel_files=path_to_excel_files, type_name=type_name, type_pdc_ov=type_pdc_ov,
        load=load, path_to_mpe=path_to_mpe, lpf=lpf)

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
    print('{0}/166 comuni'.format(comuni_monitored))
    print()

    return comuni_measure, comuni_monitored


def get_comuni_measures(comuni_excel_map, save_tex=False, temp_tex=False):

    comuni_pdc_ov_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire',
        FOLDER_COMUNI_EXCEL)
    comuni_pds_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Prov di sanatoria',
        FOLDER_COMUNI_EXCEL)
    comuni_pdc_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire',
        FOLDER_COMUNI_EXCEL, type_pdc_ov=False)
    comuni_cila_measure, comuni_monitored = get_comuni_measure(
        comuni_excel_map, 'Controllo CILA',
        FOLDER_COMUNI_EXCEL)

    if save_tex:
        measurement_05_pratiche_header = [
            'Concluse con SA',
            'Concluse',
            'con sospensioni',
            'con CdS',
            'Durata [gg]',
            'Termine [gg]',
            'Avviate',
            'Arretrate']
        tex_file_header = measurement_05_pratiche_header

        comuni_pdc_measure.index = measurement_05_pratiche_header
        comuni_pdc_ov_measure.index = measurement_05_pratiche_header
        comuni_pds_measure.index = measurement_05_pratiche_header
        comuni_cila_measure.index = measurement_05_pratiche_header

        measurement_05_series = {
            'Permesso di Costruire OV': comuni_pdc_ov_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int)}
        measurement_05 = pd.DataFrame(measurement_05_series).T

        measurement_05b_series = {
            'Permesso di Costruire': comuni_pdc_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int),
            'Controllo della CILA': comuni_cila_measure.apply(np.rint).astype(int)}
        measurement_05b = pd.DataFrame(measurement_05b_series).T

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_' + PERIODO_MONITORAGGIO + '.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_05.columns = tex_file_header
            baseline_styler = measurement_05.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            caption = 'PAT-PNRR | Procedimenti Edilizi | Misurazione ' + PERIODO_MONITORAGGIO
            if temp_tex:
                caption += ' | PARZIALE'
            table_tex_content = baseline_styler.to_latex(
                caption=caption, label='pat-pnrr_mpe_' + PERIODO_MONITORAGGIO, position='!htbp',
                position_float="centering", hrules=True)
            table_tex_file.write(table_tex_content)

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_' + PERIODO_MONITORAGGIO + 'b.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_05b.columns = tex_file_header
            baseline_styler = measurement_05b.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            caption='PAT-PNRR | Procedimenti Edilizi | Misurazione ' + PERIODO_MONITORAGGIO + 'b'
            if temp_tex:
                caption += ' | PARZIALE'
            table_tex_content = baseline_styler.to_latex(
                caption=caption, label='pat-pnrr_mpe_' + PERIODO_MONITORAGGIO + 'b', position='!htbp',
                position_float="centering", hrules=True)
            table_tex_file.write(table_tex_content)

    return True


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


    # list_excel, list_xls = get_list_excel(FOLDER_COMUNI_EXCEL,
    #                                       missing=True)
    # for comune in list_excel:
    #     print(comune)


    # comune_name = 'Dro'
    # name_excel_file = '079_Dro.xlsx'
    # path_to_excel_files = FOLDER_COMUNI_EXCEL
    # print('controllo il file excel del comune di {0}'.format(comune_name))
    # comune = ComuneExcel(name_excel_file, path_to_excel_files, comune_name)
    # comune.check_headers_excel()
    # comune.check_dataframes_excel()

    # comuni_dataframe_org = comune.get_comune_dataframe('ORGANICO')
    # comune_dataframe_pdc = comune.get_comune_dataframe('Permessi di Costruire')
    # comune_dataframe_pds = comune.get_comune_dataframe('Prov di sanatoria')
    # comune_dataframe_cila = comune.get_comune_dataframe('Controllo CILA')
    
    # comune_measure_series_pdc = comune.get_comune_measure_series('Permessi di Costruire')
    # comune_measure_series_pds = comune.get_comune_measure_series('Prov di sanatoria')
    # comune_measure_series_cila = comune.get_comune_measure_series('Controllo CILA')


    # load = True
    # comuni_dataframe_org_05 = get_comuni_dataframe(
    #     comuni_excel_map, 'ORGANICO', FOLDER_COMUNI_EXCEL,
    #     load=load)
    # comuni_dataframe_pdc_05 = get_comuni_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', FOLDER_COMUNI_EXCEL,
    #     load=load, pf='l_02')
    # comuni_dataframe_pds_05 = get_comuni_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', FOLDER_COMUNI_EXCEL,
    #     load=load, pf='l_02')
    # comuni_dataframe_cila_05 = get_comuni_dataframe(
    #     comuni_excel_map, 'Controllo CILA', FOLDER_COMUNI_EXCEL,
    #     load=load)

    # comuni_measure_dataframe_org = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'ORGANICO', FOLDER_COMUNI_EXCEL,
    #     load=load)
    # comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', FOLDER_COMUNI_EXCEL,
    #     type_pdc_ov=True, load=load)
    # comuni_measure_dataframe_pdc_ov.loc[:, 'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2024q3-4'].sort_values(ascending=False)

    # comuni_measure_dataframe_pdc = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', FOLDER_COMUNI_EXCEL,
    #     type_pdc_ov=False, load=load)
    # comuni_measure_dataframe_pds = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', FOLDER_COMUNI_EXCEL,
    #     load=load)
    # comuni_measure_dataframe_cila = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Controllo CILA', FOLDER_COMUNI_EXCEL,
    #     load=load)


    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC ordinario', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC in variante', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC in deroga', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC convenzionato', type_pdc_ov=False, load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC asseverato', type_pdc_ov=False, load=False)
    #
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='PdC in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='Provvedimenti in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #                    FOLDER_COMUNI_EXCEL,
    #                    type_name='Regolarizzazione', load=False)


    # check_comuni_excel(FOLDER_COMUNI_EXCEL)
    # get_comuni_dataframes(comuni_excel_map, load=False)
    # check_comuni_dataframes(comuni_excel_map)
    get_comuni_measures_dataframe(comuni_excel_map, load=False)
    get_comuni_measures(comuni_excel_map, save_tex=False)

    # load = True
    # lpf = True
    # comuni_pdc_ov_measure, comuni_monitored = get_comuni_measure(comuni_excel_map, 'Permessi di Costruire',
    #     FOLDER_COMUNI_EXCEL, load=load, lpf=lpf)
    # comuni_pds_measure, comuni_monitored = get_comuni_measure(comuni_excel_map, 'Prov di sanatoria',
    #     FOLDER_COMUNI_EXCEL, load=load, lpf=lpf)
