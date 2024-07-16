"""
    PAT-PNRR 3a misurazione (June 2023)
    Monitoraggio Procedimenti Edilizi v2.x
    Francesco Melchiori, 2023
"""


import os
import shelve
import datetime
import difflib

import numpy as np
import pandas as pd

from pat_pnrr_mpe.pat_pnrr_comuni_excel_mapping import *


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


def get_list_excel(path_base=None, missing=False, pat_pnrr_misurazione=3):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'

    # list_excel = []
    # for file in os.listdir(path_base + 'pat_pnrr_mpe\\pat_pnrr_3a_misurazione_tabelle_comunali\\'):
    #     if file.find('xls') != -1:
    #         list_excel.append(file)

    file_excel_column = 1
    if pat_pnrr_misurazione == 3:
        file_excel_column = 1
    elif pat_pnrr_misurazione == 4:
        file_excel_column = 2

    if missing:
        list_excel = [comune[0] for comune in comuni_excel_map
                      if comune[file_excel_column] is None]
    else:
        list_excel = [(comune[file_excel_column], comune[0]) for comune in comuni_excel_map
                      if comune[file_excel_column] is not None]

    return list_excel


def check_comuni_excel():
    list_excel = get_list_excel()
    for file, comune_name in list_excel:
        comune_excel = ComuneExcel(file, comune_name)
        comune_excel.check_headers_excel()
        comune_excel.check_dataframes_excel()

    return True


def get_comuni_dataframe(comuni_excel_map, sheet_name, load=True, path_base=False, lpf=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'
    path_shelve = path_base + 'pat_pnrr_mpe\\pat_pnrr_3a_misurazione_tabelle_comunali\\'

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
        comuni_excel_map = [(comune[0], comune[1])
                            for comune in comuni_excel_map if comune[1] is not None]
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
    
    if lpf:
        if sheet_name=='Permessi di Costruire':
            # REQUEST 20240513_01 | pdc-ov non conclusi durata netta > 120 gg
            # - pdc-ov, non conclusi, mpe 03
            #   - data fine semestre (31/12/2022) - data inizio pratica - sospensione (se c'e')
            #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
            measure_end_date = pd.Timestamp('2022-12-31')
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
            print('numero pdc-ov mpe-03 non conclusi con giorni sospensioni nulli = ' + \
                  str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
            
            sheet_suffix += '_ov'
            
        if sheet_name=='Prov di sanatoria':
            # REQUEST 20240515_01 | pds non conclusi durata netta > 120 gg
            # - pds, non conclusi, mpe 03
            #   - data fine semestre (31/12/2022) - data inizio pratica - sospensione (se c'e')
            #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
            measure_end_date = pd.Timestamp('2022-12-31')
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
            print('numero pds mpe-03 non conclusi con giorni sospensioni nulli = ' + \
                  str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
        
        comuni_dataframe.drop(
            comuni_dataframe[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].index,
            inplace=True)

        comuni_dataframe_shelve = shelve.open(path_shelve + 'comuni_dataframe' + \
                                              sheet_suffix + '_lpf')
        comuni_dataframe_shelve['comuni_dataframe'] = comuni_dataframe
        comuni_dataframe_shelve.close()

    return comuni_dataframe


def get_comuni_measure_dataframe(comuni_excel_map, sheet_name, type_name=False, type_pdc_ov=True,
                                 load=True, path_base=False, lpf=False):
    if not path_base:
        path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\'
    path_shelve = path_base + 'pat_pnrr_mpe\\pat_pnrr_3a_misurazione_tabelle_comunali\\'

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
        path_shelve_file = path_shelve + 'comuni_measure_dataframe' + sheet_suffix
        if lpf:
            path_shelve_file += '_lpf'
        comuni_measure_dataframe_shelve = shelve.open(path_shelve_file)
        comuni_measure_dataframe = comuni_measure_dataframe_shelve['comuni_measure_dataframe']
        comuni_measure_dataframe_shelve.close()
    else:
        comuni_excel_map = [(comune[0], comune[1])
                            for comune in comuni_excel_map if comune[1] is not None]
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
                                                                     type_pdc_ov, lpf=lpf)
            comuni_measure_dataframe.append(comune_measure_series)
        comuni_measure_dataframe = pd.DataFrame(comuni_measure_dataframe, index=comuni_names)

        path_shelve_file = path_shelve + 'comuni_measure_dataframe' + sheet_suffix
        if lpf:
            path_shelve_file += '_lpf'
        comuni_measure_dataframe_shelve = shelve.open(path_shelve_file)
        comuni_measure_dataframe_shelve['comuni_measure_dataframe'] = comuni_measure_dataframe
        comuni_measure_dataframe_shelve.close()

    return comuni_measure_dataframe


def get_comuni_dataframes(comuni_excel_map, load=True):

    comuni_dataframe_pdc = get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire', load=load)
    comuni_dataframe_pds = get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria', load=load)
    comuni_dataframe_cila = get_comuni_dataframe(
        comuni_excel_map, 'Controllo CILA', load=load)

    return True


def get_comuni_measures_dataframe(comuni_excel_map, load=True):

    comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(
        comuni_excel_map, 'Permessi di Costruire', load=load)
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
                       measure_period='2022q3-4', load=True, lpf=False):

    comuni_measure_dataframe = get_comuni_measure_dataframe(
        comuni_excel_map=comuni_excel_map, sheet_name=sheet_name, type_name=type_name,
        type_pdc_ov=type_pdc_ov, load=load, lpf=lpf)

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

    return comuni_measure


def get_comuni_measures(comuni_excel_map, save_tex=False):

    comuni_pdc_measure = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=False)
    comuni_pdc_ov_measure = get_comuni_measure(
        comuni_excel_map, 'Permessi di Costruire')
    comuni_pds_measure = get_comuni_measure(
        comuni_excel_map, 'Prov di sanatoria')
    comuni_cila_measure = get_comuni_measure(
        comuni_excel_map, 'Controllo CILA')

    if save_tex:
        measurement_03_pratiche_header = [
            'Concluse con SA',
            'Concluse',
            'con sospensioni',
            'con CdS',
            'Durata [gg]',
            'Termine [gg]',
            'Avviate',
            'Arretrate']
        tex_file_header = measurement_03_pratiche_header

        comuni_pdc_measure.index = measurement_03_pratiche_header
        comuni_pdc_ov_measure.index = measurement_03_pratiche_header
        comuni_pds_measure.index = measurement_03_pratiche_header
        comuni_cila_measure.index = measurement_03_pratiche_header

        measurement_03_series = {
            'Permesso di Costruire OV': comuni_pdc_ov_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int)}
        measurement_03 = pd.DataFrame(measurement_03_series).T

        measurement_03b_series = {
            'Permesso di Costruire': comuni_pdc_measure.apply(np.rint).astype(int),
            'Provvedimento di Sanatoria': comuni_pds_measure.apply(np.rint).astype(int),
            'Controllo della CILA': comuni_cila_measure.apply(np.rint).astype(int)}
        measurement_03b = pd.DataFrame(measurement_03b_series).T

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_2022q3-4.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_03.columns = tex_file_header
            baseline_styler = measurement_03.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            table_tex_content = baseline_styler.to_latex(
                caption='PAT-PNRR | Procedimenti Edilizi | Misurazione 2022Q3-4',
                label='pat-pnrr_mpe_2022q3-4', position='!htbp', position_float="centering",
                hrules=True)
            table_tex_file.write(table_tex_content)

        tex_file_name = ('pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/'
                         'pat-pnrr_mpe_2022q3-4b.tex')
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_03b.columns = tex_file_header
            baseline_styler = measurement_03b.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            table_tex_content = baseline_styler.to_latex(
                caption='PAT-PNRR | Procedimenti Edilizi | Misurazione 2022Q3-4b',
                label='pat-pnrr_mpe_2022q3-4b', position='!htbp', position_float="centering",
                hrules=True)
            table_tex_file.write(table_tex_content)
    return True


class ComuneExcel:

    def __init__(self, path_file, comune_name='Test', path_base=''):
        if path_base == '':
            self.path_base = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\' \
                             'pat_pnrr_mpe\\pat_pnrr_3a_misurazione_tabelle_comunali\\'
        else:
            self.path_base = path_base
        self.path_file = path_file
        self.excel_path = self.path_base + self.path_file
        self.comune_name = comune_name
        self.excel_structure = {
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
                    'presentazione',
                    'termine',
                    'conclusione',
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
                    'sospensioni'
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
        
        # correzione arretrati 202405 | inserimento pratiche pdc e pds arretrate emerse a posteriori
        if sheet_name == 'Permessi di Costruire':
            sheet_name_correction_202405 = 'correzione_arretrati_pdc_202405'
            names = [
                    'comune',
                    'tipologia_pratica',
                    'id_pratica',
                    'data_inizio_pratica',
                    'giorni_termine_normativo',
                    'data_fine_pratica',
                    'data_fine_pratica_silenzio-assenso',
                    'conferenza_servizi',
                    'tipologia_massima_sospensione',
                    'giorni_sospensioni'
                ]
            usecols = [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                ]
        elif sheet_name == 'Prov di sanatoria':
            sheet_name_correction_202405 = 'correzione_arretrati_pds_202405'
            names = [
                    'comune',
                    'tipologia_pratica',
                    'id_pratica',
                    'data_inizio_pratica',
                    'giorni_termine_normativo',
                    'data_fine_pratica',
                    'tipologia_massima_sospensione',
                    'giorni_sospensioni'
                ]
            usecols = [
                    0, 1, 2, 3, 4, 5, 6, 7
                ]
        if (sheet_name == 'Permessi di Costruire') | (sheet_name == 'Prov di sanatoria'):
            excel_path_correction_202405 = self.path_base + sheet_name_correction_202405 + '.xls'
            comuni_correction_202405_dataframe = get_dataframe_excel(
                excel_path_correction_202405, sheet_name_correction_202405, names, usecols,
                skiprows=1, droprows=[], dtype=None)
            change_mask = comuni_correction_202405_dataframe.loc[:, 'comune'] != \
                self.comune_name
            comuni_correction_202405_dataframe.drop(
                comuni_correction_202405_dataframe[change_mask].index, inplace=True)
            if not comuni_correction_202405_dataframe.empty:
                comune_dataframe = pd.concat([comune_dataframe,
                                              comuni_correction_202405_dataframe],
                                              axis='rows', join='outer')
                comune_dataframe.reset_index(drop=True, inplace=True)

        if sheet_name == 'Permessi di Costruire':
            comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_pdc[0], inplace=True)
            comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(60, inplace=True)
            comune_dataframe.loc[:, 'conferenza_servizi'].fillna('NO', inplace=True)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
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
                    'string').str.contains('10/8/25022', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '10/8/2022'
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('03/05/20023', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '03/05/2023'
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
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
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'] < \
                          pd.Timestamp('2022-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica_silenzio-assenso'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
            comune_dataframe.loc[change_mask, 'data_fine_pratica_silenzio-assenso'] = pd.NaT

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
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('30 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 30
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('90 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 90
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
                if type(conferenza_servizi_originale) is float:
                    if conferenza_servizi_originale == 0:
                        conferenza_servizi_originale = 'no'
                    elif conferenza_servizi_originale != 0:
                        conferenza_servizi_originale = 'si'
                if type(conferenza_servizi_originale) is str:
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
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
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
                    'string').str.contains('04/072022', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = '04/07/2022'
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
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
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('30 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 30
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('60 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 60
                change_mask = comune_dataframe.loc[:, 'giorni_termine_normativo'].astype(
                    'string').str.contains('90 days', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'giorni_termine_normativo'] = 90
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

        if sheet_name == 'Controllo CILA':
            comune_dataframe.loc[:, 'tipologia_pratica'].fillna(types_cila[0], inplace=True)
            comune_dataframe.loc[:, 'giorni_termine_normativo'].fillna(0, inplace=True)
            comune_dataframe.loc[:, 'tipologia_massima_sospensione'].fillna('', inplace=True)
            comune_dataframe.loc[:, 'giorni_sospensioni'].fillna(0, inplace=True)

            try:
                comune_dataframe['data_inizio_pratica'] = pd.to_datetime(
                    comune_dataframe['data_inizio_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_inizio_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_inizio_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_inizio_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)

            if comune_dataframe.loc[:, 'data_fine_pratica'].dtype == 'O':
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] == ' '
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
                change_mask = comune_dataframe.loc[:, 'data_fine_pratica'].astype(
                    'string').str.contains('non concluso', case=False, na=False, regex=False)
                comune_dataframe.loc[change_mask, 'data_fine_pratica'] = None
            try:
                comune_dataframe['data_fine_pratica'] = pd.to_datetime(
                    comune_dataframe['data_fine_pratica'],
                    errors='raise', dayfirst=True)
            except:
                print('data_fine_pratica is UNKNOWN: ')
                print(comune_dataframe.loc[:, 'data_fine_pratica'])
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] < \
                          pd.Timestamp('2022-06-30 23:59:59.999')
            comune_dataframe.drop(comune_dataframe[change_mask].index, inplace=True)
            change_mask = comune_dataframe.loc[:, 'data_fine_pratica'] > \
                          pd.Timestamp('2022-12-31 23:59:59.999')
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
                                  measure_period='2022q3-4', lpf=False):
        if lpf:
            path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'
            path_to_excel_files = 'pat_pnrr_3a_misurazione_tabelle_comunali\\'
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
        # elif sheet_name == 'Prov di sanatoria':
        #     numero_pratiche_concluse_con_silenzio_assenso = 0
        #     filter_mask = comune_dataframe.loc[:, 'data_fine_pratica'].isna() == False
        #     filter_mask = filter_mask & (comune_dataframe.loc[:, 'conferenza_servizi'] == True)
        #     filter_mask = filter_mask & filter_type
        #     numero_pratiche_concluse_con_provvedimento_espresso_con_conferenza_servizi = \
        #         comune_dataframe[filter_mask].__len__()
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
        filter_mask = filter_mask & (comune_dataframe.loc[:, 'giorni_sospensioni']
                                     > pd.Timedelta(0, unit='D'))
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


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


    # list_excel = get_list_excel(missing=True)
    # check_comuni_excel()

    # comune = ComuneExcel('009_Baselga_Edilizia.xls', 'Baselga di Pinè')
    # comune = ComuneExcel('222_Ton_Edilizia.xlsx', 'Ton')

    # comune_dataframe_pdc = comune.get_comune_dataframe('Permessi di Costruire')
    # comune_dataframe_pds = comune.get_comune_dataframe('Prov di sanatoria')
    # comune_dataframe_cila = comune.get_comune_dataframe('Controllo CILA')

    # comune_measure_series_pdc = comune.get_comune_measure_series('Permessi di Costruire')
    # comune_measure_series_pds = comune.get_comune_measure_series('Prov di sanatoria')
    # comune_measure_series_cila = comune.get_comune_measure_series('Controllo CILA')


    # load = True
    # lpf = True
    # comuni_dataframe_pdc_03 = get_comuni_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', load=load, lpf=lpf)
    # comuni_dataframe_pds_03 = get_comuni_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', load=load, lpf=lpf)
    # comuni_dataframe_cila_03 = get_comuni_dataframe(
    #     comuni_excel_map, 'Controllo CILA', load=load)

    # load=False
    # comuni_measure_dataframe_pdc_ov = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', load=load)
    # comuni_measure_dataframe_pdc = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', type_pdc_ov=False, load=load)
    # comuni_measure_dataframe_pds = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', load=load)
    # comuni_measure_dataframe_cila = get_comuni_measure_dataframe(
    #     comuni_excel_map, 'Controllo CILA', load=load)
    # get_comuni_dataframes(comuni_excel_map, load=False)


    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire')
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria')
    # get_comuni_measure(comuni_excel_map, 'Controllo CILA')

    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire', 'PdC ordinario', load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire', 'PdC in variante', load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire', 'PdC in deroga', load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire', 'PdC convenzionato', load=False)
    # get_comuni_measure(comuni_excel_map, 'Permessi di Costruire', 'PdC asseverato', load=False)

    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria', 'PdC in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria', 'Provvedimenti in Sanatoria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Prov di sanatoria', 'Regolarizzazione', load=False)

    # get_comuni_measure(comuni_excel_map, 'Controllo CILA', 'CILA ordinaria', load=False)
    # get_comuni_measure(comuni_excel_map, 'Controllo CILA', 'CILA con sanzione', load=False)


    get_comuni_measures_dataframe(comuni_excel_map, load=False)
    get_comuni_measures(comuni_excel_map)

    # load = True
    # lpf = False
    # comuni_pdc_ov_measure = get_comuni_measure(
    #     comuni_excel_map, 'Permessi di Costruire', load=load, lpf=lpf)
    # comuni_pds_measure = get_comuni_measure(
    #     comuni_excel_map, 'Prov di sanatoria', load=load, lpf=lpf)


    # --


    # ATTENZIONE! MEDIA VS. MEDIA DI MEDIE

    # x = pd.DataFrame([1])
    # y = pd.DataFrame([3, 5, 7])
    # print(pd.concat([x, y]).mean())
    # print(np.mean([x.mean(), y.mean()]))


    # TEST 4a MISURAZIONE

    # path = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\doc\\'
    # comune = ComuneExcel(path_file='pat_pnrr_4a_misurazione_20230929_test.xls',
    #                      comune_name='test', path_base=path)
    # comune_dataframe_pdc = comune.get_comune_dataframe('Permessi di Costruire')
    # comune_dataframe_pds = comune.get_comune_dataframe('Prov di sanatoria')
    # comune_dataframe_cila = comune.get_comune_dataframe('Controllo CILA')


    # REQUEST 20240513_01 | pdc-ov non conclusi durata netta > 120 gg
    # - pdc-ov, non conclusi, mpe 03
    #   - data fine semestre (31/12/2022) - data inizio pratica - sospensione (se c'e')
    #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
    # measure_end_date = pd.Timestamp('2022-12-31')
    # filter_type = (comuni_dataframe_pdc_03.loc[:, 'tipologia_pratica'] ==
    #                   'PdC ordinario') ^ \
    #               (comuni_dataframe_pdc_03.loc[:, 'tipologia_pratica'] ==
    #                   'PdC in variante')
    # filter_mask = filter_type & (
    #     comuni_dataframe_pdc_03.loc[:, 'data_fine_pratica_silenzio-assenso'].isna())
    # filter_mask = filter_mask & (
    #     comuni_dataframe_pdc_03.loc[:, 'data_fine_pratica'].isna())
    # filtro_pratiche_non_concluse = (\
    #     measure_end_date - \
    #     comuni_dataframe_pdc_03.loc[filter_mask, 'data_inizio_pratica'] - \
    #     comuni_dataframe_pdc_03.loc[filter_mask, 'giorni_sospensioni']) > \
    #         pd.to_timedelta(120, errors='coerce', unit='D')
    # filtro_no_trento = comuni_dataframe_pdc_03.loc[:, 'comune'] != 'Trento'
    
    # pratiche_non_concluse = comuni_dataframe_pdc_03[
    #     filter_mask & filtro_pratiche_non_concluse]
    # filtro_non_concluse_giorni_sospensioni_nulli = \
    #     pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
    #         pd.to_timedelta(0, errors='coerce', unit='D')
    # pratiche_non_concluse_giorni_sospensioni_nulli = pratiche_non_concluse[
    #     filtro_non_concluse_giorni_sospensioni_nulli]
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
    #     filtro_non_concluse_giorni_sospensioni_nulli.sum()
    # print('numero pdc-ov mpe-03 non conclusi con giorni sospensioni nulli = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
    # comuni_dataframe_pdc_03[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].to_csv(
    #     'pat-pnrr_edilizia_pdc_ov_mpe_03_non_concluse_no_sospensioni_request_20240513_01_02.csv')
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento = \
    #     (filtro_non_concluse_giorni_sospensioni_nulli & \
    #      filtro_no_trento).sum()
    # print('numero pdc-ov mpe-03 non conclusi con giorni sospensioni nulli senza trento = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento))


    # REQUEST 20240515_01 | pds non conclusi durata netta > 120 gg
    # - pds, non conclusi, mpe 03
    #   - data fine semestre (31/12/2022) - data inizio pratica - sospensione (se c'e')
    #     - quante pratiche risultanti > 120 gg con sospensioni nulle?
    # measure_end_date = pd.Timestamp('2022-12-31')
    # filter_mask = (
    #     comuni_dataframe_pds_03.loc[:, 'data_fine_pratica'].isna())
    # filtro_pratiche_non_concluse = (\
    #     measure_end_date - \
    #     comuni_dataframe_pds_03.loc[filter_mask, 'data_inizio_pratica'] - \
    #     comuni_dataframe_pds_03.loc[filter_mask, 'giorni_sospensioni']) > \
    #         pd.to_timedelta(120, errors='coerce', unit='D')
    # filtro_no_trento = comuni_dataframe_pds_03.loc[:, 'comune'] != 'Trento'
    
    # pratiche_non_concluse = comuni_dataframe_pds_03[
    #     filter_mask & filtro_pratiche_non_concluse]
    # filtro_non_concluse_giorni_sospensioni_nulli = \
    #     pratiche_non_concluse.loc[:, 'giorni_sospensioni'] == \
    #         pd.to_timedelta(0, errors='coerce', unit='D')
    # pratiche_non_concluse_giorni_sospensioni_nulli = pratiche_non_concluse[
    #     filtro_non_concluse_giorni_sospensioni_nulli]
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli = \
    #     filtro_non_concluse_giorni_sospensioni_nulli.sum()
    # print('numero pds mpe-03 non conclusi con giorni sospensioni nulli = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli))
    # comuni_dataframe_pds_03[filter_mask & filtro_non_concluse_giorni_sospensioni_nulli].to_csv(
    #     'pat-pnrr_edilizia_pds_mpe_03_non_concluse_no_sospensioni_request_20240515_01_02.csv')
    # numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento = \
    #     (filtro_non_concluse_giorni_sospensioni_nulli & \
    #      filtro_no_trento).sum()
    # print('numero pds mpe-03 non conclusi con giorni sospensioni nulli senza trento = ' + \
    #       str(numero_pratiche_non_concluse_giorni_sospensioni_nulli_no_trento))
