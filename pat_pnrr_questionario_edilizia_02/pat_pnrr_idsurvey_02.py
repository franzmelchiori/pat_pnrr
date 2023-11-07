"""
    PAT-PNRR Measurement December 2022
    ISPAT Survey Processing System
    Francesco Melchiori, 2022
"""


import numpy as np
import pandas as pd


def get_pat_comuni_dataframe_idsurvey_02(path_base=''):

    # DATA SETUP
    idsurvey_comuni = np.loadtxt(path_base + 'idsurvey_data\\idsurvey_comuni.csv', dtype='U33',
                                 delimiter=',', skiprows=1, usecols=0, encoding='utf8')
    idsurvey_id = np.loadtxt(path_base + 'idsurvey_data\\idsurvey_comuni.csv', dtype='i4',
                             delimiter=',', skiprows=1, usecols=1, encoding='utf8')
    idsurvey_id_to_comune = dict(zip(idsurvey_id, idsurvey_comuni))

    idsurvey_data_id = np.loadtxt(path_base + 'idsurvey_data\\idsurvey_data.csv', dtype='i4',
                                  delimiter=',', skiprows=1,
                                  usecols=3,
                                  encoding='utf8')
    idsurvey_data_comuni = np.array([idsurvey_id_to_comune[id_comune]
                                     for id_comune in idsurvey_data_id])

    idsurvey_data_measurement_permesso_costruire_labels = np.array([
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q1-2',  # 11
        'numero_permessi_costruire_conclusi_senza_sospensioni_2022q1-2',  # 4
        'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2',  # 7
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2',  # 10
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',  # 12
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2022q1-2',  # 15
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2'])  # 18
    idsurvey_data_measurement_permesso_costruire = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(11, 4, 7, 10, 12, 15, 18),
        encoding='utf8')
    idsurvey_dataframe_measurement_permesso_costruire = pd.DataFrame(
        idsurvey_data_measurement_permesso_costruire,
        columns=idsurvey_data_measurement_permesso_costruire_labels,
        index=idsurvey_data_comuni)
    numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1_2 =\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2']
    numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1_2.name =\
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1-2'
    numero_permessi_costruire_2022q1_2 =\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2022q1-2'] +\
        idsurvey_dataframe_measurement_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2']
    numero_permessi_costruire_2022q1_2.name =\
        'numero_permessi_costruire_2022q1-2'
    pat_comuni_dataframe_idsurvey_measurement_permesso_costruire = pd.concat(
        [idsurvey_dataframe_measurement_permesso_costruire,
         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1_2,
         numero_permessi_costruire_2022q1_2],
        axis='columns', join='outer')

    idsurvey_data_measurement_controllo_cila_labels = np.array([
        'numero_controlli_cila_conclusi_senza_sospensioni_2022q1-2',  # 5
        'numero_controlli_cila_conclusi_con_sospensioni_2022q1-2',  # 8
        'giornate_durata_media_controlli_cila_conclusi_2022q1-2',  # 13
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2022q1-2',  # 16
        'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2'])  # 19
    idsurvey_data_measurement_controllo_cila = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(5, 8, 13, 16, 19),
        encoding='utf8')
    idsurvey_dataframe_measurement_controllo_cila = pd.DataFrame(
        idsurvey_data_measurement_controllo_cila,
        columns=idsurvey_data_measurement_controllo_cila_labels,
        index=idsurvey_data_comuni)
    numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1_2 =\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2022q1-2']
    numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1_2.name =\
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1-2'
    numero_controlli_cila_2022q1_2 =\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_non_conclusi_non_scaduti_termini_2022q1-2'] +\
        idsurvey_dataframe_measurement_controllo_cila[
            'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2']
    numero_controlli_cila_2022q1_2.name =\
        'numero_controlli_cila_2022q1-2'
    pat_comuni_dataframe_idsurvey_measurement_controllo_cila = pd.concat(
        [idsurvey_dataframe_measurement_controllo_cila,
         numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1_2,
         numero_controlli_cila_2022q1_2],
        axis='columns', join='outer')

    idsurvey_data_measurement_sanatoria_labels = np.array([
        'numero_sanatorie_concluse_senza_sospensioni_2022q1-2',  # 6
        'numero_sanatorie_concluse_con_sospensioni_2022q1-2',  # 9
        'giornate_durata_media_sanatorie_concluse_2022q1-2',  # 14
        'numero_sanatorie_non_concluse_non_scaduti_termini_2022q1-2',  # 17
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2'])  # 20
    idsurvey_data_measurement_sanatoria = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(6, 9, 14, 17, 20),
        encoding='utf8')
    idsurvey_dataframe_measurement_sanatoria = pd.DataFrame(
        idsurvey_data_measurement_sanatoria,
        columns=idsurvey_data_measurement_sanatoria_labels,
        index=idsurvey_data_comuni)
    numero_sanatorie_concluse_con_provvedimento_espresso_2022q1_2 =\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2022q1-2']
    numero_sanatorie_concluse_con_provvedimento_espresso_2022q1_2.name =\
        'numero_sanatorie_concluse_con_provvedimento_espresso_2022q1-2'
    numero_sanatorie_2022q1_2 =\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2022q1-2'] +\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_non_concluse_non_scaduti_termini_2022q1-2'] +\
        idsurvey_dataframe_measurement_sanatoria[
            'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2']
    numero_sanatorie_2022q1_2.name =\
        'numero_sanatorie_2022q1-2'
    pat_comuni_dataframe_idsurvey_measurement_sanatoria = pd.concat(
        [idsurvey_dataframe_measurement_sanatoria,
         numero_sanatorie_concluse_con_provvedimento_espresso_2022q1_2,
         numero_sanatorie_2022q1_2],
        axis='columns', join='outer')

    pat_comuni_dataframe_idsurvey = pd.concat(
        [pat_comuni_dataframe_idsurvey_measurement_permesso_costruire,
         pat_comuni_dataframe_idsurvey_measurement_controllo_cila,
         pat_comuni_dataframe_idsurvey_measurement_sanatoria],
        axis='columns', join='outer')

    return pat_comuni_dataframe_idsurvey


if __name__ == '__main__':

    get_pat_comuni_dataframe_idsurvey_02('pat_pnrr_questionario_edilizia_02\\')
