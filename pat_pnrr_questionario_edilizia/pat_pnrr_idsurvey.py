"""
    PAT-PNRR Baseline June 2022
    ISPAT Survey Processing System
    Francesco Melchiori, 2022
"""


import shelve
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_pat_comuni_dataframe_idsurvey(path_base=''):

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

    idsurvey_data_baseline_permesso_costruire_labels = np.array([
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',  # 16
        'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',  # 4
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',  # 8
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',  # 12
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',  # 17
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',  # 20
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'])  # 24
    idsurvey_data_baseline_permesso_costruire = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(16, 4, 8, 12, 17, 20, 24),
        encoding='utf8')
    idsurvey_dataframe_baseline_permesso_costruire = pd.DataFrame(
        idsurvey_data_baseline_permesso_costruire,
        columns=idsurvey_data_baseline_permesso_costruire_labels,
        index=idsurvey_data_comuni)
    numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4']
    numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3_4.name =\
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4'
    numero_permessi_costruire_2021q3_4 =\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4']
    numero_permessi_costruire_2021q3_4.name =\
        'numero_permessi_costruire_2021q3-4'
    pat_comuni_dataframe_idsurvey_baseline_permesso_costruire = pd.concat(
        [idsurvey_dataframe_baseline_permesso_costruire,
         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3_4,
         numero_permessi_costruire_2021q3_4],
        axis='columns', join='outer')

    idsurvey_data_baseline_controllo_cila_labels = np.array([
        'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',  # 5
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',  # 9
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',  # 13
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',  # 18
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',  # 21
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4'])  # 25
    idsurvey_data_baseline_controllo_cila = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(5, 9, 13, 18, 21, 25),
        encoding='utf8')
    idsurvey_dataframe_baseline_controllo_cila = pd.DataFrame(
        idsurvey_data_baseline_controllo_cila,
        columns=idsurvey_data_baseline_controllo_cila_labels,
        index=idsurvey_data_comuni)
    numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4']
    numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3_4.name =\
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4'
    numero_controlli_cila_2021q3_4 =\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4']
    numero_controlli_cila_2021q3_4.name =\
        'numero_controlli_cila_2021q3-4'
    pat_comuni_dataframe_idsurvey_baseline_controllo_cila = pd.concat(
        [idsurvey_dataframe_baseline_controllo_cila,
         numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3_4,
         numero_controlli_cila_2021q3_4],
        axis='columns', join='outer')

    idsurvey_data_baseline_sanatoria_labels = np.array([
        'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',  # 6
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',  # 10
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',  # 14
        'giornate_durata_media_sanatorie_concluse_2021q3-4',  # 19
        'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',  # 22
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'])  # 26
    idsurvey_data_baseline_sanatoria = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv',
        dtype='i4', delimiter=',', skiprows=1,
        usecols=(6, 10, 14, 19, 22, 26),
        encoding='utf8')
    idsurvey_dataframe_baseline_sanatoria = pd.DataFrame(
        idsurvey_data_baseline_sanatoria,
        columns=idsurvey_data_baseline_sanatoria_labels,
        index=idsurvey_data_comuni)
    numero_sanatorie_concluse_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4']
    numero_sanatorie_concluse_con_provvedimento_espresso_2021q3_4.name =\
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4'
    numero_sanatorie_2021q3_4 =\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']
    numero_sanatorie_2021q3_4.name =\
        'numero_sanatorie_2021q3-4'
    pat_comuni_dataframe_idsurvey_baseline_sanatoria = pd.concat(
        [idsurvey_dataframe_baseline_sanatoria,
         numero_sanatorie_concluse_con_provvedimento_espresso_2021q3_4,
         numero_sanatorie_2021q3_4],
        axis='columns', join='outer')

    idsurvey_data_issue_labels = np.array([
        'portale_online_ufficio_tecnico',
        'protocollazione_digitale_ufficio_tecnico',
        'gestionale_digitale_ufficio_tecnico',
        'ordine_fase_critica_accesso_atti',
        'ordine_fase_critica_acquisizione_domanda',
        'ordine_fase_critica_istruttoria_domanda',
        'ordine_fase_critica_acquisizione_autorizzazioni',
        'ordine_fase_critica_acquisizione_integrazioni',
        'ordine_fase_critica_rilascio_provvedimento',
        'ordine_fattore_critico_completezza_pratica',
        'ordine_fattore_critico_complessita_pratica',
        'ordine_fattore_critico_volume_pratiche',
        'ordine_fattore_critico_varieta_pratiche',
        'ordine_fattore_critico_interpretazione_norme',
        'ordine_fattore_critico_interpretazione_atti',
        'ordine_fattore_critico_reperibilita_informazioni',
        'ordine_fattore_critico_molteplicita_interlocutori',
        'ordine_fattore_critico_adempimenti_amministrativi',
        'ordine_fattore_critico_mancanza_formazione',
        'ordine_fattore_critico_altro_fattore'])
    idsurvey_data_issues = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv', dtype='U7',
        delimiter=',', skiprows=1,
        usecols=(28, 29,
                 31, 33, 34, 35, 36, 37, 38, 39,
                 40, 41, 42, 43, 44, 45, 46, 47, 48, 49),
        encoding='utf8')
    idsurvey_data_issues[idsurvey_data_issues == ''] = 0
    idsurvey_data_issues = idsurvey_data_issues.astype(dtype='i4')
    idsurvey_dataframe_issues = pd.DataFrame(idsurvey_data_issues,
                                             columns=idsurvey_data_issue_labels,
                                             index=idsurvey_data_comuni)
    idsurvey_data_text_issue_labels = np.array([
        'nome_protocollazione_digitale_ufficio_tecnico',
        'nome_gestionale_digitale_ufficio_tecnico',
        'nome_fattore_critico_altro_fattore'])
    idsurvey_data_text_issues = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv', dtype='U256',
        delimiter=',', skiprows=1,
        usecols=(30, 32,
                 50),
        encoding='utf8')
    idsurvey_dataframe_text_issues = pd.DataFrame(idsurvey_data_text_issues,
                                                  columns=idsurvey_data_text_issue_labels,
                                                  index=idsurvey_data_comuni)
    pat_comuni_dataframe_issues = pd.concat(
        [idsurvey_dataframe_issues,
         idsurvey_dataframe_text_issues],
        axis='columns', join='outer')

    pat_comuni_dataframe_idsurvey = pd.concat(
        [pat_comuni_dataframe_idsurvey_baseline_permesso_costruire,
         pat_comuni_dataframe_idsurvey_baseline_controllo_cila,
         pat_comuni_dataframe_idsurvey_baseline_sanatoria,
         pat_comuni_dataframe_issues],
        axis='columns', join='outer')

    return pat_comuni_dataframe_idsurvey


def get_pat_comuni_dataframe_idsurvey_times(path_base=''):

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

    idsurvey_data_time_labels = np.array([
        'data_inizio_compilazione',
        'data_fine_compilazione'])
    idsurvey_data_times = np.loadtxt(
        path_base + 'idsurvey_data\\idsurvey_data.csv', dtype='U19',
        delimiter=',', skiprows=1,
        usecols=(0, 1),
        encoding='utf8')
    pat_comuni_dataframe_idsurvey_times = pd.DataFrame(idsurvey_data_times,
                                                       columns=idsurvey_data_time_labels,
                                                       index=idsurvey_data_comuni)

    return pat_comuni_dataframe_idsurvey_times


if __name__ == '__main__':

    # DATA SETUP
    idsurvey_comuni = np.loadtxt('./idsurvey_data/idsurvey_comuni.csv', dtype='U33',
                                 delimiter=',', skiprows=1, usecols=0, encoding='utf8')
    idsurvey_id = np.loadtxt('./idsurvey_data/idsurvey_comuni.csv', dtype='i4',
                             delimiter=',', skiprows=1, usecols=1, encoding='utf8')
    idsurvey_id_to_comune = dict(zip(idsurvey_id, idsurvey_comuni))

    clustering_results = shelve.open('clustering_results')
    clustering_labels = clustering_results['labels']
    clustering_results.close()
    idsurvey_id_to_cluster = dict(zip(idsurvey_id, clustering_labels))

    # idsurvey_survey = np.loadtxt('./idsurvey_data/idsurvey_survey.csv', dtype='U329, U12, U67',
    #                              delimiter=',', skiprows=1, usecols=(0, 1, 2), encoding='utf8')

    # array_to_analyze = idsurvey_survey
    # for column in ['f'+str(i) for i in range(len(array_to_analyze[0]))]:
    #     row_lenghts = np.array([len(row_element) for row_element in array_to_analyze[column]])
    #     print(np.amax(row_lenghts), array_to_analyze[column][np.argmax(row_lenghts)])

    idsurvey_data_id = np.loadtxt('./idsurvey_data/idsurvey_data.csv', dtype='i4',
                                  delimiter=',', skiprows=1,
                                  usecols=3,
                                  encoding='utf8')
    idsurvey_data_comuni = np.array([idsurvey_id_to_comune[id_comune]
                                     for id_comune in idsurvey_data_id])

    idsurvey_data_clusters = np.array([idsurvey_id_to_cluster[id_comune]
                                       for id_comune in idsurvey_data_id])
    idsurvey_dataframe_clusters = pd.DataFrame(idsurvey_data_clusters,
                                               columns=['cluster_label'],
                                               index=idsurvey_data_comuni)

    # BASELINE PROCESSING
    # idsurvey_data_baseline_labels = np.array([
    #     'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',  # 4
    #     'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',  # 5
    #     'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',  # 6
    #     'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',  # 8
    #     'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',  # 9
    #     'numero_sanatorie_concluse_con_sospensioni_2021q3-4',  # 10
    #     'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',  # 12
    #     'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',  # 13
    #     'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',  # 14
    #     'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',  # 16
    #     'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',  # 17
    #     'giornate_durata_media_controlli_cila_conclusi_2021q3-4',  # 18
    #     'giornate_durata_media_sanatorie_concluse_2021q3-4',  # 19
    #     'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',  # 20
    #     'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',  # 21
    #     'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',  # 22
    #     'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',  # 24
    #     'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',  # 25
    #     'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'])  # 26
    # idsurvey_data_baseline = np.loadtxt('./idsurvey_data/idsurvey_data.csv', dtype='i4',
    #                                     delimiter=',', skiprows=1,
    #                                     usecols=(4, 5, 6, 8, 9,
    #                                              10, 12, 13, 14, 16, 17, 18, 19,
    #                                              20, 21, 22, 24, 25, 26),
    #                                     encoding='utf8')
    # idsurvey_dataframe_baseline = pd.DataFrame(idsurvey_data_baseline,
    #                                            columns=idsurvey_data_baseline_labels,
    #                                            index=idsurvey_data_comuni)

    idsurvey_data_baseline_permesso_costruire_labels = np.array([
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',  # 16
        'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',  # 4
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',  # 8
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',  # 12
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',  # 17
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',  # 20
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'])  # 24
    idsurvey_data_baseline_permesso_costruire = np.loadtxt('./idsurvey_data/idsurvey_data.csv',
                                                           dtype='i4', delimiter=',', skiprows=1,
                                                           usecols=(16, 4, 8, 12, 17, 20, 24),
                                                           encoding='utf8')
    idsurvey_dataframe_baseline_permesso_costruire = pd.DataFrame(
        idsurvey_data_baseline_permesso_costruire,
        columns=idsurvey_data_baseline_permesso_costruire_labels,
        index=idsurvey_data_comuni)

    idsurvey_data_baseline_controllo_cila_labels = np.array([
        'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',  # 5
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',  # 9
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',  # 13
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',  # 18
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',  # 21
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4'])  # 25
    idsurvey_data_baseline_controllo_cila = np.loadtxt('./idsurvey_data/idsurvey_data.csv',
                                                       dtype='i4', delimiter=',', skiprows=1,
                                                       usecols=(5, 9, 13, 18, 21, 25),
                                                       encoding='utf8')
    idsurvey_dataframe_baseline_controllo_cila = pd.DataFrame(
        idsurvey_data_baseline_controllo_cila,
        columns=idsurvey_data_baseline_controllo_cila_labels,
        index=idsurvey_data_comuni)

    idsurvey_data_baseline_sanatoria_labels = np.array([
        'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',  # 6
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',  # 10
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',  # 14
        'giornate_durata_media_sanatorie_concluse_2021q3-4',  # 19
        'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',  # 22
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'])  # 26
    idsurvey_data_baseline_sanatoria = np.loadtxt('./idsurvey_data/idsurvey_data.csv',
                                                  dtype='i4', delimiter=',', skiprows=1,
                                                  usecols=(6, 10, 14, 19, 22, 26),
                                                  encoding='utf8')
    idsurvey_dataframe_baseline_sanatoria = pd.DataFrame(
        idsurvey_data_baseline_sanatoria,
        columns=idsurvey_data_baseline_sanatoria_labels,
        index=idsurvey_data_comuni)

    # baseline_permesso_costruire
    numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4']
    numero_permessi_costruire_2021q3_4 =\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_permesso_costruire[
            'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4']
    print('')
    print('PAT-PNRR | BASELINE | Giugno 2022')
    print('')
    print("Risposta al questionario ISPAT sull'edilizia ricevuta dai seguenti comuni:")
    for ordine, comune in enumerate(idsurvey_dataframe_baseline_permesso_costruire.index):
        print('    ' + str(ordine + 1) + '. ' + comune)
    print('')
    print('Denominazione procedura: Procedimenti Edilizi: Permesso di Costruire')
    print('Concluse con silenzio assenso (numero): '+str(
          idsurvey_dataframe_baseline_permesso_costruire[
              'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | Totali (numero): '+str(
          numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3_4
          .mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con sospensioni (numero): '+str(
          idsurvey_dataframe_baseline_permesso_costruire[
              'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con Conferenza dei Servizi (numero): '+str(
          idsurvey_dataframe_baseline_permesso_costruire[
              'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | Durata media (giornate): '+str(
          idsurvey_dataframe_baseline_permesso_costruire[
              'giornate_durata_media_permessi_costruire_conclusi_2021q3-4'
          ].mean().round(2)))
    print('Termine massimo (giornate): 60')
    print('Avviate (numero): '+str(
          numero_permessi_costruire_2021q3_4
          .mean().round(2)))
    print('Arretrato (numero): '+str(
          idsurvey_dataframe_baseline_permesso_costruire[
              'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'
          ].mean().round(2)))
    print('Note: media dei comuni trentini')

    # baseline_controllo_cila
    numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4']
    numero_controlli_cila_2021q3_4 =\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_controllo_cila[
            'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4']
    print('')
    print('Denominazione procedura: Procedimenti Edilizi: Controllo della CILA')
    print('Concluse con silenzio assenso (numero): 0')
    print('Concluse con provvedimento espresso | Totali (numero): '+str(
          numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3_4
          .mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con sospensioni (numero): '+str(
          idsurvey_dataframe_baseline_controllo_cila[
              'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con Conferenza dei Servizi (numero): '+str(
          idsurvey_dataframe_baseline_controllo_cila[
              'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | Durata media (giornate): '+str(
          idsurvey_dataframe_baseline_controllo_cila[
              'giornate_durata_media_controlli_cila_conclusi_2021q3-4'
          ].mean().round(2)))
    print('Termine massimo (giornate): 0')  # non esiste termine massimo per controlli cila
    print('Avviate (numero): '+str(
          numero_controlli_cila_2021q3_4
          .mean().round(2)))
    print('Arretrato (numero): '+str(
          idsurvey_dataframe_baseline_controllo_cila[
              'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4'
          ].mean().round(2)))
    print('Note: media dei comuni trentini')

    # baseline_sanatoria
    numero_sanatorie_concluse_con_provvedimento_espresso_2021q3_4 =\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4']
    numero_sanatorie_2021q3_4 =\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4'] +\
        idsurvey_dataframe_baseline_sanatoria[
            'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']
    print('')
    print('Denominazione procedura: Procedimenti Edilizi: Sanatoria')
    print('Concluse con silenzio assenso (numero): 0')
    print('Concluse con provvedimento espresso | Totali (numero): '+str(
          numero_sanatorie_concluse_con_provvedimento_espresso_2021q3_4
          .mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con sospensioni (numero): '+str(
          idsurvey_dataframe_baseline_sanatoria[
              'numero_sanatorie_concluse_con_sospensioni_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | di cui con Conferenza dei Servizi (numero): '+str(
          idsurvey_dataframe_baseline_sanatoria[
              'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'
          ].mean().round(2)))
    print('Concluse con provvedimento espresso | Durata media (giornate): '+str(
          idsurvey_dataframe_baseline_sanatoria[
              'giornate_durata_media_sanatorie_concluse_2021q3-4'
          ].mean().round(2)))
    print('Termine massimo (giornate): 60')
    print('Avviate (numero): '+str(
          numero_sanatorie_2021q3_4
          .mean().round(2)))
    print('Arretrato (numero): '+str(
          idsurvey_dataframe_baseline_sanatoria[
              'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'
          ].mean().round(2)))
    print('Note: media dei comuni trentini')

    # ISSUE PROCESSING
    idsurvey_data_issue_labels = np.array([
        'portale_online_ufficio_tecnico',
        'protocollazione_digitale_ufficio_tecnico',
        'gestionale_digitale_ufficio_tecnico',
        'ordine_fase_critica_accesso_atti',
        'ordine_fase_critica_acquisizione_domanda',
        'ordine_fase_critica_istruttoria_domanda',
        'ordine_fase_critica_acquisizione_autorizzazioni',
        'ordine_fase_critica_acquisizione_integrazioni',
        'ordine_fase_critica_rilascio_provvedimento',
        'ordine_fattore_critico_completezza_pratica',
        'ordine_fattore_critico_complessita_pratica',
        'ordine_fattore_critico_volume_pratiche',
        'ordine_fattore_critico_varieta_pratiche',
        'ordine_fattore_critico_interpretazione_norme',
        'ordine_fattore_critico_interpretazione_atti',
        'ordine_fattore_critico_reperibilita_informazioni',
        'ordine_fattore_critico_molteplicita_interlocutori',
        'ordine_fattore_critico_adempimenti_amministrativi',
        'ordine_fattore_critico_mancanza_formazione',
        'ordine_fattore_critico_altro_fattore'])
    idsurvey_data_issues = np.loadtxt('./idsurvey_data/idsurvey_data.csv', dtype='U7',
                                      delimiter=',', skiprows=1,
                                      usecols=(28, 29,
                                               31, 33, 34, 35, 36, 37, 38, 39,
                                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49),
                                      encoding='utf8')
    idsurvey_data_issues[idsurvey_data_issues == ''] = 0
    idsurvey_data_issues = idsurvey_data_issues.astype(dtype='i4')
    idsurvey_dataframe_issues = pd.DataFrame(idsurvey_data_issues,
                                             columns=idsurvey_data_issue_labels,
                                             index=idsurvey_data_comuni)

    idsurvey_data_text_issue_labels = np.array([
        'nome_protocollazione_digitale_ufficio_tecnico',
        'nome_gestionale_digitale_ufficio_tecnico',
        'nome_fattore_critico_altro_fattore'])
    idsurvey_data_text_issues = np.loadtxt('./idsurvey_data/idsurvey_data.csv', dtype='U256',
                                           delimiter=',', skiprows=1,
                                           usecols=(30, 32,
                                                    50),
                                           encoding='utf8')
    idsurvey_dataframe_text_issues = pd.DataFrame(idsurvey_data_text_issues,
                                                  columns=idsurvey_data_text_issue_labels,
                                                  index=idsurvey_data_comuni)

    # phase issues chart
    selected_phase_issues = idsurvey_dataframe_issues.iloc[:, 3:9]
    selected_phase_issues_bigger_0_gain = (selected_phase_issues != 0) * -7
    selected_phase_issues_1_gain = (selected_phase_issues == 1) * 3
    selected_phase_issues_2_gain = (selected_phase_issues == 2) * 1
    weighted_phase_issues = np.sum(np.abs(selected_phase_issues +
                                          selected_phase_issues_bigger_0_gain) +
                                   selected_phase_issues_1_gain +
                                   selected_phase_issues_2_gain)
    phase_issues_to_plot = weighted_phase_issues.sort_values(ascending=False)

    ax = plt.subplot(polar=True)
    theta = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    values = phase_issues_to_plot.values
    values = values * 100 / max(values)
    width = 2 * np.pi / 6
    width -= width * 0.1
    tick_label = [label[len('ordine_fase_critica_'):].replace('_', ' ').capitalize()
                  for label in phase_issues_to_plot.index]
    colors = plt.get_cmap('plasma').colors[50:]
    colors = colors[::-round(len(colors) / 6)]
    ax.grid(axis='x', alpha=0.25)
    ax.spines['polar'].set_visible(False)
    ax.bar(theta, values, width=width, tick_label=tick_label, color=colors, alpha=0.9,
           antialiased=True)
    for text_label in ax.get_xticklabels():
        text_label.set_size(6)
    for text_label in ax.get_yticklabels():
        text_label.set_size(6)
    plt.savefig('pat-pnrr_procedimenti_edilizi_fasi_critiche_' +
                datetime.datetime.today().strftime('%Y%m%d') +
                '.png', dpi=600)
    # plt.show()
    plt.close()

    # factor issues chart
    selected_factor_issues = idsurvey_dataframe_issues.iloc[:, 9:19]
    selected_factor_issues_bigger_0_gain = (selected_factor_issues != 0) * -11
    selected_factor_issues_1_gain = (selected_factor_issues == 1) * 3
    selected_factor_issues_2_gain = (selected_factor_issues == 2) * 1
    weighted_factor_issues = np.sum(np.abs(selected_factor_issues +
                                           selected_factor_issues_bigger_0_gain) +
                                    selected_factor_issues_1_gain +
                                    selected_factor_issues_2_gain)
    factor_issues_to_plot = weighted_factor_issues.sort_values(ascending=False)

    ax = plt.subplot(polar=True)
    theta = np.linspace(0.0, 2 * np.pi, 10, endpoint=False)
    values = factor_issues_to_plot.values
    values = values * 100 / max(values)
    width = 2 * np.pi / 10
    width -= width * 0.1
    tick_label = [label[len('ordine_fattore_critico_'):].replace('_', ' ').capitalize()
                  for label in factor_issues_to_plot.index]
    colors = plt.get_cmap('plasma').colors[50:]
    colors = colors[::-round(len(colors) / 10)]
    ax.grid(axis='x', alpha=0.25)
    ax.spines['polar'].set_visible(False)
    ax.bar(theta, values, width=width, tick_label=tick_label, color=colors, alpha=0.9,
           antialiased=True)
    for text_label in ax.get_xticklabels():
        text_label.set_size(6)
    for text_label in ax.get_yticklabels():
        text_label.set_size(6)
    plt.savefig('pat-pnrr_procedimenti_edilizi_fattori_critici_' +
                datetime.datetime.today().strftime('%Y%m%d') +
                '.png', dpi=600)
    # plt.show()
    plt.close()
