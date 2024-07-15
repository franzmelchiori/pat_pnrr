"""
    PAT-PNRR Monitoring Analyzer
    Francesco Melchiori, 2024
"""


import subprocess
import shelve
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from pat_pnrr_clustering.pat_pnrr_clustering import *
from pat_pnrr_questionario_edilizia.pat_pnrr_idsurvey import *
from pat_pnrr_questionario_edilizia.pat_pnrr_raccolta_puntuale_trento import *
from pat_pnrr_questionario_edilizia.pat_pnrr_raccolta_puntuale import *
from pat_pnrr_questionario_edilizia_02.pat_pnrr_idsurvey_02 import *
from pat_pnrr_mpe.pat_pnrr_comuni_excel_mapping import *
from pat_pnrr_mpe import pat_pnrr_3a_misurazione as pat_pnrr_3a
from pat_pnrr_mpe import pat_pnrr_4a_misurazione as pat_pnrr_4a
from pat_pnrr_mpe import pat_pnrr_5a_misurazione as pat_pnrr_5a


def update_dataframe_subset(dataframe_to_update, dataframe_subset):

    dataframe_to_update.loc[dataframe_subset.index, dataframe_subset.columns] = dataframe_subset

    return True

def get_pat_comunita_valle():
    pat_comuni_toponimo = np.loadtxt(
        'pat_pnrr_clustering\\ispat_statistiche_base_20220314.csv',
        dtype='U33', delimiter=',', skiprows=1, usecols=0, encoding='utf8')

    ispat_comunita_valle = np.loadtxt(
        'pat_pnrr_questionario_edilizia_02\\doc\\ispat_comunita_valle.csv',
        dtype='U33', delimiter=',', usecols=2, encoding='utf8')

    pat_comunita_valle_dataframe_ispat = pd.DataFrame(
        ispat_comunita_valle,
        columns=['pat_comunita_valle'],
        index=pat_comuni_toponimo)

    return pat_comunita_valle_dataframe_ispat


def get_pat_comuni_dataframe(load=True):

    if load:
        pat_comuni_shelve = shelve.open('pat_comuni_dataframe')
        pat_comuni_dataframe = pat_comuni_shelve['pat_comuni_dataframe']
        pat_comuni_shelve.close()

    else:
        pat_comuni_dataframe_ispat =\
            get_pat_comuni_dataframe_ispat(
                'pat_pnrr_clustering\\',
                kmeans_clustering_original=True)
        pat_comunita_valle_dataframe_ispat = \
            get_pat_comunita_valle()
        pat_comuni_dataframe_idsurvey =\
            get_pat_comuni_dataframe_idsurvey(
                'pat_pnrr_questionario_edilizia\\')
        pat_comuni_dataframe_idsurvey_02 =\
            get_pat_comuni_dataframe_idsurvey_02(
                'pat_pnrr_questionario_edilizia_02\\')
        pat_comuni_dataframe_excel_03 =\
            pat_pnrr_3a.get_comuni_measures_dataframe(
                comuni_excel_map)
        pat_comuni_dataframe_excel_04 =\
            pat_pnrr_4a.get_comuni_measures_dataframe(
                comuni_excel_map)
        pat_comuni_dataframe_excel_05 =\
            pat_pnrr_5a.get_comuni_measures_dataframe(
                comuni_excel_map)
        pat_comuni_dataframe = pd.concat(
            [pat_comuni_dataframe_ispat,
             pat_comunita_valle_dataframe_ispat,
             pat_comuni_dataframe_idsurvey,
             pat_comuni_dataframe_idsurvey_02,
             pat_comuni_dataframe_excel_03,
             pat_comuni_dataframe_excel_04,
             pat_comuni_dataframe_excel_05],
            axis='columns', join='outer')

        pat_dataframe_raccolta_puntuale_trento =\
            get_pat_comuni_dataframe_raccolta_puntuale_trento(
                'pat_pnrr_questionario_edilizia\\')
        update_dataframe_subset(pat_comuni_dataframe, pat_dataframe_raccolta_puntuale_trento)

        pat_dataframe_raccolta_puntuale =\
            get_pat_dataframe_raccolta_puntuale(
                'pat_pnrr_questionario_edilizia\\')
        update_dataframe_subset(pat_comuni_dataframe, pat_dataframe_raccolta_puntuale)

        clustering_label_map = {
            0: 'Cluster 2: comuni medio-piccoli (31)',
            1: 'Cluster 1: comuni piccoli (130)',
            2: 'Cluster 3: comuni medi (3)',
            3: 'Cluster 4: Rovereto',
            4: 'Cluster 5: Trento'}
        pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] =\
            pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'].map(clustering_label_map)

        phase_issues_labels = [
            'pat_comuni_kmeans_clustering_labels',
            'ordine_fase_critica_accesso_atti',
            'ordine_fase_critica_acquisizione_domanda',
            'ordine_fase_critica_istruttoria_domanda',
            'ordine_fase_critica_acquisizione_autorizzazioni',
            'ordine_fase_critica_acquisizione_integrazioni',
            'ordine_fase_critica_rilascio_provvedimento']
        factor_issues_labels = [
            'pat_comuni_kmeans_clustering_labels',
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
            'ordine_fattore_critico_altro_fattore']

        issues_map = {
            0: 0,
            1: 13,
            2: 10,
            3: 8,
            4: 7,
            5: 6,
            6: 5,
            7: 4,
            8: 3,
            9: 2,
            10: 1}
        for phase_issues_label in phase_issues_labels[1:]:
            pat_comuni_dataframe[phase_issues_label] = \
                pat_comuni_dataframe[phase_issues_label].map(issues_map)
        for factor_issues_label in factor_issues_labels[1:]:
            pat_comuni_dataframe[factor_issues_label] = \
                pat_comuni_dataframe[factor_issues_label].map(issues_map)

        # CORRECTIONS June 30, 2022
        # Lavis
        pat_comuni_dataframe.loc['Lavis',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'] -= 15
        pat_comuni_dataframe.loc['Lavis',
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] += 11
        pat_comuni_dataframe.loc['Lavis',
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] += 4
        pat_comuni_dataframe.loc['Lavis',
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'] -= 26
        pat_comuni_dataframe.loc['Lavis',
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] += 5
        pat_comuni_dataframe.loc['Lavis',
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] += 21
        # Castello Tesino
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'] -= 11
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] += 9
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] += 2
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'] -= 5
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] += 4
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] += 1
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4'] -= 4
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'] += 0
        pat_comuni_dataframe.loc['Castello Tesino',
            'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4'] += 4
        # Fai della Paganella
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'] -= 5
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'] += 0
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4'] += 5
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'] -= 5
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_sanatorie_concluse_con_sospensioni_2021q3-4'] += 0
        pat_comuni_dataframe.loc['Fai della Paganella',
            'numero_sanatorie_concluse_senza_sospensioni_2021q3-4'] += 5

        # CORRECTIONS October 6, 2022
        # Bieno
        pat_comuni_dataframe.loc['Bieno',
            'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4'] *= -1

        # CORRECTIONS November 29, 2022
        # Civezzano
        pat_comuni_dataframe.loc['Civezzano',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 13
        # Lavis
        pat_comuni_dataframe.loc['Lavis',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 17
        pat_comuni_dataframe.loc['Lavis',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 17
        # Livo
        pat_comuni_dataframe.loc['Livo',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 6
        pat_comuni_dataframe.loc['Livo',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 6
        # Molveno
        pat_comuni_dataframe.loc['Molveno',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 8
        pat_comuni_dataframe.loc['Molveno',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 8
        # Nogaredo
        pat_comuni_dataframe.loc['Nogaredo',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 9
        pat_comuni_dataframe.loc['Nogaredo',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 9
        # Pomarolo
        pat_comuni_dataframe.loc['Pomarolo',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 6
        pat_comuni_dataframe.loc['Pomarolo',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 6
        # Sanzeno
        pat_comuni_dataframe.loc['Sanzeno',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 6
        # Spiazzo
        pat_comuni_dataframe.loc['Spiazzo',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 9
        pat_comuni_dataframe.loc['Spiazzo',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 9
        # Tenno
        pat_comuni_dataframe.loc['Tenno',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 10
        pat_comuni_dataframe.loc['Tenno',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 10
        # Ville d'Anaunia
        pat_comuni_dataframe.loc["Ville d'Anaunia",
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 24
        pat_comuni_dataframe.loc["Ville d'Anaunia",
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 24
        # Lona-Lases
        pat_comuni_dataframe.loc['Lona-Lases',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 4
        pat_comuni_dataframe.loc['Lona-Lases',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 4
        # Cimone
        pat_comuni_dataframe.loc['Cimone',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 1
        # Contà
        pat_comuni_dataframe.loc['Contà',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 2
        pat_comuni_dataframe.loc['Contà',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 1
        pat_comuni_dataframe.loc['Contà',
            'numero_permessi_costruire_conclusi_senza_sospensioni_2022q1-2'] += 1
        # Caderzone Terme
        pat_comuni_dataframe.loc['Caderzone Terme',
            'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'] -= 2
        pat_comuni_dataframe.loc['Caderzone Terme',
            'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'] += 2

        # CORRECTIONS September 20, 2023
        giornate_durata_media = [
            'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
            'giornate_durata_media_sanatorie_concluse_2021q3-4',
            'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',
            'giornate_durata_media_sanatorie_concluse_2022q1-2',
            'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4',
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4']
        pat_comuni_dataframe.loc[:, giornate_durata_media] = \
            pat_comuni_dataframe.loc[:, giornate_durata_media].replace(0, np.nan)

        pat_comuni_shelve = shelve.open('pat_comuni_dataframe')
        pat_comuni_shelve['pat_comuni_dataframe'] = pat_comuni_dataframe
        pat_comuni_shelve.close()

    return pat_comuni_dataframe


def print_baselines(pat_comuni_dataframe, mean=False, model_baselines=False,
                    exclude_clusters_3_4_5=False, termine_scadenza=60,
                    save_tex=False, save_csv=False):

    baseline_permessi_costruire_labels = [
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'numero_permessi_costruire_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4']
    baseline_controlli_cila_labels = [
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'numero_controlli_cila_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4']
    baseline_sanatorie_labels = [
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'numero_sanatorie_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']

    baseline_pratiche_header = [
        'Concluse con SA',
        'Concluse',
        'con sospensioni',
        'con CdS',
        'Durata [gg]',
        'Termine [gg]',
        'Avviate',
        'Arretrate']
    tex_file_header = baseline_pratiche_header

    if exclude_clusters_3_4_5:
        filter_clusters = (
            pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                'Cluster 1: comuni piccoli (130)') |\
            (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                'Cluster 2: comuni medio-piccoli (31)')
    else:
        filter_clusters = (
            pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] !=
                'none')

    baseline_permessi_costruire_166_rows = \
        pat_comuni_dataframe[baseline_permessi_costruire_labels][filter_clusters]
    baseline_permessi_costruire_166_rows.insert(
        5, 'giornate_termine_scadenza_2021q3-4',
        np.ones(baseline_permessi_costruire_166_rows.shape[0])*termine_scadenza)
    # baseline_permessi_costruire_166_rows = \
    #     baseline_permessi_costruire_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_permessi_costruire_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_permessi_costruire_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Permesso di Costruire | '
                        'Definizione della Baseline dei 166 Comuni Trentini',
                label='pat-pnrr_baseline_permessi_costruire_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    baseline_controlli_cila_166_rows = \
        pat_comuni_dataframe[baseline_controlli_cila_labels][filter_clusters]
    baseline_controlli_cila_166_rows.insert(
        0, 'numero_controlli_cila_conclusi_con_silenzio-assenso_2021q3-4',
        np.zeros(baseline_controlli_cila_166_rows.shape[0]))
    baseline_controlli_cila_166_rows.insert(
        5, 'giornate_termine_scadenza_2021q3-4',
        np.zeros(baseline_controlli_cila_166_rows.shape[0]))
    # baseline_controlli_cila_166_rows = \
    #     baseline_controlli_cila_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_controlli_cila_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_controlli_cila_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Controllo della CILA | '
                        'Definizione della Baseline dei 166 Comuni Trentini',
                label='pat-pnrr_baseline_controlli_cila_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    baseline_sanatorie_166_rows = \
        pat_comuni_dataframe[baseline_sanatorie_labels][filter_clusters]
    baseline_sanatorie_166_rows.insert(
        0, 'numero_sanatorie_concluse_con_silenzio-assenso_2021q3-4',
        np.zeros(baseline_sanatorie_166_rows.shape[0]))
    baseline_sanatorie_166_rows.insert(
        5, 'giornate_termine_scadenza_2021q3-4',
        np.ones(baseline_sanatorie_166_rows.shape[0])*termine_scadenza)
    # baseline_sanatorie_166_rows = \
    #     baseline_sanatorie_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_sanatorie_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_sanatorie_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Provvedimento di Sanatoria | '
                        'Definizione della Baseline dei 166 Comuni Trentini',
                label='pat-pnrr_baseline_sanatorie_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    if mean or model_baselines:
        baseline_permessi_costruire = baseline_permessi_costruire_166_rows\
            .mean().apply(np.ceil).astype(int)
        baseline_controlli_cila = baseline_controlli_cila_166_rows\
            .mean().apply(np.ceil).astype(int)
        baseline_sanatorie = baseline_sanatorie_166_rows\
            .mean().apply(np.ceil).astype(int)
    else:
        baseline_permessi_costruire = baseline_permessi_costruire_166_rows.sum().astype(int)
        baseline_controlli_cila = baseline_controlli_cila_166_rows.sum().astype(int)
        baseline_sanatorie = baseline_sanatorie_166_rows.sum().astype(int)
    baseline_permessi_costruire[
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_permessi_costruire_conclusi_2021q3-4']\
        [filter_clusters].mean())
    baseline_permessi_costruire['giornate_termine_scadenza_2021q3-4'] = termine_scadenza
    baseline_controlli_cila[
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_controlli_cila_conclusi_2021q3-4']\
        [filter_clusters].mean())
    baseline_controlli_cila['giornate_termine_scadenza_2021q3-4'] = 0
    baseline_sanatorie[
        'giornate_durata_media_sanatorie_concluse_2021q3-4'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_sanatorie_concluse_2021q3-4']\
        [filter_clusters].mean())
    baseline_sanatorie['giornate_termine_scadenza_2021q3-4'] = termine_scadenza

    baseline_permessi_costruire.index = baseline_pratiche_header
    baseline_controlli_cila.index = baseline_pratiche_header
    baseline_sanatorie.index = baseline_pratiche_header
    baseline_series = {'Permesso di Costruire': baseline_permessi_costruire,
                       'Provvedimento di Sanatoria': baseline_sanatorie,
                       'Controllo della CILA': baseline_controlli_cila}
    baseline = pd.DataFrame(baseline_series).T

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/pat-pnrr_mpe_2021q3-4.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            # table_tex_content = baseline.to_latex(
            #     header=tex_file_header, longtable=True, escape=False,
            #     caption='Procedimenti Edilizi | Definizione della Baseline | '
            #             'SA: silenzio-assenso, CdS: Conferenza dei Servizi',
            #     label='pat-pnrr_baseline')\
            #     .replace('<NA>', '-')\
            #     .replace('Continued on next page', 'Continua sulla prossima pagina')
            baseline.columns = tex_file_header
            baseline_styler = baseline.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            table_tex_content = baseline_styler.to_latex(
                caption='PAT-PNRR | Procedimenti Edilizi | Misurazione 2021Q3-4',
                label='pat-pnrr_mpe_2021q3-4', position='!htbp', position_float="centering",
                hrules=True)
            table_tex_file.write(table_tex_content)

    # MODEL BASELINES
    if model_baselines and not exclude_clusters_3_4_5:

        def get_baseline_model(baseline_166_rows, baseline):

            baseline_model = ((
                (baseline_166_rows.loc[['Contà', 'Pieve di Bono-Prezzo']].mean() * 130) +
                (baseline_166_rows.loc[['Baselga di Pinè', 'Moena']].mean() * 31) +
                (baseline_166_rows.loc[['Arco', 'Pergine Valsugana']].mean() * 3) +
                baseline_166_rows.loc['Rovereto'] +
                baseline_166_rows.loc['Trento']) / 166).apply(np.ceil).astype(int)

            baseline_model.index = baseline_pratiche_header

            baseline_difference = (((baseline_model * 100) / baseline) - 100)
            baseline_difference[np.isnan(baseline_difference)] = 0
            baseline_difference = baseline_difference.apply(np.ceil).astype(int)

            return baseline_model, baseline_difference

        baseline_permessi_costruire_model, baseline_permessi_costruire_difference =\
            get_baseline_model(baseline_permessi_costruire_166_rows, baseline_permessi_costruire)
        baseline_controlli_cila_model, baseline_controlli_cila_difference =\
            get_baseline_model(baseline_controlli_cila_166_rows, baseline_controlli_cila)
        baseline_sanatorie_model, baseline_sanatorie_difference =\
            get_baseline_model(baseline_sanatorie_166_rows, baseline_sanatorie)

        print('')
        print('Procedimenti Edilizi: Permessi di Costruire | Definizione della MODEL Baseline')
        print(baseline_permessi_costruire_model)
        print('Procedimenti Edilizi: Permessi di Costruire | Differenza dalla Baseline')
        print(baseline_permessi_costruire_difference)
        print('')
        print('Procedimenti Edilizi: Controlli delle CILA | Definizione della MODEL Baseline')
        print(baseline_controlli_cila_model)
        print('Procedimenti Edilizi: Controlli delle CILA | Differenza dalla Baseline')
        print(baseline_controlli_cila_difference)
        print('')
        print('Procedimenti Edilizi: Sanatorie | Definizione della MODEL Baseline')
        print(baseline_sanatorie_model)
        print('Procedimenti Edilizi: Sanatorie | Differenza dalla Baseline')
        print(baseline_sanatorie_difference)

    if not exclude_clusters_3_4_5:

        baseline_permessi_costruire_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             baseline_permessi_costruire_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_permessi_costruire_5_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_permessi_costruire_5_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Permesso di Costruire | '
                        'Definizione della Baseline dei 5 cluster di Comuni Trentini',
                label='pat-pnrr_baseline_permessi_costruire_5_rows')
                .replace('pat_comuni_kmeans_clustering_labels', '')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    if not exclude_clusters_3_4_5:

        baseline_controlli_cila_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             baseline_controlli_cila_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_controlli_cila_5_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_controlli_cila_5_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Controllo della CILA | '
                        'Definizione della Baseline dei 5 cluster di Comuni Trentini',
                label='pat-pnrr_baseline_controlli_cila_5_rows')
                .replace('pat_comuni_kmeans_clustering_labels', '')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    if not exclude_clusters_3_4_5:

        baseline_sanatorie_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             baseline_sanatorie_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_baseline_sanatorie_5_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(baseline_sanatorie_5_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Provvedimento di Sanatoria | '
                        'Definizione della Baseline dei 5 cluster di Comuni Trentini',
                label='pat-pnrr_baseline_sanatorie_5_rows')
                .replace('pat_comuni_kmeans_clustering_labels', '')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    if save_csv and not exclude_clusters_3_4_5:
        baseline.to_csv('baseline.csv')
        baseline_permessi_costruire_5_rows.to_csv('baseline_permessi_costruire_5_rows.csv')
        baseline_controlli_cila_5_rows.to_csv('baseline_controlli_cila_5_rows.csv')
        baseline_sanatorie_5_rows.to_csv('baseline_sanatorie_5_rows.csv')
        baseline_permessi_costruire_166_rows.to_csv('baseline_permessi_costruire_166_rows.csv')
        baseline_controlli_cila_166_rows.to_csv('baseline_controlli_cila_166_rows.csv')
        baseline_sanatorie_166_rows.to_csv('baseline_sanatorie_166_rows.csv')

    print('')
    print('Procedimenti Edilizi: Permesso di Costruire | Definizione della Baseline')
    print(baseline_permessi_costruire)
    print('')
    print('Procedimenti Edilizi: Controllo della CILA | Definizione della Baseline')
    print(baseline_controlli_cila)
    print('')
    print('Procedimenti Edilizi: Provvedimento di Sanatoria | Definizione della Baseline')
    print(baseline_sanatorie)

    return True


def print_measurement_02(pat_comuni_dataframe, mean=False, measurement_5_rows=False,
                         exclude_clusters_3_4_5=False, termine_scadenza=60,
                         save_tex=False, save_csv=False):

    measurement_02_permessi_costruire_labels = [
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q1-2',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1-2',
        'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2',
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',
        'numero_permessi_costruire_2022q1-2',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2']
    measurement_02_controlli_cila_labels = [
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1-2',
        'numero_controlli_cila_conclusi_con_sospensioni_2022q1-2',
        'giornate_durata_media_controlli_cila_conclusi_2022q1-2',
        'numero_controlli_cila_2022q1-2',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2']
    measurement_02_sanatorie_labels = [
        'numero_sanatorie_concluse_con_provvedimento_espresso_2022q1-2',
        'numero_sanatorie_concluse_con_sospensioni_2022q1-2',
        'giornate_durata_media_sanatorie_concluse_2022q1-2',
        'numero_sanatorie_2022q1-2',
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2']

    measurement_02_pratiche_header = [
        'Concluse con SA',
        'Concluse',
        'con sospensioni',
        'con CdS',
        'Durata [gg]',
        'Termine [gg]',
        'Avviate',
        'Arretrate']
    tex_file_header = measurement_02_pratiche_header

    if exclude_clusters_3_4_5:
        filter_clusters = (
            pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                'Cluster 1: comuni piccoli (130)') |\
            (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                'Cluster 2: comuni medio-piccoli (31)')
    else:
        filter_clusters = (
            pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] !=
                'none')

    measurement_02_permessi_costruire_166_rows = \
        pat_comuni_dataframe[measurement_02_permessi_costruire_labels][filter_clusters]
    measurement_02_permessi_costruire_166_rows.insert(
        5, 'giornate_termine_scadenza_2022q1-2',
        np.ones(measurement_02_permessi_costruire_166_rows.shape[0])*termine_scadenza)
    # measurement_02_permessi_costruire_166_rows = \
    #     measurement_02_permessi_costruire_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_measurement_02_permessi_costruire_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(measurement_02_permessi_costruire_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Permesso di Costruire | '
                        '2a Misurazione dei 166 Comuni Trentini',
                label='pat-pnrr_measurement_02_permessi_costruire_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    measurement_02_controlli_cila_166_rows = \
        pat_comuni_dataframe[measurement_02_controlli_cila_labels][filter_clusters]
    measurement_02_controlli_cila_166_rows.insert(
        0, 'numero_controlli_cila_conclusi_con_silenzio-assenso_2022q1-2',
        np.zeros(measurement_02_controlli_cila_166_rows.shape[0]))
    measurement_02_controlli_cila_166_rows.insert(
        3, 'numero_controlli_cila_conclusi_con_conferenza_servizi_2022q1-2',
        np.zeros(measurement_02_controlli_cila_166_rows.shape[0]))
    measurement_02_controlli_cila_166_rows.insert(
        5, 'giornate_termine_scadenza_2022q1-2',
        np.zeros(measurement_02_controlli_cila_166_rows.shape[0]))
    # measurement_02_controlli_cila_166_rows = \
    #     measurement_02_controlli_cila_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_measurement_02_controlli_cila_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(measurement_02_controlli_cila_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Controllo della CILA | '
                        '2a Misurazione dei 166 Comuni Trentini',
                label='pat-pnrr_measurement_02_controlli_cila_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    measurement_02_sanatorie_166_rows = \
        pat_comuni_dataframe[measurement_02_sanatorie_labels][filter_clusters]
    measurement_02_sanatorie_166_rows.insert(
        0, 'numero_sanatorie_concluse_con_silenzio-assenso_2022q1-2',
        np.zeros(measurement_02_sanatorie_166_rows.shape[0]))
    measurement_02_sanatorie_166_rows.insert(
        3, 'numero_sanatorie_concluse_con_conferenza_servizi_2022q1-2',
        np.zeros(measurement_02_sanatorie_166_rows.shape[0]))
    measurement_02_sanatorie_166_rows.insert(
        5, 'giornate_termine_scadenza_2022q1-2',
        np.ones(measurement_02_sanatorie_166_rows.shape[0])*termine_scadenza)
    # measurement_02_sanatorie_166_rows = \
    #     measurement_02_sanatorie_166_rows.astype('Int64')

    if save_tex and not exclude_clusters_3_4_5:
        tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                        'pat-pnrr_measurement_02_sanatorie_166_rows.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            table_tex_file.write(measurement_02_sanatorie_166_rows.to_latex(
                header=tex_file_header, longtable=True, escape=False,
                caption='Procedimenti Edilizi: Provvedimento di Sanatoria | '
                        '2a Misurazione dei 166 Comuni Trentini',
                label='pat-pnrr_measurement_02_sanatorie_166_rows')
                .replace('<NA>', '-')
                .replace('Continued on next page', 'Continua sulla prossima pagina'))

    if save_csv and not exclude_clusters_3_4_5:
        measurement_02_permessi_costruire_166_rows.to_csv(
            'measurement_02_permessi_costruire_166_rows.csv')
        measurement_02_controlli_cila_166_rows.to_csv(
            'measurement_02_controlli_cila_166_rows.csv')
        measurement_02_sanatorie_166_rows.to_csv(
            'measurement_02_sanatorie_166_rows.csv')

    if mean:
        measurement_02_permessi_costruire = measurement_02_permessi_costruire_166_rows\
            .mean().apply(np.ceil).astype(int)
        measurement_02_controlli_cila = measurement_02_controlli_cila_166_rows\
            .mean().apply(np.ceil).astype(int)
        measurement_02_sanatorie = measurement_02_sanatorie_166_rows\
            .mean().apply(np.ceil).astype(int)
    else:
        measurement_02_permessi_costruire = measurement_02_permessi_costruire_166_rows.sum().astype(int)
        measurement_02_controlli_cila = measurement_02_controlli_cila_166_rows.sum().astype(int)
        measurement_02_sanatorie = measurement_02_sanatorie_166_rows.sum().astype(int)
    measurement_02_permessi_costruire[
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_permessi_costruire_conclusi_2022q1-2']\
        [filter_clusters].mean())
    measurement_02_permessi_costruire['giornate_termine_scadenza_2022q1-2'] = termine_scadenza
    measurement_02_controlli_cila[
        'giornate_durata_media_controlli_cila_conclusi_2022q1-2'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_controlli_cila_conclusi_2022q1-2']\
        [filter_clusters].mean())
    measurement_02_controlli_cila['giornate_termine_scadenza_2022q1-2'] = 0
    measurement_02_sanatorie[
        'giornate_durata_media_sanatorie_concluse_2022q1-2'] = math.ceil(
        pat_comuni_dataframe['giornate_durata_media_sanatorie_concluse_2022q1-2']\
        [filter_clusters].mean())
    measurement_02_sanatorie['giornate_termine_scadenza_2022q1-2'] = termine_scadenza

    measurement_02_permessi_costruire.index = measurement_02_pratiche_header
    measurement_02_controlli_cila.index = measurement_02_pratiche_header
    measurement_02_sanatorie.index = measurement_02_pratiche_header
    measurement_02_series = {'Permesso di Costruire': measurement_02_permessi_costruire,
                             'Provvedimento di Sanatoria': measurement_02_sanatorie,
                             'Controllo della CILA': measurement_02_controlli_cila}
    measurement_02 = pd.DataFrame(measurement_02_series).T

    if save_tex and not exclude_clusters_3_4_5:
        # tex_file_name = 'pat_pnrr_report_monitoraggio/pat-pnrr_measurement_02.tex'
        # with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
        #     table_tex_file.write(measurement_02.to_latex(
        #         header=tex_file_header, longtable=True, escape=False,
        #         caption='Procedimenti Edilizi | 2a Misurazione',
        #         label='pat-pnrr_measurement_02')
        #         .replace('<NA>', '-')
        #         .replace('Continued on next page', 'Continua sulla prossima pagina'))
        tex_file_name = 'pat_pnrr_mpe/relazione_tecnica/pat-mpe_measures/pat-pnrr_mpe_2022q1-2.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
            measurement_02.columns = tex_file_header
            baseline_styler = measurement_02.style
            baseline_styler.applymap_index(lambda v: "rotatebox:{90}--rwrap", axis=1)
            table_tex_content = baseline_styler.to_latex(
                caption='PAT-PNRR | Procedimenti Edilizi | Misurazione 2022Q1-2',
                label='pat-pnrr_mpe_2022q1-2', position='!htbp', position_float="centering",
                hrules=True)
            table_tex_file.write(table_tex_content)

    if save_csv and not exclude_clusters_3_4_5:
        measurement_02.to_csv(
            'measurement_02.csv')

    if measurement_5_rows and not exclude_clusters_3_4_5:

        measurement_02_permessi_costruire_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             measurement_02_permessi_costruire_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

        if save_tex:
            tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                            'pat-pnrr_measurement_02_permessi_costruire_5_rows.tex'
            with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
                table_tex_file.write(measurement_02_permessi_costruire_5_rows.to_latex(
                    header=tex_file_header, longtable=True, escape=False,
                    caption='Procedimenti Edilizi: Permesso di Costruire | '
                            '2a Misurazione dei 5 cluster di Comuni Trentini',
                    label='pat-pnrr_measurement_02_permessi_costruire_5_rows')
                    .replace('pat_comuni_kmeans_clustering_labels', '')
                    .replace('Continued on next page', 'Continua sulla prossima pagina'))

        measurement_02_controlli_cila_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             measurement_02_controlli_cila_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

        if save_tex:
            tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                            'pat-pnrr_measurement_02_controlli_cila_5_rows.tex'
            with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
                table_tex_file.write(measurement_02_controlli_cila_5_rows.to_latex(
                    header=tex_file_header, longtable=True, escape=False,
                    caption='Procedimenti Edilizi: Controllo della CILA | '
                            '2a Misurazione dei 5 cluster di Comuni Trentini',
                    label='pat-pnrr_measurement_02_controlli_cila_5_rows')
                    .replace('pat_comuni_kmeans_clustering_labels', '')
                    .replace('Continued on next page', 'Continua sulla prossima pagina'))

        measurement_02_sanatorie_5_rows = pd.concat(
            [pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'],
             measurement_02_sanatorie_166_rows], axis='columns', join='outer').groupby(
            'pat_comuni_kmeans_clustering_labels').mean().apply(np.ceil).astype(int)

        if save_tex:
            tex_file_name = 'pat_pnrr_report_monitoraggio/' \
                            'pat-pnrr_measurement_02_sanatorie_5_rows.tex'
            with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
                table_tex_file.write(measurement_02_sanatorie_5_rows.to_latex(
                    header=tex_file_header, longtable=True, escape=False,
                    caption='Procedimenti Edilizi: Provvedimento di Sanatoria | '
                            '2a Misurazione dei 5 cluster di Comuni Trentini',
                    label='pat-pnrr_measurement_02_sanatorie_5_rows')
                    .replace('pat_comuni_kmeans_clustering_labels', '')
                    .replace('Continued on next page', 'Continua sulla prossima pagina'))

        if save_csv:
            measurement_02_permessi_costruire_5_rows.to_csv(
                'measurement_02_permessi_costruire_5_rows.csv')
            measurement_02_controlli_cila_5_rows.to_csv(
                'measurement_02_controlli_cila_5_rows.csv')
            measurement_02_sanatorie_5_rows.to_csv(
                'measurement_02_sanatorie_5_rows.csv')

    print('')
    print('Procedimenti Edilizi: Permesso di Costruire | 2a Misurazione')
    print(measurement_02_permessi_costruire)
    print('')
    print('Procedimenti Edilizi: Controllo della CILA | 2a Misurazione')
    print(measurement_02_controlli_cila)
    print('')
    print('Procedimenti Edilizi: Provvedimento di Sanatoria | 2a Misurazione')
    print(measurement_02_sanatorie)

    return True


def show_baselines(pat_comuni_dataframe):

    baseline_permessi_costruire_labels = [
        'pat_comuni_kmeans_clustering_labels',
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'numero_permessi_costruire_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4']
    baseline_controlli_cila_labels = [
        'pat_comuni_kmeans_clustering_labels',
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'numero_controlli_cila_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4']
    baseline_sanatorie_labels = [
        'pat_comuni_kmeans_clustering_labels',
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'numero_sanatorie_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']

    baseline_permessi_costruire = pat_comuni_dataframe[baseline_permessi_costruire_labels].groupby(
        'pat_comuni_kmeans_clustering_labels').mean()
    baseline_controlli_cila = pat_comuni_dataframe[baseline_controlli_cila_labels].groupby(
        'pat_comuni_kmeans_clustering_labels').mean()
    baseline_sanatorie = pat_comuni_dataframe[baseline_sanatorie_labels].groupby(
        'pat_comuni_kmeans_clustering_labels').mean()

    baseline_size = baseline_permessi_costruire.shape[1]-1
    colors = plt.get_cmap('plasma').colors[50:]
    color_clusters = colors[::-round(len(colors) / baseline_size)]

    baseline_permessi_costruire.plot.barh(
        # title='PAT-PNRR | Procedimenti Edilizi: Permesso di Costruire | 2021Q3-4',
        color=color_clusters, fontsize=15)
    baseline_sanatorie.plot.barh(
        # title='PAT-PNRR | Procedimenti Edilizi: Provvedimento di Sanatoria | 2021Q3-4',
        color=color_clusters, fontsize=15)
    baseline_controlli_cila.plot.barh(
        # title='PAT-PNRR | Procedimenti Edilizi: Controllo della CILA | 2021Q3-4',
        color=color_clusters, fontsize=15)
    plt.show()

    return True


def show_measurement_02(pat_comuni_dataframe):

    baseline_permessi_costruire_tempi = pat_comuni_dataframe[
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4']
    baseline_permessi_costruire_arretrati = pat_comuni_dataframe[
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4']

    measurement_02_permessi_costruire_tempi = pat_comuni_dataframe[
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2']
    measurement_02_permessi_costruire_arretrati = pat_comuni_dataframe[
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2']

    pat_comuni_kmeans_clustering = pat_comuni_dataframe[
        'pat_comuni_kmeans_clustering_labels']

    differenza_permessi_costruire_tempi = \
        baseline_permessi_costruire_tempi - measurement_02_permessi_costruire_tempi
    filter_out_comuni_nan = differenza_permessi_costruire_tempi.isna() == False
    differenza_permessi_costruire_tempi = \
        differenza_permessi_costruire_tempi[filter_out_comuni_nan]
    pat_comuni_kmeans_clustering = \
        pat_comuni_kmeans_clustering[filter_out_comuni_nan]

    differenza_permessi_costruire_tempi_cluster_1 = differenza_permessi_costruire_tempi[
        pat_comuni_kmeans_clustering == 'Cluster 1: comuni piccoli (130)'].sort_values()
    differenza_permessi_costruire_tempi_cluster_2 = differenza_permessi_costruire_tempi[
        pat_comuni_kmeans_clustering == 'Cluster 2: comuni medio-piccoli (31)'].sort_values()
    differenza_permessi_costruire_tempi_cluster_3_4_5 = differenza_permessi_costruire_tempi[
        (pat_comuni_kmeans_clustering == 'Cluster 3: comuni medi (3)') |
        (pat_comuni_kmeans_clustering == 'Cluster 4: Rovereto') |
        (pat_comuni_kmeans_clustering == 'Cluster 5: Trento')].sort_values()

    differenza_permessi_costruire_arretrati = \
        baseline_permessi_costruire_arretrati - measurement_02_permessi_costruire_arretrati
    differenza_permessi_costruire_arretrati = \
        differenza_permessi_costruire_arretrati[filter_out_comuni_nan]

    differenza_permessi_costruire_arretrati_cluster_1 = differenza_permessi_costruire_arretrati[
        pat_comuni_kmeans_clustering == 'Cluster 1: comuni piccoli (130)'].sort_values()
    differenza_permessi_costruire_arretrati_cluster_2 = differenza_permessi_costruire_arretrati[
        pat_comuni_kmeans_clustering == 'Cluster 2: comuni medio-piccoli (31)'].sort_values()
    differenza_permessi_costruire_arretrati_cluster_3_4_5 = differenza_permessi_costruire_arretrati[
        (pat_comuni_kmeans_clustering == 'Cluster 3: comuni medi (3)') |
        (pat_comuni_kmeans_clustering == 'Cluster 4: Rovereto') |
        (pat_comuni_kmeans_clustering == 'Cluster 5: Trento')].sort_values()

    baseline_controlli_cila_tempi = pat_comuni_dataframe[
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4']
    baseline_controlli_cila_arretrati = pat_comuni_dataframe[
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4']

    measurement_02_controlli_cila_tempi = pat_comuni_dataframe[
        'giornate_durata_media_controlli_cila_conclusi_2022q1-2']
    measurement_02_controlli_cila_arretrati = pat_comuni_dataframe[
        'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2']

    differenza_controlli_cila_tempi = \
        baseline_controlli_cila_tempi - measurement_02_controlli_cila_tempi

    differenza_controlli_cila_tempi_cluster_1 = differenza_controlli_cila_tempi[
        differenza_permessi_costruire_tempi_cluster_1.index]
    differenza_controlli_cila_tempi_cluster_2 = differenza_controlli_cila_tempi[
        differenza_permessi_costruire_tempi_cluster_2.index]
    differenza_controlli_cila_tempi_cluster_3_4_5 = differenza_controlli_cila_tempi[
        differenza_permessi_costruire_tempi_cluster_3_4_5.index]

    differenza_controlli_cila_arretrati = \
        baseline_controlli_cila_arretrati - measurement_02_controlli_cila_arretrati

    differenza_controlli_cila_arretrati_cluster_1 = differenza_controlli_cila_arretrati[
        differenza_permessi_costruire_arretrati_cluster_1.index]
    differenza_controlli_cila_arretrati_cluster_2 = differenza_controlli_cila_arretrati[
        differenza_permessi_costruire_arretrati_cluster_2.index]
    differenza_controlli_cila_arretrati_cluster_3_4_5 = differenza_controlli_cila_arretrati[
        differenza_permessi_costruire_arretrati_cluster_3_4_5.index]

    baseline_sanatorie_tempi = pat_comuni_dataframe[
        'giornate_durata_media_sanatorie_concluse_2021q3-4']
    baseline_sanatorie_arretrati = pat_comuni_dataframe[
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']

    measurement_02_sanatorie_tempi = pat_comuni_dataframe[
        'giornate_durata_media_sanatorie_concluse_2022q1-2']
    measurement_02_sanatorie_arretrati = pat_comuni_dataframe[
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2']

    differenza_sanatorie_tempi = \
        baseline_sanatorie_tempi - measurement_02_sanatorie_tempi

    differenza_sanatorie_tempi_cluster_1 = differenza_sanatorie_tempi[
        differenza_permessi_costruire_tempi_cluster_1.index]
    differenza_sanatorie_tempi_cluster_2 = differenza_sanatorie_tempi[
        differenza_permessi_costruire_tempi_cluster_2.index]
    differenza_sanatorie_tempi_cluster_3_4_5 = differenza_sanatorie_tempi[
        differenza_permessi_costruire_tempi_cluster_3_4_5.index]

    differenza_sanatorie_arretrati = \
        baseline_sanatorie_arretrati - measurement_02_sanatorie_arretrati

    differenza_sanatorie_arretrati_cluster_1 = differenza_sanatorie_arretrati[
        differenza_permessi_costruire_arretrati_cluster_1.index]
    differenza_sanatorie_arretrati_cluster_2 = differenza_sanatorie_arretrati[
        differenza_permessi_costruire_arretrati_cluster_2.index]
    differenza_sanatorie_arretrati_cluster_3_4_5 = differenza_sanatorie_arretrati[
        differenza_permessi_costruire_arretrati_cluster_3_4_5.index]

    fig, axs = plt.subplots(3, 3, sharey=True, gridspec_kw=dict(width_ratios=[130, 31, 5]))

    axs[0, 0].bar(differenza_permessi_costruire_tempi_cluster_1.index,
                  differenza_permessi_costruire_tempi_cluster_1,
                  color=np.where(differenza_permessi_costruire_tempi_cluster_1 > 0, 'g', 'r'))
    axs[0, 1].bar(differenza_permessi_costruire_tempi_cluster_2.index,
                  differenza_permessi_costruire_tempi_cluster_2,
                  color=np.where(differenza_permessi_costruire_tempi_cluster_2 > 0, 'g', 'r'))
    axs[0, 2].bar(differenza_permessi_costruire_tempi_cluster_3_4_5.index,
                  differenza_permessi_costruire_tempi_cluster_3_4_5,
                  color=np.where(differenza_permessi_costruire_tempi_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[0, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.set_xticks([])
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[1, 0].bar(differenza_controlli_cila_tempi_cluster_1.index,
                  differenza_controlli_cila_tempi_cluster_1,
                  color=np.where(differenza_controlli_cila_tempi_cluster_1 > 0, 'g', 'r'))
    axs[1, 1].bar(differenza_controlli_cila_tempi_cluster_2.index,
                  differenza_controlli_cila_tempi_cluster_2,
                  color=np.where(differenza_controlli_cila_tempi_cluster_2 > 0, 'g', 'r'))
    axs[1, 2].bar(differenza_controlli_cila_tempi_cluster_3_4_5.index,
                  differenza_controlli_cila_tempi_cluster_3_4_5,
                  color=np.where(differenza_controlli_cila_tempi_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[1, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.set_xticks([])
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[2, 0].bar(differenza_sanatorie_tempi_cluster_1.index,
                  differenza_sanatorie_tempi_cluster_1,
                  color=np.where(differenza_sanatorie_tempi_cluster_1 > 0, 'g', 'r'))
    axs[2, 1].bar(differenza_sanatorie_tempi_cluster_2.index,
                  differenza_sanatorie_tempi_cluster_2,
                  color=np.where(differenza_sanatorie_tempi_cluster_2 > 0, 'g', 'r'))
    axs[2, 2].bar(differenza_sanatorie_tempi_cluster_3_4_5.index,
                  differenza_sanatorie_tempi_cluster_3_4_5,
                  color=np.where(differenza_sanatorie_tempi_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[2, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0, 0].set_title('Comuni piccoli', fontsize=8)
    axs[0, 1].set_title('Comuni medi', fontsize=8)
    axs[0, 2].set_title('Comuni grandi', fontsize=8)
    axs[0, 0].set_ylabel('Permessi di Costruire [gg]', fontsize=10)
    axs[1, 0].set_ylabel('Controlli delle CILA [gg]', fontsize=10)
    axs[2, 0].set_ylabel('Provvedimenti di Sanatorie [gg]', fontsize=10)

    fig.suptitle('PAT-PNRR | Differenze fra Baseline e 2a Misurazione | '
                 'Tempi [giorni] di conclusione delle pratiche | '
                 'Procedimenti Edilizi 2022Q1-2', fontsize=12)

    plt.show()
    plt.close()

    fig, axs = plt.subplots(3, 3, sharey=True, gridspec_kw=dict(width_ratios=[130, 31, 5]))

    axs[0, 0].bar(differenza_permessi_costruire_arretrati_cluster_1.index,
                  differenza_permessi_costruire_arretrati_cluster_1,
                  color=np.where(differenza_permessi_costruire_arretrati_cluster_1 > 0, 'g', 'r'))
    axs[0, 1].bar(differenza_permessi_costruire_arretrati_cluster_2.index,
                  differenza_permessi_costruire_arretrati_cluster_2,
                  color=np.where(differenza_permessi_costruire_arretrati_cluster_2 > 0, 'g', 'r'))
    axs[0, 2].bar(differenza_permessi_costruire_arretrati_cluster_3_4_5.index,
                  differenza_permessi_costruire_arretrati_cluster_3_4_5,
                  color=np.where(differenza_permessi_costruire_arretrati_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[0, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.set_xticks([])
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[1, 0].bar(differenza_controlli_cila_arretrati_cluster_1.index,
                  differenza_controlli_cila_arretrati_cluster_1,
                  color=np.where(differenza_controlli_cila_arretrati_cluster_1 > 0, 'g', 'r'))
    axs[1, 1].bar(differenza_controlli_cila_arretrati_cluster_2.index,
                  differenza_controlli_cila_arretrati_cluster_2,
                  color=np.where(differenza_controlli_cila_arretrati_cluster_2 > 0, 'g', 'r'))
    axs[1, 2].bar(differenza_controlli_cila_arretrati_cluster_3_4_5.index,
                  differenza_controlli_cila_arretrati_cluster_3_4_5,
                  color=np.where(differenza_controlli_cila_arretrati_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[1, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.set_xticks([])
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[2, 0].bar(differenza_sanatorie_arretrati_cluster_1.index,
                  differenza_sanatorie_arretrati_cluster_1,
                  color=np.where(differenza_sanatorie_arretrati_cluster_1 > 0, 'g', 'r'))
    axs[2, 1].bar(differenza_sanatorie_arretrati_cluster_2.index,
                  differenza_sanatorie_arretrati_cluster_2,
                  color=np.where(differenza_sanatorie_arretrati_cluster_2 > 0, 'g', 'r'))
    axs[2, 2].bar(differenza_sanatorie_arretrati_cluster_3_4_5.index,
                  differenza_sanatorie_arretrati_cluster_3_4_5,
                  color=np.where(differenza_sanatorie_arretrati_cluster_3_4_5 > 0, 'g', 'r'))
    for ax in axs[2, :]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8, rotation=90)
        ax.yaxis.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0, 0].set_title('Comuni piccoli', fontsize=8)
    axs[0, 1].set_title('Comuni medi', fontsize=8)
    axs[0, 2].set_title('Comuni grandi', fontsize=8)
    axs[0, 0].set_ylabel('Permessi di Costruire [n]', fontsize=10)
    axs[1, 0].set_ylabel('Controlli delle CILA [n]', fontsize=10)
    axs[2, 0].set_ylabel('Provvedimenti di Sanatorie [n]', fontsize=10)

    fig.suptitle('PAT-PNRR | Differenze fra Baseline e 2a Misurazione | '
                 'Pratiche arretrate [numero] | '
                 'Procedimenti Edilizi 2022Q1-2', fontsize=12)

    plt.show()
    plt.close()

    grandezza_comunale = pat_comuni_dataframe['pat_comuni_popolazione']
    classificazione_comunale = pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels']
    classificazione_comunale_labels = [
        'Cluster 1: comuni piccoli (130) ∝ popolazione',
        'Cluster 2: comuni medio-piccoli (31) ∝ popolazione',
        'Cluster 3: comuni medi (3) ∝ popolazione',
        'Cluster 4: Rovereto ∝ popolazione',
        'Cluster 5: Trento ∝ popolazione']
    classificazione_comunale_map = {
        'Cluster 1: comuni piccoli (130)': 0,
        'Cluster 2: comuni medio-piccoli (31)': 1,
        'Cluster 3: comuni medi (3)': 2,
        'Cluster 4: Rovereto': 3,
        'Cluster 5: Trento': 4}

    numero_pratiche_avviate_2022q1_2 = \
        pat_comuni_dataframe['numero_permessi_costruire_2022q1-2'] + \
        pat_comuni_dataframe['numero_controlli_cila_2022q1-2'] + \
        pat_comuni_dataframe['numero_sanatorie_2022q1-2']
    numero_pratiche_arretrate_2022q1_2 = \
        pat_comuni_dataframe['numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2'] + \
        pat_comuni_dataframe['numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2'] + \
        pat_comuni_dataframe['numero_sanatorie_non_concluse_scaduti_termini_2022q1-2']
    giornate_pratiche_elaborate_2022q1_2 = \
        pat_comuni_dataframe['giornate_durata_media_permessi_costruire_conclusi_2022q1-2'] + \
        pat_comuni_dataframe['giornate_durata_media_controlli_cila_conclusi_2022q1-2'] + \
        pat_comuni_dataframe['giornate_durata_media_sanatorie_concluse_2022q1-2']

    arretrato_comunale_2022q1_2 = numero_pratiche_arretrate_2022q1_2 / \
                                  numero_pratiche_avviate_2022q1_2
    elaborazione_comunale_2022q1_2 = giornate_pratiche_elaborate_2022q1_2 / \
                                     numero_pratiche_avviate_2022q1_2
    classificazione_comunale = classificazione_comunale.map(classificazione_comunale_map)

    filter_out_comuni_nan = arretrato_comunale_2022q1_2.isna() == False
    arretrato_comunale_2022q1_2 = arretrato_comunale_2022q1_2[filter_out_comuni_nan]
    elaborazione_comunale_2022q1_2 = elaborazione_comunale_2022q1_2[filter_out_comuni_nan]
    grandezza_comunale_2022q1_2 = grandezza_comunale[filter_out_comuni_nan]
    classificazione_comunale_2022q1_2 = classificazione_comunale[filter_out_comuni_nan]

    numero_pratiche_avviate_2021q3_4 = \
        pat_comuni_dataframe['numero_permessi_costruire_2021q3-4'] + \
        pat_comuni_dataframe['numero_controlli_cila_2021q3-4'] + \
        pat_comuni_dataframe['numero_sanatorie_2021q3-4']
    numero_pratiche_arretrate_2021q3_4 = \
        pat_comuni_dataframe['numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'] + \
        pat_comuni_dataframe['numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4'] + \
        pat_comuni_dataframe['numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']
    giornate_pratiche_elaborate_2021q3_4 = \
        pat_comuni_dataframe['giornate_durata_media_permessi_costruire_conclusi_2021q3-4'] + \
        pat_comuni_dataframe['giornate_durata_media_controlli_cila_conclusi_2021q3-4'] + \
        pat_comuni_dataframe['giornate_durata_media_sanatorie_concluse_2021q3-4']

    arretrato_comunale_2021q3_4 = numero_pratiche_arretrate_2021q3_4 / \
                                  numero_pratiche_avviate_2021q3_4
    elaborazione_comunale_2021q3_4 = giornate_pratiche_elaborate_2021q3_4 / \
                                     numero_pratiche_avviate_2021q3_4

    filter_out_comuni_nan = arretrato_comunale_2021q3_4.isna() == False
    arretrato_comunale_2021q3_4 = arretrato_comunale_2021q3_4[filter_out_comuni_nan]
    elaborazione_comunale_2021q3_4 = elaborazione_comunale_2021q3_4[filter_out_comuni_nan]
    grandezza_comunale_2021q3_4 = grandezza_comunale[filter_out_comuni_nan]
    classificazione_comunale_2021q3_4 = classificazione_comunale[filter_out_comuni_nan]

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    plot1 = axs[0].scatter(arretrato_comunale_2021q3_4, elaborazione_comunale_2021q3_4,
        c=classificazione_comunale_2021q3_4, s=grandezza_comunale_2021q3_4, alpha=0.5)
    axs[1].scatter(arretrato_comunale_2022q1_2, elaborazione_comunale_2022q1_2,
                   c=classificazione_comunale_2022q1_2, s=grandezza_comunale_2022q1_2, alpha=0.5)
    for ax in axs[:]:
        for label in ax.get_xticklabels(which='major'):
            label.set(fontsize=8)
        ax.set_xbound(upper=0.6)
        ax.set_ybound(lower=-30, upper=80)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axs[0].set_xlabel('pratiche arretrate / pratiche avviate 2021Q3-4', fontsize=10)
    axs[0].set_ylabel('giornate elaborazione / pratiche avviate 2021Q3-4', fontsize=10)
    axs[1].set_xlabel('pratiche arretrate / pratiche avviate 2022Q1-2', fontsize=10)
    axs[1].set_ylabel('giornate elaborazione / pratiche avviate 2022Q1-2', fontsize=10)
    axs[0].legend(plot1.legend_elements()[0], classificazione_comunale_labels, prop={'size': 12})

    fig.suptitle('PAT-PNRR | Differenze fra Baseline e 2a Misurazione | '
                 'Pratiche arretrate [/avviate] ed elaborate [/avviate] | '
                 'Procedimenti Edilizi 2022Q1-2', fontsize=12)

    plt.show()
    plt.close()

    return True


def show_measurement_03(pat_comuni_dataframe):
    filter_giornate_durata_media_permessi_costruire_conclusi = [
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4'
    ]
    filter_giornate_durata_media_controlli_cila_conclusi = [
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2022q1-2',
        'giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso_2022q3-4'
    ]
    filter_giornate_durata_media_sanatorie_concluse = [
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2022q1-2',
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'
    ]

    filter_numero_permessi_costruire_non_conclusi_scaduti_termini = [
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'
    ]
    filter_numero_controlli_cila_non_conclusi_scaduti_termini = [
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2',
        'numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'
    ]
    filter_numero_sanatorie_non_concluse_scaduti_termini = [
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4'
    ]

    filters_type = [
        filter_giornate_durata_media_permessi_costruire_conclusi,
        filter_giornate_durata_media_controlli_cila_conclusi,
        filter_giornate_durata_media_sanatorie_concluse,
        filter_numero_permessi_costruire_non_conclusi_scaduti_termini,
        filter_numero_controlli_cila_non_conclusi_scaduti_termini,
        filter_numero_sanatorie_non_concluse_scaduti_termini
    ]
    filters_name = [
        'PdC [gg]',
        'CILA [gg]',
        'PdS [gg]',
        'PdC [arr]',
        'CILA [arr]',
        'PdS [arr]'
    ]
    clasters_name = [
        'Cluster 1: comuni piccoli (130)',
        'Cluster 2: comuni medio-piccoli (31)',
        'Cluster 3: comuni medi (3)',
        'Cluster 4: Rovereto',
        'Cluster 5: Trento'
    ]

    fig, axs = plt.subplots(2, 3, sharey=True)
    for ax, filter_type, filter_name in zip(axs.flat, filters_type, filters_name):
        for cluster_name in clasters_name:
            filter_cluster = \
                pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] == cluster_name
            ax.plot(pat_comuni_dataframe.loc[filter_cluster, filter_type].mean())
            ax.set_ylabel(filter_name, fontsize=10)
            if 'arr' in filter_name:
                ax.set_xticklabels(['2021q3-4', '2022q1-2', '2022q3-4'])
            else:
                ax.set_xticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)


    fig.suptitle('PAT-PNRR | Andamento di tempi ed arretrati dei '
                 'Procedimenti Edilizi oggetto di monitoraggio | '
                 'Misurazioni 2021Q3-4, 2022Q1-2 e 2022Q3-4', fontsize=12)
    fig.legend(clasters_name, prop={'size': 12}, loc='lower center')
    plt.show()
    # plt.close()

    return True


def show_issues(pat_comuni_dataframe):

    def plot_issues(issues_series, issues_to_plot, issues_dataframe=False, title='', font_size=15):
        plt.figure()
        ax = plt.subplot(polar=True)
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.25)
        ax.spines['polar'].set_visible(False)
        for text_label in ax.get_xticklabels():
            text_label.set_size(font_size)
        for text_label in ax.get_yticklabels():
            text_label.set_size(font_size)
        colors = plt.get_cmap('plasma').colors[50:]

        if issues_dataframe is not False:
            issues_amount = issues_dataframe.shape[1]
            clusters_amount = issues_dataframe.shape[0]
            color_issues = 'black'
            alpha_issues = 0.25
            color_clusters = colors[::-round(len(colors) / clusters_amount)]
        else:
            issues_amount = issues_series.shape[0]
            color_issues = colors[::-round(len(colors) / issues_amount)]
            alpha_issues = 0.9

        values = issues_series[issues_to_plot]
        width = 2 * np.pi / issues_amount
        width -= width * 0.1
        theta = np.linspace(0, 2 * np.pi, issues_amount, endpoint=False)
        tick_label = [label.replace('ordine_fase_critica_', '')
                      .replace('ordine_fattore_critico_', '')
                      .replace('_', ' ').capitalize()
                      for label in issues_to_plot]
        ax.bar(theta, values, width=width, tick_label=tick_label, color=color_issues,
               alpha=alpha_issues, antialiased=True)

        if issues_dataframe is not False:
            for cluster_number, cluster_name in enumerate(issues_dataframe.index):
                color_cluster = color_clusters[cluster_number]
                values = issues_dataframe.loc[cluster_name][issues_to_plot]
                width = 2 * np.pi / (issues_amount * clusters_amount)
                width -= width * 0.1
                theta = np.linspace(0 + (((clusters_amount - 1) / 2) * width),
                                    2 * np.pi + (((clusters_amount - 1) / 2) * width),
                                    issues_amount, endpoint=False) - (cluster_number * width)
                ax.bar(theta, values, width=width, color=color_cluster,
                       alpha=0.9, antialiased=True)
            ax.legend(['Comuni PAT (166)'] + [cluster for cluster in issues_dataframe.index],
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=clusters_amount+1)

        # plt.savefig('pat-pnrr_procedimenti_edilizi_fasi_critiche_' +
        #             datetime.datetime.today().strftime('%Y%m%d') +
        #             '.png', dpi=600)
        # plt.close()
        plt.show()
        return True

    phase_issues_labels = [
        'pat_comuni_kmeans_clustering_labels',
        'ordine_fase_critica_accesso_atti',
        'ordine_fase_critica_acquisizione_domanda',
        'ordine_fase_critica_istruttoria_domanda',
        'ordine_fase_critica_acquisizione_autorizzazioni',
        'ordine_fase_critica_acquisizione_integrazioni',
        'ordine_fase_critica_rilascio_provvedimento']
    factor_issues_labels = [
        'pat_comuni_kmeans_clustering_labels',
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
        'ordine_fattore_critico_altro_fattore']

    # PLOT PHASE ISSUE CHARTS
    phase_issues_series = pat_comuni_dataframe[phase_issues_labels[1:]].sum()
    phase_issues_series = (phase_issues_series * 100 / phase_issues_series.max())
    phase_issues_to_plot = phase_issues_series.sort_values(ascending=False).index

    phase_issues_dataframe = pat_comuni_dataframe[phase_issues_labels].groupby(
        'pat_comuni_kmeans_clustering_labels').sum().T
    phase_issues_dataframe = (phase_issues_dataframe * 100 / phase_issues_dataframe.max()).T

    # plot_issues(phase_issues_series, phase_issues_to_plot, False,
    #             'PAT-PNRR | Procedimenti edilizi: fasi critiche | 2021Q3-4')
    plot_issues(phase_issues_series, phase_issues_to_plot, phase_issues_dataframe)  # ,
                # 'PAT-PNRR | Procedimenti edilizi: fasi critiche | 2021Q3-4')

    # PLOT FACTOR ISSUE CHARTS
    factor_issues_series = pat_comuni_dataframe[factor_issues_labels[1:]].sum()
    factor_issues_series = (factor_issues_series * 100 / factor_issues_series.max())
    factor_issues_to_plot = factor_issues_series.sort_values(ascending=False).index

    factor_issues_dataframe = pat_comuni_dataframe[factor_issues_labels].groupby(
        'pat_comuni_kmeans_clustering_labels').sum().T
    factor_issues_dataframe = (factor_issues_dataframe * 100 / factor_issues_dataframe.max()).T

    # plot_issues(factor_issues_series, factor_issues_to_plot, False,
    #             'PAT-PNRR | Procedimenti edilizi: fattori critici | 2021Q3-4')
    plot_issues(factor_issues_series, factor_issues_to_plot, factor_issues_dataframe)  # ,
                # 'PAT-PNRR | Procedimenti edilizi: fattori critici | 2021Q3-4')

    return True


def print_survey_report_images(pat_comuni_dataframe, images_load=True, tex_load=True,
                               report_load=True):

    survey_report_volume_metrics = [
        'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'
    ]

    survey_report_volume_metric_labels = [
        '1a', '1b', '1c', '2a', '2b', '2c', '3', '4', '6a', '6b', '6c', '7a', '7b', '7c'
    ]

    survey_report_time_metrics = [
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4'
    ]

    survey_report_time_metric_labels = [
        '5a', '5b', '5c'
    ]

    comuni_nodata = pat_comuni_dataframe.index.isin(['Mori', 'Nomi', 'Novaledo'])
    comuni_cluster_1 = pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==\
        'Cluster 1: comuni piccoli (130)'
    comuni_cluster_2 = pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==\
        'Cluster 2: comuni medio-piccoli (31)'
    comuni_cluster_3_4_5 = (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                            'Cluster 3: comuni medi (3)') | \
                           (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                            'Cluster 4: Rovereto') | \
                           (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
                            'Cluster 5: Trento')
    comuni_clusters = [comuni_cluster_1, comuni_cluster_2, comuni_cluster_3_4_5]

    if not images_load:
        for cluster_n, comuni_cluster_n in enumerate(comuni_clusters):
        # for cluster_n, comuni_cluster_n in enumerate([comuni_cluster_1]):
            survey_report_volume_values = pat_comuni_dataframe[~comuni_nodata & comuni_cluster_n][
                survey_report_volume_metrics]
            survey_report_volume_ticks = np.arange(
                0, survey_report_volume_values.values.max()*1.075, 10)
            survey_report_volume_range = (
                survey_report_volume_ticks[0], survey_report_volume_ticks[-1])

            survey_report_time_values = pat_comuni_dataframe[~comuni_nodata & comuni_cluster_n][
                survey_report_time_metrics]
            survey_report_time_ticks = np.arange(
                0, survey_report_time_values.values.max()*1.075, 10)
            survey_report_time_range = (survey_report_time_ticks[0], survey_report_time_ticks[-1])

            for comune_n in range(survey_report_volume_values.shape[0]):
            # for comune_n in range(1):
                survey_report_comune_volumes = survey_report_volume_values[comune_n:comune_n + 1]
                survey_report_comune_times = survey_report_time_values[comune_n:comune_n + 1]
                survey_report_comune_name = survey_report_comune_volumes.index[0]

                if cluster_n == 0:
                    cluster_index = '1'
                if cluster_n == 1:
                    cluster_index = '2'
                if cluster_n == 2:
                    cluster_index = '3-4-5'

                fig, ax = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[4.7, 1]))
                fig.set_size_inches(15, 5)

                ax[0].set_title('Volumi di pratiche', fontsize=12)
                ax[0].set_ylabel('pratiche')
                plot1 = ax[0].violinplot(survey_report_volume_values,
                                         showmeans=False, showmedians=True)
                plot2 = ax[0].boxplot(survey_report_comune_volumes)
                ax[0].yaxis.grid(True)
                ax[0].set_xticks([y + 1 for y in range(len(survey_report_volume_metric_labels))],
                                 labels=survey_report_volume_metric_labels)
                ax[0].set_yticks(survey_report_volume_ticks)
                ax[0].set_ylim(survey_report_volume_range)
                ax[0].legend([plot1['bodies'][0], plot2['medians'][0]],
                             ['Cluster ' + str(cluster_index), survey_report_comune_name],
                             prop={'size': 12})
                ax[0].spines['top'].set_visible(False)
                ax[0].spines['right'].set_visible(False)
                ax[0].spines['bottom'].set_visible(False)
                ax[0].spines['left'].set_visible(False)

                ax[1].set_title('Tempi di lavorazione', fontsize=12)
                ax[1].set_ylabel('giorni')
                ax[1].violinplot(survey_report_time_values, showmeans=False, showmedians=True)
                ax[1].boxplot(survey_report_comune_times)
                ax[1].yaxis.grid(True)
                ax[1].set_xticks([y + 1 for y in range(len(survey_report_time_metric_labels))],
                                 labels=survey_report_time_metric_labels)
                ax[1].set_yticks(survey_report_time_ticks)
                ax[1].set_ylim(survey_report_time_range)
                ax[1].spines['top'].set_visible(False)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['bottom'].set_visible(False)
                ax[1].spines['left'].set_visible(False)

                fig.supxlabel('Numero della domanda nel questionario', fontsize=12)
                fig.suptitle('PAT-PNRR | Report del 1° Questionario ISPAT-PNRR | '
                             'Procedimenti Edilizi 2021Q3-4', fontsize=16)

                survey_report_comune_edited_name = survey_report_comune_name
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'à', 'a')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'è', 'e')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'é', 'e')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'ù', 'u')

                fig.savefig('pat_pnrr_survey_reports\\images\\pat-pnrr_survey_report_cluster_' +
                            cluster_index + '_' + survey_report_comune_edited_name,
                            dpi=300, bbox_inches='tight', pad_inches=0.25)
                plt.close(fig)

    if not tex_load:

        for cluster_n, comuni_cluster_n in enumerate(comuni_clusters):
        # for cluster_n, comuni_cluster_n in enumerate([comuni_cluster_1]):
            survey_report_volume_values = pat_comuni_dataframe[~comuni_nodata & comuni_cluster_n][
                survey_report_volume_metrics]

            for comune_n in range(survey_report_volume_values.shape[0]):
            # for comune_n in range(1):
                survey_report_comune_volumes = survey_report_volume_values[comune_n:comune_n + 1]
                survey_report_comune_name = survey_report_comune_volumes.index[0]

                if cluster_n == 0:
                    cluster_index = '1'
                if cluster_n == 1:
                    cluster_index = '2'
                if cluster_n == 2:
                    cluster_index = '3-4-5'

                with open('pat_pnrr_survey_reports\\pat-pnrr_survey_report_template.tex') as f1:
                    pat_pnrr_survey_report = f1.read()
                f1.close()

                survey_report_comune_edited_name = survey_report_comune_name
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'à', 'a')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'è', 'e')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'é', 'e')
                survey_report_comune_edited_name = survey_report_comune_edited_name.replace(
                    'ù', 'u')

                pat_pnrr_survey_report = pat_pnrr_survey_report.replace(
                    'pat-pnrr_survey_report_image_placeholder',
                    'pat_pnrr_survey_reports/images/pat-pnrr_survey_report_cluster_' +
                    cluster_index + '_' + survey_report_comune_edited_name
                )

                f2 = open('pat_pnrr_survey_reports\\texes\\pat-pnrr_survey_report_cluster_' +
                         cluster_index + '_' + survey_report_comune_name + '.tex', 'a')
                f2.write(pat_pnrr_survey_report)
                f2.close()

    if not report_load:

        for cluster_n, comuni_cluster_n in enumerate(comuni_clusters):
        # for cluster_n, comuni_cluster_n in enumerate([comuni_cluster_1]):
            survey_report_volume_values = pat_comuni_dataframe[~comuni_nodata & comuni_cluster_n][
                survey_report_volume_metrics]

            for comune_n in range(survey_report_volume_values.shape[0]):
            # for comune_n in range(1):
                survey_report_comune_volumes = survey_report_volume_values[comune_n:comune_n + 1]
                survey_report_comune_name = survey_report_comune_volumes.index[0]

                if cluster_n == 0:
                    cluster_index = '1'
                if cluster_n == 1:
                    cluster_index = '2'
                if cluster_n == 2:
                    cluster_index = '3-4-5'

                subprocess.call(['pdflatex', '-output-directory',
                                 'pat_pnrr_survey_reports\\pdfs\\',
                                 'pat_pnrr_survey_reports\\texes\\pat-pnrr_survey_report_cluster_'
                                 + cluster_index + '_' + survey_report_comune_name + '.tex'])

    return True


def print_pat_comuni_meeting(pat_comuni_dataframe, save_csv=False):

    pat_comuni_meeting_1 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 1: comuni piccoli (130)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Val di Non') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valle di Sole'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_2 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 1: comuni piccoli (130)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Alta Valsugana e Bersntol') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Altipiani Cimbri') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valle di Cembra') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Val di Fiemme'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_3 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 1: comuni piccoli (130)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valsugana e Tesino') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Primiero') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Comun General de Fascia'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_4 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 1: comuni piccoli (130)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Vallagarina') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          "Territorio Val d'Adige") |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Rotaliana-Königsberg') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Paganella'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_5 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] !=
         'Cluster 1: comuni piccoli (130)') &
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] !=
         'Cluster 2: comuni medio-piccoli (31)')].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_6 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 1: comuni piccoli (130)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Giudicarie') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Alto Garda e Ledro') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valle dei Laghi'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_7 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 2: comuni medio-piccoli (31)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Val di Non') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valle di Sole') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Paganella') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Rotaliana-Königsberg') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Giudicarie') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Alto Garda e Ledro') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valle dei Laghi'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    pat_comuni_meeting_8 = pat_comuni_dataframe[
        (pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'] ==
         'Cluster 2: comuni medio-piccoli (31)') &
        ((pat_comuni_dataframe['pat_comunita_valle'] ==
          'Val di Fiemme') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Comun General de Fascia') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Primiero') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Valsugana e Tesino') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Alta Valsugana e Bersntol') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Altipiani Cimbri') |
         (pat_comuni_dataframe['pat_comunita_valle'] ==
          'Vallagarina'))].sort_values(
        by='pat_comunita_valle')[['pat_comunita_valle', 'pat_comuni_kmeans_clustering_labels']]

    if save_csv:
        pat_comuni_meeting_1.to_csv('pat_comuni_meeting_1_202210.csv')
        pat_comuni_meeting_2.to_csv('pat_comuni_meeting_2_202210.csv')
        pat_comuni_meeting_3.to_csv('pat_comuni_meeting_3_202210.csv')
        pat_comuni_meeting_4.to_csv('pat_comuni_meeting_4_202210.csv')
        pat_comuni_meeting_5.to_csv('pat_comuni_meeting_5_202210.csv')
        pat_comuni_meeting_6.to_csv('pat_comuni_meeting_6_202210.csv')
        pat_comuni_meeting_7.to_csv('pat_comuni_meeting_7_202210.csv')
        pat_comuni_meeting_8.to_csv('pat_comuni_meeting_8_202210.csv')

    print()
    print('Comuni PAT | Meeting 1')
    print(pat_comuni_meeting_1)
    print()
    print('Comuni PAT | Meeting 2')
    print(pat_comuni_meeting_2)
    print()
    print('Comuni PAT | Meeting 3')
    print(pat_comuni_meeting_3)
    print()
    print('Comuni PAT | Meeting 4')
    print(pat_comuni_meeting_4)
    print()
    print('Comuni PAT | Meeting 5')
    print(pat_comuni_meeting_5)
    print()
    print('Comuni PAT | Meeting 6')
    print(pat_comuni_meeting_6)
    print()
    print('Comuni PAT | Meeting 7')
    print(pat_comuni_meeting_7)
    print()
    print('Comuni PAT | Meeting 8')
    print(pat_comuni_meeting_8)

    return True


def cluster_baseline(pat_comuni_dataframe):

    ispat_feature_labels = [
        'pat_comuni_popolazione',
        'pat_comuni_superficie_km2',
        'pat_comuni_ricettivita',
        'pat_comuni_turisticita',
        'pat_comuni_composito_turismo',
        'pat_comuni_locali_asia_1000_residenti',
        'pat_comuni_addetti_marketing_1000_residenti',
        'pat_comuni_addetti_servizi_1000_residenti',
        'pat_comuni_altitudine_m',
        'pat_comuni_famiglie',
        'pat_comuni_edifici',
        'pat_comuni_abitazioni',
        'pat_comuni_ufficio_tecnico_sue_ore_2020',
        'pat_comuni_urbanistica_programmazione_territorio_ore_2020',
        'pat_comuni_viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020',
        'pat_comuni_totale_generale_lavori_ore_2020',
        'pat_comuni_dipendenti_2020',
        'pat_comuni_permessi_costruire_residenziale_2019',
        'pat_comuni_permessi_costruire_residenziale_2020',
        'pat_comuni_permessi_costruire_residenziale_2021',
        'pat_comuni_dia_residenziale_2019',
        'pat_comuni_dia_residenziale_2020',
        'pat_comuni_dia_residenziale_2021',
        'pat_comuni_permessi_costruire_non_residenziale_2019',
        'pat_comuni_permessi_costruire_non_residenziale_2020',
        'pat_comuni_permessi_costruire_non_residenziale_2021',
        'pat_comuni_dia_non_residenziale_2019',
        'pat_comuni_dia_non_residenziale_2020',
        'pat_comuni_dia_non_residenziale_2021',
        'pat_comuni_edifici_pubblici_non_residenziale_2019',
        'pat_comuni_edifici_pubblici_non_residenziale_2020',
        'pat_comuni_edifici_pubblici_non_residenziale_2021',
        'pat_comuni_ristrutturazioni_concessioni_2019',
        'pat_comuni_ristrutturazioni_concessioni_2020',
        'pat_comuni_ristrutturazioni_concessioni_2021',
        'pat_comuni_ristrutturazioni_dia_cila_2019',
        'pat_comuni_ristrutturazioni_dia_cila_2020',
        'pat_comuni_ristrutturazioni_dia_cila_2021']

    baseline_feature_labels = [
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'numero_permessi_costruire_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'numero_controlli_cila_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'numero_sanatorie_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4']

    # not responders for ISPAT survey: substitution with their cluster mean
    # pat_comuni_dataframe[
    #     pat_comuni_dataframe[
    #         'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'].isna() == True][
    #     pat_comuni_dataframe[
    #         'pat_comuni_kmeans_clustering_labels'] == 'Cluster 2: comuni medio-piccoli (31)'][
    #     baseline_feature_labels]
    pat_comuni_dataframe.loc['Mori', baseline_feature_labels] = \
    pat_comuni_dataframe[
        pat_comuni_dataframe[
            'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'].isna() == False][
        pat_comuni_dataframe[
            'pat_comuni_kmeans_clustering_labels'] == 'Cluster 2: comuni medio-piccoli (31)'][
        baseline_feature_labels].mean()

    # pat_comuni_dataframe[
    #     pat_comuni_dataframe[
    #         'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'].isna() == True][
    #     pat_comuni_dataframe[
    #         'pat_comuni_kmeans_clustering_labels'] == 'Cluster 1: comuni piccoli (130)'][
    #     baseline_feature_labels]
    pat_comuni_dataframe.loc['Nomi', baseline_feature_labels] = \
    pat_comuni_dataframe.loc['Novaledo', baseline_feature_labels] = \
    pat_comuni_dataframe[
        pat_comuni_dataframe[
            'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'].isna() == False][
        pat_comuni_dataframe[
            'pat_comuni_kmeans_clustering_labels'] == 'Cluster 1: comuni piccoli (130)'][
        baseline_feature_labels].mean()

    # isolation of trento
    pat_comuni_data = pat_comuni_dataframe[
        ispat_feature_labels+baseline_feature_labels].drop(['Trento'])
    pat_comuni_toponimo = pat_comuni_dataframe[
        ispat_feature_labels+baseline_feature_labels].drop(['Trento']).index

    # standardization
    scaler = preprocessing.StandardScaler().fit(pat_comuni_data)
    pat_comuni_data = scaler.transform(pat_comuni_data)

    # kmeans clustering
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pat_comuni_data)
    y_kmeans = kmeans.predict(pat_comuni_data)
    # y_kmeans = np.vectorize({0: 1, 1: 3, 2: 0, 3: 2}.get)(y_kmeans)
    centers = kmeans.cluster_centers_

    n_clusters = 4
    n_features = pat_comuni_data.shape[1]
    n_pat_comuni_cluster_selected = 30
    pat_comuni_clusters = pairwise_distances_argmin_min(pat_comuni_data, centers)
    pat_comuni_selezionati_data = np.zeros((n_clusters, n_features))
    for n_cluster in range(n_clusters):
        print('CLUSTER', n_cluster + 1)
        print('')
        ids_cluster = np.where(pat_comuni_clusters[0] == n_cluster)[0]
        ids_cluster_sorted = np.argsort(pat_comuni_clusters[1][ids_cluster])
        pat_comuni_cluster_selected = pat_comuni_toponimo[ids_cluster][ids_cluster_sorted] \
            [:n_pat_comuni_cluster_selected]
        pat_comuni_selezionati_data[n_cluster] = pat_comuni_data[ids_cluster] \
            [ids_cluster_sorted][0]
        print('    Raccolta dati da questionario:', pat_comuni_cluster_selected,
              'sul totale di', sum(y_kmeans == n_cluster))
        print('')

    # pca
    pca = PCA(n_components=3)
    pca.fit(pat_comuni_data)
    X = pca.transform(pat_comuni_data)
    Xs = pca.transform(pat_comuni_selezionati_data)
    centers = pca.transform(centers)

    if True:
        # scatterplot2D
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
        plt.scatter(Xs[:, 0], Xs[:, 1], marker='*', c=range(n_clusters), s=50, alpha=1,
                    cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], marker='2', c=range(4), s=100, alpha=0.75,
                    cmap='viridis')
        plt.savefig("pat-pnrr_kmeans_clustering_pca_baseline", dpi=300)
        plt.show()

    if False:
        # scatterplot3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='2', c=range(4), s=100,
                   alpha=0.75, cmap='viridis')
        plt.show()

    clustering_label_map = {
        0: 'Cluster 2: comuni medio-piccoli (31)',
        1: 'Cluster 1: comuni piccoli (130)',
        2: 'Cluster 3: comuni medi (3)',
        3: 'Cluster 4: Rovereto'}
    pat_comuni_baseline_clustering_labels = pd.DataFrame(
        np.vectorize(clustering_label_map.get)(y_kmeans),
        columns=['pat_comuni_baseline_clustering_labels'],
        index=pat_comuni_toponimo)
    pat_comuni_kmeans_clustering_labels = pd.DataFrame(
        pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels'].drop(
            ['Trento']))

    clustering_difference = pat_comuni_baseline_clustering_labels[
                                'pat_comuni_baseline_clustering_labels'] !=\
                            pat_comuni_kmeans_clustering_labels[
                                'pat_comuni_kmeans_clustering_labels']
    pat_comuni_clustering_difference = pd.concat(
        [pat_comuni_kmeans_clustering_labels, pat_comuni_baseline_clustering_labels],
        axis='columns', join='outer')[clustering_difference]

    print('')
    print('ISPAT-Baseline clustering difference:')
    print('')
    print(pat_comuni_clustering_difference)

    return True


def cluster_tools(pat_comuni_dataframe):

    tool_feature_labels = [
        'portale_online_ufficio_tecnico',
        'nome_protocollazione_digitale_ufficio_tecnico',
        'nome_gestionale_digitale_ufficio_tecnico']
    volume_pratiche_labels = [
        'numero_permessi_costruire_2021q3-4',
        'numero_controlli_cila_2021q3-4',
        'numero_sanatorie_2021q3-4'
    ]
    no_pitre_202306 = [
        'Arco', 'Borgo Valsugana', 'Civezzano', 'Cles', 'Drena', 'Dro', 'Grigno', 'Lavis',
        'Levico Terme', 'Nago-Torbole', 'Ospedaletto', 'Pinzolo', 'Rovereto', 'Scurelle'
    ]
    no_giscom_202306 = [
        "Borgo d'Anaunia", 'Capriana', 'Dimaro Folgaria', 'Drena', 'Dro', 'Lavis', 'Lona-Lases',
        'Ossana', 'Peio', 'Pellizzano', 'Rovereto', 'Trento'
    ]

    pat_comuni_data = pat_comuni_dataframe[
        tool_feature_labels + volume_pratiche_labels].drop([
        'Mori', 'Nomi', 'Novaledo', 'Pieve di Bono-Prezzo'])
    pat_comuni_toponimo = pat_comuni_dataframe[
        tool_feature_labels + volume_pratiche_labels].drop([
        'Mori', 'Nomi', 'Novaledo', 'Pieve di Bono-Prezzo']).index

    pat_comuni_protocollo = pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico']
    pat_comuni_protocollo = pat_comuni_protocollo.str.lower()

    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'pitre|pi.tre|protocollo informatico trentino|pi 3|p3|ptre|giscom|gicomx')] = 'pitre'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'sicraweb')] = 'maggioli sicraweb'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'datagraph')] = 'datagraph'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'municipia')] = 'infor municipia'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'apkappa')] = 'apkappa'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'halley')] = 'halley'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'hypersic')] = 'hypersic'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'jente')] = 'jente'
    pat_comuni_protocollo[pat_comuni_protocollo.str.contains(
        'siqa')] = 'siqa'
    pat_comuni_protocollo[(pat_comuni_protocollo == '') | (pat_comuni_protocollo == ' ')] = ''

    pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] = pat_comuni_protocollo

    pat_comuni_gestionale = pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico']
    pat_comuni_gestionale = pat_comuni_gestionale.str.lower()

    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'giscom|gis-com|giscloud|geopatner')] = 'giscom'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'globo')] = 'globo'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'excel|excell')] = ''
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'wince')] = 'wince'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'municipia')] = 'infor municipia'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'siqa')] = 'siqa'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'apkappa')] = 'apkappa'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'civilia')] = 'civilia'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'hypersic')] = 'hypersic'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'peo')] = ''
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'alice')] = 'maggioli alice'
    pat_comuni_gestionale[pat_comuni_gestionale.str.contains(
        'programma')] = '?'
    pat_comuni_gestionale[(pat_comuni_gestionale == '') | (pat_comuni_gestionale == ' ')] = ''

    pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] = pat_comuni_gestionale

    print("")
    print("Edilizia PAT: strumenti ICT")
    print("")
    print("    Portale+PITre+GIScom " + str(pat_comuni_data[
        (pat_comuni_data['portale_online_ufficio_tecnico'] == 2) &
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'giscom')].shape[0]))
    print("    Portale+GIScom " + str(pat_comuni_data[
        (pat_comuni_data['portale_online_ufficio_tecnico'] == 2) &
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'giscom')].shape[0]))
    print("    Portale+PITre " + str(pat_comuni_data[
        (pat_comuni_data['portale_online_ufficio_tecnico'] == 2) &
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'giscom')].shape[0]))
    print("    Portale+misti " + str(pat_comuni_data[
        (pat_comuni_data['portale_online_ufficio_tecnico'] == 2) &
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'giscom')].shape[0]))
    print("    nessuno " + str(pat_comuni_data[
        (pat_comuni_data['portale_online_ufficio_tecnico'] == 1)].shape[0]))

    print("")
    print("Edilizia PAT: portale web")
    print("")
    print("    GIScom+PITre " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'giscom')].shape[0]))
    print("    GIScom (no PITre) " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'giscom')].shape[0]))
    print("    Civilia+PITre " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'civilia')].shape[0]))
    print("    Gestionale+PITre (no GIScom o Civilia) " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'giscom') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'civilia') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != '')].shape[0]))
    print("    PITre (no gestionale) " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == '')].shape[0]))
    print("    misti (no PITre o GIScom o Civilia) " + str(pat_comuni_data[
        ((pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != 'pitre') &
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != '') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'giscom') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'civilia')) |
        ((pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] != 'pitre') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'giscom') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != 'civilia') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] != ''))].shape[0]))
    print("    nessuno " + str(pat_comuni_data[
        (pat_comuni_data['nome_protocollazione_digitale_ufficio_tecnico'] == '') &
        (pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == '')].shape[0]))

    print("")
    print("Edilizia PAT: volumi pratiche (Permessi Costruire, Sanatorie, Controlli CILA)")
    print("")
    print("    Totale pratiche " + str(pat_comuni_dataframe[volume_pratiche_labels].sum().sum()))
    print("    Pratiche GIScom " + str(pat_comuni_data[
        pat_comuni_data['nome_gestionale_digitale_ufficio_tecnico'] == 'giscom'][
        volume_pratiche_labels].sum().sum()))
    print("    Pratiche Trento " + str(pat_comuni_data.loc['Trento'][
        volume_pratiche_labels].sum().sum()))
    print("")

    return True


def show_survey_times():

    pat_comuni_dataframe_idsurvey_times = get_pat_comuni_dataframe_idsurvey_times(
        'pat_pnrr_questionario_edilizia\\')
    pat_comuni_dataframe_idsurvey_times['data_inizio_compilazione'] = pd.to_datetime(
        pat_comuni_dataframe_idsurvey_times['data_inizio_compilazione'],
        format='%Y-%m-%d %H:%M:%S')
    pat_comuni_dataframe_idsurvey_times['data_fine_compilazione'] = pd.to_datetime(
        pat_comuni_dataframe_idsurvey_times['data_fine_compilazione'],
        format='%Y-%m-%d %H:%M:%S')

    for i in range(pat_comuni_dataframe_idsurvey_times.shape[0]):
        plt.plot([pat_comuni_dataframe_idsurvey_times['data_inizio_compilazione'][i],
                  pat_comuni_dataframe_idsurvey_times['data_fine_compilazione'][i]],
                 [i, i], c='black', alpha=0.25)
    plt.scatter(pat_comuni_dataframe_idsurvey_times['data_fine_compilazione'],
                range(pat_comuni_dataframe_idsurvey_times.shape[0]), c='green', marker='x')
    plt.scatter(pat_comuni_dataframe_idsurvey_times['data_inizio_compilazione'],
                range(pat_comuni_dataframe_idsurvey_times.shape[0]), c='black', alpha=0.25)
    plt.yticks(range(pat_comuni_dataframe_idsurvey_times.shape[0]),
               pat_comuni_dataframe_idsurvey_times.index)
    plt.yticks(fontsize=7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(fontsize=7, rotation=30)
    survey_deadline_01 = pd.Timestamp('2022-05-06 17:00:00')
    plt.axvline(x=survey_deadline_01, color='orange')
    survey_deadline_02 = pd.Timestamp('2022-05-20 17:00:00')
    plt.axvline(x=survey_deadline_02, color='red')
    # plt.savefig("pat-pnrr_idsurvey_times", dpi=300)
    plt.show()

    return True


def check_transit_03_04_05():
    ''' controllo se dalla 3a alla 5a misurazione esistono pratiche pdc e pds
        per le quali i giorni totali di sospensioni sono decrescenti
    '''
    
    comuni_dataframe_pdc_03 = pat_pnrr_3a.get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire')
    comuni_dataframe_pds_03 = pat_pnrr_3a.get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria')
    comuni_dataframe_pdc_04 = pat_pnrr_4a.get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire')
    comuni_dataframe_pds_04 = pat_pnrr_4a.get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria')
    comuni_dataframe_pdc_05 = pat_pnrr_5a.get_comuni_dataframe(
        comuni_excel_map, 'Permessi di Costruire', 'pat_pnrr_5a_misurazione_tabelle_comunali\\')
    comuni_dataframe_pds_05 = pat_pnrr_5a.get_comuni_dataframe(
        comuni_excel_map, 'Prov di sanatoria', 'pat_pnrr_5a_misurazione_tabelle_comunali\\')
    
    list_excel_03 = pat_pnrr_3a.get_list_excel()
    list_excel_04 = pat_pnrr_4a.get_list_excel()
    comuni_received_03 = [comune_excel[1] for comune_excel in list_excel_03]
    comuni_received_04 = [comune_excel[1] for comune_excel in list_excel_04]
    comuni_dataframe_checks = [
        # ['pdc', comuni_dataframe_pdc_03, comuni_dataframe_pdc_04,
        #     'sospensioni decrescenti', 'tra 3a e 4a misurazione'],
        # ['pdc', comuni_dataframe_pdc_04, comuni_dataframe_pdc_05,
        #     'sospensioni decrescenti', 'tra 4a e 5a misurazione'],
        # ['pds', comuni_dataframe_pds_03, comuni_dataframe_pds_04,
        #     'sospensioni decrescenti', 'tra 3a e 4a misurazione'],
        # ['pds', comuni_dataframe_pds_04, comuni_dataframe_pds_05,
        #     'sospensioni decrescenti', 'tra 4a e 5a misurazione'],
        ['pdc', comuni_dataframe_pdc_03, comuni_dataframe_pdc_04,
            'concluse transitate', 'tra 3a e 4a misurazione'],
        ['pdc', comuni_dataframe_pdc_04, comuni_dataframe_pdc_05,
            'concluse transitate', 'tra 4a e 5a misurazione'],
        ['pds', comuni_dataframe_pds_03, comuni_dataframe_pds_04,
            'concluse transitate', 'tra 3a e 4a misurazione'],
        ['pds', comuni_dataframe_pds_04, comuni_dataframe_pds_05,
            'concluse transitate', 'tra 4a e 5a misurazione']]

    issued_messages = []
    for comune in comuni_received_03:
        for comuni_dataframe_check in comuni_dataframe_checks:
            comuni_tipo_pratica = comuni_dataframe_check[0]
            comuni_dataframe_before = comuni_dataframe_check[1]
            comuni_dataframe_after = comuni_dataframe_check[2]
            comuni_tipo_controllo = comuni_dataframe_check[3]
            comuni_messaggio_misurazioni = comuni_dataframe_check[4]

            filter_mask_comuni_before = comuni_dataframe_before.loc[:, 'comune'] == comune
            filter_mask_comuni_after = comuni_dataframe_after.loc[:, 'comune'] == comune
            filter_mask = filter_mask_comuni_before & filter_mask_comuni_after

            if filter_mask.sum() > 0:
                comuni_dataframe_before = comuni_dataframe_before[filter_mask]
                comuni_dataframe_after = comuni_dataframe_after[filter_mask]
                for index in comuni_dataframe_before.index:
                    pratica = comuni_dataframe_before.loc[index]
                    if pratica.data_inizio_pratica in \
                        comuni_dataframe_after.data_inizio_pratica.values:
                        if comuni_tipo_controllo == 'sospensioni decrescenti':
                            index_giorni_sospensioni_before = comuni_dataframe_before[
                                comuni_dataframe_before.data_inizio_pratica.values == \
                                    pratica.data_inizio_pratica
                                ].index[0]
                            index_giorni_sospensioni_after = comuni_dataframe_after[
                                comuni_dataframe_after.data_inizio_pratica.values == \
                                    pratica.data_inizio_pratica
                                ].index[0]
                            giorni_sospensioni_before = comuni_dataframe_before.loc[
                                index_giorni_sospensioni_before].giorni_sospensioni
                            giorni_sospensioni_after = comuni_dataframe_after.loc[
                                index_giorni_sospensioni_after].giorni_sospensioni
                            if giorni_sospensioni_before > giorni_sospensioni_after:
                                giorni_sospensioni_differenza = giorni_sospensioni_before - \
                                    giorni_sospensioni_after
                                message = \
                                    'sospensioni decrescenti' + \
                                    ' di ' + str(giorni_sospensioni_differenza) + \
                                    ' nel comune di ' + comune + \
                                    ' della pratica ' + comuni_tipo_pratica + \
                                    ' con data inizio ' + str(pratica.data_inizio_pratica) + \
                                    ' ' + comuni_messaggio_misurazioni
                                issued_messages.append(message)
                        if comuni_tipo_controllo == 'concluse transitate':
                            # - data inizio riportata 1 o piu' volte nella misurazione n
                            #    - data inizio misurazione n riportata 1 o piu' volte nella misurazione n+1
                            #        ! riportare tutte le pratiche concluse nella misurazione n
                            pratica_conclusa_before = comuni_dataframe_before[
                                comuni_dataframe_before.data_inizio_pratica.values == \
                                    pratica.data_inizio_pratica
                                ].data_fine_pratica.isna() == False
                            if comuni_tipo_pratica == 'pdc':
                                pratica_conclusa_before |= comuni_dataframe_before[
                                comuni_dataframe_before.data_inizio_pratica.values == \
                                    pratica.data_inizio_pratica
                                ].loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False
                            pratica_conclusa_before = sum(pratica_conclusa_before) > 0
                            if pratica_conclusa_before:
                                message = \
                                    'almeno una pratica ' + comuni_tipo_pratica + \
                                    ' con data inizio ' + str(pratica.data_inizio_pratica) + \
                                    ' nel comune di ' + comune + \
                                    ' che risulta conclusa e poi presente' + \
                                    ' ' + comuni_messaggio_misurazioni
                                issued_messages.append(message)
                            pass
    for issued_message in issued_messages:
        print(issued_message)


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


    # LOAD DATAFRAME COMUNI
    pat_comuni_dataframe = get_pat_comuni_dataframe(load=True)


    # PRINT BASELINES
    # print_baselines(pat_comuni_dataframe, save_tex=True)  # , exclude_clusters_3_4_5=True)  # , save_tex=True, save_csv=True)
    # SHOW BASELINE CHARTS
    # show_baselines(pat_comuni_dataframe)
    # SHOW ISSUE CHARTS
    # show_issues(pat_comuni_dataframe)

    # PRINT MEASUREMENT 02
    # print_measurement_02(pat_comuni_dataframe, save_tex=True)  # , exclude_clusters_3_4_5=True)  # , save_tex=True, save_csv=True)
    # SHOW MEASUREMENT 02 CHARTS
    # show_measurement_02(pat_comuni_dataframe)
    # PRINT SURVEY REPORT MEASUREMENT 02
    # print_survey_report_images(pat_comuni_dataframe)
    # PRINT COMUNI PAT MEETINGS MEASUREMENT 03
    # print_pat_comuni_meeting(pat_comuni_dataframe)

    # PRINT MEASUREMENT 03
    # pat_pnrr_3a.get_comuni_measures_dataframe(comuni_excel_map, load=True)
    # pat_pnrr_3a.get_comuni_measures(comuni_excel_map, save_tex=True)
    # SHOW MEASUREMENT 03 CHARTS (no more valid, after new baseline 202306)
    # show_measurement_03(pat_comuni_dataframe)

    # PRINT MEASUREMENT 04
    # pat_pnrr_4a.get_comuni_measures_dataframe(comuni_excel_map, load=True)
    # pat_pnrr_4a.get_comuni_measures(comuni_excel_map, save_tex=True)

    # PRINT MEASUREMENT 05
    # pat_pnrr_5a.get_comuni_measures_dataframe(comuni_excel_map, load=True)
    # pat_pnrr_5a.get_comuni_measures(comuni_excel_map, save_tex=True)


    # CLUSTER BASELINE
    # cluster_baseline(pat_comuni_dataframe)
    
    # CLUSTER TOOLS
    # cluster_tools(pat_comuni_dataframe)
    
    # PRINT MODEL BASELINES
    # print_baselines(pat_comuni_dataframe, model_baselines=True)
    
    # SHOW SURVEY TIMES
    # show_survey_times()


    # REQUEST 20240424
    #     - dati disaggregati di tutte le pratiche pdc e pds
    #         - pratiche concluse nella 4a misurazione
    #             - con data di presentazione antecedente al 01/07/2022
    #         - pratiche concluse nella 5a misurazione
    #             - con data di presentazione antecedente al 01/07/2022
    # comuni_dataframe_pdc_04 = pat_pnrr_4a.get_comuni_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', 'pat_pnrr_4a_misurazione_tabelle_comunali\\')
    # comuni_dataframe_pds_04 = pat_pnrr_4a.get_comuni_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', 'pat_pnrr_4a_misurazione_tabelle_comunali\\')
    # comuni_dataframe_pdc_05 = pat_pnrr_5a.get_comuni_dataframe(
    #     comuni_excel_map, 'Permessi di Costruire', 'pat_pnrr_5a_misurazione_tabelle_comunali\\')
    # comuni_dataframe_pds_05 = pat_pnrr_5a.get_comuni_dataframe(
    #     comuni_excel_map, 'Prov di sanatoria', 'pat_pnrr_5a_misurazione_tabelle_comunali\\')
    # 
    # filter_mask_pdc_04 = ((comuni_dataframe_pdc_04.loc[:, 'data_fine_pratica'].isna() == False) | \
    #     (comuni_dataframe_pdc_04.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False)) & \
    #     (comuni_dataframe_pdc_04.loc[:, 'data_inizio_pratica'] < pd.Timestamp('2022-07-01 23:59:59.999'))
    # filter_mask_pds_04 = (comuni_dataframe_pds_04.loc[:, 'data_fine_pratica'].isna() == False) & \
    #     (comuni_dataframe_pds_04.loc[:, 'data_inizio_pratica'] < pd.Timestamp('2022-07-01 23:59:59.999'))
    # filter_mask_pdc_05 = ((comuni_dataframe_pdc_05.loc[:, 'data_fine_pratica'].isna() == False) | \
    #     (comuni_dataframe_pdc_05.loc[:, 'data_fine_pratica_silenzio-assenso'].isna() == False)) & \
    #     (comuni_dataframe_pdc_05.loc[:, 'data_inizio_pratica'] < pd.Timestamp('2022-07-01 23:59:59.999'))
    # filter_mask_pds_05 = (comuni_dataframe_pds_05.loc[:, 'data_fine_pratica'].isna() == False) & \
    #     (comuni_dataframe_pds_05.loc[:, 'data_inizio_pratica'] < pd.Timestamp('2022-07-01 23:59:59.999'))
    # 
    # comuni_dataframe_pdc = pd.concat(
    #     [comuni_dataframe_pdc_04[filter_mask_pdc_04],
    #      comuni_dataframe_pdc_05[filter_mask_pdc_05]],
    #     axis='rows', join='outer')
    # comuni_dataframe_pdc.reset_index(drop=True, inplace=True)
    # comuni_dataframe_pdc.to_csv('pat-pnrr_edilizia_pdc_request_20240424.csv')
    # comuni_dataframe_pds = pd.concat(
    #     [comuni_dataframe_pds_04[filter_mask_pds_04],
    #      comuni_dataframe_pds_05[filter_mask_pds_05]],
    #     axis='rows', join='outer')
    # comuni_dataframe_pds.reset_index(drop=True, inplace=True)
    # comuni_dataframe_pds.to_csv('pat-pnrr_edilizia_pds_request_20240424.csv')


    # REQUEST 20240429 | giorni sospensioni decrescenti
    #     - dati disaggregati di tutte le pratiche pdc e pds dalla 3a alla 5a misurazione
    #         - segnalare quelle con sospensioni totali decrescenti
    #     - dati disaggregati di tutte le pratiche pdc e pds dalla 3a alla 5a misurazione
    #         - segnalare quelle con cambio di tipologia
    # REQUEST 20240503 | pratiche concluse ma transitate
    #     - id pratica inaffidabile occorrerebbe controllo manuale e diretto sul comune
    #     - data inizio e fine pratica unici indicatori potenzialmente affidabili
    #         - data inizio riportata 1 o piu' volte nella misurazione n
    #            - data inizio misurazione n riportata 1 o piu' volte nella misurazione n+1
    #                ! riportare tutte le pratiche concluse nella misurazione n
    # check_transit_03_04_05()


    # REQUEST 20240505 | solo pratiche concluse
    #     - comuni che dichiarano solo pratiche pdc e pds concluse
    #         - 3a, 4a e 5a misurazione


    # REQUEST 20240515_01 | 20240513_01_02 | pdc-ov non conclusi durata netta > 120 gg
    #     * tutta l'analisi anche per i pds | DONE
    #     * stessa estrazione sulla 3a e 4a misurazione
    #     - analisi
    #         - son sempre alcuni comuni che non tracciano le sospensioni
    #         - quanti comuni coinvolti
    #     - cercare di tracciarle fino alla 3a misurazione


    # REQUEST 20240515_02 | 20240513_02 | pdc-ov avviati durata lorda > 600 gg
    #     * tutta l'analisi anche per i pds | DONE
    #     - analisi
    #         - quanti comuni coinvolti
    #         - incidenza sulla durata media e sull'arretrato nella 5a misurazione
    #     - cercare di tracciarle fino alla 3a misurazione

    # TODO: controllo progressione crescente delle sospensioni pratica per pratica dalla 3a alla 5a misurazione
