"""
    PAT-PNRR Monitoring Reporter
    Francesco Melchiori, 2024
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

from pat_pnrr_monitoring_analyzer import get_pat_comuni_dataframe
from pat_pnrr_mpe.pat_pnrr_comuni_excel_mapping import comuni_excel_map


def get_comuni_performance_trends(pat_comuni_dataframe, time_limit=-1):
    def get_comuni_performance_measures(performance_measure_labels, time_limit=time_limit):
        giornate_durata_media = pat_comuni_dataframe.loc[:, performance_measure_labels[0]]
        if time_limit != -1:
            if giornate_durata_media.max() > time_limit:
                giornate_durata_media.mask(giornate_durata_media > time_limit, time_limit,
                                           inplace=True)
        if performance_measure_labels[1]:
            giornate_termine_massimo = pat_comuni_dataframe.loc[:, performance_measure_labels[1]]
        else:
            giornate_termine_massimo = 60
        numero_pratiche_arretrate = pat_comuni_dataframe.loc[:, performance_measure_labels[2]]
        numero_pratiche_avviate = pat_comuni_dataframe.loc[:, performance_measure_labels[3]]

        comuni_durata_measures = giornate_durata_media / giornate_termine_massimo
        # - giornate_durata_media e' una misura al lordo delle sospensioni per i pareri terzi
        # - i target ministeriali sono posti rispetto all'attesa dell'utenza finale
        # ! giornate_termine_massimo e' un limite normativo che non comprende i pareri terzi

        comuni_arretrato_measures = numero_pratiche_arretrate / numero_pratiche_avviate
        comuni_performance_measures = (comuni_durata_measures.pow(2) +
                                       comuni_arretrato_measures.pow(2)).pow(0.5)

        comuni_durata_measures[comuni_durata_measures.isna()] = 0
        comuni_arretrato_measures[comuni_arretrato_measures.isna()] = 0
        comuni_performance_measures[comuni_performance_measures.isna()] = 0

        return comuni_durata_measures, comuni_arretrato_measures, comuni_performance_measures

    performance_measure_labels_pdc_2021q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        None,
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_2021q3-4']
    performance_measure_labels_pds_2021q3_4 = [
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        None,
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4',
        'numero_sanatorie_2021q3-4']

    performance_measure_labels_pdc_2022q1_2 = [
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',
        None,
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2',
        'numero_permessi_costruire_2022q1-2']
    performance_measure_labels_pds_2022q1_2 = [
        'giornate_durata_media_sanatorie_concluse_2022q1-2',
        None,
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2',
        'numero_sanatorie_2022q1-2']

    performance_measure_labels_pdc_2022q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q3-4',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4',
        'numero_permessi_costruire_avviati_2022q3-4']
    performance_measure_labels_pdc_ov_2022q3_4 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2022q3-4',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4',
        'numero_permessi_costruire_ov_avviati_2022q3-4']
    performance_measure_labels_pds_2022q3_4 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2022q3-4',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4',
        'numero_sanatorie_avviate_2022q3-4']

    performance_measure_labels_pdc_2023q1_2 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q1-2',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2',
        'numero_permessi_costruire_avviati_2023q1-2']
    performance_measure_labels_pdc_ov_2023q1_2 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2023q1-2',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2',
        'numero_permessi_costruire_ov_avviati_2023q1-2']
    performance_measure_labels_pds_2023q1_2 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2023q1-2',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2',
        'numero_sanatorie_avviate_2023q1-2']

    performance_measure_labels_pdc_2023q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q3-4',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4',
        'numero_permessi_costruire_avviati_2023q3-4']
    performance_measure_labels_pdc_ov_2023q3_4 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2023q3-4',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4',
        'numero_permessi_costruire_ov_avviati_2023q3-4']
    performance_measure_labels_pds_2023q3_4 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2023q3-4',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4',
        'numero_sanatorie_avviate_2023q3-4']

    performance_trends_pdc_2021q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_2021q3_4)
    performance_trends_pdc_2022q1_2 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_2022q1_2)
    performance_trends_pdc_2022q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_2022q3_4)
    performance_trends_pdc_ov_2022q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_ov_2022q3_4)
    performance_trends_pdc_2023q1_2 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_2023q1_2)
    performance_trends_pdc_ov_2023q1_2 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_ov_2023q1_2)
    performance_trends_pdc_2023q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_2023q3_4)
    performance_trends_pdc_ov_2023q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pdc_ov_2023q3_4)
    
    performance_trends_pds_2021q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pds_2021q3_4)
    performance_trends_pds_2022q1_2 = \
        get_comuni_performance_measures(performance_measure_labels_pds_2022q1_2)
    performance_trends_pds_2022q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pds_2022q3_4)
    performance_trends_pds_2023q1_2 = \
        get_comuni_performance_measures(performance_measure_labels_pds_2023q1_2)
    performance_trends_pds_2023q3_4 = \
        get_comuni_performance_measures(performance_measure_labels_pds_2023q3_4)

    comuni_durata_trends = pd.concat([
        performance_trends_pdc_2021q3_4[0],
        performance_trends_pdc_2022q1_2[0],
        performance_trends_pdc_2022q3_4[0],
        performance_trends_pdc_ov_2022q3_4[0],
        performance_trends_pdc_2023q1_2[0],
        performance_trends_pdc_ov_2023q1_2[0],
        performance_trends_pdc_2023q3_4[0],
        performance_trends_pdc_ov_2023q3_4[0],
        performance_trends_pds_2021q3_4[0],
        performance_trends_pds_2022q1_2[0],
        performance_trends_pds_2022q3_4[0],
        performance_trends_pds_2023q1_2[0],
        performance_trends_pds_2023q3_4[0]],
        keys=['pdc_durata_2021q3_4', 'pdc_durata_2022q1_2', 'pdc_durata_2022q3_4',
              'pdc_ov_durata_2022q3_4', 'pdc_durata_2023q1_2', 'pdc_ov_durata_2023q1_2',
              'pdc_durata_2023q3_4', 'pdc_ov_durata_2023q3_4',
              'pds_durata_2021q3_4', 'pds_durata_2022q1_2', 'pds_durata_2022q3_4',
              'pds_durata_2023q1_2', 'pds_durata_2023q3_4'],
        axis='columns', join='outer')

    comuni_arretrato_trends = pd.concat([
        performance_trends_pdc_2021q3_4[1],
        performance_trends_pdc_2022q1_2[1],
        performance_trends_pdc_2022q3_4[1],
        performance_trends_pdc_ov_2022q3_4[1],
        performance_trends_pdc_2023q1_2[1],
        performance_trends_pdc_ov_2023q1_2[1],
        performance_trends_pdc_2023q3_4[1],
        performance_trends_pdc_ov_2023q3_4[1],
        performance_trends_pds_2021q3_4[1],
        performance_trends_pds_2022q1_2[1],
        performance_trends_pds_2022q3_4[1],
        performance_trends_pds_2023q1_2[1],
        performance_trends_pds_2023q3_4[1]],
        keys=['pdc_arretrato_2021q3_4', 'pdc_arretrato_2022q1_2', 'pdc_arretrato_2022q3_4',
              'pdc_ov_arretrato_2022q3_4', 'pdc_arretrato_2023q1_2', 'pdc_ov_arretrato_2023q1_2',
              'pdc_arretrato_2023q3_4', 'pdc_ov_arretrato_2023q3_4',
              'pds_arretrato_2021q3_4', 'pds_arretrato_2022q1_2', 'pds_arretrato_2022q3_4',
              'pds_arretrato_2023q1_2', 'pds_arretrato_2023q3_4'],
        axis='columns', join='outer')

    comuni_performance_trends = pd.concat([
        performance_trends_pdc_2021q3_4[2],
        performance_trends_pdc_2022q1_2[2],
        performance_trends_pdc_2022q3_4[2],
        performance_trends_pdc_ov_2022q3_4[2],
        performance_trends_pdc_2023q1_2[2],
        performance_trends_pdc_ov_2023q1_2[2],
        performance_trends_pdc_2023q3_4[2],
        performance_trends_pdc_ov_2023q3_4[2],
        performance_trends_pds_2021q3_4[2],
        performance_trends_pds_2022q1_2[2],
        performance_trends_pds_2022q3_4[2],
        performance_trends_pds_2023q1_2[2],
        performance_trends_pds_2023q3_4[2]],
        keys=['pdc_2021q3_4', 'pdc_2022q1_2', 'pdc_2022q3_4', 'pdc_ov_2022q3_4', 'pdc_2023q1_2',
              'pdc_ov_2023q1_2','pdc_2023q3_4', 'pdc_ov_2023q3_4',
              'pds_2021q3_4', 'pds_2022q1_2', 'pds_2022q3_4', 'pds_2023q1_2', 'pds_2023q3_4'],
        axis='columns', join='outer')

    return comuni_durata_trends, comuni_arretrato_trends, comuni_performance_trends


def get_comuni_scores(comuni_performance_trends, pdc_measure_labels, pds_measure_labels):
    comuni_pdc_scores = comuni_performance_trends.loc[:, pdc_measure_labels].mean(axis=1)
    comuni_pds_scores = comuni_performance_trends.loc[:, pds_measure_labels].mean(axis=1)
    comuni_scores = (comuni_pdc_scores.pow(2) + comuni_pds_scores.pow(2)).pow(0.5)
    return comuni_pdc_scores, comuni_pds_scores, comuni_scores


def print_comuni_performance_charts(pat_comuni_dataframe,
                                    comuni_durata_trends, comuni_arretrato_trends,
                                    comuni_performance_trends, mpe_number=3,
                                    just_provincia=True, no_trento=True,
                                    just_one=False, save_charts=True):
    if mpe_number == 3:
        pdc_measure_labels = ['pdc_2022q1_2', 'pdc_2022q3_4']
        pds_measure_labels = ['pds_2022q1_2', 'pds_2022q3_4']
        mpe_number_label = '03'
        periodo_label = '2022'
        pdc_durata_labels = ['pdc_durata_2022q1_2', 'pdc_durata_2022q3_4']
        pdc_arretrato_labels = ['pdc_arretrato_2022q1_2', 'pdc_arretrato_2022q3_4']
        pds_durata_labels = ['pds_durata_2022q1_2', 'pds_durata_2022q3_4']
        pds_arretrato_labels = ['pds_arretrato_2022q1_2', 'pds_arretrato_2022q3_4']
    
    elif mpe_number == 4:
        pdc_measure_labels = ['pdc_2022q3_4', 'pdc_2023q1_2']
        pds_measure_labels = ['pds_2022q3_4', 'pds_2023q1_2']
        mpe_number_label = '04'
        periodo_label = '2022Q3-4 2023Q1-2'
        pdc_durata_labels = ['pdc_durata_2022q3_4', 'pdc_durata_2023q1_2']
        pdc_arretrato_labels = ['pdc_arretrato_2022q3_4', 'pdc_arretrato_2023q1_2']
        pds_durata_labels = ['pds_durata_2022q3_4', 'pds_durata_2023q1_2']
        pds_arretrato_labels = ['pds_arretrato_2022q3_4', 'pds_arretrato_2023q1_2']
    
    elif mpe_number == 5:
        pdc_measure_labels = ['pdc_2023q1_2', 'pdc_2023q3_4']
        pds_measure_labels = ['pds_2023q1_2', 'pds_2023q3_4']
        mpe_number_label = '05'
        periodo_label = '2023'
        pdc_durata_labels = ['pdc_durata_2023q1_2', 'pdc_durata_2023q3_4']
        pdc_arretrato_labels = ['pdc_arretrato_2023q1_2', 'pdc_arretrato_2023q3_4']
        pdc_durata_measure_labels = [
            'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2',
            'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q3-4']
        pdc_avviato_measure_labels = [
            'numero_permessi_costruire_avviati_2023q1-2',
            'numero_permessi_costruire_avviati_2023q3-4']
        pdc_arretrato_measure_labels = [
            'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2',
            'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4']
        pds_durata_labels = ['pds_durata_2023q1_2', 'pds_durata_2023q3_4']
        pds_arretrato_labels = ['pds_arretrato_2023q1_2', 'pds_arretrato_2023q3_4']
        pds_durata_measure_labels = [
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2',
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4']
        pds_avviato_measure_labels = [
            'numero_sanatorie_avviate_2023q1-2',
            'numero_sanatorie_avviate_2023q3-4']
        pds_arretrato_measure_labels = [
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2',
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4']
        ore_tecnici_settimana_label = 'ore_tecnici_settimana_2023q3-4'
    
    comuni_pdc_durata = comuni_durata_trends.loc[:, pdc_durata_labels].mean(axis=1)
    comuni_pdc_arretrato = comuni_arretrato_trends.loc[:, pdc_arretrato_labels].mean(axis=1)
    comuni_pds_durata = comuni_durata_trends.loc[:, pds_durata_labels].mean(axis=1)
    comuni_pds_arretrato = comuni_arretrato_trends.loc[:, pds_arretrato_labels].mean(axis=1)

    comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
        comuni_performance_trends, pdc_measure_labels, pds_measure_labels)

    # sistemare il calcolo di comuni_pdc_pds_durata e comuni_pdc_pds_arretrato
    if mpe_number >= 5:
        ore_tecnici_settimana = pat_comuni_dataframe.loc[:, ore_tecnici_settimana_label]
        comuni_pdc_pds_durata = \
            pat_comuni_dataframe.loc[:, pdc_durata_measure_labels].mean(axis=1) + \
            pat_comuni_dataframe.loc[:, pds_durata_measure_labels].mean(axis=1)
        comuni_pdc_pds_avviato = \
            pat_comuni_dataframe.loc[:, pdc_avviato_measure_labels].sum(axis=1) + \
            pat_comuni_dataframe.loc[:, pds_avviato_measure_labels].sum(axis=1)
        comuni_pdc_pds_arretrato = \
            pat_comuni_dataframe.loc[:, pdc_arretrato_measure_labels].sum(axis=1) + \
            pat_comuni_dataframe.loc[:, pds_arretrato_measure_labels].sum(axis=1)

    pdc_pds_score_max = comuni_performance_trends.loc[:,
                        pdc_measure_labels + pds_measure_labels].max(axis=None)
    pdc_pds_score_ticks = np.arange(0, pdc_pds_score_max, 1)
    pdc_pds_score_range = (pdc_pds_score_ticks[0], pdc_pds_score_ticks[-1])

    classificazione_comunale = pat_comuni_dataframe['pat_comuni_kmeans_clustering_labels']
    classificazione_comunale_labels = [
        'Cluster 1: comuni piccoli (130)',
        'Cluster 2: comuni medio-piccoli (31)',
        'Cluster 3: comuni medi (3)',
        'Cluster 4: Rovereto',
        'Cluster 5: Trento']
    classificazione_comunale_map = {
        'Cluster 1: comuni piccoli (130)': 0,
        'Cluster 2: comuni medio-piccoli (31)': 1,
        'Cluster 3: comuni medi (3)': 2,
        'Cluster 4: Rovereto': 3,
        'Cluster 5: Trento': 4}
    classificazione_comunale_color = {
        'Cluster 1: comuni piccoli (130)': 'lime',
        'Cluster 2: comuni medio-piccoli (31)': 'green',
        'Cluster 3: comuni medi (3)': 'orange',
        'Cluster 4: Rovereto': 'orangered',
        'Cluster 5: Trento': 'darkred'}
    # classificazione_comunale_color = classificazione_comunale.map(classificazione_comunale_color)
    classificazione_comunale = classificazione_comunale.map(classificazione_comunale_map)
    grandezza_comunale = classificazione_comunale+10+pow(classificazione_comunale+1, 3)

    fig, ax = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[0.25, 0.25, 0.25, 0.25]),
                           layout='constrained')
    fig.set_size_inches(15, 5)

    ax[0].set_title(periodo_label, fontsize=12)
    plot1 = ax[0].scatter(comuni_pdc_arretrato, comuni_pdc_durata,
                          c=classificazione_comunale, marker='o', s=grandezza_comunale, alpha=0.5)
    ax[0].set_xlim(-0.05, 0.7)
    ax[0].set_ylim(-0.05, 6.5)
    ax[0].set_xlabel('Arretrati/avviato PdC')
    ax[0].set_ylabel('Durata/termine PdC')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    ax[1].set_title(periodo_label, fontsize=12)
    plot2 = ax[1].scatter(comuni_pds_arretrato, comuni_pds_durata,
                          c=classificazione_comunale, marker='o', s=grandezza_comunale, alpha=0.5)
    ax[1].set_xlim(-0.05, 0.7)
    ax[1].set_ylim(-0.05, 6.5)
    ax[1].set_xlabel('Arretrati/avviato PdS')
    ax[1].set_ylabel('Durata/termine PdS')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    ax[2].set_title(periodo_label, fontsize=12)
    plot3 = ax[2].scatter(comuni_pds_scores, comuni_pdc_scores,
                          c=classificazione_comunale, marker='o', s=grandezza_comunale, alpha=0.5)
    ax[2].set_xlim(-0.05, 6.5)
    ax[2].set_ylim(-0.05, 6.5)
    ax[2].set_xlabel('Pressione PdS')
    ax[2].set_ylabel('Pressione PdC')
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False)

    ax[3].set_title('Pressione ' + periodo_label, fontsize=12)
    plot4 = ax[3].violinplot(comuni_scores, showmeans=True, showextrema=False, showmedians=False)
    Nx, Ny = 1, 1000
    imgArr = np.tile(np.linspace(0, 1, Ny), (Nx, 1)).T
    ymin, ymax = 0, 8.5
    xmin, xmax = ax[3].get_xlim()
    path = Path(plot4['bodies'][0].get_paths()[0].vertices)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax[3].add_patch(patch)
    ax[3].imshow(imgArr, origin="lower", extent=[xmin, xmax, ymin, ymax],
                 aspect="auto", cmap=mpl.colormaps['turbo'], clip_path=patch)
    ax[3].hlines(0, 0, 2, colors='r', linewidth=0)
    ax[3].set_xticks([], [])
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False)

    fig.legend(plot1.legend_elements()[0], classificazione_comunale_labels,
               prop={'size': 12}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    fig.savefig('pat_pnrr_mpe\\relazione_tecnica\\'
                'pat_pnrr_performance_chart_provincia_' + mpe_number_label,
                dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)

    if mpe_number >= 5:
        fig, ax = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[0.25, 0.25, 0.25, 0.25]),
                               layout='constrained')
        fig.set_size_inches(15, 5)

        if no_trento:
            ore_tecnici_settimana.Trento = 0
            comuni_pdc_pds_durata.Trento = 0
            comuni_pdc_pds_avviato.Trento = 0
            comuni_pdc_pds_arretrato.Trento = 0
            comuni_scores.Trento = 0
            grandezza_comunale.Trento = 0

        # scatter di Avviato PdC+PdS 2023 ed Ore elaborazione/settimana 2023
        ax[0].set_title(periodo_label, fontsize=12)
        plot1 = ax[0].scatter(ore_tecnici_settimana, comuni_pdc_pds_avviato,
                              c=classificazione_comunale,
                              marker='o', s=grandezza_comunale, alpha=0.5)
        ax[0].set_xlabel('Elaborazione [ore/settimana]')
        ax[0].set_ylabel('Avviati PdC+PdS')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)

        # scatter di Durata media PdC+PdS [gg] 2023 ed Ore elaborazione/settimana 2023
        ax[1].set_title(periodo_label, fontsize=12)
        plot2 = ax[1].scatter(ore_tecnici_settimana, comuni_pdc_pds_durata,
                              c=classificazione_comunale,
                              marker='o', s=grandezza_comunale, alpha=0.5)
        ax[1].set_xlabel('Elaborazione [ore/settimana]')
        ax[1].set_ylabel('Durata [gg] PdC+PdS')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)

        # scatter di Arretrato PdC+PdS 2023 ed Ore elaborazione/settimana 2023
        ax[2].set_title(periodo_label, fontsize=12)
        plot3 = ax[2].scatter(ore_tecnici_settimana, comuni_pdc_pds_arretrato,
                              c=classificazione_comunale,
                              marker='o', s=grandezza_comunale, alpha=0.5)
        ax[2].set_xlabel('Elaborazione [ore/settimana]')
        ax[2].set_ylabel('Arretrati PdC+PdS')
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[2].spines['left'].set_visible(False)

        # scatter di Pressione 2023 ed Ore elaborazione/settimana 2023
        ax[3].set_title(periodo_label, fontsize=12)
        plot4 = ax[3].scatter(ore_tecnici_settimana, comuni_scores,
                        c=classificazione_comunale, marker='o', s=grandezza_comunale, alpha=0.5)
        ax[3].set_ylim(0, 8.5)
        ax[3].set_xlabel('Elaborazione [ore/settimana]')
        ax[3].set_ylabel('Pressione')
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].spines['left'].set_visible(False)

        fig.legend(plot1.legend_elements()[0], classificazione_comunale_labels,
                prop={'size': 12}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)
        fig.savefig('pat_pnrr_mpe\\relazione_tecnica\\'
                    'pat_pnrr_performance_organico_chart_provincia_' + mpe_number_label,
                    dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)

    if just_provincia:
        return

    pdc_measure_labels = ['pdc_2021q3_4', 'pdc_2022q1_2', 'pdc_2022q3_4', 'pdc_2023q1_2',
                          'pdc_2023q3_4']
    pds_measure_labels = ['pds_2021q3_4', 'pds_2022q1_2', 'pds_2022q3_4', 'pds_2023q1_2',
                          'pds_2023q3_4']
    for comune in comuni_excel_map:
        print('produco le dashboard per il comune di ' + comune[0])

        # dashboard comunale
        #   a. (+) comune nello scatter pdc durata/arretrato ultimi 12 mesi
        #   b. (+) comune nello scatter pds durata/arretrato ultimi 12 mesi

        fig, ax = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[0.4, 0.4, 0.2]))
        fig.set_size_inches(15, 5)

        # ax[0].set_title('Andamento PdC e PdS', fontsize=12)
        plot1 = ax[0].plot(np.arange(0, pdc_measure_labels.__len__(), 1),
                           comuni_performance_trends.loc[comune[0], pdc_measure_labels].values,
                           label='Pressione PdC', c='grey', linestyle='dotted', marker='o',
                           alpha=0.75)
        plot2 = ax[0].plot(np.arange(0, pds_measure_labels.__len__(), 1),
                           comuni_performance_trends.loc[comune[0], pds_measure_labels].values,
                           label='Pressione PdS', c='grey', linestyle='dashed', marker='x',
                           alpha=0.75)
        # ax[0].yaxis.grid(True)
        ax[0].set_yticks(pdc_pds_score_ticks)
        ax[0].set_ylim(pdc_pds_score_range)
        ax[0].set_xticks(range(len(pdc_measure_labels)),
                         labels=[label.lstrip('pdc_').replace('q', 'Q').replace('_', '-')
                                 for label in pdc_measure_labels])
        ax[0].legend()
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)

        ax[1].set_title('Ultimi 12 mesi', fontsize=12)
        plot1 = ax[1].scatter(comuni_pds_scores, comuni_pdc_scores,
                              c='gray', marker='o', s=grandezza_comunale, alpha=0.5)
        plot2 = ax[1].scatter(comuni_pds_scores.loc[comune[0]], comuni_pdc_scores.loc[comune[0]],
                              label=comune[0], c='r', marker='D', s=grandezza_comunale[comune[0]])
        ax[1].set_xlim(-0.05, 6.5)
        ax[1].set_ylim(-0.05, 6.5)
        ax[1].set_xlabel('Pressione PdS')
        ax[1].set_ylabel('Pressione PdC')
        ax[1].legend()
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)

        ax[2].set_title('Pressione', fontsize=12)
        plot = ax[2].violinplot(comuni_scores, showmeans=True, showextrema=False,
                                showmedians=False)
        Nx, Ny = 1, 1000
        imgArr = np.tile(np.linspace(0, 1, Ny), (Nx, 1)).T
        ymin, ymax = 0, 8.5
        xmin, xmax = ax[2].get_xlim()
        path = Path(plot['bodies'][0].get_paths()[0].vertices)
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax[2].add_patch(patch)
        ax[2].imshow(imgArr, origin="lower", extent=[xmin, xmax, ymin, ymax],
                     aspect="auto", cmap=mpl.colormaps['turbo'], clip_path=patch)
        ax[2].hlines(comuni_scores[comune[0]], 0, 2, colors='r', linewidth=2)
        ax[2].text(0, comuni_scores[comune[0]] + 0.1,
                           s='{0}: {1:.2f}'.format(comune[0], comuni_scores[comune[0]]))
        ax[2].set_xticks([], [])
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[2].spines['left'].set_visible(False)

        if save_charts:
            comune_edited_name = comune[0]
            comune_edited_name = comune_edited_name.replace('à', 'a')
            comune_edited_name = comune_edited_name.replace('è', 'e')
            comune_edited_name = comune_edited_name.replace('é', 'e')
            comune_edited_name = comune_edited_name.replace('ù', 'u')
            fig.savefig('pat_pnrr_mpe\\relazione_tecnica\\pat_pnrr_performance_charts\\'
                        'pat_pnrr_performance_chart_' + comune_edited_name,
                        dpi=300, bbox_inches='tight', pad_inches=0.25)
        plt.close(fig)

        if mpe_number >= 5:
            fig, ax = plt.subplots(ncols=4,
                                   gridspec_kw=dict(width_ratios=[0.25, 0.25, 0.25, 0.25]),
                                   layout='constrained')
            fig.set_size_inches(15, 5)

            if no_trento:
                ore_tecnici_settimana.Trento = 0
                comuni_pdc_pds_durata.Trento = 0
                comuni_pdc_pds_avviato.Trento = 0
                comuni_pdc_pds_arretrato.Trento = 0
                comuni_scores.Trento = 0
                grandezza_comunale.Trento = 0

            # scatter di Avviato PdC+PdS 2023 ed Ore elaborazione/settimana 2023
            ax[0].set_title(periodo_label, fontsize=12)
            plot1_1 = ax[0].scatter(ore_tecnici_settimana, comuni_pdc_pds_avviato,
                                    c='gray', marker='o', s=grandezza_comunale, alpha=0.5)
            plot1_2 = ax[0].scatter(ore_tecnici_settimana.loc[comune[0]],
                                    comuni_pdc_pds_avviato.loc[comune[0]],
                                    label=comune[0],
                                    c='r', marker='D', s=grandezza_comunale[comune[0]])
            ax[0].set_xlabel('Elaborazione [ore/settimana]')
            ax[0].set_ylabel('Avviati PdC+PdS')
            ax[0].legend()
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].spines['left'].set_visible(False)

            # scatter di Durata media PdC+PdS [gg] 2023 ed Ore elaborazione/settimana 2023
            ax[1].set_title(periodo_label, fontsize=12)
            plot2_1 = ax[1].scatter(ore_tecnici_settimana, comuni_pdc_pds_durata,
                                    c='grey', marker='o', s=grandezza_comunale, alpha=0.5)
            plot2_2 = ax[1].scatter(ore_tecnici_settimana.loc[comune[0]],
                                    comuni_pdc_pds_durata.loc[comune[0]],
                                    c='r', marker='D', s=grandezza_comunale[comune[0]])
            ax[1].set_xlabel('Elaborazione [ore/settimana]')
            ax[1].set_ylabel('Durata [gg] PdC+PdS')
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].spines['left'].set_visible(False)

            # scatter di Arretrato PdC+PdS 2023 ed Ore elaborazione/settimana 2023
            ax[2].set_title(periodo_label, fontsize=12)
            plot3_1 = ax[2].scatter(ore_tecnici_settimana, comuni_pdc_pds_arretrato,
                                    c='grey', marker='o', s=grandezza_comunale, alpha=0.5)
            plot3_1 = ax[2].scatter(ore_tecnici_settimana.loc[comune[0]],
                                    comuni_pdc_pds_arretrato.loc[comune[0]],
                                    c='r', marker='D', s=grandezza_comunale[comune[0]])
            ax[2].set_xlabel('Elaborazione [ore/settimana]')
            ax[2].set_ylabel('Arretrati PdC+PdS')
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].spines['left'].set_visible(False)

            # scatter di Pressione 2023 ed Ore elaborazione/settimana 2023
            ax[3].set_title(periodo_label, fontsize=12)
            plot4_1 = ax[3].scatter(ore_tecnici_settimana, comuni_scores,
                                    c='grey', marker='o', s=grandezza_comunale, alpha=0.5)
            plot4_2 = ax[3].scatter(ore_tecnici_settimana.loc[comune[0]],
                                    comuni_scores.loc[comune[0]],
                                    c='r', marker='D', s=grandezza_comunale[comune[0]])
            ax[3].set_ylim(0, 8.5)
            ax[3].set_xlabel('Elaborazione [ore/settimana]')
            ax[3].set_ylabel('Pressione')
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].spines['left'].set_visible(False)

            if save_charts:
                comune_edited_name = comune[0]
                comune_edited_name = comune_edited_name.replace('à', 'a')
                comune_edited_name = comune_edited_name.replace('è', 'e')
                comune_edited_name = comune_edited_name.replace('é', 'e')
                comune_edited_name = comune_edited_name.replace('ù', 'u')
                fig.savefig('pat_pnrr_mpe\\relazione_tecnica\\pat_pnrr_performance_charts\\'
                            'pat_pnrr_performance_organico_chart_' + comune_edited_name,
                            dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close(fig)

        if just_one:
            break
    return comuni_scores


def print_comuni_performance_tables(pat_comuni_dataframe, just_one=False, save_tables=True):
    pat_comuni_dataframe[
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2021q3_4'] = 60
    pat_comuni_dataframe[
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q1_2'] = 60
    pat_comuni_dataframe[
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2021q3_4'] = 60
    pat_comuni_dataframe[
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2022q1_2'] = 60

    performance_labels = [
        'Durata [gg]',  # 'Durata media [gg]',
        'Termine [gg]',  # 'Termine mediano [gg]',
        'Arretrati',  # 'Pratiche arretrate',
        'Avviati']  # 'Pratiche avviate']

    performance_measure_labels_pdc_2021q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2021q3_4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_2021q3-4']
    performance_measure_labels_pdc_2022q1_2 = [
        'giornate_durata_media_permessi_costruire_conclusi_2022q1-2',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q1_2',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2',
        'numero_permessi_costruire_2022q1-2']
    performance_measure_labels_pdc_2022q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q3-4',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4',
        'numero_permessi_costruire_avviati_2022q3-4']
    performance_measure_labels_pdc_2023q1_2 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q1-2',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2',
        'numero_permessi_costruire_avviati_2023q1-2']
    performance_measure_labels_pdc_2023q3_4 = [
        'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q3-4',
        'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4',
        'numero_permessi_costruire_avviati_2023q3-4']

    performance_measure_labels_pdc_ov_2022q3_4 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2022q3-4',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4',
        'numero_permessi_costruire_ov_avviati_2022q3-4']
    performance_measure_labels_pdc_ov_2023q1_2 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2023q1-2',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2',
        'numero_permessi_costruire_ov_avviati_2023q1-2']
    performance_measure_labels_pdc_ov_2023q3_4 = [
        'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_permessi_costruire_ov_avviati_2023q3-4',
        'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4',
        'numero_permessi_costruire_ov_avviati_2023q3-4']

    performance_measure_labels_pds_2021q3_4 = [
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2021q3_4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4',
        'numero_sanatorie_2021q3-4']
    performance_measure_labels_pds_2022q1_2 = [
        'giornate_durata_media_sanatorie_concluse_2022q1-2',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2022q1_2',
        'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2',
        'numero_sanatorie_2022q1-2']
    performance_measure_labels_pds_2022q3_4 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2022q3-4',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4',
        'numero_sanatorie_avviate_2022q3-4']
    performance_measure_labels_pds_2023q1_2 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2023q1-2',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2',
        'numero_sanatorie_avviate_2023q1-2']
    performance_measure_labels_pds_2023q3_4 = [
        'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4',
        'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2023q3-4',
        'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4',
        'numero_sanatorie_avviate_2023q3-4']

    measure_labels = [
        ['PdC 2021Q3-4', performance_measure_labels_pdc_2021q3_4],
        ['PdC 2022Q1-2', performance_measure_labels_pdc_2022q1_2],
        ['PdC 2022Q3-4', performance_measure_labels_pdc_2022q3_4],
        ['PdC 2023Q1-2', performance_measure_labels_pdc_2023q1_2],
        ['PdC 2023Q3-4', performance_measure_labels_pdc_2023q3_4],
        ['PdC-OV 2022Q3-4', performance_measure_labels_pdc_ov_2022q3_4],
        ['PdC-OV 2023Q1-2', performance_measure_labels_pdc_ov_2023q1_2],
        ['PdC-OV 2023Q3-4', performance_measure_labels_pdc_ov_2023q3_4],
        ['PdS 2021Q3-4', performance_measure_labels_pds_2021q3_4],
        ['PdS 2022Q1-2', performance_measure_labels_pds_2022q1_2],
        ['PdS 2022Q3-4', performance_measure_labels_pds_2022q3_4],
        ['PdS 2023Q1-2', performance_measure_labels_pds_2023q1_2],
        ['PdS 2023Q3-4', performance_measure_labels_pds_2023q3_4]
    ]

    for comune in comuni_excel_map:
        print('produco la tabella per il comune di ' + comune[0])
        
        comune_performance_table = []
        for column_label, original_performance_labels in measure_labels:
            comune_performance_measures = pat_comuni_dataframe.loc[comune[0], original_performance_labels]
            comune_performance_measures.index = performance_labels
            comune_performance_measures.name = column_label
            comune_performance_measures[comune_performance_measures.isna()] = -1
            comune_performance_table.append(comune_performance_measures.astype(int))
        if save_tables:
            comune_edited_name = comune[0]
            comune_edited_name = comune_edited_name.replace('à', 'a')
            comune_edited_name = comune_edited_name.replace('è', 'e')
            comune_edited_name = comune_edited_name.replace('é', 'e')
            comune_edited_name = comune_edited_name.replace('ù', 'u')
            tex_file_name = 'pat_pnrr_mpe/relazione_tecnica/pat_pnrr_performance_tables/' \
                            'pat_pnrr_performance_table_{0}.tex'.format(comune_edited_name)
            with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
                table_tex_file.write(pd.DataFrame(comune_performance_table).to_latex(
                    longtable=True, escape=False,
                    label='pat_pnrr_performance_table_{0}'.format(comune_edited_name))
                    .replace('-1', '')
                    .replace('Continued on next page', 'Continua sulla prossima pagina'))
        if just_one:
            break
    return


def print_comuni_performance_list(just_one=False, save_tables=True):
    list_tex = ''
    for comune in comuni_excel_map:
        print('produco la relazione per il comune di ' + comune[0])

        comune_edited_name = comune[0]
        comune_edited_name = comune_edited_name.replace('à', 'a')
        comune_edited_name = comune_edited_name.replace('è', 'e')
        comune_edited_name = comune_edited_name.replace('é', 'e')
        comune_edited_name = comune_edited_name.replace('ù', 'u')
        list_tex += r'\subsection{{{0}}}'.format(comune[0]) + '\n' + \
                    r'\input{pat_pnrr_performance_tables/' + \
                    r'pat_pnrr_performance_table_{0}.tex}}'.format(comune_edited_name) + '\n' + \
                    r'\begin{center}' + '\n' \
                    r'  \includegraphics[height=5cm]{' + \
                    r'pat_pnrr_performance_charts/' + \
                    r'pat_pnrr_performance_chart_{0}.png}}'.format(comune_edited_name) + '\n' + \
                    r'\end{center}' + '\n' + \
                    r'\begin{center}' + '\n' \
                    r'  \includegraphics[height=5cm]{' + \
                    r'pat_pnrr_performance_charts/' + \
                    r'pat_pnrr_performance_organico_chart_{0}.png}}'.format(comune_edited_name) + '\n' + \
                    r'\end{center}' + '\n' + \
                    r'\clearpage' + '\n\n'
        if just_one:
            break

    if save_tables:
        tex_file_name = 'pat_pnrr_mpe/relazione_tecnica/' \
                        'pat_pnrr_performance_list.tex'
        with open(tex_file_name, 'w', encoding="utf-8") as list_tex_file:
            list_tex_file.write(list_tex)
    return


def print_comuni_pressure_list(comuni_performance_trends):
    pdc_measure_labels = ['pdc_2023q1_2', 'pdc_2023q3_4']
    pds_measure_labels = ['pds_2023q1_2', 'pds_2023q3_4']
    comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
        comuni_performance_trends, pdc_measure_labels, pds_measure_labels)

    tex_file_name = 'pat_pnrr_mpe/relazione_tecnica/' \
                    'pat_pnrr_pressure_list.tex'
    with open(tex_file_name, 'w', encoding="utf-8") as table_tex_file:
        table_tex_file.write(pd.DataFrame(comuni_scores.sort_values(ascending=False),
            columns=['Pressione']).to_latex(float_format="{:.2f}".format,
            longtable=True, escape=False, label='pat_pnrr_pressure_list')
            .replace('Continued on next page', 'Continua sulla prossima pagina'))
    return


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pat_comuni_dataframe = get_pat_comuni_dataframe()

    comuni_durata_trends, comuni_arretrato_trends, comuni_performance_trends = \
        get_comuni_performance_trends(pat_comuni_dataframe, time_limit=365)
    # pdc_measure_labels = ['pdc_2023q1_2', 'pdc_2023q3_4']
    # pds_measure_labels = ['pds_2023q1_2', 'pds_2023q3_4']
    # comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    #     comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
    # comuni_scores.to_csv('pat-pnrr_edilizia_pressione_2023.csv')

    # for mpe_number in [3, 4, 5]:
    for mpe_number in [5]:
        print_comuni_performance_charts(pat_comuni_dataframe,
                                        comuni_durata_trends, comuni_arretrato_trends,
                                        comuni_performance_trends, mpe_number=mpe_number,
                                        just_provincia=False, no_trento=False,
                                        just_one=False, save_charts=True)
    print_comuni_performance_tables(pat_comuni_dataframe, just_one=False, save_tables=True)
    print_comuni_performance_list(just_one=False, save_tables=True)
    print_comuni_pressure_list(comuni_performance_trends)

    # TODO: ! scatter Pressione NETTA dei PdC/PdS: Pressione NETTA dei PdC/PdS con Pd = durata media NETTA [gg] / termine massimo [gg]
    
    # TODO: ! graficare la distribuzione della pressione per una soluzione di misure comunali compatibili con i target

    # TODO: andamento posizione nella lista di comuni per pressione complessiva
    # TODO: mappa 166 comuni: localizzazione pressione
