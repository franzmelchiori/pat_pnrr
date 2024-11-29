import json
import numpy as np
import pandas as pd
from flask import render_template, url_for, request
from markupsafe import escape, Markup

from pat_pnrr.pat_pnrr_mpe_server import app
from pat_pnrr_monitoring_analyzer import get_pat_comuni_dataframe
from pat_pnrr_monitoring_reporter import get_comuni_performance_trends, get_comuni_scores
from pat_pnrr.pat_pnrr_mpe import pat_pnrr_5a_misurazione
from pat_pnrr.pat_pnrr_mpe import pat_pnrr_6a_misurazione


pd.options.mode.copy_on_write = True


target = {
    'pdc_ov_durata': 109,
    'pdc_ov_arretrati': 324,
    'pds_durata': 130,
    'pds_arretrati': 300}

pat_comuni_dataframe = get_pat_comuni_dataframe()
comuni_popolazione = pat_comuni_dataframe.pat_comuni_popolazione
classificazione_comunale_map = {
    'Cluster 1: comuni piccoli (130)': 0,
    'Cluster 2: comuni medio-piccoli (31)': 1,
    'Cluster 3: comuni medi (3)': 2,
    'Cluster 4: Rovereto': 3,
    'Cluster 5: Trento': 4}
classificazione_comunale = pat_comuni_dataframe.pat_comuni_kmeans_clustering_labels
classificazione_comunale = classificazione_comunale.map(classificazione_comunale_map)

# TODO: modulare le ore e gli scores per misurazione (es. 2023q3-2024q2, 2023)
ore_tecnici_settimana = pat_comuni_dataframe.loc[:, 'ore_tecnici_settimana_2024q1-2']
ore_tecnici_settimana.loc[ore_tecnici_settimana.isna()] = \
    pat_comuni_dataframe.loc[:, 'ore_tecnici_settimana_2023q3-4'].loc[ore_tecnici_settimana.isna()]
ore_tecnici_settimana.loc[ore_tecnici_settimana.isna()] = 0
comuni_durata_trends, comuni_durata_netta_trends, \
comuni_arretrato_trends, \
comuni_performance_trends, comuni_performance_netta_trends= \
    get_comuni_performance_trends(pat_comuni_dataframe)  #, time_limit=365)

pdc_measure_labels = ['pdc_2021q3_4', 'pdc_2022q1_2']
pds_measure_labels = ['pds_2021q3_4', 'pds_2022q1_2']
comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
scatter_pressione_data_2021q3_2022q2 = pd.concat([
    classificazione_comunale,
    comuni_pds_scores,
    comuni_pdc_scores,
    comuni_popolazione,
    comuni_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_score',
        'pdc_score',
        'popolazione',
        'score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')

pdc_measure_labels = ['pdc_2022q1_2', 'pdc_2022q3_4']
pds_measure_labels = ['pds_2022q1_2', 'pds_2022q3_4']
comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
scatter_pressione_data_2022 = pd.concat([
    classificazione_comunale,
    comuni_pds_scores,
    comuni_pdc_scores,
    comuni_popolazione,
    comuni_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_score',
        'pdc_score',
        'popolazione',
        'score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')

pdc_measure_labels = ['pdc_2022q3_4', 'pdc_2023q1_2']
pds_measure_labels = ['pds_2022q3_4', 'pds_2023q1_2']
pdc_net_measure_labels = ['pdc_performance_netta_2022q3_4', 'pdc_performance_netta_2023q1_2']
pds_net_measure_labels = ['pds_performance_netta_2022q3_4', 'pds_performance_netta_2023q1_2']
comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
comuni_pdc_net_scores, comuni_pds_net_scores, comuni_net_scores = get_comuni_scores(
    comuni_performance_netta_trends, pdc_net_measure_labels, pds_net_measure_labels)
scatter_pressione_data_2022q3_2023q2 = pd.concat([
    classificazione_comunale,
    comuni_pds_scores,
    comuni_pdc_scores,
    comuni_popolazione,
    comuni_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_score',
        'pdc_score',
        'popolazione',
        'score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')
scatter_pressione_netta_data_2022q3_2023q2 = pd.concat([
    classificazione_comunale,
    comuni_pds_net_scores,
    comuni_pdc_net_scores,
    comuni_popolazione,
    comuni_net_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_net_score',
        'pdc_net_score',
        'popolazione',
        'net_score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')

pdc_measure_labels = ['pdc_2023q1_2', 'pdc_2023q3_4']
pds_measure_labels = ['pds_2023q1_2', 'pds_2023q3_4']
pdc_net_measure_labels = ['pdc_performance_netta_2023q1_2', 'pdc_performance_netta_2023q3_4']
pds_net_measure_labels = ['pds_performance_netta_2023q1_2', 'pds_performance_netta_2023q3_4']
comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
comuni_pdc_net_scores, comuni_pds_net_scores, comuni_net_scores = get_comuni_scores(
    comuni_performance_netta_trends, pdc_net_measure_labels, pds_net_measure_labels)
scatter_pressione_data_2023 = pd.concat([
    classificazione_comunale,
    comuni_pds_scores,
    comuni_pdc_scores,
    comuni_popolazione,
    comuni_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_score',
        'pdc_score',
        'popolazione',
        'score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')
scatter_pressione_netta_data_2023 = pd.concat([
    classificazione_comunale,
    comuni_pds_net_scores,
    comuni_pdc_net_scores,
    comuni_popolazione,
    comuni_net_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_net_score',
        'pdc_net_score',
        'popolazione',
        'net_score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')

pdc_measure_labels = ['pdc_2023q3_4', 'pdc_2024q1_2']
pds_measure_labels = ['pds_2023q3_4', 'pds_2024q1_2']
pdc_net_measure_labels = ['pdc_performance_netta_2023q3_4', 'pdc_performance_netta_2024q1_2']
pds_net_measure_labels = ['pds_performance_netta_2023q3_4', 'pds_performance_netta_2024q1_2']
comuni_pdc_scores, comuni_pds_scores, comuni_scores = get_comuni_scores(
    comuni_performance_trends, pdc_measure_labels, pds_measure_labels)
comuni_pdc_net_scores, comuni_pds_net_scores, comuni_net_scores = get_comuni_scores(
    comuni_performance_netta_trends, pdc_net_measure_labels, pds_net_measure_labels)
scatter_pressione_data_2023q3_2024q2 = pd.concat([
    classificazione_comunale,
    comuni_pds_scores,
    comuni_pdc_scores,
    comuni_popolazione,
    comuni_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_score',
        'pdc_score',
        'popolazione',
        'score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')
scatter_pressione_netta_data_2023q3_2024q2 = pd.concat([
    classificazione_comunale,
    comuni_pds_net_scores,
    comuni_pdc_net_scores,
    comuni_popolazione,
    comuni_net_scores,
    ore_tecnici_settimana,
    pd.Series(pat_comuni_dataframe.index, pat_comuni_dataframe.index)],
    keys=[
        'cluster_comune',
        'pds_net_score',
        'pdc_net_score',
        'popolazione',
        'net_score',
        'ore_tecnici_settimana',
        'nome_comune'],
    axis='columns', join='outer')

chart_provincia_area_time_avviato_pdc_pds_series = {
    'pdc_avviato': [
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_2021q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_2022q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_avviati_2022q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_avviati_2023q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_avviati_2023q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_avviati_2024q1-2'].sum().astype(int))],
    'pds_avviato': [
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_2021q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_2022q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_avviate_2022q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_avviate_2023q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_avviate_2023q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_avviate_2024q1-2'].sum().astype(int))]}
chart_provincia_line_time_durata_pdc_pds_series = {
    'pdc_durata': [
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_2021q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_2022q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2024q1-2'].mean()).astype(int))],
    'pds_durata': [
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_2021q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_2022q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2024q1-2'].mean()).astype(int))]}
chart_provincia_line_time_durata_netta_pdc_pds_series = {
    'pdc_durata': [
        str(),
        str(),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_permessi_costruire_conclusi_con_provvedimento_espresso_2023q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_permessi_costruire_conclusi_con_provvedimento_espresso_2024q1-2'].mean()).astype(int))],
    'pds_durata': [
        str(),
        str(),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_sanatorie_concluse_con_provvedimento_espresso_2023q3-4'].mean()).astype(int)),
        str(np.ceil(pat_comuni_dataframe.loc[:, 'giornate_durata_media_netta_sanatorie_concluse_con_provvedimento_espresso_2024q1-2'].mean()).astype(int))]}
chart_provincia_area_time_arretrato_pdc_pds_series = {
    'pdc_arretrato': [
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2024q1-2'].sum().astype(int))],
    'pds_arretrato': [
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4'].sum().astype(int)),
        str(pat_comuni_dataframe.loc[:, 'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2024q1-2'].sum().astype(int))]}
chart_comuni_scatter_cluster_pop_pressione_pdc_pds_series = [{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 4].iloc[:, 1:])]}]
chart_comuni_scatter_cluster_pop_pressione_netta_pdc_pds_series = [{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 4].iloc[:, 1:])]
    },{
    'comuni_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 0].iloc[:, 1:])],
    'comuni_medio_piccoli':[list(a) for a in np.array(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 1].iloc[:, 1:])],
    'comuni_medi':[list(a) for a in np.array(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 2].iloc[:, 1:])],
    'rovereto':[list(a) for a in np.array(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 3].iloc[:, 1:])],
    'trento':[list(a) for a in np.array(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 4].iloc[:, 1:])]}]
chart_comuni_box_cluster_pressione_series = [{
    'comuni_piccoli': list(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 0].score),
    'comuni_medio_piccoli': list(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 1].score),
    'comuni_medi': list(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 2].score),
    'rovereto': list(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 3].score),
    'trento': list(scatter_pressione_data_2021q3_2022q2[scatter_pressione_data_2021q3_2022q2.cluster_comune == 4].score)
    },{
    'comuni_piccoli': list(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 0].score),
    'comuni_medio_piccoli': list(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 1].score),
    'comuni_medi': list(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 2].score),
    'rovereto': list(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 3].score),
    'trento': list(scatter_pressione_data_2022[scatter_pressione_data_2022.cluster_comune == 4].score)
    },{
    'comuni_piccoli': list(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 0].score),
    'comuni_medio_piccoli': list(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 1].score),
    'comuni_medi': list(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 2].score),
    'rovereto': list(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 3].score),
    'trento': list(scatter_pressione_data_2022q3_2023q2[scatter_pressione_data_2022q3_2023q2.cluster_comune == 4].score)
    },{
    'comuni_piccoli': list(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 0].score),
    'comuni_medio_piccoli': list(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 1].score),
    'comuni_medi': list(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 2].score),
    'rovereto': list(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 3].score),
    'trento': list(scatter_pressione_data_2023[scatter_pressione_data_2023.cluster_comune == 4].score)
    },{
    'comuni_piccoli': list(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 0].score),
    'comuni_medio_piccoli': list(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 1].score),
    'comuni_medi': list(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 2].score),
    'rovereto': list(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 3].score),
    'trento': list(scatter_pressione_data_2023q3_2024q2[scatter_pressione_data_2023q3_2024q2.cluster_comune == 4].score)}]
chart_comuni_box_cluster_pressione_netta_series = [{
    'comuni_piccoli': list(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 0].net_score),
    'comuni_medio_piccoli': list(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 1].net_score),
    'comuni_medi': list(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 2].net_score),
    'rovereto': list(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 3].net_score),
    'trento': list(scatter_pressione_netta_data_2022q3_2023q2[scatter_pressione_netta_data_2022q3_2023q2.cluster_comune == 4].net_score)
    },{
    'comuni_piccoli': list(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 0].net_score),
    'comuni_medio_piccoli': list(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 1].net_score),
    'comuni_medi': list(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 2].net_score),
    'rovereto': list(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 3].net_score),
    'trento': list(scatter_pressione_netta_data_2023[scatter_pressione_netta_data_2023.cluster_comune == 4].net_score)
    },{
    'comuni_piccoli': list(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 0].net_score),
    'comuni_medio_piccoli': list(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 1].net_score),
    'comuni_medi': list(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 2].net_score),
    'rovereto': list(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 3].net_score),
    'trento': list(scatter_pressione_netta_data_2023q3_2024q2[scatter_pressione_netta_data_2023q3_2024q2.cluster_comune == 4].net_score)}]
chart_provincia_gauge_pressione_timelapse_series = [
    str(scatter_pressione_data_2021q3_2022q2.score.mean()),
    str(scatter_pressione_data_2022.score.mean()),
    str(scatter_pressione_data_2022q3_2023q2.score.mean()),
    str(scatter_pressione_data_2023.score.mean()),
    str(scatter_pressione_data_2023q3_2024q2.score.mean())]
chart_provincia_gauge_pressione_netta_timelapse_series = [
    str(scatter_pressione_netta_data_2022q3_2023q2.net_score.mean()),
    str(scatter_pressione_netta_data_2023.net_score.mean()),
    str(scatter_pressione_netta_data_2023q3_2024q2.net_score.mean())]
rank_pressione_index_2021q3_2022q2 = scatter_pressione_data_2021q3_2022q2.score.sort_values().index
rank_pressione_index_2022 = scatter_pressione_data_2022.score.sort_values().index
rank_pressione_index_2022q3_2023q2 = scatter_pressione_data_2022q3_2023q2.score.sort_values().index
rank_pressione_index_2023 = scatter_pressione_data_2023.score.sort_values().index
rank_pressione_index_2023q3_2024q2 = scatter_pressione_data_2023q3_2024q2.score.sort_values().index
chart_comuni_rank_time_pressione_series = [[
    comune,
    [str(np.array(range(1, 167))[rank_pressione_index_2021q3_2022q2 == comune][0]),
     str(np.array(range(1, 167))[rank_pressione_index_2022 == comune][0]),
     str(np.array(range(1, 167))[rank_pressione_index_2022q3_2023q2 == comune][0]),
     str(np.array(range(1, 167))[rank_pressione_index_2023 == comune][0]),
     str(np.array(range(1, 167))[rank_pressione_index_2023q3_2024q2 == comune][0])],
    str(scatter_pressione_data_2023q3_2024q2.loc[comune].score)]
    for comune in pat_comuni_dataframe.index]


@app.route('/')
def index():
    btnradio_mpe = request.args.get('btnradio_mpe')
    if not btnradio_mpe:
        btnradio_mpe = 'btnradio_mpe_2024Q1_2'
    durata_netta = request.args.get('durata_netta')
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
            btnradio_mpe = btnradio_mpe,
            durata_netta = durata_netta,
            chart_provincia_area_time_avviato_pdc_pds_series = chart_provincia_area_time_avviato_pdc_pds_series,
            chart_provincia_line_time_durata_pdc_pds_series = chart_provincia_line_time_durata_pdc_pds_series,
            chart_provincia_line_time_durata_netta_pdc_pds_series = chart_provincia_line_time_durata_netta_pdc_pds_series,
            chart_provincia_area_time_arretrato_pdc_pds_series = chart_provincia_area_time_arretrato_pdc_pds_series,
            chart_comuni_scatter_cluster_pop_pressione_pdc_pds_series = chart_comuni_scatter_cluster_pop_pressione_pdc_pds_series,
            chart_comuni_scatter_cluster_pop_pressione_netta_pdc_pds_series = chart_comuni_scatter_cluster_pop_pressione_netta_pdc_pds_series,
            chart_comuni_box_cluster_pressione_series = chart_comuni_box_cluster_pressione_series,
            chart_comuni_box_cluster_pressione_netta_series = chart_comuni_box_cluster_pressione_netta_series,
            chart_provincia_gauge_pressione_timelapse_series = chart_provincia_gauge_pressione_timelapse_series,
            chart_provincia_gauge_pressione_netta_timelapse_series = chart_provincia_gauge_pressione_netta_timelapse_series,
            chart_comuni_rank_time_pressione_series = chart_comuni_rank_time_pressione_series)
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
            btnradio_mpe = btnradio_mpe,
            durata_netta = durata_netta,
            chart_provincia_area_time_avviato_pdc_pds_series = chart_provincia_area_time_avviato_pdc_pds_series,
            chart_provincia_line_time_durata_pdc_pds_series = chart_provincia_line_time_durata_pdc_pds_series,
            chart_provincia_line_time_durata_netta_pdc_pds_series = chart_provincia_line_time_durata_netta_pdc_pds_series,
            chart_provincia_area_time_arretrato_pdc_pds_series = chart_provincia_area_time_arretrato_pdc_pds_series,
            chart_comuni_scatter_cluster_pop_pressione_pdc_pds_series = chart_comuni_scatter_cluster_pop_pressione_pdc_pds_series,
            chart_comuni_scatter_cluster_pop_pressione_netta_pdc_pds_series = chart_comuni_scatter_cluster_pop_pressione_netta_pdc_pds_series,
            chart_comuni_box_cluster_pressione_series = chart_comuni_box_cluster_pressione_series,
            chart_comuni_box_cluster_pressione_netta_series = chart_comuni_box_cluster_pressione_netta_series,
            chart_provincia_gauge_pressione_timelapse_series = chart_provincia_gauge_pressione_timelapse_series,
            chart_provincia_gauge_pressione_netta_timelapse_series = chart_provincia_gauge_pressione_netta_timelapse_series,
            chart_comuni_rank_time_pressione_series = chart_comuni_rank_time_pressione_series)

@app.route('/misure')
def misure():
    return render_template('misure.html')

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
