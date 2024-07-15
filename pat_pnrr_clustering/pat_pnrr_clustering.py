"""
    PAT-PNRR Clustering
    Cluster di Comuni Trentini
    Francesco Melchiori, 2022-2024
"""


import shelve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.decomposition import PCA

from pat_pnrr_mpe.pat_pnrr_comuni_excel_mapping import *
from pat_pnrr_mpe import pat_pnrr_3a_misurazione as pat_pnrr_3a
from pat_pnrr_mpe import pat_pnrr_4a_misurazione as pat_pnrr_4a
from pat_pnrr_mpe import pat_pnrr_5a_misurazione as pat_pnrr_5a


def get_pat_comuni_dataframe_ispat(path_base='', kmeans_clustering_original=True):

    pat_comuni_toponimo = np.loadtxt(
        path_base + 'ispat_statistiche_base_20220314.csv', dtype='U33', delimiter=',', skiprows=1,
        usecols=0, encoding='utf8')

    pat_comuni_popolazione, \
    pat_comuni_superficie_km2, \
    pat_comuni_ricettivita, \
    pat_comuni_turisticita, \
    pat_comuni_composito_turismo, \
    pat_comuni_locali_asia_1000_residenti, \
    pat_comuni_addetti_marketing_1000_residenti, \
    pat_comuni_addetti_servizi_1000_residenti, \
    pat_comuni_altitudine_m, \
    pat_comuni_famiglie, \
    pat_comuni_edifici, \
    pat_comuni_abitazioni = np.loadtxt(
        path_base + 'ispat_statistiche_base_20220314.csv',
        dtype='f8', delimiter=',', skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12), encoding='utf8', unpack=True)

    # ERRATA (ispat feature): pat_comuni_altitudine_m
    # CORRIGE:
    # pat_comuni_altitudine_m = np.loadtxt(
    #     path_base + 'trentino_statistiche_base_20210101.csv',
    #     dtype='f8', delimiter=',', skiprows=1,
    #     usecols=3, encoding='utf8', unpack=True)

    pat_comuni_ufficio_tecnico_sue_ore_2020, \
    pat_comuni_urbanistica_programmazione_territorio_ore_2020, \
    pat_comuni_viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020, \
    pat_comuni_totale_generale_lavori_ore_2020, \
    pat_comuni_dipendenti_2020, \
    pat_comuni_permessi_costruire_residenziale_2019, \
    pat_comuni_permessi_costruire_residenziale_2020, \
    pat_comuni_permessi_costruire_residenziale_2021, \
    pat_comuni_dia_residenziale_2019, \
    pat_comuni_dia_residenziale_2020, \
    pat_comuni_dia_residenziale_2021, \
    pat_comuni_permessi_costruire_non_residenziale_2019, \
    pat_comuni_permessi_costruire_non_residenziale_2020, \
    pat_comuni_permessi_costruire_non_residenziale_2021, \
    pat_comuni_dia_non_residenziale_2019, \
    pat_comuni_dia_non_residenziale_2020, \
    pat_comuni_dia_non_residenziale_2021, \
    pat_comuni_edifici_pubblici_non_residenziale_2019, \
    pat_comuni_edifici_pubblici_non_residenziale_2020, \
    pat_comuni_edifici_pubblici_non_residenziale_2021, \
    pat_comuni_ristrutturazioni_concessioni_2019, \
    pat_comuni_ristrutturazioni_concessioni_2020, \
    pat_comuni_ristrutturazioni_concessioni_2021, \
    pat_comuni_ristrutturazioni_dia_cila_2019, \
    pat_comuni_ristrutturazioni_dia_cila_2020, \
    pat_comuni_ristrutturazioni_dia_cila_2021 = np.loadtxt(
        path_base + 'ispat_statistiche_edilizia_20220314.csv',
        dtype='f8', delimiter=',', skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26), encoding='utf8', unpack=True)

    pat_comuni_data = np.stack([
        pat_comuni_popolazione,
        pat_comuni_superficie_km2,
        pat_comuni_ricettivita,
        pat_comuni_turisticita,
        pat_comuni_composito_turismo,
        pat_comuni_locali_asia_1000_residenti,
        pat_comuni_addetti_marketing_1000_residenti,
        pat_comuni_addetti_servizi_1000_residenti,
        pat_comuni_altitudine_m,
        pat_comuni_famiglie,
        pat_comuni_edifici,
        pat_comuni_abitazioni,
        pat_comuni_ufficio_tecnico_sue_ore_2020,
        pat_comuni_urbanistica_programmazione_territorio_ore_2020,
        pat_comuni_viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020,
        pat_comuni_totale_generale_lavori_ore_2020,
        pat_comuni_dipendenti_2020,
        pat_comuni_permessi_costruire_residenziale_2019,
        pat_comuni_permessi_costruire_residenziale_2020,
        pat_comuni_permessi_costruire_residenziale_2021,
        pat_comuni_dia_residenziale_2019,
        pat_comuni_dia_residenziale_2020,
        pat_comuni_dia_residenziale_2021,
        pat_comuni_permessi_costruire_non_residenziale_2019,
        pat_comuni_permessi_costruire_non_residenziale_2020,
        pat_comuni_permessi_costruire_non_residenziale_2021,
        pat_comuni_dia_non_residenziale_2019,
        pat_comuni_dia_non_residenziale_2020,
        pat_comuni_dia_non_residenziale_2021,
        pat_comuni_edifici_pubblici_non_residenziale_2019,
        pat_comuni_edifici_pubblici_non_residenziale_2020,
        pat_comuni_edifici_pubblici_non_residenziale_2021,
        pat_comuni_ristrutturazioni_concessioni_2019,
        pat_comuni_ristrutturazioni_concessioni_2020,
        pat_comuni_ristrutturazioni_concessioni_2021,
        pat_comuni_ristrutturazioni_dia_cila_2019,
        pat_comuni_ristrutturazioni_dia_cila_2020,
        pat_comuni_ristrutturazioni_dia_cila_2021], axis=1)

    pat_comuni_data_labels = [
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

    pat_comuni_dataframe_ispat = pd.DataFrame(
        pat_comuni_data,
        columns=pat_comuni_data_labels,
        index=pat_comuni_toponimo)

    # attivazioni peo
    pat_comuni_attivazioni_peo = np.loadtxt(
        path_base + 'consorziocomuni_attivazioni_peo_20220311.csv',
        dtype='?', delimiter=',', skiprows=1,
        usecols=1, encoding='utf8')

    pat_comuni_dataframe_attivazioni_peo = pd.DataFrame(
        pat_comuni_attivazioni_peo,
        columns=['pat_comuni_attivazioni_peo'],
        index=pat_comuni_toponimo)

    # kmeans_clustering_original = True
    if kmeans_clustering_original:
        clustering_results = shelve.open(path_base + 'clustering_results')
        clustering_labels = clustering_results['labels']
        clustering_selection_labels = clustering_results['selection_labels']
        clustering_results.close()
    else:
        # trento isolation
        pat_comuni_data = np.delete(pat_comuni_data, 154, 0)

        # standardization
        scaler = preprocessing.StandardScaler().fit(pat_comuni_data)
        pat_comuni_data = scaler.transform(pat_comuni_data)

        # normalization
        # pat_comuni_data = preprocessing.normalize(pat_comuni_data, norm='l2')

        # kmeans clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(pat_comuni_data)
        y_kmeans = kmeans.predict(pat_comuni_data)
        centers = kmeans.cluster_centers_

        clustering_labels = np.insert(y_kmeans, 154, n_clusters)
        clustering_selection_labels = np.zeros((166), dtype=bool)

    pat_comuni_dataframe_kmeans_clustering_labels = pd.DataFrame(
        clustering_labels,
        columns=['pat_comuni_kmeans_clustering_labels'],
        index=pat_comuni_toponimo)

    pat_comuni_dataframe_kmeans_clustering_selection_labels = pd.DataFrame(
        clustering_selection_labels,
        columns=['pat_comuni_dataframe_kmeans_clustering_selection_labels'],
        index=pat_comuni_toponimo)
    pat_comuni_dataframe_kmeans_clustering_selection_labels.loc['Trento'] = True

    if not kmeans_clustering_original:
        n_pat_comuni_cluster_selected = 3
        pat_comuni_clusters = pairwise_distances_argmin_min(pat_comuni_data, centers)
        for n_cluster in range(n_clusters):
            ids_cluster = np.where(pat_comuni_clusters[0] == n_cluster)[0]
            ids_cluster_sorted = np.argsort(pat_comuni_clusters[1][ids_cluster])
            pat_comuni_cluster_selected = pat_comuni_toponimo[ids_cluster][ids_cluster_sorted]\
                [:n_pat_comuni_cluster_selected]
            pat_comuni_dataframe_kmeans_clustering_selection_labels.loc[
                pat_comuni_cluster_selected] = True

        clustering_results = shelve.open(path_base + 'clustering_results')
        clustering_results['labels'] = pat_comuni_dataframe_kmeans_clustering_labels
        clustering_results['selection_labels'] = pat_comuni_dataframe_kmeans_clustering_selection_labels
        clustering_results.close()

    pat_comuni_dataframe_ispat = pd.concat(
        [pat_comuni_dataframe_ispat,
         pat_comuni_dataframe_attivazioni_peo,
         pat_comuni_dataframe_kmeans_clustering_labels,
         pat_comuni_dataframe_kmeans_clustering_selection_labels],
        axis='columns', join='outer')

    return pat_comuni_dataframe_ispat


def cluster_comuni_2022():
    # settings
    n_clusters = 4
    n_pat_comuni_cluster_selected = 166

    # features

    # ispat_statistiche_base_20220314.csv

    # pat_comuni_popolazione
    # pat_comuni_superficie_km2
    # pat_comuni_ricettivita
    # pat_comuni_turisticita
    # pat_comuni_composito_turismo
    # pat_comuni_locali_asia_1000_residenti
    # pat_comuni_addetti_marketing_1000_residenti
    # pat_comuni_addetti_servizi_1000_residenti
    # pat_comuni_altitudine_m
    # pat_comuni_famiglie
    # pat_comuni_edifici
    # pat_comuni_abitazioni

    # ispat_statistiche_edilizia_20220314.csv

    # ufficio_tecnico_sue_ore_2020
    # urbanistica_programmazione_territorio_ore_2020
    # viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020
    # totale_generale_lavori_ore_2020
    # dipendenti_2020
    # permessi_costruire_residenziale_2019
    # permessi_costruire_residenziale_2020
    # permessi_costruire_residenziale_2021
    # dia_residenziale_2019
    # dia_residenziale_2020
    # dia_residenziale_2021
    # permessi_costruire_non_residenziale_2019
    # permessi_costruire_non_residenziale_2020
    # permessi_costruire_non_residenziale_2021
    # dia_non_residenziale_2019
    # dia_non_residenziale_2020
    # dia_non_residenziale_2021
    # edifici_pubblici_non_residenziale_2019
    # edifici_pubblici_non_residenziale_2020
    # edifici_pubblici_non_residenziale_2021
    # ristrutturazioni_concessioni_2019
    # ristrutturazioni_concessioni_2020
    # ristrutturazioni_concessioni_2021
    # ristrutturazioni_dia_cila_2019
    # ristrutturazioni_dia_cila_2020
    # ristrutturazioni_dia_cila_2021

    # missing data

    # cila_scia_ricevute_2018
    # cila_scia_ricevute_2019
    # cila_scia_ricevute_2020
    # opere_pubbliche_senza_collaudo_2018
    # opere_pubbliche_senza_collaudo_2019
    # opere_pubbliche_senza_collaudo_2020
    # puc_approvati_2018
    # puc_approvati_2019
    # puc_approvati_2020
    # convenzioni_altri_enti_vero_2020
    # pianificazione_sovracomunale_vero_2020
    # capofila_convenzioni_vero_2020
    # associazioni_altri_enti_vero_2020

    # https://scikit-learn.org/stable/modules/impute.html#imputation-of-missing-values
    # https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

    # data load

    # pat_comuni_toponimo = np.loadtxt('trentino_statistiche_base_20210101.csv', dtype='U33',
    #                                  delimiter=',', skiprows=1, usecols=0, encoding='utf8')
    # pat_comuni_popolazione, \
    # pat_comuni_superficie_km2, \
    # pat_comuni_altitudine_m = np.loadtxt('trentino_statistiche_base_20210101.csv', dtype='f8',
    #                                      delimiter=',', skiprows=1, usecols=(1, 2, 3),
    #                                      encoding='utf8', unpack=True)
    # pat_comuni_ricettivita, \
    # pat_comuni_turisticita, \
    # pat_comuni_antropizzazione = np.loadtxt('trentino_statistiche_turismo_2020.csv',
    #                                         dtype='f8', delimiter=',', skiprows=1,
    #                                         usecols=(1, 2, 3), encoding='utf8', unpack=True)

    pat_comuni_toponimo = np.loadtxt('ispat_statistiche_base_20220314.csv', dtype='U33',
                                     delimiter=',', skiprows=1, usecols=0, encoding='utf8')
    pat_comuni_popolazione, \
        pat_comuni_superficie_km2, \
        pat_comuni_ricettivita, \
        pat_comuni_turisticita, \
        pat_comuni_composito_turismo, \
        pat_comuni_locali_asia_1000_residenti, \
        pat_comuni_addetti_marketing_1000_residenti, \
        pat_comuni_addetti_servizi_1000_residenti, \
        pat_comuni_altitudine_m, \
        pat_comuni_famiglie, \
        pat_comuni_edifici, \
        pat_comuni_abitazioni = np.loadtxt(
            'ispat_statistiche_base_20220314.csv', dtype='f8', delimiter=',', skiprows=1,
            usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12), encoding='utf8', unpack=True)
    pat_comuni_ufficio_tecnico_sue_ore_2020, \
        pat_comuni_urbanistica_programmazione_territorio_ore_2020, \
        pat_comuni_viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020, \
        pat_comuni_totale_generale_lavori_ore_2020, \
        pat_comuni_dipendenti_2020, \
        pat_comuni_permessi_costruire_residenziale_2019, \
        pat_comuni_permessi_costruire_residenziale_2020, \
        pat_comuni_permessi_costruire_residenziale_2021, \
        pat_comuni_dia_residenziale_2019, \
        pat_comuni_dia_residenziale_2020, \
        pat_comuni_dia_residenziale_2021, \
        pat_comuni_permessi_costruire_non_residenziale_2019, \
        pat_comuni_permessi_costruire_non_residenziale_2020, \
        pat_comuni_permessi_costruire_non_residenziale_2021, \
        pat_comuni_dia_non_residenziale_2019, \
        pat_comuni_dia_non_residenziale_2020, \
        pat_comuni_dia_non_residenziale_2021, \
        pat_comuni_edifici_pubblici_non_residenziale_2019, \
        pat_comuni_edifici_pubblici_non_residenziale_2020, \
        pat_comuni_edifici_pubblici_non_residenziale_2021, \
        pat_comuni_ristrutturazioni_concessioni_2019, \
        pat_comuni_ristrutturazioni_concessioni_2020, \
        pat_comuni_ristrutturazioni_concessioni_2021, \
        pat_comuni_ristrutturazioni_dia_cila_2019, \
        pat_comuni_ristrutturazioni_dia_cila_2020, \
        pat_comuni_ristrutturazioni_dia_cila_2021 = np.loadtxt(
            'ispat_statistiche_edilizia_20220314.csv', dtype='f8', delimiter=',', skiprows=1,
            usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26), encoding='utf8', unpack=True)
    pat_comuni_data = np.stack([
        pat_comuni_popolazione,
        pat_comuni_superficie_km2,
        pat_comuni_ricettivita,
        pat_comuni_turisticita,
        pat_comuni_composito_turismo,
        pat_comuni_locali_asia_1000_residenti,
        pat_comuni_addetti_marketing_1000_residenti,
        pat_comuni_addetti_servizi_1000_residenti,
        pat_comuni_altitudine_m,
        pat_comuni_famiglie,
        pat_comuni_edifici,
        pat_comuni_abitazioni,
        pat_comuni_ufficio_tecnico_sue_ore_2020,
        pat_comuni_urbanistica_programmazione_territorio_ore_2020,
        pat_comuni_viabilita_circolazione_stradale_illuminazione_pubblica_ore_2020,
        pat_comuni_totale_generale_lavori_ore_2020,
        pat_comuni_dipendenti_2020,
        pat_comuni_permessi_costruire_residenziale_2019,
        pat_comuni_permessi_costruire_residenziale_2020,
        pat_comuni_permessi_costruire_residenziale_2021,
        pat_comuni_dia_residenziale_2019,
        pat_comuni_dia_residenziale_2020,
        pat_comuni_dia_residenziale_2021,
        pat_comuni_permessi_costruire_non_residenziale_2019,
        pat_comuni_permessi_costruire_non_residenziale_2020,
        pat_comuni_permessi_costruire_non_residenziale_2021,
        pat_comuni_dia_non_residenziale_2019,
        pat_comuni_dia_non_residenziale_2020,
        pat_comuni_dia_non_residenziale_2021,
        pat_comuni_edifici_pubblici_non_residenziale_2019,
        pat_comuni_edifici_pubblici_non_residenziale_2020,
        pat_comuni_edifici_pubblici_non_residenziale_2021,
        pat_comuni_ristrutturazioni_concessioni_2019,
        pat_comuni_ristrutturazioni_concessioni_2020,
        pat_comuni_ristrutturazioni_concessioni_2021,
        pat_comuni_ristrutturazioni_dia_cila_2019,
        pat_comuni_ristrutturazioni_dia_cila_2020,
        pat_comuni_ristrutturazioni_dia_cila_2021], axis=1)
    n_features = pat_comuni_data.shape[1]

    # attivazioni peo
    pat_comuni_attivazioni_peo = np.loadtxt('consorziocomuni_attivazioni_peo_20220311.csv',
                                            dtype='?', delimiter=',', skiprows=1, usecols=1,
                                            encoding='utf8')

    # trento isolation
    pat_trento_data = pat_comuni_data[154]
    pat_comuni_toponimo = np.delete(pat_comuni_toponimo, 154, 0)
    pat_comuni_data = np.delete(pat_comuni_data, 154, 0)
    pat_comuni_attivazioni_peo = np.delete(pat_comuni_attivazioni_peo, 154, 0)

    # standardization
    scaler = preprocessing.StandardScaler().fit(pat_comuni_data)
    pat_comuni_data = scaler.transform(pat_comuni_data)

    # normalization
    # pat_comuni_data = preprocessing.normalize(pat_comuni_data, norm='l2')

    if True:
        # kmeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(pat_comuni_data)
        y_kmeans = kmeans.predict(pat_comuni_data)
        centers = kmeans.cluster_centers_
        clustering_results = shelve.open('clustering_results')
        clustering_results['labels'] = np.insert(y_kmeans, 154, n_clusters)
        clustering_results.close()
        pat_comuni_clusters = pairwise_distances_argmin_min(pat_comuni_data, centers)
        pat_comuni_selezionati_data = np.zeros((n_clusters, n_features))
        pat_comuni_peo_selezionati_data = np.zeros((n_clusters, n_features))
        for n_cluster in range(n_clusters):
            print('CLUSTER', n_cluster+1)
            print('')
            ids_cluster = np.where(pat_comuni_clusters[0] == n_cluster)[0]
            ids_cluster_sorted = np.argsort(pat_comuni_clusters[1][ids_cluster])
            pat_comuni_cluster_selected = pat_comuni_toponimo[ids_cluster][ids_cluster_sorted]\
                [:n_pat_comuni_cluster_selected]
            pat_comuni_selezionati_data[n_cluster] = pat_comuni_data[ids_cluster]\
                [ids_cluster_sorted][0]
            print('    Raccolta dati da questionario o di persona:', pat_comuni_cluster_selected,
                  'sul totale di', sum(y_kmeans == n_cluster))
            print('')
            ids_cluster_peo = np.where(pat_comuni_clusters[0]\
                                       [pat_comuni_attivazioni_peo] == n_cluster)[0]
            ids_cluster_peo_sorted = np.argsort(pat_comuni_clusters[1]\
                                                [pat_comuni_attivazioni_peo][ids_cluster_peo])
            if ids_cluster_peo.any() and ids_cluster_peo_sorted.any():
                pat_comuni_cluster_peo_selected = pat_comuni_toponimo[pat_comuni_attivazioni_peo]\
                    [ids_cluster_peo][ids_cluster_peo_sorted][:n_pat_comuni_cluster_selected]
                pat_comuni_peo_selezionati_data[n_cluster] = pat_comuni_data\
                    [pat_comuni_attivazioni_peo][ids_cluster_peo][ids_cluster_peo_sorted][0]
                print('    Raccolta dati da sistema informatico:', pat_comuni_cluster_peo_selected,
                      'sul totale di', sum(y_kmeans == n_cluster))
            else:
                print('    Raccolta dati da sistema informatico: []', 'sul totale di',
                      sum(y_kmeans == n_cluster))
            print('')
            print('')

        # pca
        pca = PCA(n_components=4)
        pca.fit(pat_comuni_data)
        X_pca = pca.transform(pat_comuni_data)
        Xs_pca = pca.transform(pat_comuni_selezionati_data)
        centers_pca = pca.transform(centers)

        # X = pat_comuni_data
        # Xs = pat_comuni_selezionati_data
        # centers = centers
        X = X_pca
        Xs = Xs_pca
        centers = centers_pca

        if False:
            # scatterplot2D-01
            for i in range(3):
                plt.scatter(X[:, i], X[:, i + 1], c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
                plt.scatter(Xs[:, i], Xs[:, i + 1], marker='*', c=range(n_clusters),s=50, alpha=1,
                            cmap='viridis')
                plt.scatter(centers[:, 0], centers[:, 1], marker='2', c=range(n_clusters), s=100,
                            alpha=0.75, cmap='viridis')
                plt.savefig("pat-pnrr_kmeans_clustering_pca_" + str(i), dpi=300)
                plt.show()

        if False:
            # scatterplot3D
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
            ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], marker='*', c=range(n_clusters), s=50,
                       alpha=1, cmap='viridis')
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='2',
                       c=range(n_clusters), s=100, alpha=0.75, cmap='viridis')
            plt.show()

    if False:
        # dbscan clustering
        dbscan = DBSCAN(eps=3, min_samples=2)
        dbscan.fit(pat_comuni_data)
        y_dbscan = dbscan.labels_
        print(y_dbscan)

        # pca
        pca = PCA(n_components=4)
        pca.fit(pat_comuni_data)
        X_pca = pca.transform(pat_comuni_data)

        # X = pat_comuni_data
        X = X_pca

        for i in range(3):
            plt.scatter(X[:, i], X[:, i+1], c=y_dbscan, s=50, alpha=0.5, cmap='viridis')
            plt.show()


def cluster_comuni_2024(n_clusters):
    n_clusters = n_clusters

    # LOAD DATA
    comuni_measures_dataframe_mpe_3 = pat_pnrr_3a.get_comuni_measures_dataframe(
        comuni_excel_map, load=True)
    comuni_measures_dataframe_mpe_4 = pat_pnrr_4a.get_comuni_measures_dataframe(
        comuni_excel_map, load=True)
    comuni_measures_dataframe_mpe_5 = pat_pnrr_5a.get_comuni_measures_dataframe(
        comuni_excel_map, load=True)

    # SELECT DATA and FILL NANs
    numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q3-4']],
        axis='columns', join='outer')
    numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso.ffill(
        axis='columns', inplace=True)

    giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso_2023q3-4']],
        axis='columns', join='outer')
    giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.ffill(
        axis='columns', inplace=True)
    giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.bfill(
        axis='columns', inplace=True)
    giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso.fillna(
        value=60, axis='columns', inplace=True)
    
    numero_permessi_costruire_ov_avviati = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_permessi_costruire_ov_avviati_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_permessi_costruire_ov_avviati_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_permessi_costruire_ov_avviati_2023q3-4']],
        axis='columns', join='outer')
    numero_permessi_costruire_ov_avviati.ffill(
        axis='columns', inplace=True)
    
    numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo_2023q3-4']],
        axis='columns', join='outer')
    numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo.ffill(
        axis='columns', inplace=True)
    
    numero_sanatorie_concluse_con_provvedimento_espresso = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_sanatorie_concluse_con_provvedimento_espresso_2023q3-4']],
        axis='columns', join='outer')
    numero_sanatorie_concluse_con_provvedimento_espresso.ffill(
        axis='columns', inplace=True)
    
    giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q3-4']],
        axis='columns', join='outer')
    giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.ffill(
        axis='columns', inplace=True)
    giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.bfill(
        axis='columns', inplace=True)
    giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso.fillna(
        value=60, axis='columns', inplace=True)
    
    numero_sanatorie_avviate = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_sanatorie_avviate_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_sanatorie_avviate_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_sanatorie_avviate_2023q3-4']],
        axis='columns', join='outer')
    numero_sanatorie_avviate.ffill(
        axis='columns', inplace=True)
    
    numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo = pd.concat([
        comuni_measures_dataframe_mpe_3[
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4'],
        comuni_measures_dataframe_mpe_4[
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2'],
        comuni_measures_dataframe_mpe_5[
            'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q3-4']],
        axis='columns', join='outer')
    numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo.ffill(
        axis='columns', inplace=True)
    
    comuni_measures_dataframe_mpe = pd.concat([
        numero_permessi_costruire_ov_conclusi_con_provvedimento_espresso,
        giornate_durata_media_permessi_costruire_ov_conclusi_con_provvedimento_espresso,
        numero_permessi_costruire_ov_avviati,
        numero_permessi_costruire_ov_arretrati_non_conclusi_scaduto_termine_massimo,
        numero_sanatorie_concluse_con_provvedimento_espresso,
        giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso,
        numero_sanatorie_avviate,
        numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo],
        axis='columns', join='outer')
    n_features = comuni_measures_dataframe_mpe.shape[1]
    
    # STANDARDIZATION
    scaler = preprocessing.StandardScaler().fit(comuni_measures_dataframe_mpe)
    comuni_tensors_dataframe_mpe = scaler.transform(comuni_measures_dataframe_mpe)

    # NORMALIZATION
    # comuni_tensors_dataframe_mpe = preprocessing.normalize(comuni_tensors_dataframe_mpe, norm='l2')

    # KMEANS
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(comuni_tensors_dataframe_mpe)
    y_kmeans = kmeans.predict(comuni_tensors_dataframe_mpe)
    centers = kmeans.cluster_centers_
    
    # PRINT CLUSTERS
    pat_comuni_clusters = pd.Series(
        data=pairwise_distances_argmin_min(comuni_tensors_dataframe_mpe, centers)[0],
        index=comuni_measures_dataframe_mpe.index,
        name='pat_comune_cluster')
    for n_cluster in range(n_clusters):
        print('PAT comuni cluster ', n_cluster+1, ':')
        for comune in pat_comuni_clusters[pat_comuni_clusters == n_cluster].index:
            print('    ' + comune)
        print('')

    # PCA
    pca = PCA(n_components=4)
    pca.fit(comuni_tensors_dataframe_mpe)
    X_pca = pca.transform(comuni_tensors_dataframe_mpe)
    centers_pca = pca.transform(centers)

    # SCATTER 2D
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                marker='2', c=range(n_clusters), s=100, alpha=0.75, cmap='viridis')
    plt.show()

    # SCATTER 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
               c=y_kmeans, s=50, alpha=0.5, cmap='viridis')
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2],
               marker='2', c=range(n_clusters), s=100, alpha=0.75, cmap='viridis')
    plt.show()

    return


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    cluster_comuni_2024(n_clusters = 7)
