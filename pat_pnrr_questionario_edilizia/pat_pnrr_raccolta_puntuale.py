"""
    PAT-PNRR Baseline June 2022
    ISPAT Survey Processing System
    In person collection of data
    Francesco Melchiori, 2022
"""


import numpy as np
import pandas as pd


def find_string_dataframe(dataframe, find_string_row='TOTAL',
                          find_string_column='senza sospensioni'):

    find_mask_column = dataframe.apply(
        lambda row: row.astype(str).str.contains(find_string_column).any(), axis=0)
    find_mask_row = dataframe.apply(
        lambda column: column.astype(str).str.contains(find_string_row).any(), axis=1)

    find_index_column = find_mask_column[find_mask_column].index[0]
    find_index_row = find_mask_row[find_mask_row].index[0]

    return find_index_column, find_index_row


def get_series_xlsx(path_file_xlsx):

    labels_raccolta_puntuale = ['numero_pratiche_concluse_senza_sospensioni',
                                'numero_pratiche_concluse_con_sospensioni',
                                'numero_pratiche_concluse_con_conferenza_servizi',
                                'numero_pratiche_concluse_con_silenzio-assenso',
                                'giornate_durata_media_pratiche_concluse',
                                'numero_pratiche_non_concluse_non_scaduti_termini',
                                'numero_pratiche_non_concluse_scaduti_termini']

    dataframe_xlsx = pd.read_excel(path_file_xlsx, sheet_name=None, header=None)

    for key in dataframe_xlsx.keys():
        if 'costruire' in key.lower():
            dataframe_permessi_costruire = dataframe_xlsx[key]
        elif 'cila' in key.lower():
            dataframe_controllo_cila = dataframe_xlsx[key]
        elif 'sanatorie' in key.lower():
            dataframe_sanatorie = dataframe_xlsx[key]

    find_index_column, find_index_row = find_string_dataframe(dataframe_permessi_costruire)
    series_permessi_costruire = dataframe_permessi_costruire.iloc[
                                    find_index_row][find_index_column:find_index_column+7]
    series_permessi_costruire.index = labels_raccolta_puntuale
    series_permessi_costruire.name = 'series_permessi_costruire'
    series_permessi_costruire = pd.to_numeric(
        series_permessi_costruire, errors='coerce').fillna(0).astype('i4')

    find_index_column, find_index_row = find_string_dataframe(dataframe_controllo_cila)
    series_controllo_cila = dataframe_controllo_cila.iloc[
                                find_index_row][find_index_column:find_index_column+7]
    series_controllo_cila.index = labels_raccolta_puntuale
    series_controllo_cila.name = 'series_controllo_cila'
    series_controllo_cila = pd.to_numeric(
        series_controllo_cila, errors='coerce').fillna(0).astype('i4')

    find_index_column, find_index_row = find_string_dataframe(dataframe_sanatorie)
    series_sanatorie = dataframe_sanatorie.iloc[
                           find_index_row][find_index_column:find_index_column+7]
    series_sanatorie.index = labels_raccolta_puntuale
    series_sanatorie.name = 'series_sanatorie'
    series_sanatorie = pd.to_numeric(
        series_sanatorie, errors='coerce').fillna(0).astype('i4')

    return series_permessi_costruire, series_controllo_cila, series_sanatorie


def get_dataframe_xlsx(comune, path_file_xlsx):

    dataframe_xlsx = pd.read_excel(path_file_xlsx, sheet_name=None, header=None)

    for key in dataframe_xlsx.keys():
        if 'costruire' in key.lower():
            dataframe_permessi_costruire = dataframe_xlsx[key]
        elif 'cila' in key.lower():
            dataframe_controllo_cila = dataframe_xlsx[key]
        elif 'sanatorie' in key.lower():
            dataframe_sanatorie = dataframe_xlsx[key]

    find_index_column, find_index_row = find_string_dataframe(dataframe_permessi_costruire)
    series_permessi_costruire = dataframe_permessi_costruire.iloc[
                                    find_index_row][find_index_column:find_index_column+7]
    series_permessi_costruire = pd.to_numeric(
        series_permessi_costruire, errors='coerce').fillna(0).astype('i4')
    series_permessi_costruire.reset_index(drop=True, inplace=True)

    numero_permessi_costruire_conclusi_con_provvedimento_espresso =\
        series_permessi_costruire[0] +\
        series_permessi_costruire[1] +\
        series_permessi_costruire[2]
    numero_permessi_costruire = \
        numero_permessi_costruire_conclusi_con_provvedimento_espresso +\
        series_permessi_costruire[3] +\
        series_permessi_costruire[5] +\
        series_permessi_costruire[6]

    raccolta_puntuale_comune_baseline_permesso_costruire_labels = np.array([
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_permessi_costruire_2021q3-4'])
    raccolta_puntuale_comune_baseline_permesso_costruire = [
        series_permessi_costruire[3],  # numero_pratiche_concluse_con_silenzio-assenso
        series_permessi_costruire[0],  # numero_pratiche_concluse_senza_sospensioni
        series_permessi_costruire[1],  # numero_pratiche_concluse_con_sospensioni
        series_permessi_costruire[2],  # numero_pratiche_concluse_con_conferenza_servizi
        series_permessi_costruire[4],  # giornate_durata_media_pratiche_concluse
        series_permessi_costruire[5],  # numero_pratiche_non_concluse_non_scaduti_termini
        series_permessi_costruire[6],  # numero_pratiche_non_concluse_scaduti_termini
        numero_permessi_costruire_conclusi_con_provvedimento_espresso,
        numero_permessi_costruire]
    pat_dataframe_raccolta_puntuale_comune_baseline_permesso_costruire = pd.DataFrame(
        [raccolta_puntuale_comune_baseline_permesso_costruire],
        columns=raccolta_puntuale_comune_baseline_permesso_costruire_labels,
        index=[comune])

    find_index_column, find_index_row = find_string_dataframe(dataframe_controllo_cila)
    series_controllo_cila = dataframe_controllo_cila.iloc[
                                    find_index_row][find_index_column:find_index_column+7]
    series_controllo_cila = pd.to_numeric(
        series_controllo_cila, errors='coerce').fillna(0).astype('i4')
    series_controllo_cila.reset_index(drop=True, inplace=True)

    numero_controlli_cila_conclusi_con_provvedimento_espresso =\
        series_controllo_cila[0] +\
        series_controllo_cila[1] +\
        series_controllo_cila[2]
    numero_controlli_cila = \
        numero_controlli_cila_conclusi_con_provvedimento_espresso +\
        series_controllo_cila[5] +\
        series_controllo_cila[6]

    raccolta_puntuale_comune_baseline_controllo_cila_labels = np.array([
        'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_controlli_cila_2021q3-4'])
    raccolta_puntuale_comune_baseline_controllo_cila = [
        series_controllo_cila[0],  # numero_pratiche_concluse_senza_sospensioni
        series_controllo_cila[1],  # numero_pratiche_concluse_con_sospensioni
        series_controllo_cila[2],  # numero_pratiche_concluse_con_conferenza_servizi
        series_controllo_cila[4],  # giornate_durata_media_pratiche_concluse
        series_controllo_cila[5],  # numero_pratiche_non_concluse_non_scaduti_termini
        series_controllo_cila[6],  # numero_pratiche_non_concluse_scaduti_termini
        numero_controlli_cila_conclusi_con_provvedimento_espresso,
        numero_controlli_cila]
    pat_dataframe_raccolta_puntuale_comune_baseline_controllo_cila = pd.DataFrame(
        [raccolta_puntuale_comune_baseline_controllo_cila],
        columns=raccolta_puntuale_comune_baseline_controllo_cila_labels,
        index=[comune])

    find_index_column, find_index_row = find_string_dataframe(dataframe_sanatorie)
    series_sanatorie = dataframe_sanatorie.iloc[
                           find_index_row][find_index_column:find_index_column+7]
    series_sanatorie = pd.to_numeric(
        series_sanatorie, errors='coerce').fillna(0).astype('i4')
    series_sanatorie.reset_index(drop=True, inplace=True)

    numero_sanatorie_concluse_con_provvedimento_espresso =\
        series_sanatorie[0] +\
        series_sanatorie[1] +\
        series_sanatorie[2]
    numero_sanatorie = \
        numero_sanatorie_concluse_con_provvedimento_espresso +\
        series_sanatorie[5] +\
        series_sanatorie[6]

    raccolta_puntuale_comune_baseline_sanatoria_labels = np.array([
        'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4',
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4',
        'numero_sanatorie_2021q3-4'])
    raccolta_puntuale_comune_baseline_sanatoria = [
        series_sanatorie[0],  # numero_pratiche_concluse_senza_sospensioni
        series_sanatorie[1],  # numero_pratiche_concluse_con_sospensioni
        series_sanatorie[2],  # numero_pratiche_concluse_con_conferenza_servizi
        series_sanatorie[4],  # giornate_durata_media_pratiche_concluse
        series_sanatorie[5],  # numero_pratiche_non_concluse_non_scaduti_termini
        series_sanatorie[6],  # numero_pratiche_non_concluse_scaduti_termini
        numero_sanatorie_concluse_con_provvedimento_espresso,
        numero_sanatorie]
    pat_dataframe_raccolta_puntuale_comune_baseline_sanatoria = pd.DataFrame(
        [raccolta_puntuale_comune_baseline_sanatoria],
        columns=raccolta_puntuale_comune_baseline_sanatoria_labels,
        index=[comune])

    pat_dataframe_raccolta_puntuale_comune = pd.concat(
        [pat_dataframe_raccolta_puntuale_comune_baseline_permesso_costruire,
         pat_dataframe_raccolta_puntuale_comune_baseline_controllo_cila,
         pat_dataframe_raccolta_puntuale_comune_baseline_sanatoria],
        axis='columns', join='outer')

    return pat_dataframe_raccolta_puntuale_comune


def print_series_raccolta_puntuale(series_permessi_costruire,
                                   series_controllo_cila,
                                   series_sanatorie,
                                   comune=''):

    if comune != '':
        print('')
        print('Series della raccolta puntuale presso il Comune di ' + comune)
    print('')
    print(series_permessi_costruire)
    print('')
    print(series_controllo_cila)
    print('')
    print(series_sanatorie)
    print('')

    return True


def get_series_raccolte_puntuali():

    path_raccolte_puntuali = {
        'Rovereto':
            'raccolta_puntuale_02_rovereto/File per Incontro Comuni_Rovereto.xlsx',
        'Pergine Valsugana':
            'raccolta_puntuale_03_pergine_valsugana/PERGINE - File per Incontro Comuni.xlsx',
        'Moena':
            'raccolta_puntuale_05_moena/File per Incontro Comuni_Moena.xlsx',
        'Baselga di Pine':
            'raccolta_puntuale_06_baselga_pine/File per Incontro Comuni_Baselga di Pinè.xlsx',
        'Pieve di Bono Prezzo':
            'raccolta_puntuale_07_pieve_bono_prezzo/File per Incontro Comuni Pieve.xlsx'
    }

    series_raccolte_puntuali = {}
    for comune in path_raccolte_puntuali.keys():
        series_permessi_costruire,\
        series_controllo_cila,\
        series_sanatorie = get_series_xlsx(path_raccolte_puntuali[comune])
        series_raccolte_puntuali[comune] = {
            'permessi di costruire': series_permessi_costruire,
            'controllo delle CILA': series_controllo_cila,
            'sanatorie': series_sanatorie
        }

    return series_raccolte_puntuali


def get_pat_dataframe_raccolta_puntuale(path_base=''):

    path_raccolte_puntuali = {
        'Rovereto':
            path_base + 'raccolta_puntuale_02_rovereto\\'
                        'File per Incontro Comuni_Rovereto.xlsx',
        'Pergine Valsugana':
            path_base + 'raccolta_puntuale_03_pergine_valsugana\\'
                        'PERGINE - File per Incontro Comuni.xlsx',
        'Moena':
            path_base + 'raccolta_puntuale_05_moena\\'
                        'File per Incontro Comuni_Moena.xlsx',
        'Baselga di Pinè':
            path_base + 'raccolta_puntuale_06_baselga_pine\\'
                        'File per Incontro Comuni_Baselga di Pinè.xlsx',
        'Pieve di Bono-Prezzo':
            path_base + 'raccolta_puntuale_07_pieve_bono_prezzo\\'
                        'File per Incontro Comuni Pieve.xlsx'
    }

    dataframe_raccolte_puntuali = {}
    for comune in path_raccolte_puntuali.keys():
        dataframe_raccolte_puntuali[comune] = get_dataframe_xlsx(
            comune, path_raccolte_puntuali[comune])

    pat_dataframe_raccolta_puntuale = pd.concat(
        [dataframe_raccolte_puntuali[comune] for comune in dataframe_raccolte_puntuali.keys()],
        axis='rows', join='outer')

    return pat_dataframe_raccolta_puntuale


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    series_raccolte_puntuali = get_series_raccolte_puntuali()
    for comune in series_raccolte_puntuali.keys():
        print_series_raccolta_puntuale(
            series_raccolte_puntuali[comune]['permessi di costruire'],
            series_raccolte_puntuali[comune]['controllo delle CILA'],
            series_raccolte_puntuali[comune]['sanatorie'],
            comune
        )
