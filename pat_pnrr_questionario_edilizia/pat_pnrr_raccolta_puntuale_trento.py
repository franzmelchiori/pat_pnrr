"""
    PAT-PNRR Baseline June 2022
    Processing System with Building Data from Comune di Trento
    Francesco Melchiori, 2022
"""


import numpy as np
import pandas as pd


def get_dataframe_xlsx(path_file_xlsx, pratiche_concluse=True, controlli_cila=False):

    if pratiche_concluse:
        names = ['data_ricezione', 'data_conclusione', 'stato_pratica',
                 'giornate_sospensione', 'giornate_termine']
        usecols = [5, 6, 7, 10, 11]
        if controlli_cila:
            names.pop()
            usecols.pop()
    else:
        names = ['data_ricezione',
                 'giornate_sospensione', 'giornate_termine']
        usecols = [4, 9, 10]
        if controlli_cila:
            names.pop()
            usecols.pop()

    dataframe_xlsx = pd.read_excel(path_file_xlsx, names=names, usecols=usecols)

    dataframe_xlsx.drop(
        dataframe_xlsx.index[dataframe_xlsx['data_ricezione'].isna()],
        inplace=True)
    dataframe_xlsx.drop(
        dataframe_xlsx.index[dataframe_xlsx['data_ricezione'] == 'Data domanda'],
        inplace=True)
    dataframe_xlsx.drop(
        dataframe_xlsx.index[dataframe_xlsx['data_ricezione'] == '29/12/0000'],
        inplace=True)
    if pratiche_concluse:
        dataframe_xlsx.drop(
            dataframe_xlsx.index[dataframe_xlsx['stato_pratica'] != 'RILASCIATA'],
            inplace=True)
        dataframe_xlsx.drop(columns='stato_pratica', inplace=True)
    dataframe_xlsx.drop_duplicates(inplace=True)

    return dataframe_xlsx


def get_baseline(pratiche_concluse_con_espressione,
                 pratiche_non_concluse,
                 pratiche_concluse_con_silenzio_assenso=pd.DataFrame(),
                 format_questionario=False):
    numero_pratiche_concluse_con_espressione = len(pratiche_concluse_con_espressione)
    numero_pratiche_concluse_con_sospensioni = len(pratiche_concluse_con_espressione[
        pratiche_concluse_con_espressione['giornate_sospensione'] > 0])
    numero_pratiche_concluse_senza_sospensioni = len(pratiche_concluse_con_espressione[
        pratiche_concluse_con_espressione['giornate_sospensione'] == 0])
    numero_pratiche_concluse_con_conferenza = 0
    giornate_durata_media_pratiche_concluse = (
        pd.to_datetime(pratiche_concluse_con_espressione['data_conclusione']) -
        pd.to_datetime(pratiche_concluse_con_espressione['data_ricezione'])).mean().days
    if 'giornate_termine' in pratiche_concluse_con_espressione.columns:
        giornate_durata_media_termine = (
            pratiche_concluse_con_espressione['giornate_termine']).mean().round(2)
    else:
        giornate_durata_media_termine = 0

    pratiche_non_concluse = pratiche_non_concluse[
        pd.to_datetime(pratiche_non_concluse['data_ricezione']) <
            pd.Timestamp('2021-12-31 23:59:59.999')]
        # pratiche_non_concluse['data_ricezione'] < datetime(2021, 12, 31, 23, 59)]
    if 'giornate_termine' in pratiche_non_concluse.columns:
        pratiche_non_concluse_scaduti_termini = pratiche_non_concluse[
            ((pd.Timestamp('2021-12-31 23:59:59.999') -
              pd.to_datetime(pratiche_non_concluse['data_ricezione'])).dt.days -
            # ((datetime(2021, 12, 31, 23, 59) - pratiche_non_concluse['data_ricezione']).dt.days -
             pratiche_non_concluse['giornate_sospensione']) >
            pratiche_non_concluse['giornate_termine']]
        numero_pratiche_non_concluse_scaduti_termini = len(
            pratiche_non_concluse_scaduti_termini)
        pratiche_non_concluse_non_scaduti_termini = pratiche_non_concluse[
            ((pd.Timestamp('2021-12-31 23:59:59.999') -
              pd.to_datetime(pratiche_non_concluse['data_ricezione'])).dt.days -
            # ((datetime(2021, 12, 31, 23, 59) - pratiche_non_concluse['data_ricezione']).dt.days -
             pratiche_non_concluse['giornate_sospensione']) <=
            pratiche_non_concluse['giornate_termine']]
        numero_pratiche_non_concluse_non_scaduti_termini = len(
            pratiche_non_concluse_non_scaduti_termini)
    else:
        numero_pratiche_non_concluse_scaduti_termini = 0
        numero_pratiche_non_concluse_non_scaduti_termini = len(
            pratiche_non_concluse)

    if not pratiche_concluse_con_silenzio_assenso.empty:
        numero_pratiche_concluse_con_silenzio_assenso = len(
            pratiche_concluse_con_silenzio_assenso)
    else:
        numero_pratiche_concluse_con_silenzio_assenso = 0

    numero_pratiche =\
        numero_pratiche_concluse_con_silenzio_assenso +\
        numero_pratiche_concluse_con_espressione +\
        numero_pratiche_non_concluse_scaduti_termini +\
        numero_pratiche_non_concluse_non_scaduti_termini

    baseline = [
        numero_pratiche_concluse_con_silenzio_assenso,
        numero_pratiche_concluse_con_espressione,
        numero_pratiche_concluse_con_sospensioni,
        numero_pratiche_concluse_con_conferenza,
        giornate_durata_media_pratiche_concluse,
        giornate_durata_media_termine,
        numero_pratiche,
        numero_pratiche_non_concluse_scaduti_termini]

    questionario = [
        numero_pratiche_concluse_senza_sospensioni,
        numero_pratiche_concluse_con_sospensioni,
        numero_pratiche_concluse_con_conferenza,
        numero_pratiche_concluse_con_silenzio_assenso,
        giornate_durata_media_pratiche_concluse,
        numero_pratiche_non_concluse_non_scaduti_termini,
        numero_pratiche_non_concluse_scaduti_termini]

    return baseline, questionario


def print_baseline(nome_procedura, baseline):

    print('Denominazione procedura: Procedimenti Edilizi: ' + nome_procedura)
    print('Concluse con silenzio assenso (numero): ' +
          str(baseline[0]))
    print('Concluse con provvedimento espresso | Totali (numero): ' +
          str(baseline[1]))
    print('Concluse con provvedimento espresso | di cui con sospensioni (numero): ' +
          str(baseline[2]))
    print('Concluse con provvedimento espresso | di cui con Conferenza dei Servizi (numero): ' +
          str(baseline[3]))
    print('Concluse con provvedimento espresso | Durata media (giornate): ' +
          str(baseline[4]))
    print('Termine massimo (giornate): ' +
          str(baseline[5]))
    print('Avviate (numero): ' +
          str(baseline[6]))
    print('Arretrato (numero): ' +
          str(baseline[7]))
    print('')


def print_questionario(nome_procedura, questionario):

    print('Denominazione procedura: Procedimenti Edilizi: ' + nome_procedura)
    print('Pratiche concluse senza sospensioni 2021q3-4 (numero): ' +
          str(questionario[0]))
    print('Pratiche concluse con sospensioni 2021q3-4 (numero): ' +
          str(questionario[1]))
    print('Pratiche concluse con Conferenza dei Servizi 2021q3-4 (numero): ' +
          str(questionario[2]))
    print('Pratiche concluse con silenzio-assenso 2021q3-4 (numero): ' +
          str(questionario[3]))
    print('Durata media pratiche concluse 2021q3-4 (giornate): ' +
          str(questionario[4]))
    print('Pratiche non concluse non scaduti termini 2021q3-4 (numero): ' +
          str(questionario[5]))
    print('Pratiche non concluse scaduti termini 2021q3-4 (numero): ' +
          str(questionario[6]))
    print('')


def get_pat_comuni_dataframe_raccolta_puntuale_trento(path_base=''):

    permessi_costruire_conclusi_con_silenzio_assenso = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Conlusioni con silenzio assenso.xlsx'),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Presentazioni e conclusioni con silenzio assenso.xlsx')],
        ignore_index=True)

    permessi_costruire_conclusi_con_espressione = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Conclusioni senza silenzio assenso.xlsx'),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Presentazioni e conclusioni senza silenzio assenso.xlsx')],
        ignore_index=True)

    permessi_costruire_non_conclusi = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Conclusioni_Assenso_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'Conclusioni_Senza Assenso_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Permessi di costruire\\'
                        'PdC_pendenti_2022.xlsx',
            pratiche_concluse=False)],
        ignore_index=True)

    baseline, questionario = get_baseline(permessi_costruire_conclusi_con_espressione,
                                          permessi_costruire_non_conclusi,
                                          permessi_costruire_conclusi_con_silenzio_assenso)

    raccolta_puntuale_trento_baseline_permesso_costruire_labels = np.array([
        'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4',
        'numero_permessi_costruire_conclusi_senza_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4',
        'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_permessi_costruire_conclusi_2021q3-4',
        'numero_permessi_costruire_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4',
        'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_permessi_costruire_2021q3-4'])
    raccolta_puntuale_trento_baseline_permesso_costruire = [
        questionario[3],
        questionario[0],
        questionario[1],
        questionario[2],
        questionario[4],
        questionario[5],
        questionario[6],
        baseline[1],
        baseline[6]]
    pat_dataframe_raccolta_puntuale_trento_baseline_permesso_costruire = pd.DataFrame(
        [raccolta_puntuale_trento_baseline_permesso_costruire],
        columns=raccolta_puntuale_trento_baseline_permesso_costruire_labels,
        index=['Trento'])

    controlli_cila_conclusi = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Controllo delle CILA\\'
                        'Conclusioni controlli_CILA.xlsx',
            controlli_cila=True),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Controllo delle CILA\\'
                        'Depositi e conclusioni controlli_CILA.xlsx',
            controlli_cila=True)],
        ignore_index=True)

    controlli_cila_non_conclusi = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Controllo delle CILA\\'
                        'Conclusioni_CILA_2022.xlsx',
            pratiche_concluse=False, controlli_cila=True)],
        ignore_index=True)

    baseline, questionario = get_baseline(controlli_cila_conclusi,
                                          controlli_cila_non_conclusi)

    raccolta_puntuale_trento_baseline_controllo_cila_labels = np.array([
        'numero_controlli_cila_conclusi_senza_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4',
        'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_controlli_cila_conclusi_2021q3-4',
        'numero_controlli_cila_non_conclusi_non_scaduti_termini_2021q3-4',
        'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4',
        'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4',
        'numero_controlli_cila_2021q3-4'])
    raccolta_puntuale_trento_baseline_controllo_cila = [
        questionario[0],
        questionario[1],
        questionario[2],
        questionario[4],
        questionario[5],
        questionario[6],
        baseline[1],
        baseline[6]]
    pat_dataframe_raccolta_puntuale_trento_baseline_controllo_cila = pd.DataFrame(
        [raccolta_puntuale_trento_baseline_controllo_cila],
        columns=raccolta_puntuale_trento_baseline_controllo_cila_labels,
        index=['Trento'])

    sanatorie_concluse = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Sanatorie\\'
                        'Conclusioni_Sanatorie.xlsx'),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Sanatorie\\'
                        'Depositi e conclusioni_Sanatorie.xlsx')],
        ignore_index=True)

    sanatorie_non_concluse = pd.concat([
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Sanatorie\\'
                        'Conclusioni_Sanatorie_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            path_base + 'raccolta_puntuale_01_trento\\'
                        'Sanatorie\\'
                        'Sanatorie_Pendenti_2022.xlsx',
            pratiche_concluse=False)],
        ignore_index=True)

    baseline, questionario = get_baseline(sanatorie_concluse,
                                          sanatorie_non_concluse)

    raccolta_puntuale_trento_baseline_sanatoria_labels = np.array([
        'numero_sanatorie_concluse_senza_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_sospensioni_2021q3-4',
        'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4',
        'giornate_durata_media_sanatorie_concluse_2021q3-4',
        'numero_sanatorie_non_concluse_non_scaduti_termini_2021q3-4',
        'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4',
        'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4',
        'numero_sanatorie_2021q3-4'])
    raccolta_puntuale_trento_baseline_sanatoria = [
        questionario[0],
        questionario[1],
        questionario[2],
        questionario[4],
        questionario[5],
        questionario[6],
        baseline[1],
        baseline[6]]
    pat_dataframe_raccolta_puntuale_trento_baseline_sanatoria = pd.DataFrame(
        [raccolta_puntuale_trento_baseline_sanatoria],
        columns=raccolta_puntuale_trento_baseline_sanatoria_labels,
        index=['Trento'])

    pat_dataframe_raccolta_puntuale_trento = pd.concat(
        [pat_dataframe_raccolta_puntuale_trento_baseline_permesso_costruire,
         pat_dataframe_raccolta_puntuale_trento_baseline_controllo_cila,
         pat_dataframe_raccolta_puntuale_trento_baseline_sanatoria],
        axis='columns', join='outer')

    return pat_dataframe_raccolta_puntuale_trento


if __name__ == '__main__':

    permessi_costruire_conclusi_con_silenzio_assenso = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Conlusioni con silenzio assenso.xlsx'),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Presentazioni e conclusioni con silenzio assenso.xlsx')],
        ignore_index=True)

    permessi_costruire_conclusi_con_espressione = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Conclusioni senza silenzio assenso.xlsx'),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Presentazioni e conclusioni senza silenzio assenso.xlsx')],
        ignore_index=True)

    permessi_costruire_non_conclusi = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Conclusioni_Assenso_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/Conclusioni_Senza Assenso_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Permessi di costruire/PdC_pendenti_2022.xlsx',
            pratiche_concluse=False)],
        ignore_index=True)

    baseline, questionario = get_baseline(permessi_costruire_conclusi_con_espressione,
                                          permessi_costruire_non_conclusi,
                                          permessi_costruire_conclusi_con_silenzio_assenso)

    # print_baseline('Permessi di Costruire', baseline)
    print_questionario('Permessi di Costruire', questionario)

    controlli_cila_conclusi = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Controllo delle CILA/Conclusioni controlli_CILA.xlsx',
            controlli_cila=True),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Controllo delle CILA/Depositi e conclusioni controlli_CILA.xlsx',
            controlli_cila=True)],
        ignore_index=True)

    controlli_cila_non_conclusi = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Controllo delle CILA/Conclusioni_CILA_2022.xlsx',
            pratiche_concluse=False, controlli_cila=True)],
        ignore_index=True)

    baseline, questionario = get_baseline(controlli_cila_conclusi,
                                          controlli_cila_non_conclusi)

    # print_baseline('Controllo delle CILA', baseline)
    print_questionario('Controllo delle CILA', questionario)

    sanatorie_concluse = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Sanatorie/Conclusioni_Sanatorie.xlsx'),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Sanatorie/Depositi e conclusioni_Sanatorie.xlsx')],
        ignore_index=True)

    sanatorie_non_concluse = pd.concat([
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Sanatorie/Conclusioni_Sanatorie_2022.xlsx',
            pratiche_concluse=False),
        get_dataframe_xlsx(
            'raccolta_puntuale_01_trento/Sanatorie/Sanatorie_Pendenti_2022.xlsx',
            pratiche_concluse=False)],
        ignore_index=True)

    baseline, questionario = get_baseline(sanatorie_concluse,
                                          sanatorie_non_concluse)

    # print_baseline('Sanatorie', baseline)
    print_questionario('Sanatorie', questionario)
