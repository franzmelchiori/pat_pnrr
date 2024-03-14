"""
    PAT-PNRR strumenti misurazione
    Monitoraggio Procedimenti Edilizi v2.x
    Francesco Melchiori, 2024
"""


import os

import pandas as pd

from .pat_pnrr_comuni_excel_mapping import *


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


def get_list_excel(path_to_excel_files, path_to_mpe=None, missing=False):
    if not path_to_mpe:
        path_to_mpe = 'C:\\projects\\franzmelchiori\\projects\\pat_pnrr\\pat_pnrr_mpe\\'

    list_xls = []
    for file in os.listdir(path_to_mpe + path_to_excel_files):
        if file.find('xls') != -1:
            list_xls.append(file)

    if missing:
        list_excel = [comune[0] for comune in comuni_excel_map
                      if comune[2] is None]
    else:
        list_excel = [(comune[2], comune[0]) for comune in comuni_excel_map
                      if comune[2] is not None]

    return list_excel, list_xls
