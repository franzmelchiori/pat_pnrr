"""
    PAT-PNRR MPE Report Comunale
    Francesco Melchiori, 2024
"""


import os
import subprocess

from .pat_pnrr_comuni_excel_mapping import *


def print_report_comunale(name_comune, version, version_to_remove=''):
    name_comune = name_comune
    name_comune_file = name_comune
    name_comune_file = name_comune_file.replace('à', 'a')
    name_comune_file = name_comune_file.replace('è', 'e')
    name_comune_file = name_comune_file.replace('é', 'e')
    name_comune_file = name_comune_file.replace('ù', 'u')
    version = version
    version_file = version
    version_file = version_file.replace('.', '-')
    version_to_remove = version_to_remove
    version_file_to_remove = version_to_remove
    version_file_to_remove = version_file_to_remove.replace('.', '-')

    with open('pat_pnrr_mpe\\report_comunale\\pat_pnrr_mpe_report_comunale.txt', \
              mode='r', encoding='utf-8') as f1:
        tex_pat_pnrr_mpe_report_comunale = f1.read()
    f1.close()

    tex_pat_pnrr_mpe_report_comunale = \
        tex_pat_pnrr_mpe_report_comunale.replace(
            '<NAME_COMUNE>', name_comune)
    tex_pat_pnrr_mpe_report_comunale = \
        tex_pat_pnrr_mpe_report_comunale.replace(
            '<NAME_COMUNE_FILE>', name_comune_file)
    tex_pat_pnrr_mpe_report_comunale = \
        tex_pat_pnrr_mpe_report_comunale.replace(
            '<VERSION>', version)
    if name_comune != 'Trento':
        tex_pat_pnrr_mpe_report_comunale = \
            tex_pat_pnrr_mpe_report_comunale.replace(
                '<NO_TRENTO>', '')  #' ; dal grafico sono esclusi i dati di Trento')
    else:
        tex_pat_pnrr_mpe_report_comunale = \
            tex_pat_pnrr_mpe_report_comunale.replace(
                '<NO_TRENTO>', '')

    try:                
        os.remove('pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file_to_remove + '.tex')
    except:
        pass
    try:
        os.remove('pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file + '.tex')
    except:
        pass
    f2 = open('pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_' +
        name_comune_file + '_' + version_file + '.tex', mode='w', encoding='utf-8')
    f2.write(tex_pat_pnrr_mpe_report_comunale)
    f2.close()

    subprocess.call(['pdflatex', '-output-directory',
                     'pat_pnrr_mpe\\report_comunale\\pdfs\\',
                     'pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_' + \
                      name_comune_file + '_' + version_file + '.tex'])
    subprocess.call(['pdflatex', '-output-directory',
                     'pat_pnrr_mpe\\report_comunale\\pdfs\\',
                     'pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_' + \
                      name_comune_file + '_' + version_file + '.tex'])
    
    try:
        os.remove('pat_pnrr_mpe\\report_comunale\\pdfs\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file + '.aux')
    except:
        pass
    try:                
        os.remove('pat_pnrr_mpe\\report_comunale\\pdfs\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file + '.idx')
    except:
        pass
    try:                
        os.remove('pat_pnrr_mpe\\report_comunale\\pdfs\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file + '.log')
    except:
        pass
    try:                
        os.remove('pat_pnrr_mpe\\report_comunale\\pdfs\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file + '.toc')
    except:
        pass
    try:                
        os.remove('pat_pnrr_mpe\\report_comunale\\pdfs\\pat_pnrr_mpe_' + \
                name_comune_file + '_' + version_file_to_remove + '.pdf')
    except:
        pass
    return


def print_report_comunali(version, version_to_remove, just_one = False):
    name_comuni = [comune[0] for comune in comuni_excel_map]
    for name_comune in name_comuni:
        print_report_comunale(name_comune, version = version, version_to_remove = version_to_remove)
        if just_one:
            break
    return


if __name__ == '__main__':
    # print_report_comunale(name_comune = 'Trento', version = 'v5.7.0', version_file_to_remove = 'v5.5.0')
    # print_report_comunale(name_comune = 'Ala', version = 'v5.7.0', version_to_remove = 'v5.5.0')
    print_report_comunali(version = 'v5.7.0', version_to_remove = 'v5.5.0')
