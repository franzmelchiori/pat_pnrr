"""
    PAT-PNRR MPE Report Comunale
    Francesco Melchiori, 2024
"""


import subprocess

from .pat_pnrr_comuni_excel_mapping import *


def print_report_comunali():
    with open('pat_pnrr_mpe\\report_comunale\\pat_pnrr_mpe_report_comunale.tex') as f1:
        tex_pat_pnrr_mpe_report_comunale = f1.read()
    f1.close()

    name_comune = 'Ledro'
    name_comune_file = name_comune
    name_comune_file = name_comune_file.replace('à', 'a')
    name_comune_file = name_comune_file.replace('è', 'e')
    name_comune_file = name_comune_file.replace('é', 'e')
    name_comune_file = name_comune_file.replace('ù', 'u')

    tex_pat_pnrr_mpe_report_comunale = \
        tex_pat_pnrr_mpe_report_comunale.replace(
            '<NAME_COMUNE>', name_comune)
    tex_pat_pnrr_mpe_report_comunale = \
        tex_pat_pnrr_mpe_report_comunale.replace(
            '<NAME_COMUNE_FILE>', name_comune_file)
    
    f2 = open('pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_report_comunale_' +
        name_comune_file + '.tex', 'a')
    f2.write(tex_pat_pnrr_mpe_report_comunale)
    f2.close()

    subprocess.call(['pdflatex', '-output-directory',
                     'pat_pnrr_mpe\\report_comunale\\pdfs\\',
                     'pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_report_comunale_' + \
                        name_comune_file + '.tex'])
    subprocess.call(['pdflatex', '-output-directory',
                     'pat_pnrr_mpe\\report_comunale\\pdfs\\',
                     'pat_pnrr_mpe\\report_comunale\\texes\\pat_pnrr_mpe_report_comunale_' + \
                        name_comune_file + '.tex'])
    return


if __name__ == '__main__':
    print_report_comunali()
