from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.orm import Session


engine = create_engine(
    'postgresql+psycopg2://postgres:admin@localhost:5432/pat-pnrr',
    echo=True, future=True)


# pat_comuni_dataframe
#     2021q3-4
#         numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4
#         numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4
#         numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4
#         giornate_durata_media_permessi_costruire_conclusi_2021q3-4
#         60
#         numero_permessi_costruire_2021q3-4
#         numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4
#     2022q1-2
#         numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q1-2
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1-2
#         numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2
#         numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2
#         giornate_durata_media_permessi_costruire_conclusi_2022q1-2
#         60
#         numero_permessi_costruire_2022q1-2
#         numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2
#     2022q3-4
#         numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q3-4
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni_2022q3-4
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2022q3-4
#         giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4
#         giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q3-4
#         numero_permessi_costruire_avviati_2022q3-4
#         numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4
#     2023q1-2
#         numero_permessi_costruire_conclusi_con_silenzio-assenso_2023q1-2
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni_2023q1-2
#         numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2023q1-2
#         giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2
#         giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q1-2
#         numero_permessi_costruire_avviati_2023q1-2
#         numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2


metadata_obj = MetaData()
comuni = Table(
    "comuni",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("nome_comune", String))  # es. Ala
monitoraggi = Table(
    "monitoraggi",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("periodo_monitoraggio", String))  # es. 2021q3-4
pdc = Table(
    "permessi_di_costruire",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("id_comune", Integer),
    Column("id_monitoraggio", Integer),
    Column("numero_permessi_costruire_conclusi_con_silenzio-assenso", Integer),
    Column("numero_permessi_costruire_conclusi_con_provvedimento_espresso", Integer),
    Column("numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni", Integer),
    Column("numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi", Integer),
    Column("giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso", Integer),
    Column("giornate_durata_mediana_termine_massimo_permessi_costruire_avviati", Integer),
    Column("numero_permessi_costruire_avviati", Integer),
    Column("numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo", Integer))
