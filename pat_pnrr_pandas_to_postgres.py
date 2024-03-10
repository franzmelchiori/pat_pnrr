"""
    PAT-PNRR
    Monitoraggio Procedimenti Edilizi
    From Pandas to PostgreSQL
    Francesco Melchiori, 2024
"""
"""
    [cli] to dump the postgres db:
        pg_dump -U postgres -d pat-pnrr > pat-pnrr.sql
    [cli] to restore the postgres db:
        createdb -U postgres -T template0 pat-pnrr
        psql -U postgres -d pat-pnrr < pat-pnrr.sql
"""


import numpy as np
from sqlalchemy import create_engine
# from sqlalchemy import text
# from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import Table, Column
from sqlalchemy import Integer, String
from sqlalchemy import ForeignKey
# from sqlalchemy import insert
from sqlalchemy import select
# from sqlalchemy import bindparam
# from sqlalchemy import func, cast
# from sqlalchemy import and_, or_
# from sqlalchemy import desc
# from sqlalchemy import update, delete
from sqlalchemy.orm import Session
from sqlalchemy.orm import registry
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

from pat_pnrr_monitoring_analyzer import get_pat_comuni_dataframe


engine = create_engine(
    'postgresql+psycopg2://postgres:admin@localhost:5432/pat-pnrr',
    echo=True, future=True)


mapper_registry = registry()
Base = mapper_registry.generate_base()
Base = declarative_base()

class Comune(Base):
    __tablename__ = 'comuni'
    id = Column(Integer, primary_key=True)
    nome_comune = Column(String)  # es. Ala
    pdc = relationship('PDC', back_populates='comune')
    pds = relationship('PDS', back_populates='comune')
    cdc = relationship('CDC', back_populates='comune')
    def __repr__(self):
        return f'Comune(id={self.id!r},\
                        nome_comune={self.nome_comune!r})'

class Monitoraggio(Base):
    __tablename__ = 'monitoraggi'
    id = Column(Integer, primary_key=True)
    periodo_monitoraggio = Column(String)  # es. 2021q3-4
    pdc = relationship('PDC', back_populates='monitoraggio')
    pds = relationship('PDS', back_populates='monitoraggio')
    cdc = relationship('CDC', back_populates='monitoraggio')
    def __repr__(self):
        return f'Monitoraggio(id={self.id!r},\
                              periodo_monitoraggio={self.periodo_monitoraggio!r})'

class PDC(Base):
    __tablename__ = 'pdc'
    id = Column(Integer, primary_key=True)
    n_pdc_conc_sa = Column(Integer)  # numero_permessi_costruire_conclusi_con_silenzio-assenso
    n_pdc_conc_pe = Column(Integer)  # numero_permessi_costruire_conclusi_con_provvedimento_espresso
    n_pdc_conc_pe_sosp = Column(Integer)  # numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni
    n_pdc_conc_pe_cds = Column(Integer)  # numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi
    gg_pdc_conc_pe = Column(Integer)  # giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso
    gg_pdc_max = Column(Integer)  # giornate_durata_mediana_termine_massimo_permessi_costruire_avviati
    n_pdc = Column(Integer)  # numero_permessi_costruire_avviati
    n_pdc_arr = Column(Integer)  # numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo
    id_comune = Column(Integer, ForeignKey('comuni.id'))
    id_monitoraggio = Column(Integer, ForeignKey('monitoraggi.id'))
    comune = relationship('Comune', back_populates='pdc')
    monitoraggio = relationship('Monitoraggio', back_populates='pdc')
    def __repr__(self):
        return f'PDC(id={self.id!r},\
                     n_pdc_conc_sa={self.n_pdc_conc_sa!r},\
                     n_pdc_conc_pe={self.n_pdc_conc_pe!r},\
                     n_pdc_conc_pe_sosp={self.n_pdc_conc_pe_sosp!r},\
                     n_pdc_conc_pe_cds={self.n_pdc_conc_pe_cds!r},\
                     gg_pdc_conc_pe={self.gg_pdc_conc_pe!r},\
                     gg_pdc_max={self.gg_pdc_max!r},\
                     n_pdc={self.n_pdc!r},\
                     n_pdc_arr={self.n_pdc_arr!r})'

class PDS(Base):
    __tablename__ = 'pds'
    id = Column(Integer, primary_key=True)
    n_pds_conc_sa = Column(Integer)  # numero_sanatorie_concluse_con_silenzio-assenso
    n_pds_conc_pe = Column(Integer)  # numero_sanatorie_concluse_con_provvedimento_espresso
    n_pds_conc_pe_sosp = Column(Integer)  # numero_sanatorie_concluse_con_provvedimento_espresso_con_sospensioni
    n_pds_conc_pe_cds = Column(Integer)  # numero_sanatorie_concluse_con_provvedimento_espresso_con_conferenza_servizi
    gg_pds_conc_pe = Column(Integer)  # giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso
    gg_pds_max = Column(Integer)  # giornate_durata_mediana_termine_massimo_sanatorie_avviate
    n_pds = Column(Integer)  # numero_sanatorie_avviate
    n_pds_arr = Column(Integer)  # numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo
    id_comune = Column(Integer, ForeignKey('comuni.id'))
    id_monitoraggio = Column(Integer, ForeignKey('monitoraggi.id'))
    comune = relationship('Comune', back_populates='pds')
    monitoraggio = relationship('Monitoraggio', back_populates='pds')
    def __repr__(self):
        return f'PDS(id={self.id!r},\
                     n_pds_conc_sa={self.n_pds_conc_sa!r},\
                     n_pds_conc_pe={self.n_pds_conc_pe!r},\
                     n_pds_conc_pe_sosp={self.n_pds_conc_pe_sosp!r},\
                     n_pds_conc_pe_cds={self.n_pds_conc_pe_cds!r},\
                     gg_pds_conc_pe={self.gg_pds_conc_pe!r},\
                     gg_pds_max={self.gg_pds_max!r},\
                     n_pds={self.n_pds!r},\
                     n_pds_arr={self.n_pds_arr!r})'

class CDC(Base):
    __tablename__ = 'cdc'
    id = Column(Integer, primary_key=True)
    n_cdc_conc_sa = Column(Integer)  # numero_controlli_cila_conclusi_con_silenzio-assenso
    n_cdc_conc_pe = Column(Integer)  # numero_controlli_cila_conclusi_con_provvedimento_espresso
    n_cdc_conc_pe_sosp = Column(Integer)  # numero_controlli_cila_conclusi_con_provvedimento_espresso_con_sospensioni
    n_cdc_conc_pe_cds = Column(Integer)  # numero_controlli_cila_conclusi_con_provvedimento_espresso_con_conferenza_servizi
    gg_cdc_conc_pe = Column(Integer)  # giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso
    gg_cdc_max = Column(Integer)  # giornate_durata_mediana_termine_massimo_controlli_cila_avviati
    n_cdc = Column(Integer)  # numero_controlli_cila_avviati
    n_cdc_arr = Column(Integer)  # numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo
    id_comune = Column(Integer, ForeignKey('comuni.id'))
    id_monitoraggio = Column(Integer, ForeignKey('monitoraggi.id'))
    comune = relationship('Comune', back_populates='cdc')
    monitoraggio = relationship('Monitoraggio', back_populates='cdc')
    def __repr__(self):
        return f'CDC(id={self.id!r},\
                     n_cdc_conc_sa={self.n_cdc_conc_sa!r},\
                     n_cdc_conc_pe={self.n_cdc_conc_pe!r},\
                     n_cdc_conc_pe_sosp={self.n_cdc_conc_pe_sosp!r},\
                     n_cdc_conc_pe_cds={self.n_cdc_conc_pe_cds!r},\
                     gg_cdc_conc_pe={self.gg_cdc_conc_pe!r},\
                     gg_cdc_max={self.gg_cdc_max!r},\
                     n_cdc={self.n_cdc!r},\
                     n_cdc_arr={self.n_cdc_arr!r})'

drop_tables = True
if drop_tables:
    PDC.__table__.drop(engine)
    PDS.__table__.drop(engine)
    CDC.__table__.drop(engine)

mapper_registry.metadata.create_all(engine)
Base.metadata.create_all(engine)


pat_comuni_dataframe = get_pat_comuni_dataframe()
# with Session(engine) as session:
#     for nome_comune in pat_comuni_dataframe.index:
#         comune = Comune(nome_comune=nome_comune)
#         session.add(comune)
#         session.commit()
#     session.close()

periodo_monitoraggi = ['2021q3-4', '2022q1-2', '2022q3-4', '2023q1-2']
# with Session(engine) as session:
#     for periodo_monitoraggio in periodo_monitoraggi:
#         monitoraggio = Monitoraggio(periodo_monitoraggio=periodo_monitoraggio)
#         session.add(monitoraggio)
#         session.commit()
#     session.close()


with Session(engine) as session:
    for nome_comune in pat_comuni_dataframe.index:
        comune = session.execute(select(Comune).where(
            Comune.nome_comune == nome_comune)).scalar_one()
        comune_dataframe = pat_comuni_dataframe.loc[nome_comune].replace(np.nan, None)
        periodo_monitoraggio = '2021q3-4'
        monitoraggio = session.execute(select(Monitoraggio).where(
            Monitoraggio.periodo_monitoraggio == periodo_monitoraggio)).scalar_one()
        pdc = PDC(
            n_pdc_conc_sa = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_silenzio-assenso_2021q3-4'],
            n_pdc_conc_pe = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2021q3-4'],
            n_pdc_conc_pe_sosp = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_sospensioni_2021q3-4'],
            n_pdc_conc_pe_cds = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_conferenza_servizi_2021q3-4'],
            gg_pdc_conc_pe = comune_dataframe[
                'giornate_durata_media_permessi_costruire_conclusi_2021q3-4'],
            gg_pdc_max = \
                60,
            n_pdc = comune_dataframe[
                'numero_permessi_costruire_2021q3-4'],
            n_pdc_arr = comune_dataframe[
                'numero_permessi_costruire_non_conclusi_scaduti_termini_2021q3-4'])
        pdc.comune = comune
        pdc.monitoraggio = monitoraggio
        pds = PDS(
            n_pds_conc_sa = \
                0,
            n_pds_conc_pe = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_2021q3-4'],
            n_pds_conc_pe_sosp = comune_dataframe[
                'numero_sanatorie_concluse_con_sospensioni_2021q3-4'],
            n_pds_conc_pe_cds = comune_dataframe[
                'numero_sanatorie_concluse_con_conferenza_servizi_2021q3-4'],
            gg_pds_conc_pe = comune_dataframe[
                'giornate_durata_media_sanatorie_concluse_2021q3-4'],
            gg_pds_max = \
                60,
            n_pds = comune_dataframe[
                'numero_sanatorie_2021q3-4'],
            n_pds_arr = comune_dataframe[
                'numero_sanatorie_non_concluse_scaduti_termini_2021q3-4'])
        pds.comune = comune
        pds.monitoraggio = monitoraggio
        cdc = CDC(
            n_cdc_conc_sa = \
                0,
            n_cdc_conc_pe = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_2021q3-4'],
            n_cdc_conc_pe_sosp = comune_dataframe[
                'numero_controlli_cila_conclusi_con_sospensioni_2021q3-4'],
            n_cdc_conc_pe_cds = comune_dataframe[
                'numero_controlli_cila_conclusi_con_conferenza_servizi_2021q3-4'],
            gg_cdc_conc_pe = comune_dataframe[
                'giornate_durata_media_controlli_cila_conclusi_2021q3-4'],
            gg_cdc_max = \
                30,
            n_cdc = comune_dataframe[
                'numero_controlli_cila_2021q3-4'],
            n_cdc_arr = comune_dataframe[
                'numero_controlli_cila_non_conclusi_scaduti_termini_2021q3-4'])
        cdc.comune = comune
        cdc.monitoraggio = monitoraggio
        periodo_monitoraggio = '2022q1-2'
        monitoraggio = session.execute(select(Monitoraggio).where(
            Monitoraggio.periodo_monitoraggio == periodo_monitoraggio)).scalar_one()
        pdc = PDC(
            n_pdc_conc_sa = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q1-2'],
            n_pdc_conc_pe = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q1-2'],
            n_pdc_conc_pe_sosp = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_sospensioni_2022q1-2'],
            n_pdc_conc_pe_cds = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_conferenza_servizi_2022q1-2'],
            gg_pdc_conc_pe = comune_dataframe[
                'giornate_durata_media_permessi_costruire_conclusi_2022q1-2'],
            gg_pdc_max = \
                60,
            n_pdc = comune_dataframe[
                'numero_permessi_costruire_2022q1-2'],
            n_pdc_arr = comune_dataframe[
                'numero_permessi_costruire_non_conclusi_scaduti_termini_2022q1-2'])
        pdc.comune = comune
        pdc.monitoraggio = monitoraggio
        pds = PDS(
            n_pds_conc_sa = \
                0,
            n_pds_conc_pe = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_2022q1-2'],
            n_pds_conc_pe_sosp = comune_dataframe[
                'numero_sanatorie_concluse_con_sospensioni_2022q1-2'],
            n_pds_conc_pe_cds = \
                0,
            gg_pds_conc_pe = comune_dataframe[
                'giornate_durata_media_sanatorie_concluse_2022q1-2'],
            gg_pds_max = \
                60,
            n_pds = comune_dataframe[
                'numero_sanatorie_2022q1-2'],
            n_pds_arr = comune_dataframe[
                'numero_sanatorie_non_concluse_scaduti_termini_2022q1-2'])
        pds.comune = comune
        pds.monitoraggio = monitoraggio
        cdc = CDC(
            n_cdc_conc_sa = \
                0,
            n_cdc_conc_pe = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q1-2'],
            n_cdc_conc_pe_sosp = comune_dataframe[
                'numero_controlli_cila_conclusi_con_sospensioni_2022q1-2'],
            n_cdc_conc_pe_cds = \
                0,
            gg_cdc_conc_pe = comune_dataframe[
                'giornate_durata_media_controlli_cila_conclusi_2022q1-2'],
            gg_cdc_max = \
                30,
            n_cdc = comune_dataframe[
                'numero_controlli_cila_2022q1-2'],
            n_cdc_arr = comune_dataframe[
                'numero_controlli_cila_non_conclusi_scaduti_termini_2022q1-2'])
        cdc.comune = comune
        cdc.monitoraggio = monitoraggio
        periodo_monitoraggio = '2022q3-4'
        monitoraggio = session.execute(select(Monitoraggio).where(
            Monitoraggio.periodo_monitoraggio == periodo_monitoraggio)).scalar_one()
        pdc = PDC(
            n_pdc_conc_sa = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_silenzio-assenso_2022q3-4'],
            n_pdc_conc_pe = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4'],
            n_pdc_conc_pe_sosp = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni_2022q3-4'],
            n_pdc_conc_pe_cds = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2022q3-4'],
            gg_pdc_conc_pe = comune_dataframe[
                'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2022q3-4'],
            gg_pdc_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2022q3-4'],
            n_pdc = comune_dataframe[
                'numero_permessi_costruire_avviati_2022q3-4'],
            n_pdc_arr = comune_dataframe[
                'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'])
        pdc.comune = comune
        pdc.monitoraggio = monitoraggio
        pds = PDS(
            n_pds_conc_sa = comune_dataframe[
                'numero_sanatorie_concluse_con_silenzio-assenso_2022q3-4'],
            n_pds_conc_pe = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'],
            n_pds_conc_pe_sosp = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_sospensioni_2022q3-4'],
            n_pds_conc_pe_cds = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_conferenza_servizi_2022q3-4'],
            gg_pds_conc_pe = comune_dataframe[
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2022q3-4'],
            gg_pds_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2022q3-4'],
            n_pds = comune_dataframe[
                'numero_sanatorie_avviate_2022q3-4'],
            n_pds_arr = comune_dataframe[
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2022q3-4'])
        pds.comune = comune
        pds.monitoraggio = monitoraggio
        cdc = CDC(
            n_cdc_conc_sa = comune_dataframe[
                'numero_controlli_cila_conclusi_con_silenzio-assenso_2022q3-4'],
            n_cdc_conc_pe = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_2022q3-4'],
            n_cdc_conc_pe_sosp = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_sospensioni_2022q3-4'],
            n_cdc_conc_pe_cds = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2022q3-4'],
            gg_cdc_conc_pe = comune_dataframe[
                'giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso_2022q3-4'],
            gg_cdc_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_controlli_cila_avviati_2022q3-4'],
            n_cdc = comune_dataframe[
                'numero_controlli_cila_avviati_2022q3-4'],
            n_cdc_arr = comune_dataframe[
                'numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo_2022q3-4'])
        cdc.comune = comune
        cdc.monitoraggio = monitoraggio
        periodo_monitoraggio = '2023q1-2'
        monitoraggio = session.execute(select(Monitoraggio).where(
            Monitoraggio.periodo_monitoraggio == periodo_monitoraggio)).scalar_one()
        pdc = PDC(
            n_pdc_conc_sa = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_silenzio-assenso_2023q1-2'],
            n_pdc_conc_pe = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2'],
            n_pdc_conc_pe_sosp = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_sospensioni_2023q1-2'],
            n_pdc_conc_pe_cds = comune_dataframe[
                'numero_permessi_costruire_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2023q1-2'],
            gg_pdc_conc_pe = comune_dataframe[
                'giornate_durata_media_permessi_costruire_conclusi_con_provvedimento_espresso_2023q1-2'],
            gg_pdc_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_permessi_costruire_avviati_2023q1-2'],
            n_pdc = comune_dataframe[
                'numero_permessi_costruire_avviati_2023q1-2'],
            n_pdc_arr = comune_dataframe[
                'numero_permessi_costruire_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2'])
        pdc.comune = comune
        pdc.monitoraggio = monitoraggio
        pds = PDS(
            n_pds_conc_sa = comune_dataframe[
                'numero_sanatorie_concluse_con_silenzio-assenso_2023q1-2'],
            n_pds_conc_pe = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'],
            n_pds_conc_pe_sosp = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_sospensioni_2023q1-2'],
            n_pds_conc_pe_cds = comune_dataframe[
                'numero_sanatorie_concluse_con_provvedimento_espresso_con_conferenza_servizi_2023q1-2'],
            gg_pds_conc_pe = comune_dataframe[
                'giornate_durata_media_sanatorie_concluse_con_provvedimento_espresso_2023q1-2'],
            gg_pds_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_sanatorie_avviate_2023q1-2'],
            n_pds = comune_dataframe[
                'numero_sanatorie_avviate_2023q1-2'],
            n_pds_arr = comune_dataframe[
                'numero_sanatorie_arretrate_non_concluse_scaduto_termine_massimo_2023q1-2'])
        pds.comune = comune
        pds.monitoraggio = monitoraggio
        cdc = CDC(
            n_cdc_conc_sa = comune_dataframe[
                'numero_controlli_cila_conclusi_con_silenzio-assenso_2023q1-2'],
            n_cdc_conc_pe = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_2023q1-2'],
            n_cdc_conc_pe_sosp = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_sospensioni_2023q1-2'],
            n_cdc_conc_pe_cds = comune_dataframe[
                'numero_controlli_cila_conclusi_con_provvedimento_espresso_con_conferenza_servizi_2023q1-2'],
            gg_cdc_conc_pe = comune_dataframe[
                'giornate_durata_media_controlli_cila_conclusi_con_provvedimento_espresso_2023q1-2'],
            gg_cdc_max = comune_dataframe[
                'giornate_durata_mediana_termine_massimo_controlli_cila_avviati_2023q1-2'],
            n_cdc = comune_dataframe[
                'numero_controlli_cila_avviati_2023q1-2'],
            n_cdc_arr = comune_dataframe[
                'numero_controlli_cila_arretrati_non_conclusi_scaduto_termine_massimo_2023q1-2'])
        cdc.comune = comune
        cdc.monitoraggio = monitoraggio
        session.commit()
        # break
    session.close()


if __name__ == '__main__':
    pass
