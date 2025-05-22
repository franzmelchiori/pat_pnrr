PAT MPE
=======

A supporto dell'attività di monitoraggio dei procedimenti edilizi (MPE) e nell'idea di strutturare tale attività in uno strumento informatico a supporto delle amministrazioni coinvolte (PAT, Comuni, Consorzio dei Comuni Trentini), si sta sviluppando (in Python, internamente al gruppo PAT PNRR Digital) la web app PAT MPE.

PAT MPE consente di estrarre i dati delle pratiche edilizie dai file Excel provenienti dai comuni trentini ogni semestre, alimentando delle basi dati (in Pandas e Postgres). Da queste ultime vengono elaborate le metriche ministeriali (ed altre, a supporto dell’analisi) per ogni comune e per l’intera provincia con evidenza di numeri, grafici ed analisi.

PAT MPE espone un'interfaccia web (navigabile tramite browser) per la visualizzazione di cruscotti dati e documentazione. Infine, si prevede di sviluppare anche delle web API per l’integrazione di PAT MPE nei gestionali trentini per l’edilizia (es. GIScom, Licenze Edilizie) al fine di consentire in futuro la raccolta automatizzata dei dati (evitando il lavoro ai comuni, ma preservando il valore del monitoraggio).


Grafici attuali
---------------

* chart | provincia | area | time | avviato [nr] pdc_pds | mpe 1-last (time)
* chart | provincia | line | time | avviato [nr] pdc_pds | mpe 1-last (time)
* chart | provincia | line | time | cluster | avviato [nr] pdc | mpe 1-last (time)
* chart | provincia | line | time | cluster | avviato [nr] pds | mpe 1-last (time)
* chart | provincia | line | time | durata [gg] pdc_pds | mpe 1-last (time)
* chart | provincia | area | time | arretrato [nr] pdc_pds | mpe 1-last (time)
* chart | comuni | scatter | cluster pop | pressione (lorda/netta) [idx] pdc pds | mpe 1-last (timelapse)
* chart | comuni | box | cluster | pressione (lorda/netta) [idx] | mpe 1-last
* chart | provincia | gauge | mean | pressione (lorda/netta) [idx] | mpe 1-last (timelapse)
* chart | comuni | pie | rank pop | pressione (lorda/netta) [idx] | mpe 1-last (timelapse)
* chart | comuni | rank | time | pressione (lorda/netta) [idx] | mpe 1-last (time)


Prossimi sviluppi
-----------------

1. Grafici
    1. Sviluppare e visualizzare nella web app i seguenti grafici e mappe:
        * chart | provincia | line | time | cluster | arretrato [nr] pdc | mpe 1-last (time)
        * chart | provincia | line | time | cluster | arretrato [nr] pds | mpe 1-last (time)
        * chart | comuni | scatter | cluster | pressione [idx]-elaborazione [ore/sett] pdc_pds | mpe 5-last (timelapse)
        * chart | comuni | scatter | cluster | avviato [nr]-elaborazione [ore/sett] pdc_pds | mpe 5-last (timelapse)
        * chart | comuni | scatter | cluster | durata [gg]-elaborazione [ore/sett] pdc_pds | mpe 5-last (timelapse)
        * chart | comuni | scatter | cluster | arretrato [nr]-elaborazione [ore/sett] pdc_pds | mpe 5-last (timelapse)
        * chart | provincia | map | pressione | mpe 1-last (timelapse?)
        * chart | pratiche | cal | avviato [nr] pdc_pds | mpe 1-last (time)
2. Analisi
    1. Sviluppare la dashboard analitica per ognuno dei 166 comuni trentini
    2. Sviluppare nuove strategie di clustering dei comuni (oltre a quello ISPAT)
    3. Sviluppare un ottimizzatore per indicare dei percorsi verso i target finali
    4. Sviluppare un sistema di previsione edilizia sul carico di lavoro comunale


Visione futura
--------------

1. Intranet PAT
    1. Integrare nella web app un modulo di autenticazione PAT
    2. Pubblicare la web app nella intranet PAT via Docker
2. Internet
    1. Sviluppare le web API per l'accesso CRUD ai dati (pratiche e misure) su Postgres ed in particolare al tracciamento delle sospensioni: questo dovrebbe consentire e rendere opportuna l'integrazione delle web API nei gestionali trentini per l'edilizia
    2. Sviluppare il front end della web app con Vue.js
    3. Sviluppare un portale per lo scaricamento di un file Excel che predispone al monitoraggio dei procedimenti edilizi ed il caricamente dello stesso file Excel compilato per il recepimento del monitoraggio, segnalando eventuali dati mancanti o errati
    4. Sviluppare un sistema di autenticazione ed autorizzazione per Internet e pubblicare la web app
