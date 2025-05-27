PAT MPE
=======

A supporto dell'attività di monitoraggio dei procedimenti edilizi (MPE) e nell'idea di strutturare tale attività in uno strumento informatico a supporto delle amministrazioni coinvolte (PAT, Comuni, Consorzio dei Comuni Trentini), si sta sviluppando (in Python, internamente al gruppo PAT PNRR Digital) la web app PAT MPE.

PAT MPE consente di estrarre i dati delle pratiche edilizie dai file Excel provenienti dai comuni trentini ogni semestre, alimentando delle basi dati (in Pandas e Postgres). Da queste ultime vengono elaborate le metriche ministeriali (ed altre, a supporto dell’analisi) per ogni comune e per l’intero territorio provinciale con evidenza di numeri, grafici ed analisi.

PAT MPE espone un'interfaccia web (navigabile tramite browser) per la visualizzazione di cruscotti dati e documentazione. Infine, si prevede di sviluppare anche delle web API per l’integrazione di PAT MPE nei gestionali trentini per l’edilizia (es. GIScom, Licenze Edilizie) al fine di consentire in futuro la raccolta completamente automatizzata dei dati (evitando il lavoro ai comuni, ma preservando il valore del monitoraggio).


Grafici attuali
---------------

* grafico provinciale ad aree con il cumulo di pratiche PdC e PdS avviate nei vari monitoraggi
* grafico provinciale a linee con il numero di pratiche PdC e PdS avviate nei vari monitoraggi
* grafico provinciale a linee con il numero di pratiche PdC avviate per i vari cluster comunali nei vari monitoraggi
* grafico provinciale a linee con il numero di pratiche PdS avviate per i vari cluster comunali nei vari monitoraggi
* grafico provinciale a linee con la durata delle pratiche PdC e PdS concluse nei vari monitoraggi
* grafico provinciale ad aree con il numero di pratiche PdC e PdS arretrate nei vari monitoraggi
* grafico comunale a nuvola animata di punti con l'indice di pressione (lorda e netta) di pratiche PdC e PdS nei vari monitoraggi
* grafico comunale a scatole percentili con l'indice di pressione (lorda e netta) di pratiche PdC e PdS per i vari cluster comunali nei vari monitoraggi
* grafico provinciale a barometro animato con l'indice di pressione (lorda e netta) nei vari monitoraggi
* grafico comunale a torta animata con fette (proporzionali alla popolazione) ordinate per indice di pressione (lorda e netta) dei vari comuni nei vari monitoraggi
* grafico comunale a linee ordinate per indice di pressione (lorda e netta) dei vari comuni nei vari monitoraggi
* grafico comunale a nuvola animata di punti con l'indice di pressione (lorda e netta) e le ore a settimana di elaborazione tecnica nei vari monitoraggi


Prossimi sviluppi
-----------------

1. Grafici
    1. Sviluppare e visualizzare nella web app i seguenti grafici e mappe:
        * grafico provinciale a linee con il numero di pratiche PdC arretrate per i vari cluster comunali nei vari monitoraggi
        * grafico provinciale a linee con il numero di pratiche PdS arretrate per i vari cluster comunali nei vari monitoraggi
        * grafico comunale a nuvola animata di punti con il numero di pratiche PdC e PdS avviate e le ore a settimana di elaborazione tecnica nei vari monitoraggi
        * grafico comunale a nuvola animata di punti con la durata delle pratiche PdC e PdS concluse e le ore a settimana di elaborazione tecnica nei vari monitoraggi
        * grafico comunale a nuvola animata di punti con il numero di pratiche PdC e PdS arretrate e le ore a settimana di elaborazione tecnica nei vari monitoraggi
        * mappa provinciale con l'indice di pressione (lorda e netta) nei vari monitoraggi
        * calendario provinciale con il numero di pratiche PdC e PdS avviate nei vari monitoraggi
2. Analisi
    1. Sviluppare un ottimizzatore per indicare dei percorsi verso i target finali
    2. Sviluppare la dashboard analitica per ognuno dei 166 comuni trentini
    3. Sviluppare nuove strategie di clustering dei comuni (oltre a quello ISPAT)
    4. Sviluppare un sistema di previsione edilizia sul carico di lavoro comunale


Visione futura
--------------

1. Pubblicazione della web app PAT MPE in Intranet PAT
    1. Integrare nella web app un modulo di autenticazione PAT
    2. Pubblicare la web app nella intranet PAT via Docker
2. Pubblicazione della web app PAT MPE in Internet
    1. Sviluppare le web API per l'accesso CRUD ai dati (pratiche e misure) su Postgres ed in particolare al tracciamento delle sospensioni: questo dovrebbe consentire e rendere opportuna l'integrazione delle web API nei gestionali trentini per l'edilizia
    2. Sviluppare il front end della web app con Vue.js
    3. Sviluppare un portale per lo scaricamento di un file Excel che predispone al monitoraggio dei procedimenti edilizi ed il caricamente dello stesso file Excel compilato per il recepimento del monitoraggio, segnalando eventuali dati mancanti o errati
    4. Sviluppare un sistema di autenticazione ed autorizzazione per Internet e pubblicare la web app
