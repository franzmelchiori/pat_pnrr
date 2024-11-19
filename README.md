# pat_pnrr
PAT MPE

A supporto dell'attivita' di monitoraggio dei procedimenti edilizi (MPE) e nell'idea di strutturare tale attivita' in un prodotto da lasciare all'amministrazione pubblica (PAT), si sta sviluppando (in Python, internamente al gruppo PAT PNRR Digital) la web app PAT MPE.

PAT MPE consente di estrarre i dati delle pratiche edilizie dai file Excel provenienti dai comuni trentini ogni semestre, alimentando delle basi dati (Pandas e Postgres). Da quest'ultime vengono elaborate le metriche ministeriali (ed altre a supporto dell’analisi) per ogni comune e per l’intera provincia, vengono generate delle relazioni tecniche (in PDF), con numeri, grafici ed analisi, per ogni comune e per l’intera provincia.

PAT MPE espone ora anche un'interfaccia web (navigabile tramite browser) la visualizzazione di cruscotti dati e documentazione. Infine, sono in via di sviluppo anche delle web API per l’integrazione di PAT MPE nei gestionali trentini per l’edilizia (es. GIScom, Licenze Edilizie) al fine di consentire la futura raccolta automatizzata dei dati (evitando il lavoro ai comuni, ma preservando il valore del monitoraggio).
