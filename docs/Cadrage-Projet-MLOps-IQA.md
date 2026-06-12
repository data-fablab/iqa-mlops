# Cadrage projet MLOps - Industrial Quality Assistant

## 1. Vision
**Industrial Quality Assistant** (IQA) est un copilote qualite pour le controle visuel de pieces `Casting`.

Objectif : rendre le controle plus objectif, tracable et exploitable, sans remplacer Sophie, l'inspectrice qualite.

Le MVP doit demontrer une boucle MLOps complete :
- inference visuelle ;
- ROI metier ;
- score d'anomalie ;
- heatmap explicable ;
- decision Vert / Orange / Rouge ;
- feedback oracle GT puis humain cible ;
- versioning donnees/modeles ;
- monitoring, drift, recalibration et reentrainement controle.

## 2. Probleme metier
Le controle visuel depend de la fatigue, de l'eclairage, de l'experience et de la subjectivite. Les defauts sont petits, parfois difficiles a localiser, et les donnees qualite ne sont pas centralisees.

Dans un contexte aeronautique et defense, le projet privilegie :
- securite de decision ;
- humain dans la boucle ;
- tracabilite ;
- gouvernance des modeles ;
- protection des donnees industrielles.

Roles : Sophie decide, Marc pilote, Thomas valide les cas process, Laurent garantit securite et donnees.

## 3. Perimetre MVP
Le MVP utilise :
```text
Casting_class1
Casting_class2
Casting_class3
```

Ces classes representent des familles de pieces differentes, regroupees dans une classe metier :
```text
business_class = Casting
source_class   = Casting_class1 | Casting_class2 | Casting_class3
```

Hors perimetre V1 : rejet critique automatique, classification fine des defauts, correlation complete defaut/process, IoT machine, Kubernetes, generation d'images comme fonctionnalite utilisateur.

## 4. Donnees
Source :
```text
D:\mar26_bds_anomalies_pieces_indus\data\raw\hss-iad
```

Dans le projet IQA, cette source est traitee comme un historique industriel rejoue, pas comme le stockage de production final. Les images peuvent rester versionnees via DVC/MinIO, mais le runtime manipule toujours des URI et passe par le meme contrat d'ingestion que la production.

Sources d'ingestion :
```text
historical_replay  -> dataset Casting rejoue pour le MVP
production_ingest  -> future arrivee camera / poste qualite / MES
```

Stockage cible :
```text
MinIO      -> fichiers images et artefacts lourds
PostgreSQL -> piece_event, lots, timestamps, source, URI, predictions, feedback
```

| Source | Train good | Test good | Test defective | Masques GT |
|---|---:|---:|---:|---:|
| `Casting_class1` | 440 | 12 | 19 | 19 |
| `Casting_class2` | 1242 | 39 | 49 | 49 |
| `Casting_class3` | 341 | 18 | 23 | 23 |
| **Total** | **2023** | **69** | **91** | **91** |

Les defauts sont tres petits : environ `0,60 %`, `0,35 %` et `0,27 %` de l'image selon la classe. Le pipeline doit conserver une logique haute resolution.

## 5. Inventaire et unite metier
Manifests locaux :
```text
D:\MLOPS\data\metadata
```

Fichiers : `casting_images_inventory.csv`, `casting_piece_events.csv`, `casting_flux_replay_plan*.csv`, `replay_scenarios.csv`.

Inventaire :
```text
2183 images principales
2023 train/good
69 test/good
91 test/defective
91 masques GT
sha256 par image
```

Nommage :
```text
YYYY-MM-DD_HH_MM_SS_mmm[-n]_vueA_vueB.jpg
```

Un evenement piece est defini par :
```text
piece_event = source_class + group_key
```

Resultat :
```text
962 evenements pieces
2183 images
40 evenements defectueux
taux defaut piece ~ 4,2 %
1 a 4 images par piece_event
```

## 6. Scenarios de replay
Deux scenarios utilisent la meme API, la meme architecture modele et les memes pipelines, mais leurs lots, datasets, metriques et versions candidates sont isoles par `scenario_id`. Seul le bootstrap V0 est partage. La simulation MVP impose trois garde-fous : amorce hors replay, drift good-only avant defauts, validation set fige hors replay.

### Production naturelle
```text
scenario_id       = production_replay_natural
scenario_type     = production
is_representative = true
ordre             = source_timestamp, source_class, event_key
```
Usage : demonstration produit, lots, dashboard Marc, feedback oracle, tracabilite. Le replay applique un warm-up class1/good puis une production-like avec defauts repartis.

### Drift controle
```text
scenario_id       = drift_domain_extension
scenario_type     = mlops_stress_test
is_representative = false
ordre             = Casting_class1 puis Casting_class2 puis Casting_class3
```
Phases :
```text
baseline_domain_class1
domain_extension_class2
domain_extension_class3
```
Usage : detection drift, recalibration, reentrainement candidat, comparaison MLflow. Le scenario cible isole une phase nouveau domaine good-only avant injection des defauts.

## 7. Modele V1
Pipeline :
```text
image 1024
-> segmenteur ROI surface fonctionnelle
-> controle qualite ROI
-> RD/Feature-AE good-only
-> score anomalie + heatmap
-> aggregation vues au niveau piece
-> decision Vert/Orange/Rouge
-> feedback oracle GT
```

Briques :
- segmenteur ROI fige : surface inspectable, non reentraine en MVP ;
- teacher ResNet18 fige : referentiel features layer2/layer3 ;
- RD/Feature-AE : modele de normalite et seul modele vivant ;
- calibration : seuils conservateurs, score top-k ;
- latence cible : p95 < 1 seconde sur poste GPU local.

## 8. Cycles de vie Feature-AE
Le MVP simplifie le cycle de vie : le segmenteur ROI reste fige, le Feature-AE concentre l'apprentissage continu.

Amorce commune, separee des replays :
```text
bootstrap_dataset_version = feature_ae_good_v001_bootstrap
contenu                   = 50 evenements pieces Casting_class1/good
volume                    = 50 images
usage                     = entrainement Feature-AE V0
roi_segmenter_version     = fixe
```

### Cycle production naturelle
Objectif : maturation progressive dans un flux usine representatif.

```text
scenario_id = production_replay_natural
V0 = amorce Casting_class1/good
V1 = V0 + conformes valides lot_001
V2 = V1 + conformes valides lot_002
V3 = V2 + faux positifs valides conformes
```

Declenchement :
```text
30 a 50 nouvelles pieces conformes validees
ROI ok uniquement
pas de defaut confirme ni defaut manque dans le train normal
```

### Cycle drift controle
Objectif : detection drift, reentrainement candidat et comparaison avant/apres.

```text
scenario_id = drift_domain_extension
V0 = amorce Casting_class1/good
V1 = V0 + Casting_class2/good valides apres alerte drift
V2 = V1 + Casting_class3/good valides apres alerte drift
class1 -> baseline stable
class2 -> drift features teacher + hausse erreur Feature-AE
class3 -> nouveau drift controle
```

Les defauts confirmes, defauts manques et ROI warning/fail restent hors train normal.

## 9. ROI et decision
Le segmenteur ROI est critique : une ROI trop restrictive peut masquer un defaut. Pour simplifier le MVP, il est fige et ne fait pas partie du cycle de reentrainement.

Chaque ROI produit :
```text
roi_quality_status = ok | warning | fail
roi_failure_reasons
surface_ratio
landmark_ratio
pattern_id
view_key
```

Regles :
- ROI vide ou presque pleine : `fail` ;
- ratio hors plage, pattern inconnu, ROI fragmentee : `warning` ou `fail` ;
- anomalie forte hors ROI : `warning`.

Decision :
```text
ROI warning -> Orange minimum
ROI fail    -> revue obligatoire
Vert        -> seulement si toutes les vues ont ROI OK et score bas
```

Aggregation :
```text
une vue Rouge -> piece Rouge
une vue Orange ou ROI warning/fail -> piece Orange
sinon -> piece Vert
```

Gouvernance ROI :
```text
roi_model_version fixe
feedback ROI conserve pour analyse
aucune promotion automatique
nouvelle version ROI hors MVP seulement
defect_coverage_min >= 0.95
taux ROI fail stable ou en baisse
ROI warning/fail hors train Feature-AE
```

## 10. Feedback
Pour accelerer le MVP, le feedback Sophie est automatise par un oracle GT. L'interface Sophie reste fonctionnelle : elle permet de saisir un verdict `human_sophie`, exploite par les memes regles que l'oracle et prioritaire en cas de divergence.

```text
feedback_source = oracle_gt | human_sophie
label good      -> Conforme
label defective -> Defaut confirme
masque GT       -> evaluation localisation
```

Exploitation :
```text
Conforme valide -> train normal candidat
Defaut confirme -> validation/test uniquement
Defaut manque -> validation prioritaire + alerte critique
ROI warning/fail -> quarantaine
```

Regle : le label GT n'est utilise qu'apres prediction, jamais pendant l'inference.

## 11. Versioning et tracabilite
Chaque prediction doit etre reliee a :
```text
image source -> sha256 -> piece_event -> scenario -> lot
-> dataset_version -> model_version -> prediction -> feedback_source
```

Trois niveaux :
```text
raw_dataset_id = hss_iad_casting_raw_v1
manifest       = iqa_casting_manifest_v001
model_dataset  = feature_ae_good_v001, v002, v003...
validation     = validation_set_v001
```

Outils :
```text
Git    -> code, docs, petits manifests
DVC    -> donnees volumineuses, exports, artefacts
MLflow -> runs, parametres, metriques, liens dataset/modele
MinIO  -> remote S3 local DVC, MLflow artifacts, heatmaps, modeles
```

Chaque run MLflow logue : dataset, manifest, scenario, train/validation counts, model_type, roi_model_version, feature_ae_version, git_commit.

Convention : `raw/hss_iad_casting_raw_v1`, `manifest/iqa_casting_manifest_v001`, `replay/*_v001`, `validation/validation_set_v001`, `model_dataset/feature_ae_good_v001`, `model/feature_ae_v001`, `model/roi_segmenter_v001_fixed`.

## 12. Architecture MVP
Briques MLOps du MVP :
```text
raw hss-iad
-> ingestion + inventory sha256
-> piece_events multi-vues
-> bootstrap Feature-AE hors replay
-> replay production-like / drift controle
-> FastAPI gateway + service inference PyTorch
-> feedback oracle GT apres prediction
-> metadata store PostgreSQL
-> object store MinIO
-> DVC remote s3://iqa-dvc
-> Prometheus/Grafana
-> Airflow DAGs ingestion/replay/lifecycle/monitoring
-> dataset candidat good-only
-> training + evaluation Feature-AE
-> MLflow tracking + model registry
-> promotion / rollback
-> /admin/reload-model
```

Contrats :
```text
Serving       = ROI fixe + teacher fixe + Feature-AE actif
Feedback      = oracle_gt au MVP, human_sophie fonctionnel et prioritaire
Training      = separe de l'usage metier
Orchestration = Airflow LocalExecutor, DAG vedette iqa_lifecycle
Storage       = MinIO local, acces via src/iqa/storage uniquement
Registry      = candidate -> test -> prod ou archived
Reload        = reserve admin/pipeline
Simulation    = dry-run predictions, lots, runs, registry
```

Endpoints minimaux :
```text
GET  /health
GET  /model/version
GET  /replay-scenarios
POST /predict
POST /piece-events/{event_id}/predict
POST /feedback
GET  /metrics
POST /admin/reload-model
```

Prediction piece : `piece_event_id`, `scenario_id`, `model_version`, `roi_model_version`, `views`, `piece_score`, `piece_status`, `decision_reasons`, `latency_ms`.

## 13. Monitoring et drift
Toutes les metriques sont filtrables par :
```text
scenario_id, lot_id, source_class, pattern_id/view_key,
model_version, dataset_version, roi_model_version
```

Metriques :
```text
Production : pieces, lots, Vert/Orange/Rouge, temps controle
ROI        : ok/warning/fail, surface_ratio, landmark_ratio
Feature-AE : score, erreur p50/p95/p99, depassement seuil
Feedback   : source, FP, FN, defauts confirmes, conformes valides
Systeme    : latence p95, taux erreur API, succes pipelines
```

Drift :
```text
1. Drift entree      -> features teacher ResNet18 layer2/layer3
2. Degradation modele -> erreur reconstruction Feature-AE
3. Drift ROI         -> ratios ROI et warning/fail, sans reentrainement ROI
```

Signal fort :
```text
features teacher qui driftent + erreur Feature-AE qui monte
= drift domaine probable
```

Fenetres :
```text
lot courant, 3 derniers lots, reference bootstrap + calibration
```

Seuils MVP :
```text
PSI < 0.10 normal
PSI 0.10-0.25 warning
PSI > 0.25 drift probable
KS p-value < 0.01 + ecart visible -> alerte drift
p95 reconstruction +30 % -> warning
p95 reconstruction +50 % -> alerte
roi_fail > 2 % sur un lot -> alerte technique
faux negatif > 0 -> alerte critique
```

## 14. Recalibration et reentrainement
Feature-AE : entrainement candidat si :
```text
30 a 50 nouvelles pieces conformes validees
ET drift confirme sur features teacher ou erreur Feature-AE
```

Autres declencheurs :
```text
taux Orange trop eleve
faux positifs confirmes nombreux
p95 reconstruction en derive
nouvelle source_class ou nouveau pattern stable
```

Faux negatif :
```text
alerte critique -> blocage promotion -> analyse seuils/ROI/heatmap
-> ajout validation prioritaire -> recalibration ou candidat si justifie
```

Le segmenteur ROI est fige, versionne et monitore, sans shadow training dans le perimetre MVP.

Gates Feature-AE :
```text
evaluation post-train obligatoire, pas seulement lot declencheur
recall defaut == 1.0 sur validation_set_v001
Image AP / Pixel AP >= prod - 0.02
taux Orange <= 30 % en V0/V1, <= 15 % en V2+
latence p95 sous objectif
tests incidents FN / ROI / rollback OK
comparaison MLflow + rollback possible
promotion = copie modele vers s3://iqa-models/prod/
```

## 15. CI/CD et tests
Tests :
- healthcheck API ;
- inference image ;
- inference piece multi-vues ;
- feedback ;
- chargement modele ;
- schemas donnees ;
- non-regression validation reelle.
- contrats API, datasets candidats, monitoring, incidents.

Cycle :
```text
Boucle code  : commit -> lint/tests -> build Docker -> tests import DAG Airflow
Boucle modele: evenement donnees -> Airflow lifecycle -> train/eval/gates
              -> MLflow candidate -> prod ou archive -> reload inference
```

La CI ne declenche jamais d'entrainement. Airflow fait vivre le modele sur evenement donnees. La validation peut rejeter un candidat ; le segmenteur ROI reste fige.

## 16. Go/no-go
Metier :
- faux negatifs ;
- faux positifs ;
- taux revue humaine ;
- temps moyen controle ;
- taux pieces bloquees ;
- adoption inspectrice.

Modele :
- Image AP, Pixel AP, AUPIMO ;
- recall defaut ;
- precision localisation ;
- stabilite par pattern/source_class ;
- drift features/scores ;
- defect coverage ROI ;
- taux ROI warning/fail.

Systeme : latence p95 < 1 seconde, taux erreur API, disponibilite, succes pipelines.

Les go/no-go sont calcules sur `validation_set_v001`, fige avant tout replay, stratifie par source_class, exclu du replay et de la calibration. Les donnees simulees servent a la demonstration et aux tests de pipeline.

## 17. Risques
Risques principaux :
- desequilibre `Casting_class2` ;
- patterns rares ;
- faux negatifs critiques ;
- ROI figee trop restrictive sur cas rares ;
- heatmaps peu lisibles ;
- confusion reel/simule ;
- surapprentissage aux lots ;
- derive progressive des scores ;
- complexite excessive MVP ;
- donnees sensibles mal protegees.

Mesures :
- equilibrage par source_class/pattern ;
- validation reelle separee ;
- feedback oracle GT pour le MVP ;
- validation process ;
- controle ROI a chaque inference ;
- segmenteur ROI fige, versionne et surveille ;
- reentrainement automatique limite au Feature-AE good-only ;
- versioning strict ;
- separation claire des scenarios ;

## 18. Recit de demonstration
Deux recits :
```text
Recit produit
-> production_replay_natural
-> usage quotidien, lots, Sophie, Marc, feedback, tracabilite

Recit MLOps
-> drift_domain_extension
-> drift teacher/AE, recalibration, candidat, MLflow
```

Boucle :
```text
piece controlee -> ROI -> Feature-AE -> heatmap -> decision Sophie
-> feedback oracle GT -> dataset versionne -> monitoring
-> entrainement candidat -> comparaison MLflow -> promotion ou rollback
```

La valeur metier reste portee par le controle reel, la tracabilite, la reduction du risque qualite et la capacite a faire vivre un modele en production de maniere gouvernee.
