# Data Contracts IQA Phase 2

Ce document decrit les contrats metadata stables utilises par les manifests CSV
legers et par la cible PostgreSQL applicative. Les fichiers lourds restent dans
MinIO/DVC ; Git ne contient que les manifests, schemas, tests et scripts.

## Chaine de tracabilite

La chaine cible est :

```text
sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback
```

Les identifiants servent a relier les faits metier sans dependre d'un chemin de
fichier, d'un ordre de ligne ou d'un run local.

## Identifiants canoniques

| Champ | Role |
| --- | --- |
| `raw_dataset_id` | Identifiant stable du dataset source brut. Valeur actuelle : `hss_iad_casting_raw_v1`. |
| `manifest_id` | Identifiant du manifest CSV publie. |
| `piece_event_id` | Identifiant stable de l'evenement piece consomme par l'API et les repos metadata. |
| `source_event_id` | Identifiant de la piece originale quand un evenement est simule en replay. |
| `scenario_id` | Scenario runtime obligatoire cote API. Exemples : `production_replay_natural`, `drift_domain_extension`. |
| `dataset_version` | Version data utilisee pour replay, validation, calibration, bootstrap ou source raw. |
| `replay_id` | Version stable du replay. Vide hors replay. |
| `validation_id` | Identifiant stable du set de validation. Vide hors validation. |
| `scenario_version` | Version du scenario replay. Vide hors replay. |
| `lot_id` | Lot production ou replay qui regroupe des pieces dans les vues API et metrics. |
| `prediction_id` | Identifiant de prediction produit par l'API. |
| `model_version` | Version modele servie ou tracee dans MLflow/API. |
| `feedback_id` | Identifiant logique futur si les feedbacks deviennent multi-evenements ; aujourd'hui le feedback oracle est upsert par `prediction_id`. |

## Manifests Phase 2

Les colonnes Phase 2 sont ajoutees en fin de manifest, sans supprimer ni
renommer les colonnes historiques :

```text
raw_dataset_id
manifest_id
piece_event_id
dataset_version
replay_id
validation_id
scenario_version
```

Contrats actuels :

| Manifest | `manifest_id` | `dataset_version` | Regle `piece_event_id` |
| --- | --- | --- | --- |
| `data/metadata/casting_piece_events.csv` | `casting_piece_events_v001` | `hss_iad_casting_raw_v1` | `piece_event_id == event_id` |
| `data/metadata/feature_ae_bootstrap_events.csv` | `feature_ae_bootstrap_events_v001` | `feature_ae_good_v001_bootstrap` | `piece_event_id == event_id` |
| `data/metadata/casting_flux_replay_plan_natural_v003.csv` | `casting_flux_replay_plan_natural_v002` | `production_replay_natural_v002` | `piece_event_id == simulated_event_id` |
| `data/metadata/casting_flux_replay_plan_natural_train_v004.csv` | `casting_flux_replay_plan_natural_train_v004` | `production_replay_natural_train_v004` | `piece_event_id == simulated_event_id` |
| `data/metadata/casting_flux_replay_plan_drift.csv` | `casting_flux_replay_plan_drift_v001` | `drift_domain_extension_v001` | `piece_event_id == simulated_event_id` |
| `data/validation/validation_set_replay_gate_v001.csv` | `validation_set_replay_gate_v001` | `validation_set_replay_gate_v001` | `piece_event_id == event_id` |
| `data/validation/validation_set_replay_gate_v002.csv` | `validation_set_replay_gate_v002` | `validation_set_replay_gate_v002` | `piece_event_id == event_id` |
| `data/validation/validation_set_replay_gate_v003.csv` | `validation_set_replay_gate_v003` | `validation_set_replay_gate_v003` | `piece_event_id == event_id` |
| `data/validation/validation_set_replay_representative_v001.csv` | `validation_set_replay_representative_v001` | `validation_set_replay_representative_v001` | `piece_event_id == event_id` |
| `data/validation/calibration_good_reference_v001.csv` | `calibration_good_reference_v001` | `calibration_good_reference_v001` | `piece_event_id == event_id` |

En replay, `source_event_id` reste l'identifiant de la piece originale et
`piece_event_id` identifie l'evenement simule servi par le scheduler replay.

## Feedback et eligibilite train

`oracle_gt` reste la source souveraine pour l'eligibilite train. Les valeurs
machine cibles restent ASCII :

```text
oracle_verdict = conforme | defective
train_eligible = true | false
train_eligibility_source = oracle_gt
```

`human_sophie` reste display-only dans cette phase et ne rend jamais un exemple
eligible au train normal.

Les datasets Feature-AE candidats suivent ces versions :

| Version | Source |
| --- | --- |
| `feature_ae_good_mvp_v001` | Socle good MVP disjoint validation/calibration/replay. |

Un sample est eligible seulement si `oracle_verdict=conforme`,
`train_eligible=true`, `train_eligibility_source=oracle_gt`, sans quarantaine et
sans statut ROI bloquant.
Les manifests materialises sont produits sous `data/model_datasets/` par
`iqa-build-feature-ae-datasets` et portent `dataset_version` ainsi que
`manifest_version`.

Le lifecycle MVP utilise `feature_ae_good_mvp_v001` comme socle stable et ajoute
les conformes vus pendant le replay pour construire les candidats de cycle.

## Validation Gate Et Representative MVP

`data/validation/validation_set_replay_gate_v001.csv` est le panel gate
AUPIMO utilise par defaut pour les runs progressifs courts. Il est fige au
niveau image afin de conserver le signal AUPIMO historique du bootstrap et de
comparer les candidats sans faire s'effondrer la metrique low-FPR sur un panel
hors distribution de gate.

La repartition actuelle par `source_class` est :

| `source_class` | Images |
| --- | ---: |
| `Casting_class1` | 13 |
| `Casting_class2` | 25 |
| `Casting_class3` | 11 |

`data/validation/validation_set_replay_gate_v002.csv` est un panel gate
candidat reconstruit au niveau `piece_event`. Il conserve les defauts reserves
disponibles et ajoute des pieces good tenues hors bootstrap, calibration et
replay naturel pour stabiliser la calibration seuils et rapprocher le contrat de
decision du replay multi-vues.

La repartition v002 par `source_class` est :

| `source_class` | Pieces |
| --- | ---: |
| `Casting_class1` | 23 |
| `Casting_class2` | 73 |
| `Casting_class3` | 38 |

La repartition v002 par label est :

| Label | Pieces |
| --- | ---: |
| `good` | 120 |
| `defective` | 14 |

Ce panel ne devient le default Airflow qu'apres validation serveur explicite de
la metrique AUPIMO, des seuils et de la classification replay.

`data/validation/validation_set_replay_gate_v003.csv` est le panel gate fixe du
Scenario B. Il est extrait du replay naturel avant entrainement progressif :
10 pieces defaut et 120 pieces good sont reservees pour la promotion, et le
replay d'entrainement devient
`data/metadata/casting_flux_replay_plan_natural_train_v004.csv`.

La repartition v003 par label est :

| Label | Pieces |
| --- | ---: |
| `good` | 120 |
| `defective` | 10 |

La repartition v003 par `source_class` est :

| `source_class` | Pieces |
| --- | ---: |
| `Casting_class1` | 23 |
| `Casting_class2` | 71 |
| `Casting_class3` | 36 |

Le replay train v004 et le gate v003 sont disjoints sur `source_event_id`, et
leur union reconstruit les 562 evenements du replay naturel v003.

`data/validation/validation_set_replay_representative_v001.csv` est fige au
niveau `piece_event` et sert aux validations completes/audit.
La repartition actuelle par `source_class` est :

| `source_class` | Pieces |
| --- | ---: |
| `Casting_class1` | 13 |
| `Casting_class2` | 39 |
| `Casting_class3` | 22 |

Ce set reste disjoint de bootstrap, calibration et replay. Il sert a verifier la
regle oracle sans alimenter le train normal. La calibration seuils utilise le
panel good separe `data/validation/calibration_good_reference_v001.csv`.

## Persistance PostgreSQL

PostgreSQL est optionnel et active explicitement par `IQA_METADATA_BACKEND=postgres`.
Le backend applicatif stocke les faits, statuts, timestamps, versions, URI et
payloads JSONB. Il ne stocke pas les images, checkpoints, masques ou heatmaps
binaires.

Le socle PostgreSQL applicatif couvre aussi les evenements MLOps et gouvernance
prepares pour les prochains lots : `lot_events`, `incident_events`,
`model_version_events`, `scenario_version_events` et
`lifecycle_trigger_events`. Ces tables exposent les identifiants de jointure
stables (`scenario_id`, `lot_id`, `piece_event_id`, `prediction_id`,
`dataset_version`, `manifest_version`, `model_version`, `trigger_reason`) tout
en conservant le payload complet en JSONB.

Les CSV restent la source operationnelle immediate jusqu'aux lots DVC/replay
suivants, mais les noms de colonnes sont alignes avec la cible PostgreSQL.
