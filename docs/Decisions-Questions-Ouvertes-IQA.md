# Propositions de decisions - Questions ouvertes IQA

Ce document synthetise les decisions a proposer au groupe pour fermer les
questions d'architecture restees ouvertes pendant le cadrage. Les formulations
sont anonymes et orientees arbitrage projet.

## Statut apres convergence Ken/IQA

Les points suivants sont maintenant fermes :

- `piece_event` est l'unite atomique de split, replay, feedback, validation,
  calibration et train.
- `calibration_set_v001` est ajoute comme set good-only etanche.
- Invariant officiel : `bootstrap ∩ calibration ∩ replay ∩ validation = vide`.
- MLflow Registry est la source de verite de la promotion et du rollback.
- Les registered models sont separes par scenario :
  `feature_ae__production_replay_natural` et
  `feature_ae__drift_domain_extension`.
- `iqa-api` et `iqa-inference` restent deux services separes.
- Le `pyproject.toml` racine est conserve en phase initiale ; la migration
  `services/` est reportee.
- Sophie reste une vitrine MVP ; `oracle_gt` pilote le workflow operationnel.
- Les faits replay portent `event_time`, `recorded_at` et `is_simulated`.

Questions ouvertes restantes :

1. Stock reel de `piece_events` defectueux par `source_class` apres extraction
   definitive des CSV.
2. Taille nominale des lots replay, a fixer apres verification du stock
   defectueux et du volume good disponible.

## Synthese courte

- Le split se fait au niveau `piece_event`, jamais au niveau image.
- `validation_set_v001` est fige hors replay, hors train et hors calibration.
- Le ROI reste fige, mais `defect_coverage` devient une gate de recevabilite par
  `source_class`.
- Pour le MVP, l'API est un vrai service FastAPI et l'inference est un service
  Docker separe ; ingestion, training et monitoring sont aussi separes.
- PostgreSQL reste un seul conteneur, mais avec trois bases logiques :
  `iqa_metadata`, `mlflow`, `airflow`.
- MLflow Registry devient la source de verite de la promotion modele ; MinIO
  stocke les artefacts.
- L'interface Sophie est une vitrine dans le MVP ; l'oracle GT automatise le
  workflow.
- Airflow evalue les conditions donnees via `iqa_monitoring` ; le commit ne
  declenche jamais le lifecycle modele.
- Le drift inattendu et l'extension de domaine planifiee sont deux regimes
  distincts.
- Les registered models MLflow sont isoles par `scenario_id`.
- En cas de retard, la boucle MLOps naturelle est prioritaire sur la performance
  ML parfaite et les dashboards avances.
- L'oracle GT repond `defective` si un masque GT non vide existe, sinon
  `conforme`.
- Le reverse proxy retenu est Nginx.

## Decisions proposees

### 1. Unite de split

Proposition : `piece_event` est l'unite atomique.

Toutes les vues d'une meme piece restent ensemble : meme split, meme scenario,
meme validation, meme replay et meme train. Les images heritent du statut de leur
`piece_event`. Les comptes par image restent utiles pour le reporting, mais ne
sont jamais une frontiere de split.

Invariant : les splits sont au niveau piece ; les images heritent de
l'affectation de leur piece.

### 2. Composition de `validation_set_v001`

Proposition : `validation_set_v001` est fige au niveau `piece_event`, stratifie
par `source_class`, exclu du replay, du train et de la calibration.

Composition cible a verifier sur les `piece_events` disponibles :

| Source class | Defectueux | Conformes |
|---|---:|---:|
| `Casting_class1` | 5 | 20 |
| `Casting_class2` | 8 | 20 |
| `Casting_class3` | 6 | 20 |
| **Total** | **19** | **60** |

Cette cible doit etre controlee contre les donnees reelles avant generation
definitive. Si une classe n'a pas assez de defectueux pour une mesure robuste, la
gate de cette classe est marquee `insufficient_evidence`, pas `passed`.

La gate `recall defaut == 1.0` est conservee, mais elle n'est interpretable que
si l'effectif minimal par classe est atteint.

### 3. ROI fige et `defect_coverage`

Proposition : `defect_coverage` est une gate de recevabilite du perimetre, pas
une gate Feature-AE.

Avant d'integrer une `source_class` dans un scenario, la couverture des defauts
GT par la ROI figee est mesuree. Si `defect_coverage < 0.95`, la classe est
declaree hors perimetre MVP ou placee en revue obligatoire. Le projet ne pretend
pas couvrir une classe que le ROI fige decoupe mal.

### 4. API, inference et services Docker

Proposition : vraie API et services Docker separes.

`iqa-api` est un vrai service FastAPI applicatif. Il expose les contrats metier
et admin : `/health`, `/predict`, `/feedback`, `/model/version`,
`/replay-scenarios`, `/metrics`, `/admin/reload-model`.

`iqa-inference` est un service Docker separe responsable de l'inference PyTorch
GPU : ROI segmenter fige, teacher ResNet18 fige, Feature-AE actif, score,
heatmap et decision Vert/Orange/Rouge.

`iqa-trainer`, `iqa-ingestion`, `iqa-replay` et `iqa-monitoring` sont des
services ou images batch separes, appeles par Airflow selon le pipeline. Cette
separation respecte la logique microservices du projet MLOps, facilite la
scalabilite legere et evite de transformer l'API en conteneur monolithique.

Pour eviter la contention GPU, Airflow utilise un pool ou verrou GPU avec
`max_active_tasks=1`. L'inference reste prioritaire pendant la demonstration ;
l'entrainement lourd doit etre controle ou suspendu si necessaire.

### 5. PostgreSQL

Proposition : une instance PostgreSQL, trois bases logiques separees.

- `iqa_metadata` : faits metier, predictions, feedback, lots, versions, incidents.
- `mlflow` : backend MLflow.
- `airflow` : metadata Airflow.

Chaque base utilise un utilisateur dedie. La sauvegarde prioritaire concerne
`iqa_metadata`. Airflow et MLflow sont reconstructibles ou restaurables
separement.

### 6. Source de verite promotion

Proposition : MLflow Registry est la source de verite de la promotion.

MinIO/S3 stocke les artefacts. L'etat `candidate`, `test`, `prod` ou `archived`
vit dans MLflow. `/admin/reload-model` lit MLflow pour connaitre la version
`prod`, puis charge l'artefact correspondant depuis MinIO.

Une copie S3 seule ne vaut jamais promotion.

### 7. Interface Sophie et oracle GT

Proposition MVP : pas de feedback humain operationnel.

L'interface Sophie est une vitrine fonctionnelle de revue : visualisation de la
piece, score, heatmap, decision proposee et parcours cible. Le workflow automatise
utilise l'oracle GT afin de rendre la demonstration reproductible.

Si `human_sophie` est active plus tard, la priorite humaine vaut pour la decision
affichee et l'experience metier, mais jamais pour contaminer le train. Le GT reste
souverain pour l'eligibilite au dataset d'entrainement.

### 8. Declenchement lifecycle

Proposition : monitoring periodique court avec condition donnees.

Airflow `iqa_monitoring` tourne sur cron court et evalue les conditions :
nouvelles pieces conformes validees, drift, seuils et gates. Si la condition est
vraie, il declenche ou arme `iqa_lifecycle`.

Pour la soutenance, un declenchement manuel reste possible, mais il doit reposer
sur une condition donnees documentee.

Formule de reference : le cron evalue, la donnee autorise, le commit ne declenche
jamais.

### 9. Baseline drift

Proposition : distinguer deux regimes de drift.

- Production naturelle : drift inattendu, baseline = bootstrap + lots recents
  valides.
- Drift controle : extension de domaine attendue class2/class3. La demonstration
  montre une boucle gouvernee : detection, validation good-only, candidat, gates,
  promotion ou rejet.

La baseline PSI/KS est versionnee comme artefact lie au `dataset_version`.

### 10. MLflow et `scenario_id`

Proposition : registered models MLflow separes par scenario.

Noms proposes :

- `feature_ae__production_replay_natural`
- `feature_ae__drift_domain_extension`

Chaque scenario a son propre stage `prod`. `/predict` et `/admin/reload-model`
prennent ou deduisent `scenario_id`. Le bootstrap V0 est l'ancetre commun, mais
les lignees divergent ensuite.

### 11. Ordre de sacrifice

Proposition : priorite a la boucle MLOps gouvernee, pas a la performance ML
parfaite.

Ordre de priorite :

1. Intouchable : scenario naturel de bout en bout.
2. Protege : tracabilite, MLflow, promotion ou rejet, reload.
3. Sacrifiable : qualite reelle du Feature-AE si les gates rejettent le candidat.
4. Sacrifiable ensuite : drift avance, incidents complexes, dashboards riches.

Message soutenance : si le modele est imparfait mais rejete par les gates, le
systeme MLOps fonctionne.

### 12. Oracle GT sans masque

Proposition : oracle deterministe par contrat dataset.

- Masque GT non vide : defaut.
- Masque GT vide : conforme.
- Pas de masque GT sur `train/good` : conforme par contrat.
- Defauts confirmes : jamais dans le train normal.

Le replay melange des pieces `train/good` conformes par contrat et des pieces test
defectueuses avec masque GT, en excluant `validation_set_v001`.

### 13. Reverse proxy

Proposition : Nginx.

Traefik est hors besoin MVP. Le routage est statique, sur station unique, avec des
services connus et des sous-chemins fixes. Nginx est donc plus simple et coherent
avec l'arborescence `deploy/nginx/`.

Les formulations `Nginx ou Traefik` doivent etre supprimees des documentations.
