# IQA Phase 3 Storytelling Soutenance

## Objectif

La démonstration présente IQA comme une plateforme industrielle complète et non comme un simple modèle de Machine Learning.

Le parcours suit trois personas complémentaires :

| Persona | Point de vue |
| --- | --- |
| Sophie | contrôle qualité terrain |
| Marc | pilotage qualité industrielle |
| Laurent | cybersécurité, gouvernance et traçabilité |

La démonstration montre que la plateforme répond aux besoins de production, de supervision, de traçabilité et de sécurité.

## Message d’ouverture

IQA assiste le contrôle visuel de pièces industrielles.

La plateforme détecte les anomalies, explique les décisions, conserve la traçabilité des données et des modèles, automatise le cycle de vie MLOps et protège les accès sensibles.

L’objectif n’est pas de remplacer l’opératrice qualité.

L’objectif est de lui fournir une aide fiable, explicable et gouvernée.

## Architecture finale

Utilisateur

↓  

Kong Gateway

↓  

Streamlit, FastAPI iqa API, MLflow, Grafana, Airflow et MinIO

FastAPI échange avec le service d’inférence, PostgreSQL, MLflow Registry, MinIO et Prometheus.

Airflow orchestre l’ingestion, le replay, l’entraînement, l’évaluation, les gates, la promotion et le reload.

## Partie 1 : Sophie, contrôle qualité terrain

### Besoin métier

Sophie inspecte les pièces et doit identifier rapidement les cas conformes, suspects ou défectueux.

Elle doit comprendre la décision du système sans perdre son rôle de contrôle humain.

### Démonstration

1. Ouvrir l’interface Streamlit de Sophie.
2. Sélectionner une pièce ou une prédiction.
3. Afficher la décision Vert, Orange ou Rouge.
4. Afficher le score et la heatmap.
5. Montrer les informations de traçabilité de la pièce.
6. Montrer une divergence entre la prédiction et l’oracle GT.
7. Montrer que le retour Sophie reste une information de revue et ne décide pas seul de l’entraînement.

### Message oral

Sophie reste au centre du processus qualité.

La plateforme attire son attention sur les pièces les plus importantes à revoir.

La décision est explicable grâce au score, à la heatmap et à la traçabilité.

Le feedback humain est visible, mais l’oracle GT reste souverain pour déterminer l’éligibilité à l’entraînement.

### Preuves attendues

| Preuve | Source |
| --- | --- |
| décision Vert, Orange ou Rouge | Streamlit et API |
| score et heatmap | Streamlit et MinIO |
| historique des prédictions | endpoint predictions |
| divergence | vue Sophie |
| feedback display only | contrats FastAPI |
| oracle GT souverain | règles de gouvernance |

## Partie 2 : Marc, pilotage qualité industrielle

### Besoin métier

Marc doit suivre la qualité par lot, comprendre les tendances, identifier les dérives et vérifier que le modèle actif reste fiable.

### Démonstration

1. Ouvrir le dashboard Marc.
2. Afficher les lots inspectés.
3. Montrer les volumes Vert, Orange et Rouge.
4. Montrer les taux Orange et Rouge.
5. Montrer les divergences et feedbacks fermés.
6. Ouvrir le suivi du cycle de vie.
7. Montrer le modèle actif dans MLflow Registry.
8. Montrer les runs MLflow et les métriques.
9. Montrer le lien avec les versions de données et les manifests.
10. Présenter la promotion et le rollback.

### Message oral

Marc ne voit pas seulement une prédiction individuelle.

Il dispose d’une vision industrielle par lot.

Il peut expliquer quelle version de données, quel run, quel modèle et quelle décision de promotion ont produit le résultat observé.

Le cycle de vie est automatisé par Airflow, mais les gates empêchent une promotion dangereuse.

### Preuves attendues

| Preuve | Source |
| --- | --- |
| résumé par lot | dashboard Marc |
| taux Vert, Orange et Rouge | API et Grafana |
| runs et métriques | MLflow |
| modèle actif | MLflow Registry |
| dataset version | DVC et manifests |
| orchestration | Airflow |
| promotion et rollback | Registry et runbooks |

## Partie 3 : Laurent, cybersécurité et gouvernance

### Besoin métier

Laurent doit vérifier que les accès sont contrôlés, que les actions sont auditables et que la plateforme sépare correctement les données, les artefacts et les modèles.

### Démonstration

1. Montrer Kong comme point d’entrée public.
2. Tester un accès sans clé et montrer le refus.
3. Tester un accès autorisé avec une clé service.
4. Tester une route admin avec une clé non autorisée.
5. Tester la route admin avec la clé admin.
6. Montrer le rate limiting.
7. Montrer les headers de sécurité.
8. Montrer les logs Kong.
9. Montrer les incidents FastAPI.
10. Montrer la séparation PostgreSQL, MinIO et MLflow.
11. Montrer les images Docker immuables et la séparation dev et prod.
12. Montrer la traçabilité complète d’une prédiction.

### Message oral

Kong protège l’entrée.

FastAPI protège la logique métier et la gouvernance IA.

PostgreSQL conserve les faits structurés.

MinIO conserve les artefacts lourds.

MLflow Registry reste la source de vérité des modèles.

DVC et les manifests assurent la traçabilité des données.

Cette séparation limite les impacts d’un incident et permet un audit de bout en bout.

### Preuves attendues

| Preuve | Source |
| --- | --- |
| accès refusé sans clé | Kong |
| accès autorisé | Kong |
| protection admin | Kong et FastAPI |
| rate limiting | Kong |
| headers sécurité | Kong |
| logs accès | Kong |
| incidents IA | FastAPI et PostgreSQL |
| artefacts lourds | MinIO |
| modèle actif | MLflow Registry |
| lineage données | DVC et manifests |
| images immuables | Docker Hub et CI |
| rollback | runbook et Registry |

## Démonstration de traçabilité complète

La démonstration suit une même pièce de bout en bout :

image → sha256 → piece_event_id → scenario_id et lot_id → dataset_version et manifest_version → model_version et MLflow run → prediction_id → feedback → incident éventuel

## Démonstration du cycle de vie MLOps

Le scénario final présente le cycle suivant :

nouveau lot → ingestion → replay → construction du dataset candidat → entraînement → évaluation → gates

Si les gates échouent, le modèle actif est conservé.

Si les gates passent, le modèle est promu dans MLflow Registry puis rechargé dans le service d’inférence.

## Bonnes pratiques à expliquer

| Bonne pratique | Justification |
| --- | --- |
| Kong en entrée | centralise les contrôles transverses |
| FastAPI comme frontière métier | conserve les règles IA et qualité |
| PostgreSQL pour les métadonnées | données structurées et auditables |
| MinIO pour les artefacts | stockage adapté aux fichiers lourds |
| MLflow Registry source de vérité | contrôle des versions actives |
| DVC pour la lineage | reproductibilité des données |
| Airflow comme orchestrateur | découple orchestration et code métier |
| tâches Airflow en conteneurs | isolation et portabilité |
| images Docker immuables | déploiement reproductible |
| gates avant promotion | réduction du risque modèle |
| oracle GT souverain | prévention du feedback poisoning |
| rollback documenté | reprise rapide après incident |

## Conclusion de la soutenance

La plateforme IQA ne se limite pas à détecter une anomalie.

Elle relie le contrôle terrain, le pilotage qualité, le cycle de vie du modèle et la gouvernance de sécurité.

Sophie comprend et revoit les décisions.

Marc pilote les lots et le cycle de vie.

Laurent contrôle les accès, les incidents et la traçabilité.

La plateforme démontre ainsi une approche MLOps industrielle, observable, reproductible et sécurisée.

## Validation IQA3 NAT07

| Exigence | Statut |
| --- | --- |
| storytelling Sophie | documenté |
| storytelling Marc | documenté |
| storytelling Laurent | documenté |
| déroulé de démonstration | documenté |
| preuves attendues | documentées |
| architecture finale Phase 3 | documentée |
| bonnes pratiques | documentées |
| conclusion soutenance | documentée |
