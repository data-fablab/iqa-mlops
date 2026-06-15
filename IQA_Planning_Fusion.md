# IQA PLANNING CONSOLIDÉ — PHASE 1 + PHASE 2

## Phase 1 — Fondations, arbitrages et tracer bullet

### Métadonnées
- Document : IQA/IQA MLOps Phase 1 - Fondations, arbitrages et tracer bullet
- Auteur : Equipe IQA
- Version : 4.0
- Dernière modification : 12/06/2026

## Jalon

| Référence | Type | Tâche | Acteur |
|------------|------|--------|--------|
| IQA1_JAL01 | JAL | Décisions Q2 à Q14 validées : piece_event, validation_set, ROI, GPU, PostgreSQL, MLflow, GT, Airflow, drift, Nginx | All |

## Bloc Adrien

- IQA1_ADR01 — Acter piece_event comme unité atomique de split, isolation, label et reporting
- IQA1_ADR02 — Vérifier casting_images_inventory.csv : chemins, labels, masques GT, sha256, 2183 images
- IQA1_ADR03 — Vérifier casting_piece_events.csv : 962 piece_events, 40 défectueux
- IQA1_ADR04 — Définir composition validation_set_v001 par piece_event
- IQA1_ADR05 — Garantir invariant bootstrap, replay et validation_set sans chevauchement
- IQA1_ADR06 — Intégrer bootstrap V0 : 50 piece_events Casting_class1 good
- IQA1_ADR07 — Intégrer plans production_replay_natural et drift_domain_extension avec scenario_id
- IQA1_ADR08 — Initialiser DVC Phase 1 sur raw, metadata, validation et model_datasets
- IQA1_ADR09 — Créer tests data Phase 1 : volumes, sha256, labels, masques, no overlap
- IQA1_ADR10 — Documenter data assumptions Phase 1

## Bloc Natacha

- IQA1_NAT01 — Définir threat model MVP : feedback poisoning, reload abusif, prediction_id invalide
- IQA1_NAT02 — Créer schémas Pydantic prediction, piece_event, feedback, incident, model_version, scenario, reload
- IQA1_NAT03 — Créer endpoints /health, /model/version, /replay-scenarios et /metrics
- IQA1_NAT04 — Créer /piece-events/{event_id}/predict avec statut Vert Orange Rouge + audit
- IQA1_NAT05 — Créer /feedback avec règle prediction_id valide et non clôturée
- IQA1_NAT06 — Acter human_sophie prioritaire affiché mais GT souverain pour eligibility_train
- IQA1_NAT07 — Bloquer feedback invalide : source inconnue, statut interdit, prédiction absente/rejouée
- IQA1_NAT08 — Préparer /admin/reload-model protégé par token admin + log
- IQA1_NAT09 — Exposer métriques sécurité IA : feedback_conflict, ai_security_incident, unsafe_train_blocked
- IQA1_NAT10 — Créer tests contrats et sécurité

## Bloc Ken

- IQA1_KEN01 — Définir contrat modèle : entrée piece_event, sorties score, heatmap_uri, ROI status, statut
- IQA1_KEN02 — Acter ROI figé et defect_coverage comme gate de recevabilité par source_class
- IQA1_KEN03 — Créer wrappers ROI segmenter, teacher ResNet18 et Feature AE
- IQA1_KEN04 — Créer predict_image minimal : score, ROI status, heatmap placeholder, latency_ms
- IQA1_KEN05 — Créer predict_piece minimal : agrégation multi vues Vert Orange Rouge
- IQA1_KEN06 — Créer le module de construction des datasets candidats avec règles de sécurité IA : good only, ROI ok, aucun défaut, aucun validation_set
- IQA1_KEN07 — Acter MLflow Registry source de vérité et MinIO stockage artefacts
- IQA1_KEN08 — Créer registry skeleton candidate, test, prod, archived par scenario_id
- IQA1_KEN09 — Créer DAG IQA_lifecycle importable avec placeholders
- IQA1_KEN10 — Créer tests modèle et registry

## Bloc Missy

- IQA1_MIS01 — Créer arborescence repo : docs, configs, src/IQA, airflow, deploy, tests, reports, models
- IQA1_MIS02 — Configurer uv, pyproject.toml, .gitignore et dépendances
- IQA1_MIS03 — Configurer PostgreSQL : un conteneur, trois bases
- IQA1_MIS04 — Créer docker-compose minimal
- IQA1_MIS05 — Créer minio-init avec buckets
- IQA1_MIS06 — Configurer Airflow LocalExecutor, pool GPU max_active_tasks 1
- IQA1_MIS07 — Acter Nginx reverse proxy et supprimer Traefik
- IQA1_MIS08 — Créer Streamlit placeholder : lots, modèle actif, statut pièce, lien feedback
- IQA1_MIS09 — Créer Makefile et CI minimale
- IQA1_MIS10 — Créer runbook Phase 1

## Intégration finale
- IQA1_JAL02 — revue croisée : arbitrages, tests data, API, modèle et infra validés

---

# Phase 2 — Microservices, suivi, versioning et boucle modèle réaliste

## Métadonnées
- Version : 4.0
- Dernière modification : 12/06/2026

## Carry-over

- IQA3_CAR01 — dvc init + remote MinIO + suivi raw/metadata/validation/model_datasets
- IQA3_CAR02 — .github/workflows : lint + pytest + import DAGs + smoke
- IQA3_CAR03 — UI Streamlit placeholder (lots, modèle actif, statut pièce, lien feedback)
- IQA3_CAR04 — Threat model + gardes feedback invalide + métriques sécurité IA

## Jalon
- IQA2_JAL01 — interfaces data, API, modèle, registry, storage et infra figées

## Bloc Adrien

- IQA2_ADR01 à IQA2_ADR19 (liste complète issue du document Phase 2)
- Brancher production_replay_natural sur la vraie API
- Créer lot scheduler naturel
- Implémenter reset et isolation par scenario_id
- Finaliser drift_domain_extension
- Implémenter validation_set_v001
- Ajouter raw_dataset_id, manifest_id, replay_id, validation_id, dataset_version
- Intégrer règle oracle
- Valider dvc push/pull MinIO
- Créer dvc.yaml
- Créer contrats metadata
- Construire model_dataset feature_ae_good_v002
- Construire model_dataset feature_ae_good_v003
- Brancher feedback conforme_validé
- Gérer quarantaine défaut confirmé
- Ajouter tests data
- Ajouter test reproductibilité
- Durcir DAGs ingestion/replay
- Ajouter liens dataset_version dans MLflow
- Livrer documentation data_contracts.md, dvc_versioning.md, validation_set.md, replay_runbook.md

## Bloc Natacha

- IQA2_NAT01 à IQA2_NAT17
- API IQA_metadata
- Finalisation /predict
- scenario_id obligatoire
- feedback human_sophie
- GT souverain
- anti feedback poisoning
- blocage eligibility_train
- sécurisation reload-model
- audit trail
- métriques filtrées
- métriques sécurité IA
- schémas Pydantic Phase 2
- erreurs API normalisées
- incidents API
- tests contrats API
- tests sécurité IA
- documentation gouvernance

## Bloc Ken

- IQA2_KEN01 — Finaliser interface Feature AE candidat : train, eval, save, load, predict
- IQA2_KEN02 — Finaliser candidate_builder good only
- IQA2_KEN03 — Implémenter train_feature_ae_v2 et v3 candidats versionnés
- IQA2_KEN04 — Implémenter evaluate_feature_ae sur validation_set_v001
- IQA2_KEN05 — Mesurer defect_coverage par source_class et bloquer si couverture < 0.95
- IQA2_KEN06 — Logger runs MLflow
- IQA2_KEN07 — Créer registered models séparés par scenario_id
- IQA2_KEN08 — Implémenter gates
- IQA2_KEN09 — Implémenter promotion MLflow
- IQA2_KEN10 — Implémenter rollback
- IQA2_KEN11 — Finaliser DAG IQA_lifecycle
- IQA2_KEN12 — Ajouter paramètres Airflow
- IQA2_KEN13 — Créer baseline drift versionnée
- IQA2_KEN14 — Brancher model_loader sur MLflow prod
- IQA2_KEN15 — Créer tests ML
- IQA2_KEN16 — Créer tests promotion
- IQA2_KEN17 — Documenter model_lifecycle.md, gates.md, mlflow_registry.md, drift_regimes.md, rollback.md

## Bloc Missy

- IQA2_MIS01 à IQA2_MIS17
- docker-compose complet
- PostgreSQL
- MinIO
- MLflow
- Airflow
- Monitoring
- GPU lock
- Nginx
- Prometheus
- Grafana
- Streamlit
- CI
- Smoke tests
- Documentation
- Démo

## Jalons Phase 2

- IQA2_JAL02 — Replay naturel branché sur API
- IQA2_JAL03 — Première boucle de promotion testable
- IQA2_JAL04 — Phase 2 livrée
