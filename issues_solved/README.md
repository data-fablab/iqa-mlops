# Issues — Tâches Ken (Phases 1 & 2)

Issues locales dérivées de [`tâche_s1_s2.md`](../tâche_s1_s2.md). Une issue par tâche, en tranches verticales (tracer bullets) avec critères d'acceptation et dépendances.

## Phase 1 — Modèle, MLflow, Gates, Registry, Lifecycle

| # | Titre | Type | Bloqué par |
|---|-------|------|-----------|
| [IQA1_KEN01](IQA1_KEN01-contrat-modele.md) | Contrat modèle | AFK | — |
| [IQA1_KEN02](IQA1_KEN02-roi-gate-defect-coverage.md) | ROI figé + gate defect_coverage | HITL | — |
| [IQA1_KEN03](IQA1_KEN03-wrappers-modeles.md) | Wrappers modèles | AFK | KEN01 |
| [IQA1_KEN04](IQA1_KEN04-predict-image.md) | predict_image minimal | AFK | KEN01, KEN03 |
| [IQA1_KEN05](IQA1_KEN05-predict-piece.md) | predict_piece minimal | AFK | KEN04 |
| [IQA1_KEN06](IQA1_KEN06-candidate-builder.md) | Datasets candidats + sécurité IA | AFK | — |
| [IQA1_KEN07](IQA1_KEN07-mlflow-minio-source-verite.md) | MLflow source de vérité + MinIO | HITL | — |
| [IQA1_KEN08](IQA1_KEN08-registry-skeleton.md) | Registry skeleton par scenario_id | AFK | KEN07 |
| [IQA1_KEN09](IQA1_KEN09-dag-lifecycle-skeleton.md) | DAG IQA_lifecycle importable | AFK | — |
| [IQA1_KEN10](IQA1_KEN10-tests-modele-registry.md) | Tests modèle et registry | AFK | KEN01, KEN08 |

## Phase 2 — Feature-AE, Gates, Promotion, Rollback

| # | Titre | Type | Bloqué par |
|---|-------|------|-----------|
| [IQA2_KEN01](IQA2_KEN01-interface-feature-ae.md) | Interface Feature AE candidat | AFK | IQA1_KEN03 |
| [IQA2_KEN02](IQA2_KEN02-candidate-builder-good-only.md) | candidate_builder good only | AFK | IQA1_KEN06 |
| [IQA2_KEN03](IQA2_KEN03-train-feature-ae-versionne.md) | train_feature_ae v2/v3 versionnés | AFK | KEN01, KEN02 |
| [IQA2_KEN04](IQA2_KEN04-evaluate-feature-ae.md) | evaluate_feature_ae | AFK | KEN03 |
| [IQA2_KEN05](IQA2_KEN05-defect-coverage-gate.md) | defect_coverage gate < 0.95 | AFK | IQA1_KEN02, KEN04 |
| [IQA2_KEN06](IQA2_KEN06-mlflow-logging.md) | Logger runs MLflow complets | AFK | IQA1_KEN07, KEN04 |
| [IQA2_KEN07](IQA2_KEN07-registered-models-par-scenario.md) | Registered models par scenario_id | AFK | IQA1_KEN08, KEN06 |
| [IQA2_KEN08](IQA2_KEN08-gates.md) | Gates de promotion | AFK | KEN05, KEN07 |
| [IQA2_KEN09](IQA2_KEN09-promotion.md) | Promotion MLflow + MinIO | AFK | KEN08 |
| [IQA2_KEN10](IQA2_KEN10-rollback.md) | Rollback | AFK | KEN09 |
| [IQA2_KEN11](IQA2_KEN11-dag-lifecycle-complet.md) | Finaliser DAG IQA_lifecycle | AFK | IQA1_KEN09, KEN02/03/04/06/09/14 |
| [IQA2_KEN12](IQA2_KEN12-airflow-params-replay.md) | Params Airflow rejeu (naturel/drift) | AFK | KEN11 |
| [IQA2_KEN13](IQA2_KEN13-baseline-drift.md) | Baseline drift versionnée | AFK | KEN04 |
| [IQA2_KEN14](IQA2_KEN14-model-loader.md) | Brancher model_loader | AFK | KEN07, KEN09 |
| [IQA2_KEN15](IQA2_KEN15-tests-ml.md) | Tests ML sécurité IA | AFK | KEN02, KEN08 |
| [IQA2_KEN16](IQA2_KEN16-tests-promotion.md) | Tests promotion / rollback | AFK | KEN09, KEN10 |
| [IQA2_KEN17](IQA2_KEN17-documentation.md) | Documentation lifecycle | AFK | KEN11/12/13/14 |
