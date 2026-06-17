# Index Docs IQA

IQA est un MVP MLOps pour le controle visuel de pieces `Casting`.

## Documents Actifs

| Besoin | Document |
| --- | --- |
| Comprendre le produit et le scope MVP | [prd-iqa-mvp.md](prd-iqa-mvp.md) |
| Comprendre l'architecture technique | [architecture-iqa.md](architecture-iqa.md) |
| Installer et exploiter le serveur | [configuration-serveur-iqa.md](configuration-serveur-iqa.md) |
| Executer les procedures Phase 1 | [runbook-phase1-iqa.md](runbook-phase1-iqa.md) |
| Comprendre le segmenteur ROI | [modele-segmentation-roi-iqa.md](modele-segmentation-roi-iqa.md) |
| Comprendre le Feature-AE | [modele-feature-ae-iqa.md](modele-feature-ae-iqa.md) |
| Reproduire les runs ML | [reproductibilite-ml-iqa.md](reproductibilite-ml-iqa.md) |
| Suivre les decisions | [decisions-iqa.md](decisions-iqa.md) et [adr/](adr/) |
| Suivre la roadmap | [roadmap-iqa.md](roadmap-iqa.md) |
| Suivre les taches | [repartition-taches-phases-1-2.md](repartition-taches-phases-1-2.md) |
| Acceder au serveur | [acces-ssh-equipe-iqa.md](acces-ssh-equipe-iqa.md) |
| Securite et gouvernance | [security/](security/) et [governance/](governance/) |


## Phase 2 API Security And Governance

| Besoin | Document |
| --- | --- |
| Comprendre les contrats API Phase 2 | [api_contracts.md](api_contracts.md) |
| Comprendre les contrats data Phase 2 | [data-contracts.md](data-contracts.md) |
| Comprendre le versioning DVC des manifests | [dvc-versioning.md](dvc-versioning.md) |
| Comprendre le set de validation fige | [validation-set.md](validation-set.md) |
| Operer les runs replay API et Airflow | [replay-runbook.md](replay-runbook.md) |
| Comprendre les regles de feedback et d'eligibilite train | [feedback_rules.md](feedback_rules.md) |
| Comprendre la gouvernance securite IA | [ai_security_governance.md](ai_security_governance.md) |
| Comprendre la chaine d'audit et de tracabilite | [audit_trail.md](audit_trail.md) |

## Vocabulaire Minimal

Le dataset Casting sert d'historique rejoue (`historical_replay`). La production
cible utilisera `production_ingest`. Le contrat cible separe les images stockees
dans MinIO et les faits metier stockes dans PostgreSQL.

Vocabulaire stockage :

- Dataset source : historique Casting immutable, stocke dans `s3://iqa-source-datasets` et/ou versionne DVC.
- Donnees ingerees : images passees par le contrat d'ingestion, stockees dans `s3://iqa-ingested-images`, avec `piece_event`, timestamps, source et URI dans PostgreSQL en cible runtime.
- Artefacts : sorties produites par les pipelines, par exemple heatmaps, modeles, runs MLflow, rapports et datasets candidats.

Regle cible : PostgreSQL stocke les faits et les URI ; MinIO stocke les fichiers lourds.

Les checkpoints PyTorch lourds sont stockes dans MinIO, principalement `s3://iqa-models`, et references par manifests Git.
