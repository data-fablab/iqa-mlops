# IQA Phase 3 Storytelling Soutenance

## Objectif

La demonstration presente IQA comme une plateforme industrielle complete, pas
comme une simple API de Machine Learning.

Le fil rouge est :

```text
objectifs metier -> architecture -> scenarios MLOps -> demo live -> interfaces,
securite et perspectives
```

## Roles de demonstration

| Role | Point de vue |
| --- | --- |
| Inspecteur Qualite | revue visuelle des pieces et decisions qualite |
| Responsable Production | pilotage des lots, statuts et actions atelier |
| Data Lineage / MLOps | tracabilite donnees, modeles, gates et registry |
| Securite / Gouvernance | protection des acces, auditabilite, gateway |

## Message d'ouverture

IQA assiste le controle visuel de pieces industrielles. La plateforme detecte
les anomalies, explique les decisions, conserve la tracabilite des donnees et
des modeles, automatise le cycle de vie MLOps et protege les acces sensibles.

L'objectif n'est pas de remplacer l'operateur qualite. L'objectif est de fournir
une aide fiable, explicable et gouvernee.

## Partie 1 - Objectifs metier et architecture

Message : on part d'un besoin industriel simple, mais on construit une boucle
MLOps complete.

Points a montrer :

- controle qualite visuel des pieces ;
- decisions lisibles Vert / Orange / Rouge ;
- separation API, inference, stockage, orchestration et monitoring ;
- MLflow comme source de verite des modeles ;
- MinIO pour les artefacts lourds ;
- Airflow pour rendre le lifecycle observable.

## Partie 2 - Scenarios MLOps

Deux scenarios structurent la soutenance.

### Scenario 1 - Lifecycle Piece B

Scenario technique :

```text
production_replay_natural_piece_b_full
```

Le modele part d'un bootstrap faible. Les cycles representent la premiere
semaine de mise en production. Les train sets grandissent, les epochs avancent,
et les gates promeuvent uniquement les checkpoints utiles.

Message : progression controlee, promotions separees classification/localisation
et refus des candidats qui n'ameliorent pas assez.

### Scenario 2 - Drift P4 et correction ciblee

Scenario technique :

```text
production_replay_natural_piece_b_to_piece_a_p4_drift
```

Piece B est stable, puis P4 arrive progressivement. La nouveaute ROI sort du
referentiel Piece B, le drift est confirme, Airflow declenche le DAG correctif
dedie, puis la gate promeut les deux roles si la correction reduit le risque P4.

Message : le systeme observe, detecte, corrige et trace la decision.

## Partie 3 - Demo live

Ordre recommande :

1. Airflow : ouvrir les DAGs `iqa_lifecycle`, `iqa_drift_piece_a_p4` et
   `iqa_drift_correction_lifecycle`.
2. MLflow : montrer les runs, les tags de scenario, les metriques et le registry.
3. MinIO : montrer les artefacts et checkpoints.
4. Grafana Lifecycle : montrer progression, train set, epochs et promotions.
5. Grafana Drift P4 : montrer chronologie, preuve ROI, DAG correctif, gate et
   modeles promus.

## Partie 4 - Interfaces, securite et perspectives

Interfaces Streamlit :

- Interface Inspecteur Qualite : revue visuelle display-only.
- Dashboard Responsable Production : lots, conformite, actions atelier.
- Data Lineage : run id, cycles, gates, promotions, MLflow et registry.

Securite :

- Kong/reverse proxy protege les entrees ;
- FastAPI conserve les controles metier ;
- les routes sensibles restent separees ;
- les artefacts lourds restent dans MinIO ;
- les modeles actifs sont gouvernes par MLflow Registry.

Perspectives :

- brancher un vrai MES/camera ;
- persister davantage de faits metier en PostgreSQL ;
- renforcer l'authentification et les roles ;
- industrialiser les alertes et le rollback ;
- etendre les scenarios de drift.

## Conclusion

IQA relie le controle terrain, le pilotage production, la gouvernance des
modeles et la securite. La valeur n'est pas seulement la prediction : c'est la
capacite a expliquer, corriger, tracer et reprendre la main quand le contexte
industriel change.
