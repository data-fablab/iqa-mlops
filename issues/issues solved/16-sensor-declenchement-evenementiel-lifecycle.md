# 16 - Sensor de declenchement evenementiel du lifecycle

Type : AFK

## What to build

Automatiser le declenchement de `iqa_lifecycle` par evenement donnees, conformement
aux consequences de l'ADR 0002 : drift confirme, lot complet, ou volume suffisant de
conformes valides. Le sensor (ou un DAG declencheur) observe l'etat
(PostgreSQL/monitoring) et lance le lifecycle sans intervention humaine.

## Acceptance criteria

- [x] Sensor/declencheur observe les signaux (drift confirme, lot complet, volume de conformes)
  (DAG `iqa_lifecycle_trigger` `@hourly` ; tache `evaluate_decision` evalue la regle
  d'evenement `evaluate_lifecycle_signal` **dans le conteneur** image data via
  `iqa-run-lifecycle-decision`)
- [x] `iqa_lifecycle` est declenche automatiquement quand une condition est remplie
  (`gate_on_decision` `ShortCircuitOperator` lit la decision conteneur ; si
  `trigger_lifecycle` -> `trigger_lifecycle` `TriggerDagRunOperator` lance `iqa_lifecycle`)
- [x] Aucun declenchement manuel requis pour la boucle nominale
  (planifie `@hourly` ; le short-circuit n'arme le trigger que sur condition remplie,
  rien ne part sur le chemin nominal d'attente)
- [x] Les conditions/seuils sont configurables (configs existantes)
  (params DAG : `scenario_id`, `conforming_validated_count`, `drift_confirmed`,
  `roi_fail_rate`, `target_stage` ; seuil `min_natural_conforming` = regle existante)
- [x] Import DagBag vert
  (meme garde `try/except ImportError -> dag=None` que les autres DAGs ; test source
  + contrat de chaine `evaluate_decision >> gate_on_decision >> trigger_lifecycle`)

## Note d'architecture

Decision metier **dans le conteneur** (ADR 0008) ; le declenchement est de la glue
Airflow native (`ShortCircuitOperator` + `TriggerDagRunOperator`) qui n'importe jamais
`iqa` (seulement `json`). La version candidate du dataset est rederivee par la tache
`lifecycle_decision` du DAG cible, donc le trigger ne relaie que les params du signal.
L'observation reelle de l'etat du store (events PostgreSQL / monitoring) est le data
plane, isole dans les sœurs runtime (18 / 23). Cette tranche cable le trigger, pas l'I/O.

## Blocked by

- 11 - Lifecycle (4/4) : promotion + reload
