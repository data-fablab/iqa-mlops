# 16 - Sensor de declenchement evenementiel du lifecycle

Type : AFK

## What to build

Automatiser le declenchement de `iqa_lifecycle` par evenement donnees, conformement
aux consequences de l'ADR 0002 : drift confirme, lot complet, ou volume suffisant de
conformes valides. Le sensor (ou un DAG declencheur) observe l'etat
(PostgreSQL/monitoring) et lance le lifecycle sans intervention humaine.

## Acceptance criteria

- [ ] Sensor/declencheur observe les signaux (drift confirme, lot complet, volume de conformes)
- [ ] `iqa_lifecycle` est declenche automatiquement quand une condition est remplie
- [ ] Aucun declenchement manuel requis pour la boucle nominale
- [ ] Les conditions/seuils sont configurables (configs existantes)
- [ ] Import DagBag vert

## Blocked by

- 11 - Lifecycle (4/4) : promotion + reload
