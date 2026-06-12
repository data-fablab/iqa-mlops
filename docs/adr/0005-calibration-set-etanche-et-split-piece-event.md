# ADR 0005 - Calibration set etanche et split `piece_event`

## Statut

Accepte.

## Contexte

Le dataset Casting contient des pieces multi-vues. Un split au niveau image peut faire fuiter une vue conforme d'une piece defectueuse dans le train good-only. La calibration des seuils Feature-AE ne doit pas consommer le validation set.

## Decision

Le `piece_event` est l'unite atomique de split, replay, feedback, validation, calibration et train.

On ajoute `calibration_set_v001` :

- good-only ;
- fige avant replay ;
- hors bootstrap ;
- hors train ;
- hors replay ;
- hors `validation_set_v001` ;
- reserve a la calibration des seuils Feature-AE.

Invariant officiel :

```text
bootstrap ∩ calibration ∩ replay ∩ validation = vide
```

Les contrats replay portent deux horloges :

- `event_time` : temps simule du flux historique rejoue ;
- `recorded_at` : temps systeme d'enregistrement.

`is_simulated` est derive de la source : `historical_replay` vaut `true`, `production_ingest` vaut `false`.

## Consequences

Le train Feature-AE reste strictement good-only, ROI-ok, hors validation, hors calibration et hors incident. Les images heritent toujours du split de leur `piece_event`.
