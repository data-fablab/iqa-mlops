# ADR 0001 - Validation set fige hors replay

## Statut

Accepte.

## Contexte

Le dataset contient peu de `piece_events` defectueux. Si tous les defauts sont utilises dans les replays et dans la calibration, les gates de promotion mesurent en partie des donnees deja vues par le systeme.

## Decision

Creer `validation_set_v001` avant tout replay :
- stratifie par `source_class` ;
- compose de pieces good et defectueuses ;
- exclu du bootstrap ;
- exclu des replays ;
- exclu de la calibration ;
- exclu des datasets candidats Feature-AE.

Les gates de promotion et go/no-go sont calculees sur ce jeu.

La decision ADR 0005 ajoute ensuite `calibration_set_v001`. L'invariant complet
devient donc :

```text
bootstrap ∩ calibration ∩ replay ∩ validation = vide
```

## Consequences

- Les plans de replay doivent exclure `validation_set_v001` et `calibration_set_v001`.
- L'evaluation devient credible pour le jury.
- Un test automatique doit garantir la disjonction entre bootstrap, calibration, replay et validation.
