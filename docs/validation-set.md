# Validation Set IQA

`validation_set_v001` est le set fige de controle au niveau `piece_event`.
Il sert a verifier les contrats data, la regle oracle GT et les gates de
qualite sans alimenter le train normal.

## Contrat

| Champ | Valeur attendue |
| --- | --- |
| `validation_set_id` | `validation_set_v001` |
| `dataset_version` | `validation_set_v001` |
| `validation_id` | `validation_set_v001` |
| `piece_event_id` | egal a `event_id` |

Repartition actuelle :

| `source_class` | Pieces |
| --- | ---: |
| `Casting_class1` | 5 |
| `Casting_class2` | 8 |
| `Casting_class3` | 7 |

## Regle oracle

La regle souveraine reste :

```text
masque GT absent ou vide -> oracle_verdict=conforme
masque GT non vide       -> oracle_verdict=defective
```

Les valeurs machine restent ASCII. `human_sophie` ne rend jamais un exemple
eligible au train normal.

## Verification

```bash
uv run --extra cpu pytest -q tests/data tests/feedback/test_oracle_gt_contract.py
```

Le set validation doit rester disjoint de bootstrap, calibration et replay.
