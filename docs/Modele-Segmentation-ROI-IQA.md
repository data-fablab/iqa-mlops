# Modele Segmentation ROI IQA

## 1. Role

Le segmenteur ROI isole la surface fonctionnelle de la piece. Il ne detecte pas les defauts.

Dans le MVP IQA, il est un composant fige utilise en amont du Feature-AE :

```text
image piece
-> ROI segmenter fige
-> masque surface fonctionnelle
-> controle qualite ROI
-> tuiles Feature-AE
-> score anomalie sur surface utile
```

Le ROI segmenter stabilise le pipeline : il limite le scoring aux zones metier pertinentes et evite que le Feature-AE apprenne ou score le fond.

## 2. Architecture retenue

Architecture unique conservee :

```text
model_type = functional_unet_resnet18_det1_context2b
classe     = FunctionalSurfaceUNetResNet18Det1Context2B
```

API publique attendue :

```python
ROI_SEGMENTER_MODEL_TYPE
FunctionalSurfaceUNetResNet18Det1Context2B
build_segmentation_model
load_roi_segmenter
load_roi_segmenter_checkpoint
mask_logits_from_output
replace_segmentation_head
surface_probability_from_logits
```

Les anciennes architectures de segmentation, variantes experimentales et exports training ne font pas partie du contrat public du MVP.

## 3. Entrees runtime

Le modele recoit une image locale et peut recevoir un contexte global :

```python
output = model(
    images,
    global_image=global_image,
    crop_box_mask=crop_box_mask,
)
```

Entrees :

- `images` : crop ou vue locale de la piece ;
- `global_image` : image globale ou contexte de la piece ;
- `crop_box_mask` : masque indiquant la zone du crop dans le contexte global.

Si `global_image` ou `crop_box_mask` sont absents, le runtime fournit des valeurs par defaut compatibles.

## 4. Sorties runtime

Le modele retourne un dictionnaire :

```python
{
    "mask_logits": ...,
    "objectness_logits": ...,
    "bbox": ...,
}
```

La sortie utile au pipeline IQA est `mask_logits`.

Le helper officiel est :

```python
mask_logits = mask_logits_from_output(output)
```

Le masque binaire ROI est derive des logits dans les modules de runtime ou
d'adaptation ROI. Pour un checkpoint multiclasses, la sortie officielle utilise
l'`argmax` semantique et conserve la classe `functional_surface`.

## 5. Utilisation par le Feature-AE

Le masque ROI est utilise pour :

- filtrer les tuiles avec trop peu de surface fonctionnelle ;
- ponderer la loss Feature-AE ;
- attenuer le background ;
- limiter le score anomalie a la surface utile ;
- produire les indicateurs ROI-ok / warning / fail.

Le masque ROI ne doit jamais etre interprete comme un masque de defaut.

Separation stricte :

```text
ROI mask       -> surface fonctionnelle, produit par le segmenteur
GT defect mask -> defauts oracle, utilise apres prediction seulement
```

## 6. Controle qualite ROI

Le pipeline doit pouvoir marquer une prediction ROI comme :

```text
ok
warning
fail
```

Cas a surveiller :

- ROI vide ou quasi vide ;
- ROI trop grande ;
- ROI fragmentee ;
- ratio de surface incompatible ;
- zone fonctionnelle decalee ;
- image inutilisable.

Regle training Feature-AE :

```text
train Feature-AE = good-only + ROI-ok uniquement
```

Les cas `warning` et `fail` restent hors train. Ils peuvent alimenter le monitoring ou les incidents rejouables.

## 7. Statut lifecycle

Dans le MVP, le ROI segmenter est fige.

Il peut etre charge pour l'inference et pour preparer les masques ROI, mais il n'est pas reentraine automatiquement par Airflow.

Le lifecycle modele automatise concerne uniquement le Feature-AE.

Implications :

- pas de DAG de retraining ROI dans le MVP ;
- pas de promotion automatique ROI ;
- version ROI tracee dans les metadata ;
- `roi_model_version` obligatoire pour les candidats replay Feature-AE.

## 8. Stockage et tracabilite

Les predictions ROI peuvent etre stockees comme artefacts ou references :

```text
MinIO       -> masques ROI, overlays, exports lourds
PostgreSQL -> URI, version modele, statut ROI, piece_event_id
```

Le stockage cible manipule des URI. Les fichiers lourds ne doivent pas etre stockes directement dans PostgreSQL.

Bucket cible :

```text
s3://iqa-roi-masks
```

## 9. Bootstrap et cycle normal

Le bootstrap Feature-AE utilise le meme contrat logique que le cycle normal :

```text
image_uri -> roi_mask_uri -> roi_quality_status -> Feature-AE
```

En Phase 1, les masques ROI bootstrap peuvent etre generes localement sous :

```text
data/processed/roi/bootstrap_v001/
```

Ce dossier reste hors Git. En cible production/replay, les masques sont stockes
dans `s3://iqa-roi-masks` et PostgreSQL conserve les URI et faits associes.

Commande bootstrap serveur :

```bash
uv run --extra cu128 --extra data iqa-generate-bootstrap-roi \
  --manifest data/metadata/feature_ae_bootstrap_events.csv \
  --image-root data/raw/hss-iad \
  --checkpoint models/roi_segmenter_v001_fixed/checkpoint.pt \
  --output-dir data/processed/roi/bootstrap_v001 \
  --roi-model-version roi_segmenter_v001_fixed \
  --device cuda
```

## 10. Non-objectifs MVP

Hors perimetre MVP :

- comparer plusieurs architectures ROI ;
- exposer un training ROI via l'API ;
- reentrainer ROI automatiquement ;
- utiliser le masque ROI comme verite terrain defaut ;
- faire porter au segmenteur la decision Vert / Orange / Rouge.

La decision qualite finale est produite apres inference Feature-AE et application des regles de scoring.
