# Modele Feature-AE IQA

## 1. Role

Le Feature-AE est le modele vivant du MVP IQA. Il detecte les anomalies sur la surface fonctionnelle deja isolee par le segmenteur ROI.

Il est le seul modele reentraine automatiquement dans la boucle MLOps :

```text
images ingerees
-> ROI segmenter fige
-> tuiles surface fonctionnelle
-> RD Feature-AE
-> score anomalie + heatmap
-> decision Vert / Orange / Rouge
-> feedback oracle GT apres prediction
-> monitoring / lifecycle / promotion MLflow
```

Le Feature-AE ne segmente pas la surface fonctionnelle. Il reconstruit des features teacher et mesure l'erreur de reconstruction.

## 2. Architecture retenue

Architecture unique conservee :

```text
model_type = reverse_distill_resnet18_dual_context_gated
classe     = ReverseDistillationGatedDualContextResNet18
teacher    = ResNetTeacherFeatures
backbone   = resnet18
layers     = layer2, layer3
```

API publique attendue :

```python
FEATURE_AE_MODEL_TYPE
DEFAULT_FEATURE_LAYERS
SUPPORTED_TEACHER_BACKBONE
TEACHER_LAYER_CHANNELS
ReverseDistillationGatedDualContextResNet18
ResNetTeacherFeatures
feature_anomaly_map
feature_reconstruction_loss
load_rd_feature_ae_gated
normalize_feature_layers
```

Les anciens aliases et factories ne font plus partie du contrat public.

## 3. Preprocessing officiel

Le mode historique `tile_256_overlap` n'est pas conserve. Il correspondait a une ancienne taille de tuiles et rendait le nom incoherent avec le champion actuel.

Mode IQA retenu :

```text
preprocessing_mode = tiled_context
image_size         = 384
context_size       = 768
tile_stride        = 192
normalization      = imagenet
sampling           = all
```

Pipeline image :

```text
ResizeLetterbox
-> ToTensor
-> Normalize ImageNet
-> tuile locale 384
-> contexte 768
```

Le meme contrat de preprocessing doit etre utilise en training et en inference. Les parametres doivent etre sauvegardes dans les metadata du checkpoint.

## 4. Separation ROI et GT defaut

Deux types de masques existent et ne doivent pas etre melanges.

### Masque ROI

Le masque ROI vient du segmenteur de surface fonctionnelle. Il sert a :

- filtrer les tuiles utilisables ;
- ponderer la loss ROI/background ;
- restreindre les score maps a la surface fonctionnelle ;
- scorer uniquement la zone metier utile.

Le masque ROI n'est pas une verite terrain defaut.

### Masque GT defaut oracle

Le masque GT defaut est utilise uniquement apres prediction pour :

- calculer les metriques pixel ;
- automatiser le feedback oracle GT ;
- alimenter les gates de promotion ;
- simuler le retour humain dans le MVP.

Il n'entre jamais dans le chemin d'inference et n'est jamais utilise comme entree du training normal.

Contrat dataset :

```text
train good       -> pas de GT fourni, masque defaut implicite vide
test good        -> GT fourni vide
test defective   -> GT fourni avec defauts
```

## 5. Training

Le training Feature-AE est good-only.

Regles d'entree :

- images conformes uniquement ;
- ROI status `ok` uniquement ;
- validation set fige exclu du train ;
- defauts confirmes exclus ;
- defauts manques exclus ;
- ROI warning/fail exclus ;
- bootstrap hors replay ;
- candidats replay versionnes par scenario.

Configuration champion cible :

```text
layers               = layer2, layer3
loss                 = l2_cosine
cosine_weight        = 0.5
layer2 weight        = 0.65
layer3 weight        = 0.35
roi_loss_weight      = 1.0
background_loss_weight = 0.02
min_roi_ratio        = 0.03
batch_size           = 16
learning_rate        = 5e-5
weight_decay         = 1e-4
epochs               = 14
repeat_factor        = 2
val_fraction         = 0.15
scheduler            = plateau
early_stopping       = 6 epochs
```

Commande type :

```powershell
uv run --extra cpu iqa-train-feature-ae `
  --category Casting_class1 `
  --manifest data\metadata\feature_ae_bootstrap_events.csv `
  --image-root D:\path\to\hss-iad `
  --output-checkpoint models\feature_ae\Casting_class1\checkpoint.pt `
  --layers layer2 layer3 `
  --preprocessing-mode tiled_context `
  --image-size 384 `
  --context-size 768 `
  --tile-stride 192 `
  --loss l2_cosine `
  --cosine-weight 0.5 `
  --layer-loss-weights layer2=0.65 layer3=0.35
```

## 6. Evaluation metier

La selection du champion ne doit pas reposer sur la loss seule.

Metriques image :

```text
image_auroc
image_ap
```

Metriques pixel avec GT oracle :

```text
pixel_ap
pixel_aupimo_1e-5_1e-3
```

Scoring cible :

```text
score_region     = functional_surface_prediction
smoothing        = median3
image_score      = topk_mean
topk_fraction    = 0.005
calibration      = per_layer / median_mad
validation_set   = validation_set_v001
```

Le validation set fige est exclu :

- du train ;
- du replay ;
- de la calibration normale.

Il sert a mesurer la performance et a prendre les decisions de promotion.

## 7. Checkpoints attendus

Le training doit produire :

```text
checkpoint_last.pt
checkpoint_best_loss.pt
checkpoint_epoch_XXX.pt
checkpoint_best_image_auroc.pt
checkpoint_best_image_ap.pt
checkpoint_best_pixel_ap.pt
checkpoint_best_pixel_aupimo_1e-5_1e-3.pt
checkpoint_best_image.pt
checkpoint_best_localization.pt
checkpoint.pt
metric_eval_best.json
metrics.json
params.json
loss_history.csv
```

`checkpoint.pt` pointe vers le meilleur checkpoint metier image si disponible, sinon vers le meilleur checkpoint loss.

Metadata minimales attendues dans le checkpoint :

```text
model_type
teacher_backbone
layers
preprocessing_mode
image_size
context_size
tile_stride
normalization
loss config
ROI config
GT contract
scenario_id
dataset_version
roi_model_version
feature_ae_version
run_name
metrics
```

## 8. Replay et lifecycle

Le bootstrap produit le V0 et reste hors replay.

Tout candidat issu d'un replay doit porter :

```text
scenario_id
dataset_version
roi_model_version
feature_ae_version
```

Scenarios connus :

```text
production_replay_natural
drift_domain_extension
```

Les scenarios restent isoles : datasets, runs, metriques, checkpoints et promotions ne doivent pas etre melanges.

## 9. Inference

L'inference charge le checkpoint actif via le runtime Feature-AE, applique le meme preprocessing, produit :

```text
score image
score map / heatmap
decision Vert / Orange / Rouge
versions modele
URI artefacts
```

Le service inference n'entraine jamais. Il sert uniquement le modele actif expose par le registry.

## 10. Non-objectifs MVP

Hors perimetre MVP :

- reentrainement du teacher ;
- reentrainement automatique du ROI segmenter ;
- usage de GT defaut pendant le training normal ;
- selection champion uniquement par loss ;
- API `/train` exposee au metier ;
- retour humain reel obligatoire.

Le retour humain Sophie reste une vitrine. Le workflow operationnel MVP est automatise par l'oracle GT apres prediction.
