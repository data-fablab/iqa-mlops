# Modele Feature-AE IQA

## 1. Role

Le Feature-AE est le modele vivant du MVP IQA. Il detecte les anomalies sur la surface fonctionnelle deja isolee par le segmenteur ROI.

Il est le seul modele destine au reentrainement automatique dans la boucle MLOps
cible :

```text
images ingerees
-> ROI segmenter fige
-> tuiles surface fonctionnelle
-> RD Feature-AE
-> score anomalie + heatmap
-> decision Vert / Orange / Rouge
-> feedback oracle GT apres prediction
-> monitoring / lifecycle / promotion MLflow cible
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

Le mode historique `tile_256_overlap` n'est pas conserve. Il correspondait a une ancienne taille de tuiles et rendait le nom incoherent avec le reference actuel.

Mode IQA retenu :

```text
preprocessing_contract_version = feature_ae_reference_v001
preprocessing_mode = tiled_context
image_size         = 384
context_size       = 768
tile_stride        = 384
normalization      = imagenet
teacher_weights    = IMAGENET1K_V1
layers             = layer2, layer3
layer_weights      = layer2=0.65, layer3=0.35
roi_mode           = soft_map
roi_threshold      = 0.5
sampling           = all
augmentation       = none
```

Pipeline image :

```text
ResizeLetterbox
-> ToTensor
-> Normalize ImageNet
-> tuile locale 384
-> contexte 768
```

Le meme contrat de preprocessing doit etre utilise en training, evaluation,
inference, bootstrap et lifecycle. Les parametres sont centralises dans
`src/iqa/training/feature_ae_contracts.py` et sauvegardes dans les metadata du
checkpoint. Les commandes reference/bootstrap/lifecycle refusent les overrides
non canoniques, sauf avec le flag explicite de test local
`--allow-noncanonical-preprocessing`.

Le chemin runtime de demonstration et de production n'utilise plus le letterbox
image entiere. Le contrat reference reconstruit une score map pleine resolution
par tuiles 384 avec contexte 768, fusionne `layer2/layer3`, applique le ROI
soft-map puis calcule le score `topk_mean` sur la surface fonctionnelle. Tout
chemin letterbox restant est un chemin legacy de test et ne doit pas alimenter
Replay, API, Sophie, Marc ou le lifecycle progressif.

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

Avant le bootstrap V0, les images `feature_ae_bootstrap_events.csv` passent par
le segmenteur ROI fige. L'index ROI produit devient l'entree qui permet de
conserver uniquement les images `good` avec `roi_quality_status=ok`.

Sortie locale Phase 1 :

```text
data/processed/roi/bootstrap_v001/roi_predictions.csv
```

Configuration reference cible :

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
metric_early_stopping = 4 epochs without business metric improvement
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

Commande bootstrap serveur recommandee :

```bash
uv run --extra cpu iqa-build-feature-ae-bootstrap \
  --image-root /path/to/hss-iad \
  --device cuda \
  --publish-minio
```

Cette commande restaure le ROI depuis MinIO, genere les ROI bootstrap si
necessaire, entraine le Feature-AE, selectionne le checkpoint reference par
metriques metier, publie le checkpoint dans `s3://iqa-models` et met a jour le
manifest Git du bootstrap. En developpement local, utiliser `--dry-run` pour
verifier la configuration sans lancer le training.

Le bootstrap serveur evalue les metriques metier a chaque epoch. L'arret
anticipe principal suit la progression metier : si aucune metrique prioritaire
ne s'ameliore pendant 4 evaluations consecutives, le training s'arrete. La
`val_loss` reste utilisee pour le scheduler LR et comme signal de stabilite,
mais ne pilote ni le reference ni l'arret principal quand les metriques metier
sont disponibles.

## 6. Evaluation metier

La selection du reference ne doit pas reposer sur la loss seule.

Metriques image :

```text
image_auroc
image_ap
```

Metriques pixel avec GT oracle :

```text
pixel_auroc
pixel_ap
pixel_aupimo_1e-5_1e-3
```

Le bootstrap initial est selectionne selon l'ordre suivant :

```text
pixel_aupimo_1e-5_1e-3
-> pixel_ap
-> image_ap
-> image_auroc
```

`val_loss` reste un garde-fou de stabilite et de debug. Elle ne doit pas
selectionner le reference si les metriques metier pointent vers un autre
checkpoint, et elle ne doit pas arreter le bootstrap avant la patience metier.

Scoring cible :

```text
score_contract   = feature_ae_reference_v001
teacher_weights  = IMAGENET1K_V1
layer_weights    = layer2=0.65, layer3=0.35
roi_mode         = soft_map
roi_threshold    = 0.5
smoothing        = median3
image_score      = topk_mean
topk_fraction    = 0.005
validation_set   = validation_set_v001
```

Le validation set fige est exclu :

- du train ;
- du replay ;
- de la calibration normale.

Il sert a mesurer la performance et a prendre les decisions de promotion.

## 6.1 Calibration des seuils runtime

Les seuils `green / orange / red` ne sont pas des constantes universelles. Ils
sont calibres par `model_version` avec le meme contrat reference que
l'inference, et les metriques metier sont calculees sur `validation_set_v001`
avec les masques GT defauts.

Commande serveur :

```bash
uv run --extra cu128 iqa-calibrate-feature-ae-reference \
  --model-version rd_feature_ae_gated_v001_bootstrap \
  --image-root /opt/iqa/iqa-mlops/data/raw/hss-iad \
  --validation-manifest data/validation/validation_set_v001.csv \
  --gt-masks-manifest data/validation/validation_gt_masks_v001.csv \
  --roi-mode soft_map \
  --layer-weights layer2=0.65 layer3=0.35 \
  --topk-fraction 0.005 \
  --device cuda \
  --write-manifest
```

La commande restaure le Feature-AE depuis MinIO, score les images avec le
contrat reference, materialise `predictions.npz`, `calibration_matrix.csv` et
`calibration_summary.json`, puis ecrit les seuils dans le manifest modele :

```text
orange = quantile p95 des scores conformes calibres
red    = quantile p99 des scores conformes calibres
```

La configuration retenue est selectionnee par priorite metier :

```text
pixel_aupimo_1e-5_1e-3
-> pixel_ap
-> image_ap
-> image_auroc
```

`val_loss` reste informative et ne peut jamais promouvoir seule un modele.

Le runtime et le runner replay/lifecycle utilisent ensuite ces seuils manifest
quand ils sont disponibles. Sans seuils calibres, le fallback historique reste
bloque explicitement le runtime quand les seuils calibres manquent afin de ne pas
masquer une calibration manquante.

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

`checkpoint.pt` pointe vers le checkpoint reference selectionne par metriques
metier. Pour le bootstrap initial, la priorite est donnee a la localisation
metier (`pixel_aupimo`, puis `pixel_ap`) avant les metriques image. La loss ne
sert pas de critere reference principal.

Metadata minimales attendues dans le checkpoint :

```text
model_type
teacher_backbone
layers
preprocessing_mode
preprocessing_contract_version
preprocessing_contract
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

Etat actuel : le bootstrap ROI et le smoke test Feature-AE GPU sont
operationnels. Le lifecycle complet train/eval/gates/promotion reste une cible
de Phase 2.

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

Le service inference n'entraine jamais. En cible, il sert uniquement le modele
actif expose par le registry.

## 10. Non-objectifs MVP

Hors perimetre MVP :

- reentrainement du teacher ;
- reentrainement automatique du ROI segmenter ;
- usage de GT defaut pendant le training normal ;
- selection reference uniquement par loss ;
- API `/train` exposee au metier ;
- retour humain reel obligatoire.

Le retour humain Sophie reste une vitrine. Le workflow operationnel MVP est automatise par l'oracle GT apres prediction.


