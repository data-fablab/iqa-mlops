# Gates de promotion IQA

## Statut

Les gates sont centralises dans `configs/promotion_gates.yaml` et evalues par
`evaluate_promotion_gates(...)`. Le DAG `iqa_lifecycle` charge cette config dans
`task_gates()` et `task_promotion()`.

Il faut distinguer :

- les gates actuellement alimentes par les metriques du DAG ;
- les gates cibles dont le calcul complet reste a brancher.

## Gates alimentes dans le DAG

### Recall defaut

`recall_defect_min` verifie que le rappel candidat atteint le seuil attendu.

### Orange rate

`orange_rate_max` limite la part d'images en decision incertaine.

### Latence

`latency_p95_ms_max` limite la latence p95 du candidat.

### Non-regression sur les 4 metriques metier

`quality_max_regression` pilote la promotion sur les **4 metriques metier**, dans
l'ordre de priorite `pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap ->
image_auroc` (ADR 0010 §6). Le gate est une **non-regression vs la baseline
prod**, pas un seuil absolu : pour chaque metrique evaluable (valeur finie cote
candidat **et** cote prod), `regression = prod - candidat` doit rester
`<= quality_max_regression[metrique]`. Le verdict global passe si toutes les
metriques evaluables passent.

Quand les masques GT sont absents, `pixel_aupimo` et `pixel_ap` ne sont pas
calculables : le gate **se replie sur `image_ap`** (champ `fallback_to_image_ap`).
La metrique de plus haute priorite disponible est reportee comme
`decisive_metric` (selection de reference respectant la priorite metier). Sans
baseline prod, la non-regression n'est pas evaluable et reste donc non bloquante.

`evaluate_quality_regression_gates(...)` est la fonction pure qui rend ce verdict ;
`evaluate_promotion_gates(...)` l'integre sous le gate `quality_regression` des que
les metriques candidat **et** prod sont fournies (sinon repli sur l'ancien gate
mono-metrique `image_ap_max_regression` quand seul `prod_ap` est disponible). Le
DAG lit la baseline prod via `fetch_latest_quality_metrics("prod")` dans
`task_gates()` / `task_promotion()`, journalise chaque verdict, et ne promeut que
si tous les gates passent. Les metriques **par classe** (class1/class2/class3)
sont loguees dans `iqa-model-quality` pour visualiser la couverture incrementale.

## Gates cibles ou partiellement branches

### Couverture defauts

`defect_coverage` est une regle de recevabilite cible : un dataset candidat ne
devrait pas passer si certaines classes de defauts sont insuffisamment couvertes.
La logique helper existe, mais le DAG ne doit pas annoncer ce gate comme complet
tant que la metrique n'est pas produite par `task_dataset()` ou `task_eval()`.

### ROI failure rate

`roi_fail_rate_max` est une cible liee au segmenter ROI fige
`roi_segmenter_v001_fixed`. La branche charge les statuts ROI dans
`task_dataset()` quand `roi_predictions_dirs` est fourni, mais le taux d'echec
ROI complet doit etre calcule et transmis aux gates avant d'etre considere comme
pleinement branche.

## Configuration

Fichier : `configs/promotion_gates.yaml`

```yaml
feature_ae:
  recall_defect_min: 1.0
  false_negative_total_max: 0
  image_ap_max_regression: 0.02
  roi_fail_rate_max: 0.10
  latency_p95_ms_max: 1000
  quality_max_regression:
    pixel_aupimo_1e-5_1e-3: 0.02
    pixel_ap: 0.02
    image_ap: 0.02
    image_auroc: 0.02

defect_coverage:
  min_coverage: 0.95

roi:
  model_version: roi_segmenter_v001_fixed
  frozen: true
```

## Implementation

- `src/iqa/promotion/gates.py` : evaluation unitaire des gates.
- `src/iqa/promotion/promotion.py` : promotion apres gates et alias MLflow.
- `src/iqa/dags/lifecycle_tasks.py` : orchestration Airflow Phase 1.
