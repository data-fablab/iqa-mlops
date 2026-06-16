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

### Regression AP

`image_ap_max_regression` compare l'AP candidat a l'AP production. Ce gate ne
peut etre evalue que si `prod_ap` est disponible dans les metriques d'evaluation.
Sans `prod_ap`, il doit rester non bloquant ou explicitement signale comme non
evaluable selon la logique de `evaluate_promotion_gates(...)`.

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
