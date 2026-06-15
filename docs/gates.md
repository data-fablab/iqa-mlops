# Gates de promotion IQA

## Décisions figées

### ROI segmenter — figé à v001

**Décision (IQA1_KEN02) :** Le modèle de segmentation ROI est figé à la version `roi_segmenter_v001_fixed`. Aucun entraînement ni remplacement ne peut avoir lieu sans une nouvelle décision explicite.

**Raison :** La ROI est considérée stable et suffisamment précise pour le périmètre actuel. Le figer évite une dérive de la segmentation qui invaliderait les évaluations des autres composants.

**Config :** `configs/promotion_gates.yaml` → section `roi`

---

### `defect_coverage` par `source_class` — gate de recevabilité

**Décision (IQA1_KEN02) :** La couverture de défauts (`defect_coverage`) par `source_class` est un **gate de recevabilité**. Un dataset ne peut pas passer en évaluation ou en promotion si la couverture est insuffisante.

**Seuil :** `0.95` (au moins 95 % des classes de défauts représentées par `source_class`)

**Implémentation (IQA2_KEN05) :** calcul du `defect_coverage` par `source_class`, blocage si `< 0.95`.

**Config :** `configs/promotion_gates.yaml` → section `defect_coverage.min_coverage`

---

## Autres gates de promotion

### Recall et couverture défauts

**Gate : `recall_defect_min: 1.0`**

- **Définition** : Taux de rappel sur les défauts détectés = TP / (TP + FN)
- **Seuil** : Aucun faux négatif toléré (recall = 100%)
- **Raison** : Manquer un défaut a un coût opérationnel élevé (produit défectueux vendu)
- **Mesure** : Calculée sur `validation_set_v001` par `source_class`

### Orange rate

**Gate : `orange_rate_max: 0.05`** (défaut : 5%)

- **Définition** : Proportion d'images classées comme "orange" (doute, uncertain) par le classifier
- **Formule** : `orange_count / total_predictions`
- **Seuil** : ≤ 5% des prédictions
- **Raison** : Un taux d'orange élevé indique une mauvaise confiance du modèle
- **Actions Orange** : Ces images requièrent une review humaine (coût augmenté)
- **Mesure** : Calculée en inférence sur `validation_set_v001`

**Configuration :**
```yaml
feature_ae:
  orange_rate_max: 0.05  # Naturel
  orange_rate_max: 0.12  # Drift (plus relâché)
```

### Latency

**Gate : `latency_p95_ms_max: 1000`** (milliseconds)

- **Définition** : Percentile 95 du temps de prédiction end-to-end
- **Inclut** :
  - Image loading + preprocessing
  - Model inference
  - Post-processing + heatmap generation
  - Network latency (si service distant)

- **Seuil** : ≤ 1000 ms (1 seconde)
- **Raison** : Temps d'attente utilisateur acceptable en production
- **Mesure** : Collectée via timing logs pendant `evaluate_feature_ae_checkpoint()`

**Configuration :**
```yaml
feature_ae:
  latency_p95_ms_max: 1000      # Naturel
  latency_p95_ms_max: 1200      # Drift (plus tolérant)
```

### AP (Average Precision) et régression

**Gate : `image_ap_max_regression: 0.02`**

- **Définition** : Régression d'AP (Average Precision) vs version en prod
- **Formule** : `ap_candidate - ap_prod` ≤ 0.02
- **Raison** : Prévenir les dégradations imperceptibles mais cumulées
- **Mesure** : Comparaison vs métriques stockées de la dernière version `prod` en MLflow

### ROI failure rate

**Gate : `roi_fail_rate_max: 0.10`**

- **Définition** : Proportion d'images où la segmentation ROI (Region of Interest) a échoué
- **Seuil** : ≤ 10%
- **Raison** : Le ROI segmenter est figé (ADR); une dégradation indique un décalage caméra ou un problème optique
- **Mesure** : Lors de l'inférence, comptabiliser les images où ROI segmentation = null ou roi_area < threshold

---

## Configuration complète

### Fichier : `configs/promotion_gates.yaml`

```yaml
promotion_gates:
  feature_ae:
    # Naturel
    recall_defect_min: 1.0
    false_negative_total_max: 0
    image_ap_max_regression: 0.02
    roi_fail_rate_max: 0.10
    latency_p95_ms_max: 1000
    orange_rate_max: 0.05

    defect_coverage:
      min_coverage: 0.95  # Au moins 95 % des source_classes représentées

  feature_ae_drift:
    # Drift (régime moins strict)
    recall_defect_min: 0.98
    false_negative_total_max: 5
    image_ap_max_regression: 0.10
    roi_fail_rate_max: 0.15
    latency_p95_ms_max: 1200
    orange_rate_max: 0.12

    defect_coverage:
      min_coverage: 0.90  # 90 % des source_classes

  roi:
    version_locked: "roi_segmenter_v001_fixed"  # Figé (ADR IQA1_KEN02)
```

### Implémentation

Voir `src/iqa/promotion/` :
- `evaluate_promotion_gates()` → évalue tous les gates
- `promote_model_with_gates()` → lance la promotion après validation

Implémentation complète des gates : **IQA2_KEN08**.
