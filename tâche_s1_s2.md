# IQA MLOps - Tâches de Ken (Phases 1 et 2)

## Phase 1 - Modèle, MLflow, Gates, Registry et Lifecycle

### IQA1_KEN01 - MOD

**Définir contrat modèle :**

* Entrée : piece_event
* Sorties :

  * score
  * heatmap_uri
  * ROI status
  * statut

**Dates :** 11/06/2026 → 11/06/2026
**Charge :** 0,25 j
**Avancement :** 100 %

---

### IQA1_KEN02 - ROI

**Acter ROI figé et defect_coverage comme gate de recevabilité par source_class**

**Dates :** 11/06/2026 → 11/06/2026
**Charge :** 0,25 j
**Avancement :** 50 %

---

### IQA1_KEN03 - MOD

**Créer wrappers :**

* ROI segmenter
* Teacher ResNet18
* Feature AE

**Dates :** 11/06/2026 → 11/06/2026
**Charge :** 0,25 j
**Avancement :** 100 %

---

### IQA1_KEN04 - MOD

**Créer predict_image minimal :**

* score
* ROI status
* heatmap placeholder
* latency_ms

**Dates :** 11/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 100 %

---

### IQA1_KEN05 - MOD

**Créer predict_piece minimal :**

* agrégation multi-vues
* statut Vert / Orange / Rouge

**Dates :** 11/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 100 %

---

### IQA1_KEN06 - DATA

**Créer le module de construction des datasets candidats avec règles de sécurité IA :**

* good only
* ROI OK
* aucun défaut
* aucun validation_set

**Dates :** 12/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 50 %

---

### IQA1_KEN07 - MLF

**Acter MLflow Registry source de vérité et MinIO stockage artefacts**

**Dates :** 12/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 100 %

---

### IQA1_KEN08 - REG

**Créer registry skeleton :**

* candidate
* test
* prod
* archived

par scenario_id

**Dates :** 12/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 50 %

---

### IQA1_KEN09 - AIR

**Créer DAG IQA_lifecycle importable avec placeholders**

**Dates :** 12/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 50 %

---

### IQA1_KEN10 - TST

**Créer tests modèle et registry**

**Dates :** 12/06/2026 → 12/06/2026
**Charge :** 0,25 j
**Avancement :** 50 %

---

# Phase 2 - Feature-AE, MLflow, Gates, Promotion et Rollback

### IQA2_KEN01 - MOD

Finaliser interface Feature AE candidat :

* train
* eval
* save
* load
* predict

**Dates :** 13/06/2026 → 14/06/2026

---

### IQA2_KEN02 - DATA

Finaliser candidate_builder good only :

* exclut validation_set
* exclut defective
* exclut ROI warning/fail

**Dates :** 13/06/2026 → 14/06/2026

---

### IQA2_KEN03 - MOD

Implémenter train_feature_ae_v2 et v3 candidats versionnés

**Dates :** 14/06/2026 → 15/06/2026

---

### IQA2_KEN04 - EVAL

Implémenter evaluate_feature_ae sur validation_set_v001 :

* AP
* recall
* Orange rate
* latency

**Dates :** 15/06/2026 → 15/06/2026

---

### IQA2_KEN05 - ROI

Mesurer defect_coverage par source_class et bloquer si couverture < 0.95

**Dates :** 15/06/2026 → 16/06/2026

---

### IQA2_KEN06 - MLF

Logger runs MLflow avec :

* params
* metrics
* artifacts
* git commit
* dataset_version
* scenario_id

**Dates :** 16/06/2026 → 16/06/2026

---

### IQA2_KEN07 - REG

Créer registered models séparés par scenario_id

**Dates :** 16/06/2026 → 16/06/2026

---

### IQA2_KEN08 - GATE

Implémenter gates :

* recall = 1.0
* aucun faux négatif
* AP prod -0.02 max
* Orange rate
* latency
* rollback

**Dates :** 16/06/2026 → 17/06/2026

---

### IQA2_KEN09 - REG

Implémenter promotion MLflow source de vérité et MinIO artefacts

**Dates :** 17/06/2026 → 17/06/2026

---

### IQA2_KEN10 - REG

Implémenter rollback via :

* transition MLflow
* restauration previous_prod

**Dates :** 17/06/2026 → 17/06/2026

---

### IQA2_KEN11 - AIR

Finaliser DAG IQA_lifecycle :

* dataset
* train
* eval
* gates
* MLflow
* promotion
* reload

**Dates :** 17/06/2026 → 18/06/2026

---

### IQA2_KEN12 - AIR

Ajouter paramètres Airflow pour rejouer lifecycle :

* naturel
* drift

par scenario_id

**Dates :** 18/06/2026 → 18/06/2026

---

### IQA2_KEN13 - MON

Créer baseline drift versionnée :

* production inattendue
* extension domaine planifiée

**Dates :** 18/06/2026 → 18/06/2026

---

### IQA2_KEN14 - INF

Brancher model_loader sur :

* MLflow prod par scenario_id
* artefact MinIO

**Dates :** 18/06/2026 → 19/06/2026

---

### IQA2_KEN15 - TST

Créer tests ML :

* no defective train
* no validation_set train
* ROI fail exclus
* gates bloquantes

**Dates :** 19/06/2026 → 19/06/2026

---

### IQA2_KEN16 - TST

Créer tests promotion :

* FN bloque
* AP insuffisante bloque
* promotion success
* rollback restore

**Dates :** 19/06/2026 → 19/06/2026

---

### IQA2_KEN17 - DOC

Documenter :

* model_lifecycle.md
* gates.md
* mlflow_registry.md
* drift_regimes.md
* rollback.md

**Dates :** 19/06/2026 → 19/06/2026

---

# Synthèse

## Phase 1

* 10 tâches
* 2,5 jours de charge
* 75 % d'avancement pondéré

## Phase 2

* 17 tâches
* 6 jours de charge
* 0 % d'avancement

## Total Ken

* 27 tâches
* 8,5 jours de charge
