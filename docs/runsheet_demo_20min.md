# Runsheet — Demo autonome class1 -> class2 -> class3 (~20 min)

## Pre-requis

1. Checkpoints pre-cuits bakes (Issue 26) :
   - `.cache/iqa/models/rd_feature_ae_class2_precuit/checkpoint.pt`
   - `.cache/iqa/models/rd_feature_ae_class3_precuit/checkpoint.pt`

2. Stack lancee avec les overrides demo :
   ```bash
   docker compose \
     -f deploy/docker-compose.yml \
     -f deploy/docker-compose.demo.yml \
     -f deploy/docker-compose.demo-runtime.yml \
     up -d
   ```

3. Variables d'environnement dans `.env` :
   ```
   IQA_GPU_DEMO_HOLD=0
   IQA_INFERENCE_CONTAINER=deploy-iqa-inference-1
   IQA_RETRAIN_COOLDOWN_SECONDS=60
   IQA_DATASET_HOST_PATH=/chemin/linux/vers/data/raw/hss-iad
   ```

4. Grafana ouvert sur le dashboard IQA (http://localhost:3000)
5. Alertmanager visible (http://localhost:9093)

---

## Sequence

### Phase 0 — Baseline class1 calme (T+0, ~2 min)

- La stack sert le modele class1-only baseline
- PatchCore couvre `[Casting_class1]`
- Envoyer quelques images class1 : tout est **Vert**
- **Point de narration** : "Le modele baseline ne connait que class1. PatchCore
  confirme que class1 est in-domain."

### Phase 1 — Stream class2, drift detecte (T+2, ~3 min)

- Lancer le replay class2 :
  ```bash
  .venv/Scripts/python.exe -m scripts.run_dual_drift_demo \
      --baseline-settle 10 --max-per-phase 20 --rate 6 \
      --phases domain_extension_class2
  ```
- Observer dans Grafana : le ratio OOD monte, PatchCore **fire**
- Alertmanager recoit `IqaDomainDriftPatchCore`
- Le proxy AE reste aveugle (contraste : AE ne voit pas le drift de domaine)
- **Point de narration** : "PatchCore detecte que class2 n'est pas dans sa
  banque memoire. L'AE ne voit rien — c'est normal, il mesure la
  reconstruction, pas le domaine."

### Phase 2 — Sensor tire, retrain class2 (T+5, ~5 min)

- Le sensor (`iqa_retrain_sensor`) detecte le drift et trigger `iqa_lifecycle`
- Le lifecycle warm-starte depuis le checkpoint pre-cuit class2
- Epochs=1, max_events=8 : cycle court (~2 min GPU)
- **Fallback** si le sensor ne tire pas dans les 2 min :
  ```bash
  python -m scripts.demo_fallback_lifecycle --class Casting_class2
  ```
- **Pendant le retrain** (narration) : "Le systeme a decide de re-entrainer.
  Il warm-starte depuis un checkpoint qui connait deja class2 — un seul epoch
  suffit pour atteindre le seuil de promotion."

### Phase 3 — Promotion, refresh, restart class2 (T+10, ~2 min)

- Le lifecycle promeut le candidat (`promotion_status=promoted`)
- `refresh_active_artifacts` copie le checkpoint et reconstruit PatchCore
  avec `[Casting_class1, Casting_class2]`
- `restart_inference` relance le conteneur avec le nouveau modele
- Observer dans Grafana : class2 passe de OOD a **Vert**
- **Point de narration** : "Le PatchCore couvre maintenant class1+class2.
  Les nouvelles images class2 sont in-domain. L'AE a aussi appris class2."

### Phase 4 — Stream class3, drift detecte (T+12, ~3 min)

- Lancer le replay class3 :
  ```bash
  .venv/Scripts/python.exe -m scripts.run_dual_drift_demo \
      --baseline-settle 10 --max-per-phase 20 --rate 6 \
      --phases domain_extension_class3
  ```
- PatchCore **fire** a nouveau : class3 hors banque
- class2 reste **Vert** (couvert par le cycle precedent)
- **Point de narration** : "Meme scenario. PatchCore voit que class3 est
  nouveau. Mais class2, deja couvert, reste vert."

### Phase 5 — Retrain class3, promotion, recuperation (T+15, ~5 min)

- Sensor tire → lifecycle warm-start class3 pre-cuit → promotion
- Refresh PatchCore `[Casting_class1, Casting_class2, Casting_class3]`
- Restart inference
- class3 passe a **Vert**
- **Fallback** :
  ```bash
  python -m scripts.demo_fallback_lifecycle --class Casting_class3
  ```
- **Point de narration** : "Trois classes couvertes. Le systeme s'est adapte
  deux fois de maniere autonome, sans intervention humaine."

### Phase 6 — Conclusion (T+20)

- Montrer le dashboard final : 3 classes vertes
- Rappeler le contraste AE vs PatchCore
- Ouvrir les questions

---

## Fallback manuel complet

Si le sensor ou Airflow pose probleme, declencher directement :

```bash
# Class2
python -m scripts.demo_fallback_lifecycle --class Casting_class2

# Attendre la fin du lifecycle (~3 min), puis :
# Class3
python -m scripts.demo_fallback_lifecycle --class Casting_class3
```

Le script POST le meme `conf` que le sensor aurait pousse. Le lifecycle
utilise le meme checkpoint pre-cuit et le meme contrat.

---

## Points de verification

| Etape | Signal | Attendu |
|-------|--------|---------|
| Phase 1 | Grafana OOD ratio | > 0.5 pour class2 |
| Phase 2 | Airflow DAG `iqa_lifecycle` | Running |
| Phase 3 | `promotion_status` dans summary.json | `promoted` |
| Phase 3 | Grafana class2 | Vert |
| Phase 4 | Grafana OOD ratio | > 0.5 pour class3 |
| Phase 5 | `promotion_status` | `promoted` |
| Phase 5 | Grafana class3 | Vert |
| Final | PatchCore covered_classes | `[class1, class2, class3]` |
