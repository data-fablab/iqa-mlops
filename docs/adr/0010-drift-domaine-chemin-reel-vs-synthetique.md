# ADR 0010 - Drift de domaine : chemin reel (pixels) vs chemin synthetique, et seam vers l'apprentissage continu

## Statut

Accepte le 2026-06-25. Complete [ADR 0006](0006-mlflow-registry-source-verite.md)
et [ADR 0008](0008-taches-airflow-comme-conteneurs.md). Concerne le scenario
controle `drift_domain_extension`.

## Contexte

Le scenario `drift_domain_extension` doit demontrer la chaine MLOps complete :
drift detecte -> alerte -> reentrainement -> recuperation. Deux facons de produire
le signal coexistent dans le code :

- **Chemin synthetique (chemin B)** : un scoreur deterministe sans GPU
  (`iqa.inference.drift_scoring`) derive la decision de `(scenario_id, source_class)`
  et d'un fichier `deploy/drift-state/state.json` que l'orchestrateur edite. Il
  prouve la *plomberie* (alerte -> sensor -> retrain -> reprise) sans toucher au GPU.
- **Chemin reel (chemin A)** : `iqa.inference.real_inference.RealFeatureAEScorer`
  calcule une vraie erreur de reconstruction reverse-distillation sur les pixels.

L'objectif a terme est le chemin reel sur les images du dataset, avec les metriques
de qualite modele (AUPIMO, pixel AP, image AP) pour valider le reentrainement. Le
chemin synthetique reste un smoke-test. Plusieurs incoherences bloquaient le chemin
reel : seuils de reconstruction codes en dur et non calibres, gates ne regardant que
`image_ap`, metriques pixel calculees puis jetees par `task_eval`, baseline deploye
pointant sur un checkpoint inadapte, et pas de substrat pour une future boucle
d'apprentissage continu.

Tension de fond a clarifier : AUPIMO / pixel_ap / image_ap sont des metriques
*supervisees* (masques GT du `validation_set_v001`). En production, les images
arrivent sans masque ni label : on ne peut pas les calculer en ligne. Elles ne
peuvent donc pas etre le signal de detection ; elles sont le signal de *validation*
de la promotion.

## Decision

### 1. Deux chemins, roles distincts

Le chemin synthetique reste un **script separe** et un smoke-test de la plomberie ;
on n'y touche pas. Le chemin reel a son **propre driver** qui rejoue
`casting_flux_replay_plan_drift.csv` (resolution `relative_paths -> file://`), poste
`/predict` par phase (class1 -> class2 -> class3), **n'edite jamais** `state.json`,
et observe alertes Prometheus + runs Airflow. La recuperation au Vert vient
exclusivement du checkpoint reentraine, pas d'une ecriture d'etat.

### 2. Detection en ligne : deux signaux complementaires

> **Revise le 2026-06-26.** La version initiale presentait le score de reconstruction comme
> signal unique de detection. L'investigation empirique (probe sur le plan drift) a montre
> que le score de **reconstruction AE ne separe pas les domaines** : class2 ~45 %, class3
> ~5 % au-dessus du p90 class1. Le ratio proxy AE ne franchit pas 0.5 de facon fiable.
> En revanche un detecteur **distance-au-nominal (PatchCore)** separe les domaines a 100 %
> (class2 et class3 au-dessus du p90 class1), mais est ~aleatoire sur le defaut.

**Signal defaut** : decision `{Vert, Orange, Rouge}` issue du score de reconstruction AE,
exposee via le proxy Prometheus (part Orange+Rouge, seuil 0.5 ; cf.
`configs/drift_proxy_calibration.yaml`). `reconstruction_p95` est exporte comme gauge
d'observabilite. AUPIMO / pixel_ap ne detectent pas en ligne : ils servent les gates de
promotion.

**Signal drift de domaine** : score PatchCore (distance kNN max-patch au nominal),
emis en in-process a cote de l'AE. Counter `iqa_domain_drift_total{regime}` + gauge
`iqa_domain_drift_score`. Regle `IqaDomainDriftPatchCore` (ratio out-of-domain > 0.5).

**Contrat des deux seuils (decision 7 de l'amendement) :**
- Seuil **par-piece** : vit dans la calibration du detecteur concerne (p90 class1 pour
  PatchCore, Orange/Rouge calibres pour l'AE). Responsabilite du detecteur.
- Seuil de **population** (ratio 0.5) : vit une seule fois dans la regle Prometheus
  correspondante. Pas de duplication entre detecteur et regle.

### 3. Deux etages de seuils, independants

- **Etage image** : seuils `Orange` / `Red` sur le score de reconstruction,
  **calibres sur la distribution baseline class1** (phase `baseline_domain_class1`
  du plan drift), persistes dans `configs/feature_ae_reconstruction_calibration.yaml`
  avec validation **HITL**, charges par le scoreur (remplacent les constantes en dur).
- **Etage proxy** : ratio 0.5 (espace des ratios [0,1]), inchange.

Ils sont independants mecaniquement (le proxy compte des etiquettes, pas des
scores), mais couples par le sens : un etage image mal cale rend le ratio
ininterpretable. Critere : class1 -> ratio ~ 0, class2/3 -> ratio ~ 1.

### 4. Baseline deploye = `rd_feature_ae_class1_baseline`

`iqa-inference` deploie le checkpoint entraine sur **class1 uniquement** (250 images,
`feature_ae_bootstrap_events.csv`), garantissant que class2/class3 sont reellement
hors-distribution. `feature_ae_good_v003` (entraine sur class1+2+3) sert d'**oracle
de reference** (preuve qu'un modele couvrant class2/3 les score Vert).
`rd_feature_ae_gated_v001_bootstrap` reste reserve au chemin synthetique.

### 5. Seam de reentrainement, vers l'apprentissage continu

Un contrat unique decouple la *source* des echantillons de tout l'aval :

```
task_dataset -> resolve_retrain_samples(RetrainTrigger) -> list[Sample]
             -> build_candidate_dataset -> train -> eval -> gates -> promote -> reload
```

- **Implementation A (maintenant)** : filtre le plan statique -> class1 + classes vues
  jusqu'a la phase declencheuse (couverture incrementale class2 puis class3).
- **Implementation C (a terme)** : requete sur un feedback store. Bascule par flag,
  **aval strictement inchange**.

Forward-compatible des maintenant : le **sensor transmet la classe declencheuse**
dans la conf, et un **journal de predictions JSONL** est ecrit par `iqa-inference`
pour chaque `/predict` (`ts, piece_event_id, scenario_id, source_class, image_uri,
score, decision, feature_ae_version`), joignable au label via `piece_event_id`.
C'est le substrat que le resolver C lira. Le seul travail de fond restant pour C
est la boucle de label (oracle / HITL / heuristique), volontairement repoussee.

### 6. Gates = non-regression vs baseline prod

`task_eval` remonte toute la chaine `pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap`
(deja calculee par l'evaluateur), loggee dans MLflow `iqa-model-quality`. Le gate
est une **non-regression vs la baseline prod** (pas un seuil absolu) avec repli sur
`image_ap` quand les masques GT sont absents. Raison : pendant la couverture
incrementale, class3 non encore traitee plomberait un seuil absolu alors que le
retrain a parfaitement couvert class2 ; la non-regression laisse passer un candidate
strictement meilleur. Les metriques par classe sont loggees pour visualiser la
couverture incrementale.

### 7. Activation, GPU, robustesse

- **Reload** : la promotion ecrit vers un chemin prod stable ; une tache finale du
  DAG appelle `/reload-model` (sans argument). Mecanisme **general** (vaut les deux
  regimes), exerce par le scenario drift.
- **GPU** : pas de hold continu en mode reel ; **lock par requete**, partage avec le
  retrain. Les `/predict` se serialisent derriere l'entrainement court ; le debit
  ralentit mais ne s'arrete pas, le ratio proxy reste calculable.
- **Echec de gate** : l'alerte **persiste** (le drift n'est pas resolu = signal
  HITL correct), le candidate est logge `rejected`, et un anti-boucle empeche le
  re-trigger en rafale. Le happy path reste garanti par construction (good_v003
  prouve que class2 est apprenable).

## Consequences

- Le chemin reel devient fonctionnel de bout en bout sur les vraies images, sans
  modifier la plomberie d'alerte ni le chemin synthetique.
- La transition A -> C (apprentissage continu) se reduit a une 2e implementation du
  resolver + une boucle de label ; tout l'aval (train/eval/gates/promote/reload) est
  ecrit une fois. Le journal de predictions accumule des maintenant les donnees
  necessaires.
- Les metriques de qualite modele (AUPIMO, pixel AP, image AP) pilotent reellement
  la promotion et sont visibles dans Grafana via MLflow `iqa-model-quality`.
- Nouvelle dependance operationnelle : un fichier de calibration HITL
  (`configs/feature_ae_reconstruction_calibration.yaml`) et un volume pour le journal
  de predictions JSONL.
- Le partage GPU par requete borne le ralentissement d'inference a la duree d'un
  retrain demo (epochs abaisses) ; en cible Kubernetes, le lock GPU devient une
  ressource de noeud (TODO deja note dans [ADR 0008](0008-taches-airflow-comme-conteneurs.md)).

## Amendement 2026-06-26 - Detecteur de drift de domaine (PatchCore) + politique de reentrainement

### A. Deux detecteurs, deux roles distincts (corrige la decision 2)

Investigation empirique (probe sur le plan drift, meme images, meme metriques) :

- Le score de **reconstruction** de l'AE detecte des **defauts** mais **ne separe pas les
  domaines** produits : sur le p90 class1, class2 ~45 % et class3 ~5 % d'images au-dessus
  du seuil -> le ratio proxy ne franchit pas 0.5 de facon fiable (cause du blocage de
  l'Issue 7).
- Un detecteur **distance-au-nominal (PatchCore)** -- backbone ImageNet WRN50, banque de
  patches nominaux class1, score = distance kNN max-patch -- **separe les domaines a 100 %**
  (class2 et class3 au-dessus du p90 class1), mais est ~aleatoire sur le **defaut** (image
  AUROC ~0.5, pixel AP au niveau du hasard).

Decision : **deux detecteurs complementaires, jamais l'un pour l'autre.**

- **Feature-AE (reconstruction)** = detecteur de **defaut** (decision Vert/Orange/Rouge,
  4 metriques metier). Inchange.
- **PatchCore (distance-au-nominal)** = detecteur de **drift de domaine**. Emis en
  in-process dans `iqa-inference` a cote de l'AE : counter `iqa_domain_drift_total{regime}`
  (alerte par ratio) + gauge `iqa_domain_drift_score` (courbe). Regle `IqaDomainDriftPatchCore`
  (ratio out-of-domain > 0.5). Detecteur enregistre (banque + calibration + manifest) sur
  disque hote + run MLflow `iqa-domain-drift` (artifact MinIO).

Deux seuils, comme l'AE : seuil **par-piece** (p90 class1) dans la calibration du detecteur ;
seuil de **population** (ratio 0.5) dans la regle Prometheus. Le proxy AE (decision 2) reste
un signal *defaut* ; il n'est plus presente comme detecteur de *drift de domaine*.

### B. Politique de reentrainement multi-signal (etend les decisions 5-7)

But : faire monter les 4 metriques de l'AE en declenchant un retrain **au bon moment**. Le
gate relatif vs prod (decision 6) reste le filet de securite : on ne promeut que du mieux.

**Triggers (l'un suffit) :**
1. **Drift de domaine** -- alerte PatchCore (nouveau domaine non couvert).
2. **Plancher metrique** -- prod `pixel_aupimo_1e-5_1e-3 < 0.15` **ou** `pixel_ap < 0.20`
   (cibles absolues seedees sur l'atteignable demontre : retrain 420 img = 0.172 / 0.268).
   **TRIGGER uniquement, jamais critere de gate** -- ne contamine pas la non-regression
   relative de la decision 6 (un plancher absolu casserait la couverture incrementale).
3. **Accumulation** -- N conformes validees accumulees (le `>= 50` existant).

**Quel retrain (scope, via `resolve_retrain_samples`) :** drift -> couverture incrementale
(class1 + classes vues) ; plancher -> tout le bon dispo du/des domaine(s) couvert(s) +
selection `checkpoint_best_*` (la cause racine du sous-entrainement etait le **volume** de
donnees, pas les epochs) ; accumulation -> conformes accumules. **Plusieurs triggers** ->
un seul retrain, **union** des scopes, priorite d'etiquetage drift > plancher > accumulation.

**Mecanisme :** un **evaluateur periodique unique en pull** (sensor-DAG) lit Prometheus
(drift), MLflow (metriques prod vs cibles), store de donnees (accumulation), appelle une
**fonction de decision pure** (`evaluate_retrain_policy`, extension de
`evaluate_lifecycle_signal`), et declenche `iqa_lifecycle` avec la conf. Le webhook-catcher
reste passif (observabilite) ; pas de push Alertmanager -> Airflow.

**Anti-boucle (etend la decision 7) :** re-trigger **seulement si une entree a change**
(donnees / classe derivante / checkpoint prod) ; apres K echecs de gate sur la meme
condition -> arret de l'auto-retrain, candidat `rejected`, alerte/breach **persistant comme
signal HITL** ; cooldown temporel minimal en garde-fou secondaire.

### Consequences de l'amendement

- Le blocage de l'Issue 7 (alerte qui ne tirait pas sur class2/3) est leve par un detecteur
  adapte, pas par un meilleur checkpoint AE.
- Les Issues 9/10 sont re-ancrees sur le signal de drift PatchCore. Nouvelles Issues 11-14
  (detecteur + serving + demo duale + durcissement) et 15-18 (politique de reentrainement).
- Le plancher metrique adresse le sous-entrainement du baseline deploye (AUPIMO 0.074 ->
  ~0.17 atteignable) **sans** dependre d'un drift -- independant de PatchCore.
- Nouvelle dependance operationnelle : detecteur PatchCore resident (~1.9 Go VRAM, mesure :
  les deux modeles tiennent sur un seul GPU, ~178 ms/piece sequentiel) et son enregistrement
  MLflow/MinIO ; section `retrain_policy` (cibles plancher) en config.
