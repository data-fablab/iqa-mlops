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

### 2. Detection en ligne = anomaly score agrege

Le signal de detection est la decision `{Vert, Orange, Rouge}` issue du score de
reconstruction, exposee via le **meme proxy Prometheus** (part Orange+Rouge dans le
regime drift, seuil 0.5 ; cf. `configs/drift_proxy_calibration.yaml`). Seul le
*producteur* du score change (synthetique -> reel) ; regles d'alerte et sensor sont
inchanges. `reconstruction_p95` est exporte comme **gauge d'observabilite**, pas
comme declencheur. AUPIMO / pixel_ap ne detectent pas en ligne : ils servent les
gates de promotion.

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
