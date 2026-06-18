# ADR 0008 - Taches Airflow comme conteneurs (DockerOperator), runtime iqa jamais importe

## Statut

Accepte le 2026-06-17. Complete [ADR 0002](0002-airflow-comme-orchestrateur.md) et
[ADR 0007](0007-architecture-services-avec-pyproject-racine.md).

## Contexte

[ADR 0007](0007-architecture-services-avec-pyproject-racine.md) pose la frontiere :
"Airflow orchestre par contrats HTTP ou commandes batch, et n'a pas vocation a
importer le runtime API/inference".

L'implementation actuelle contredit cette frontiere :

- `airflow/dags/iqa_lifecycle.py` utilise `PythonOperator` et fait
  `from iqa.dags.lifecycle_tasks import ...`. Airflow importe donc le runtime `iqa`.
- Le `docker-compose.yml` lance l'image officielle
  `apache/airflow:2.10.5-python3.12`, qui ne contient pas le package `iqa`. Le code
  tombe alors sur un placeholder ("iqa package not available in airflow image") et
  les taches ne realisent aucun travail reel.

Par ailleurs, l'objectif "le plus microservice et le plus automatise possible" et
la volonte de garder ouverte une evolution Kubernetes en fin de projet imposent un
modele d'orchestration portable.

## Decision

Une tache Airflow = un conteneur lancant une commande sur une image de service.
Airflow redevient un orchestrateur pur :

- aucune tache n'importe le runtime `iqa` ; le code metier est appele via les
  commandes `iqa-*` a l'interieur du conteneur du service concerne ;
- on supprime les `PythonOperator`/`BashOperator` qui dependaient de la presence
  du package `iqa` dans l'image Airflow ;
- chaque tache tire l'image du service adapte (decoupage par extras de
  [ADR 0007](0007-architecture-services-avec-pyproject-racine.md)), pas une image
  fourre-tout.

Le choix de l'operateur est encapsule dans une factory (`make_container_task`),
parametree par une variable d'environnement (`docker` aujourd'hui, `k8s` plus
tard). Les DAGs appellent la factory ; ils ne connaissent pas l'operateur concret.

Les images de service sont publiees sur un registre (Docker Hub) par la CI et
referencees par tag ; l'infra tierce (postgres, minio, mlflow, grafana,
prometheus, nginx, airflow) provient deja de registres.

## Consequences

- Resolution de l'incoherence avec [ADR 0007](0007-architecture-services-avec-pyproject-racine.md) :
  Airflow n'importe plus `iqa`.
- Portabilite vers Kubernetes (voir [ADR 0002](0002-airflow-comme-orchestrateur.md)) :
  passer a `KubernetesPodOperator` revient a changer la factory, pas les DAGs.
  **Kubernetes-ready by design, not tested** : le backend `k8s` de la factory
  (`_make_k8s_task`, selectionne par `IQA_AIRFLOW_BACKEND=k8s`) est un stub non
  exerce (ni CI ni runtime), volontairement leger pour que la bascule reste un
  changement de config et non une reecriture. Le mapping du lock GPU vers une
  ressource/PVC k8s reste un TODO (cf. `src/iqa/dags/operators.py`). On revendique
  donc la portabilite *par construction*, pas une portabilite validee.
- Le scheduler doit acceder au socket Docker (`/var/run/docker.sock`) en MVP, avec
  reseau partage et propagation du lock GPU au conteneur de tache. Ce privilege est
  acceptable en local ; il disparait avec la cible Kubernetes.
- Le DAG `iqa_lifecycle.py` reste la colonne vertebrale, mais ses etapes
  (dataset, train, eval, gates, mlflow, promotion, reload) deviennent des
  invocations de conteneurs.
- L'automatisation de bout en bout (declenchement par evenement donnees, promotion
  et reload en fin de DAG) reste l'objectif et s'appuie sur ce modele.
