# Runbook - Rollback serveur (applicatif)

Procedure de retour arriere **applicatif** : redeployer une version anterieure
des conteneurs IQA a partir d'une image deja publiee. Complete
`deploy_runbook.md` (section 8) et le smoke test `deploy/smoke-test.sh`.

> Portee : ce runbook couvre le rollback du **code / des conteneurs**. Le
> rollback **modele** (revenir a une version anterieure du Feature-AE) est
> distinct et passe par le MLflow Registry : voir `rollback.md`. Les deux sont
> independants ; un rollback applicatif ne change pas le modele actif.

## 1. Principe

- Les images sont publiees par la CI (`ci.yml`, job `publish-images`) avec des
  **tags immuables** (SHA git + tag `v*`), **jamais `latest`**.
- L'overlay `docker-compose.prod.yml` tire les images du registre selon
  `IQA_IMAGE_REGISTRY` et `IQA_IMAGE_TAG` (definis dans `.env`).
- Rollback = repointer `IQA_IMAGE_TAG` sur la **version precedente connue bonne**,
  puis `pull` + `up -d`. Reproductible et reversible : aucune reconstruction.

## 2. Prerequis

- Le tag a redeployer existe bien dans le registre (publie par une CI passee).
- Acces au serveur + `.env` de prod renseigne.
- Connaitre le **tag courant** et le **tag precedent** (voir section 3).

## 3. Identifier la version a redeployer

```bash
# tag actuellement deploye
grep IQA_IMAGE_TAG deploy/.env        # ou: echo $IQA_IMAGE_TAG

# tags disponibles : versions git (releases) ...
git tag --sort=-creatordate | head
# ... et/ou la liste des tags pousses sur Docker Hub (UI du registre)
```

Choisir le dernier tag **anterieur** connu stable (ex. `v0.1.0` si `v0.1.1`
pose probleme).

## 4. Procedure de rollback

```bash
cd deploy

# 1) repointer le tag immuable precedent dans .env (jamais latest)
#    edition manuelle, ou en ligne :
sed -i 's/^IQA_IMAGE_TAG=.*/IQA_IMAGE_TAG=v0.1.0/' .env

# 2) recuperer les images de cette version
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

# 3) redeployer
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 4) valider
cd ..
bash deploy/smoke-test.sh
```

`smoke-test.sh` doit ressortir tout vert (API, inference, MLflow, MinIO,
Prometheus, Grafana, Airflow, gateway). Verifier aussi `GET /model/version` et
le dashboard Grafana `IQA - Vue d'ensemble`.

## 5. Rollback partiel (un seul service)

Pour ne revenir que sur l'API par exemple :

```bash
cd deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  up -d --no-deps iqa-api
```

(le tag vient toujours de `IQA_IMAGE_TAG` ; pour un tag different par service,
surcharger ponctuellement la variable d'image concernee.)

## 6. Points d'attention

- **Base de donnees** : un rollback applicatif **ne rejoue pas** les migrations
  PostgreSQL/Airflow a l'envers. Si la version fautive a applique une migration
  destructive, restaurer la base depuis `s3://iqa-backups` (cf.
  `retention_storage.md`) avant de redeployer.
- **Tag inexistant** : si `pull` echoue, le tag n'a jamais ete publie -> choisir
  un tag present dans le registre.
- **Jamais `latest`** : toujours un tag immuable, sinon le rollback n'est pas
  reproductible.
- **Modele vs application** : si l'incident vient d'un mauvais modele promu,
  c'est un rollback **MLflow** (`rollback.md`), pas un rollback serveur.

## 7. Apres rollback

- Consigner l'incident (tag fautif, tag restaure, cause) dans le suivi
  d'exploitation.
- Corriger la cause sur une branche, repasser la CI, publier un nouveau tag, puis
  redeployer ce tag (avancer, pas rester bloque sur l'ancien).
