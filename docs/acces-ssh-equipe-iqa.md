# Acces Serveur IQA Via Tailscale

## Objectif

Permettre a l'equipe d'acceder au serveur IQA sans exposer le SSH sur Internet.
L'acces se fait via Tailscale, puis SSH sur le compte projet partage.

Les vraies valeurs de connexion sont transmises en prive, jamais dans GitHub.

```text
Tailscale      <SERVER_TAILSCALE_IP>
Utilisateur    <SSH_USER>
Hostname       <SERVER_HOSTNAME>
```

## Preparer Son Poste

Installer Tailscale :

```text
https://tailscale.com/download
```

Se connecter au tailnet du projet avec l'invitation fournie.

Sur Windows, si la commande `tailscale` n'est pas reconnue, utiliser :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
& "C:\Program Files\Tailscale\tailscale.exe" status
```

Verifier que le serveur est joignable :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" ping <SERVER_TAILSCALE_IP>
```

## Connexion SSH

Depuis un terminal :

```bash
ssh <SSH_USER>@<SERVER_TAILSCALE_IP>
```

Au premier acces, SSH demande de confirmer l'empreinte du serveur :

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Repondre :

```text
yes
```

Le mot de passe du compte projet est communique en prive par le responsable.

## Verification Apres Connexion

```bash
hostname
whoami
pwd
```

Resultat attendu :

```text
<SERVER_HOSTNAME>
<SSH_USER>
/home/<SSH_USER>
```

## Projet Et Services

Le repo est installe ici :

```bash
/opt/iqa/iqa-mlops
```

Commandes utiles :

```bash
cd /opt/iqa/iqa-mlops
git status
uv run --extra cpu --extra data pytest -q
```

Sur le serveur, les tests classiques restent en CPU. Les commandes de training
ou inference GPU doivent utiliser `--extra cu128`.

Services Docker :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env ps
```

Stack Docker avec inference/training GPU :

```bash
docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## URLs Utiles

Depuis une machine connectee a Tailscale :

```text
API via reverse proxy    http://<SERVER_TAILSCALE_IP>/api/health
API directe              http://<SERVER_TAILSCALE_IP>:8000/health
Inference                http://<SERVER_TAILSCALE_IP>:8100/health
Inference metrics        http://<SERVER_TAILSCALE_IP>:8100/metrics
MLflow                   http://<SERVER_TAILSCALE_IP>:5000
MinIO console            http://<SERVER_TAILSCALE_IP>:9001
Grafana                  http://<SERVER_TAILSCALE_IP>:3000
Prometheus               http://<SERVER_TAILSCALE_IP>:9090
```

## Regles D'usage

- Ne pas publier les vraies IPs, mots de passe ou tokens dans GitHub.
- Ne pas partager l'acces hors de l'equipe projet.
- Prevenir l'equipe avant de redemarrer Docker ou le serveur.
- Ne pas lancer de training GPU pendant une demonstration sans coordination.
- Ne pas modifier `.env` sans prevenir, car il pilote les services Docker.

## Depannage Rapide

Si Tailscale indique `Logged out` :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
```

Si SSH ne repond pas :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" ping <SERVER_TAILSCALE_IP>
```

Si MLflow affiche `Invalid Host header`, prevenir le responsable serveur : la
liste `IQA_MLFLOW_ALLOWED_HOSTS` doit inclure l'adresse utilisee.
