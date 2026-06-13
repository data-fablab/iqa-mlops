# Acces Tailscale Et SSH - Serveur IQA

## Objectif

Donner a l'equipe un acces controle au serveur IQA sans exposer directement le
SSH sur Internet et sans ouvrir le reseau local de la maison.

Le serveur IQA est accessible :

```text
LAN local      <SERVER_LAN_IP>
Tailscale      <SERVER_TAILSCALE_IP>
Utilisateur    <SSH_USER>
Hostname       <SERVER_HOSTNAME>
```

Pour l'equipe, l'adresse a utiliser est l'adresse Tailscale :

```bash
ssh <SSH_USER>@<SERVER_TAILSCALE_IP>
```

## Regle De Securite

Ne pas ouvrir le port `22` de la box Internet vers le serveur.

Tailscale cree un reseau prive chiffre entre les machines autorisees. Les
membres de l'equipe accedent au serveur via son IP Tailscale, sans rendre le
serveur visible publiquement.

Important : le serveur ne doit pas etre configure en routeur de sous-reseau
Tailscale. L'objectif est de donner acces au serveur IQA, pas au reseau local.

## Installation Tailscale

Chaque membre installe Tailscale sur son poste :

```text
https://tailscale.com/download
```

Puis il se connecte au tailnet du projet avec le compte invite par le
responsable.

Sur Windows, les commandes Tailscale peuvent etre lancees avec :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
& "C:\Program Files\Tailscale\tailscale.exe" up
```

Si la commande `tailscale` fonctionne directement dans le terminal, le chemin
complet n'est pas necessaire.

## Verification Tailscale

Depuis le poste du membre :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
& "C:\Program Files\Tailscale\tailscale.exe" ping <SERVER_TAILSCALE_IP>
```

Resultat attendu :

```text
pong from <SERVER_HOSTNAME> (<SERVER_TAILSCALE_IP>)
```

Si `status` indique `Logged out`, relancer :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
```

## Connexion SSH

Connexion au serveur :

```powershell
ssh <SSH_USER>@<SERVER_TAILSCALE_IP>
```

Au premier acces, SSH affiche une question du type :

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Repondre :

```text
yes
```

Verification apres connexion :

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

## Cle SSH Personnelle

Chaque membre doit utiliser sa propre cle SSH.

Sur Windows PowerShell :

```powershell
ssh-keygen -t ed25519 -C "prenom-iqa"
```

Accepter l'emplacement par defaut :

```text
C:\Users\<user>\.ssh\id_ed25519
```

Afficher la cle publique :

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Envoyer uniquement la cle publique au responsable serveur.

Ne jamais envoyer :

```text
id_ed25519
```

Le fichier prive reste sur le poste du membre.

## Ajout D'une Cle Sur Le Serveur

Sur le serveur, connecte en tant que `<SSH_USER>` :

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
```

Ajouter une ligne par membre :

```text
ssh-ed25519 AAAA... adrien-iqa
ssh-ed25519 AAAA... natacha-iqa
ssh-ed25519 AAAA... ken-iqa
```

Puis appliquer les permissions :

```bash
chmod 600 ~/.ssh/authorized_keys
```

## Acces Au Projet

Le repo est installe sur le serveur ici :

```bash
/opt/iqa/iqa-mlops
```

Commandes utiles :

```bash
cd /opt/iqa/iqa-mlops
git status
uv run --extra cpu --extra data pytest -q
```

Services Docker :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env ps
```

## URLs Utiles Via Tailscale

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

Pour MLflow, utiliser de preference l'URL Tailscale ci-dessus afin d'eviter les
problemes de `Host header`.

## Bonnes Pratiques Equipe

- Ne pas partager le mot de passe du compte `iqa`.
- Ne pas publier les vraies IPs du serveur dans GitHub.
- Ne pas partager les cles privees SSH.
- Une cle publique par personne.
- Retirer la cle d'un membre qui quitte le projet.
- Ne pas lancer de training GPU pendant une demonstration sans coordination.
- Prevenir l'equipe avant de redemarrer Docker ou le serveur.
- Ne pas modifier `.env` sans prevenir, car il pilote les services Docker.

## Depannage Rapide

### `tailscale` n'est pas reconnu sur Windows

Utiliser le chemin complet :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
```

### Tailscale indique `Logged out`

Relancer l'authentification :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
```

### SSH timeout

Verifier d'abord Tailscale :

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" ping <SERVER_TAILSCALE_IP>
```

Si le ping Tailscale ne repond pas, le probleme vient de Tailscale ou de
l'autorisation du poste dans le tailnet.

### Le navigateur n'ouvre pas MLflow

Tester depuis le poste :

```powershell
curl http://<SERVER_TAILSCALE_IP>:5000
```

Si le message parle de `Invalid Host header`, prevenir le responsable serveur :
la liste `IQA_MLFLOW_ALLOWED_HOSTS` doit inclure l'adresse utilisee.

## Retirer Un Acces

Sur le serveur :

```bash
nano ~/.ssh/authorized_keys
```

Supprimer la ligne correspondant au membre, puis sauvegarder.

Le changement est immediat pour les nouvelles connexions SSH.

## Durcissement Prevu

Quand toutes les connexions par cle sont testees, l'authentification SSH par mot
de passe pourra etre desactivee :

```bash
sudo nano /etc/ssh/sshd_config
```

Configuration cible :

```text
PasswordAuthentication no
PubkeyAuthentication yes
```

Puis :

```bash
sudo systemctl restart ssh
```

Important : tester une connexion SSH par cle dans une deuxieme fenetre avant de
fermer la session existante.
