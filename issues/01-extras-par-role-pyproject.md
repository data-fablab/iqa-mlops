# 01 - Decoupage des dependances en extras par role

Type : AFK

## What to build

Deplier le bloc `dependencies` du `pyproject.toml` racine en extras par role,
conformement a l'amendement de l'ADR 0007 :

- `serving` : fastapi, uvicorn, pydantic, boto3 (sans torch)
- `ml` : torch, torchvision, scikit-learn, mlflow
- `data` : pandas, pillow, boto3, psycopg

`torch`/`torchvision` quittent les dependances de base. Les extras existants
(`cpu`/`cu*`/`dev`) restent coherents. L'installation par extra doit etre
verifiable.

## Acceptance criteria

- [ ] Extras `serving`, `ml`, `data` definis dans `pyproject.toml`
- [ ] `torch`/`torchvision` ne sont plus dans `dependencies` de base
- [ ] `uv sync --extra serving` n'installe pas torch ; `--extra ml` l'installe
- [ ] Les commandes `iqa-*` de chaque role importent sans erreur avec leur extra
- [ ] `uv.lock` regenere et commite ; suite de tests existante toujours verte

## Blocked by

None - can start immediately
