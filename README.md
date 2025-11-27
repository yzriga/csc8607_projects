# Projets Deep Learning — Template

Installez l’environnement (via `pip` ou `conda`) et utilisez les scripts dans `src/`.
**Les consignes détaillées sont sur le site du cours.**

## Installation rapide
```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# conda (option)
# conda create -n dl-projets python=3.10 -y
# conda activate dl-projets
# pip install -r requirements.txt

# allocation
srun --time=1:00:00 --partition=interactive --gres=gpu:1 --cpus-per-task=2 --mem=8G --pty bash

# allocation (partition normal (durée plus longue))
srun --partition=normal --gres=gpu:1 --time=4:00:00 --cpus-per-task=2 --mem=8G --pty bash

# env
mamba activate deeplearning

# tensorboard
(sur ssh) tensorboard --logdir=runs --port=6006 --bind_all
(local) ssh -L 6006:arcadia-slurm-node-2:6006 tsp-client
(navigateur) http://localhost:6006
````

## Démarrage

* Modifiez `configs/config.yaml` selon votre projet.
* Implémentez les fonctions dans `src/`.
* Utilisez le `Makefile` pour lancer les étapes (ou exécutez les modules Python directement).

> Reportez-vous au site du cours pour les règles, livrables et barème.
