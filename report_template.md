# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : ZRIGA Yahia
- **Projet** : _Intitulé (dataset × modèle)_
- **Dépôt Git** : _URL publique_
- **Environnement** : `python == ...`, `torch == ...`, `cuda == ...`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) :
- **Type d’entrée** (image / texte / audio / séries) :
- **Tâche** (multiclasses, multi-label, régression) :
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) :
- **Nombre de classes** (`meta["num_classes"]`) :

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ? 
```
Le dataset utilisé est Tiny ImageNet, disponible sur HuggingFace sous l’identifiant :
https://huggingface.co/datasets/zh-plus/tiny-imagenet
Format :

Images couleur RGB

Dimensions 64 × 64

200 classes

Les données sont fournies sous la forme :

train : 100 000 images

valid : 10 000 images

Il n’y a pas de split test officiel → un sous-échantillon du split valid a été utilisé pour créer un split val/test.

Type d’entrée :
Images RGB → classification multiclasses.

Dimensions d’entrée attendues : (3, 64, 64)
Nombre de classes : 200
```
### 1.2 Splits et statistiques
**D2.** Donnez la taille de chaque split et le nombre de classes.  
| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
|------:|----------:|--------------------------------------------------------|
| Train |     100 000      |        Split officiel HuggingFace                                                |
| Val   |    8 000       |                80% du split “valid”                                        |
| Test  |      2 000     |                          20% du split “valid”                              |

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).
```
Le dataset officiel ne contient pas de split test.
J’ai donc appliqué un split :

train_test_split(..., test_size=0.2, random_state=42, shuffle=True)

- random seed = 42

- mélange aléatoire activé (shuffle=True)

- pas de stratification (la bibliothèque HF ne fournit pas les labels sous forme de liste compatible)
```
**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.
```
La distribution des classes a été calculée en parcourant l’ensemble du split train avec un compteur de labels.
Les résultats montrent que chaque classe contient exactement 500 images: 
```
![alt text](class_distribution.png)
```
L’histogramme généré (class_distribution.png) montre une distribution parfaitement uniforme sur les 200 classes, sans déséquilibre.

Impact sur l’entraînement :

  - Le modèle ne sera pas biaisé vers une classe plus fréquente.

  - L’accuracy reflète bien les performances globales.

  - Aucune technique de rééquilibrage n’est nécessaire (pas besoin de class weights ni oversampling).
  ```
**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).
```
En analysant les données Tiny ImageNet, nous avons constaté les points suivants :

  - Toutes les images sont désormais uniformisées en RGB 64×64 (conversion forcée en RGB dans le code pour corriger certaines images en niveaux de gris).

  - Le dataset est parfaitement équilibré : 500 images par classe, 200 classes.

  - Chaque échantillon possède un seul label → problème de classification monoclass (et non multi-label).

  - Les images ont une structure homogène : pas de variation de taille, pas de texte, pas de données manquantes ou corrompues.

  - Aucun bruit structurel particulier n’a été détecté.

  - L’organisation du dataset est propre et adaptée aux architectures convolutionnelles.
```
### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = __, center-crop = __, normalize = (mean=__, std=__)…
- Audio : resample = __ Hz, mel-spectrogram (n_mels=__, n_fft=__, hop_length=__), AmplitudeToDB…
- NLP : tokenizer = __, vocab = __, max_length = __, padding/truncation = __…
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  
```
Resize((64, 64))
ToTensor()
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])


Resize((64, 64))

    Paramètres exacts : taille de sortie = (64, 64)

    Pourquoi ?

        - Tiny ImageNet est défini avec des images 64×64, donc ce prétraitement garantit que toutes les images ont exactement la même résolution.

        - Ça simplifie énormément l’implémentation du modèle (les convolutions s’attendent à une taille fixe) et évite d’avoir à gérer des tailles variables dans les DataLoaders.

ToTensor()

    Paramètres : aucun paramètre additionnel.

    Pourquoi ?

        - Convertit les images PIL en tenseurs PyTorch de forme (C, H, W) et met les valeurs de pixels dans [0, 1].

        - C’est le format attendu par les couches nn.Conv2d, la loss CrossEntropyLoss, etc. Sans cette étape, le modèle ne pourrait pas consommer les données.

Normalize(mean, std)

    Paramètres exacts :

        - mean = [0.485, 0.456, 0.406]

        - std = [0.229, 0.224, 0.225]

    Pourquoi ?

        - Ce sont les statistiques classiques d’ImageNet (même famille de données que Tiny ImageNet).

        - La normalisation centre et met à l’échelle chaque canal, ce qui :

            - stabilise les gradients,

            - aide la convergence de l’optimisation,

            - fonctionne particulièrement bien avec les architectures à BatchNorm + ReLU comme notre réseau résiduel pré-activation.

En résumé, ces trois prétraitements garantissent que les données ont une forme compatible avec le modèle et une distribution numérique adaptée à l’entraînement profond.

```
**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?
```
Aucune différence → mêmes prétraitements (sans aléatoire).
```

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
```
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomResizedCrop(
    size=64,
    scale=(0.8, 1.0),
    ratio=(0.9, 1.1)
)

1) RandomHorizontalFlip(p = 0.5)

Paramètre précis : probabilité p = 0.5.

Pourquoi ?

  - Pour introduire une invariance horizontale réaliste : dans Tiny ImageNet, une même classe (lion, voiture, oiseau, etc.) peut apparaître orientée à gauche ou à droite.

  - Cela augmente artificiellement la diversité du dataset sans déformer l’objet.

  - Aide à éviter l’overfitting et améliore la généralisation du modèle.

2) RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.9, 1.1))

Paramètres précis :

  - size = 64 → image finale 64×64

  - scale = (0.8, 1.0) → zoom / recadrage entre 80% et 100% de l’image

  - ratio = (0.9, 1.1) → ratio largeur/hauteur très proche du carré

Pourquoi ?

  - Simule de légères variations de cadrage, zoom ou position de l’objet.

  - Permet au modèle d’être robuste à :

    - des zooms légers

    - des variations de position

    - des recadrages faibles

  - Améliore la capacité du modèle à généraliser aux images réelles.
```
**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.
```
Oui, les deux augmentations préservent la classe réelle de l’image:

1) RandomHorizontalFlip(p=0.5)

  - Ne modifie PAS la catégorie.

  - Un chien, un bus ou un oiseau reste le même objet après un flip horizontal.

  - L’opération ne change pas le contenu sémantique → label parfaitement conservé.

2) RandomResizedCrop(scale, ratio)

  - Ne modifie PAS la catégorie tant que l’objet principal reste visible, ce qui est le cas ici :

      - scale min = 0.8 → on garde au moins 80% de l’image

      - ratio presque carré → pas de déformation extrême

  - Le modèle doit justement être robuste aux variations naturelles de cadrage.

  - Donc le label est correctement préservé.
```

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.
```
Voici trois images extraites du DataLoader après application des prétraitements et des augmentations du split train :
```
![alt text](sanity_examples.png)
```
Commentaire :
Les images affichées sont cohérentes :

  - La normalisation a bien été inversée pour la visualisation, ce qui permet de vérifier que les couleurs restent plausibles.

  - Les recadrages aléatoires (RandomResizedCrop) sont visibles, avec un zoom ou un recentrage variable selon l’exemple.

  - Le flip horizontal peut se produire avec une probabilité de 0.5, ce qui enrichit la diversité visuelle du dataset.

  - Les labels montrés correspondent à des entiers dans [0, 199], ce qui est conforme au format Tiny ImageNet.

Ces sanity checks confirment que preprocessing + augmentation fonctionnent correctement.
```
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.
```
Lors de l’inspection d’un batch provenant du train_loader, nous avons obtenu :

- Images : torch.Size([64, 3, 64, 64]) -> batch de 64 images, chacune au format 3×64×64
- Labels : torch.Size([64])
- Cohérence avec meta["input_shape"] :
Dans meta, l’entrée est définie comme : meta["input_shape"] = (3, 64, 64)
-> Ce qui correspond exactement à la forme des images du batch.
Les dimensions sont donc correctes et parfaitement compatibles avec l’architecture du modèle (résidual CNN).
```
---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `accuracy` → score = `0.5%`
- **Prédiction aléatoire uniforme** — Métrique : `accuracy` → score = `0.5%`  

![alt text](image-2.png)
_Commentez en 2 lignes ce que ces chiffres impliquent._
```
Ces scores très faibles illustrent qu’un modèle non entraîné ne peut pas dépasser 0.5% d’accuracy simplement par hasard.
Ils serviront de seuil minimal : tout modèle opérationnel doit largement dépasser cette baseline.
```
### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).

- **Configuration actuelle des hyperparamètres du modèle**  
  - Nombre de blocs par stage : (B1, B2, B3) = (2, 2, 2)  
  - Canaux des stages 2 et 3 : (C2, C3) = (128, 256)  

- **Description couche par couche** :

  - **Input**  
    - Entrée : image RGB de forme (3, 64, 64).

  - **Couche d’initialisation**  
    - Conv2d(3 → 64, kernel_size=3, stride=1, padding=1)  
    - Pas de pooling à ce niveau.

  - **Stage 1** (résolution conservée, répété B1 = 2 fois)  
    Chaque bloc résiduel pré-activation contient :
    1. BatchNorm2d(64) → ReLU → Conv2d(64 → 64, 3×3, stride=1, padding=1)  
    2. BatchNorm2d(64) → ReLU → Conv2d(64 → 64, 3×3, stride=1, padding=1)  
    3. Chemin court : identité (pas de projection, même dimension)  
    4. Addition résiduelle : `out = conv2 + shortcut`

  - **Stage 2** (réduction de la résolution, répété B2 = 2 fois)

    - **Premier bloc de Stage 2** :
      - Chemin long :
        1. BatchNorm2d(64) → ReLU → Conv2d(64 → 128, 3×3, stride=2, padding=1)  
        2. BatchNorm2d(128) → ReLU → Conv2d(128 → 128, 3×3, stride=1, padding=1)
      - Chemin court :
        - Conv2d(64 → 128, kernel_size=1, stride=2) pour adapter résolution et canaux  
      - Addition résiduelle : `out = conv2 + projection`

    - **Blocs suivants de Stage 2** (ici 1 bloc supplémentaire) :
      - Même schéma que Stage 1 mais avec 128 canaux :
        - BN(128) → ReLU → Conv2d(128 → 128, 3×3, stride=1, padding=1)  
        - BN(128) → ReLU → Conv2d(128 → 128, 3×3, stride=1, padding=1)  
        - Chemin court : identité  
        - Addition résiduelle.

  - **Stage 3** (nouvelle réduction de résolution, répété B3 = 2 fois)

    - **Premier bloc de Stage 3** :
      - Chemin long :
        1. BatchNorm2d(128) → ReLU → Conv2d(128 → 256, 3×3, stride=2, padding=1)  
        2. BatchNorm2d(256) → ReLU → Conv2d(256 → 256, 3×3, stride=1, padding=1)
      - Chemin court :
        - Conv2d(128 → 256, kernel_size=1, stride=2)  
      - Addition résiduelle.

    - **Blocs suivants de Stage 3** (ici 1 bloc supplémentaire) :
      - BN(256) → ReLU → Conv2d(256 → 256, 3×3, stride=1, padding=1)  
      - BN(256) → ReLU → Conv2d(256 → 256, 3×3, stride=1, padding=1)  
      - Chemin court : identité  
      - Addition résiduelle.

  - **Tête de classification**
    - Global Average Pooling (AdaptiveAvgPool2d(output_size=1))  
      → sortie de forme (batch_size, 256, 1, 1)
    - Flatten → (batch_size, 256)
    - Linear(256 → 200) pour produire les logits de classification.

- **Loss function**  
  - Tâche : classification multi-classe à 200 classes.  
  - Fonction de perte : `nn.CrossEntropyLoss`.

- **Sortie du modèle**  
  - Forme des logits : `(batch_size, num_classes) = (batch_size, 200)`.

- **Nombre total de paramètres**  
  - Nombre de paramètres entraînables : `2 825 224`.

**Rôle des deux hyperparamètres du modèle**

- **Nombre de blocs par stage (B1, B2, B3)**  
  Cet hyperparamètre contrôle **la profondeur** du réseau.  
  - Avec (2, 2, 2), le modèle est plus léger et plus rapide à entraîner, ce qui limite le risque de sur-apprentissage et réduit le coût de calcul.  
  - Avec (3, 3, 3), le réseau devient plus profond : il peut modéliser des fonctions plus complexes, mais il est plus long à entraîner et peut sur-apprendre si la régularisation ou les données ne sont pas suffisantes.  
  Je pars de (2, 2, 2) comme configuration de base et je comparerai (3, 3, 3) dans la mini grid search.

- **Nombre de canaux des stages 2 et 3 (C2, C3)**  
  Cet hyperparamètre contrôle **la largeur** des couches convolutionnelles dans les stages profonds.  
  - (C2, C3) = (128, 256) donne un modèle plus large, avec plus de paramètres, donc une capacité de représentation plus élevée mais aussi un risque plus fort d’overfit et un coût mémoire plus élevé.  
  - (C2, C3) = (96, 192) réduit le nombre de canaux, donc le nombre de paramètres et le coût de calcul, au prix d’une capacité de représentation potentiellement plus faible.  
  Je commence avec (128, 256) comme modèle de référence, puis je testerai (96, 192) pour voir l’impact sur la convergence, l’overfit et les performances de validation.


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

- **Loss initiale attendue** (multi-classe)  
  Pour 200 classes, si les logits sont presque nuls (distribution uniforme), on s’attend à une perte de l’ordre de :

  `-log(1/200) ≈ log(200) ≈ 5.30`

- **Loss initiale observée sur un batch**  
  En faisant un forward sur un batch d’entraînement :

  - Forme du batch d’images : `(64, 3, 64, 64)`  
  - Forme du batch de labels : `(64,)`  
  - Forme de la sortie du modèle (logits) : `(64, 200)`  
  - Loss initiale mesurée : **5.36** (`CrossEntropyLoss`)
  ![alt text](image-1.png)

  Cette valeur est très proche de la valeur théorique attendue (~5.30) pour une prédiction presque uniforme sur 200 classes, ce qui montre que le modèle et la fonction de perte sont cohérents.

- **Vérification des gradients**

  En ajoutant un appel à `loss.backward()` sur ce même batch, les gradients des paramètres du modèle sont non nuls (par exemple, la norme des gradients de la première couche convolutionnelle est strictement positive).  
  Cela confirme que :
  - les shapes entrée/sortie sont correctes,
  - la loss est bien définie,
  - le modèle est prêt pour l’entraînement.
  ![alt text](image.png)
---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR    | WD     | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
|---------------------|-------|--------|-------|-------|-------------------------|----------|-------|
|                     |       |        |       |       |                         |          |       |
|                     |       |        |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


