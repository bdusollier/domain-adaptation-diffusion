# Analyse Diagnostique Approfondie : Pourquoi le Modèle Réaliste (Flux) Dégrade les Performances

**Date :** 2026-08-27  
**Expérience de référence :** `exp_20260826_150337_yolov8s_img1024_ep100_b16_lr0.0001_freeze10_valreal`  
**Dataset :** `dataset2` (365 images de test réelles, 1 374 véhicules annotés)  
**Modèles comparés :** YOLOv8s (`freeze=10`, `lr0=0.0001`, validation réelle `val_real`)

---

## 1. Synthèse des Résultats d'Évaluation sur le Jeu de Test

| Métrique (Test Set) | COCO Baseline | Réaliste (Flux) | Curriculum (Synth $\to$ Flux) | Synthétique (3D) | Meilleur |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | 0.3392 | 0.5340 | 0.5679 | **0.6369** | **Synthétique** |
| **mAP50-95** | 0.1991 | 0.2457 | 0.2755 | **0.3193** | **Synthétique** |
| **Précision** | **0.7857** | 0.7061 | 0.6487 | 0.6975 | **COCO-Base** |
| **Rappel** | 0.4003 | 0.5072 | 0.5537 | **0.5891** | **Synthétique** |

---

## 2. Décomposition Diagnostique par Taille d'Objet (Instance-Level)

Une analyse quantitative exhaustive a été menée sur l'ensemble des **1 374 véhicules** du jeu de test (seuil de confiance $\ge 0.25$, seuil IoU $\ge 0.5$) :

```
             ┌─────────────────────────────────────────────────────────────┐
   Grandes   │  Synthétique: 72.2% (324/449)                               │  Différence
  (> 96 px)  │  Réaliste   : 70.6% (317/449)                               │   -1.6 pt  (Quasi identique)
             └─────────────────────────────────────────────────────────────┘
             ┌─────────────────────────────────────────────────────────────┐
  Moyennes   │  Synthétique: 64.9% (565/870)                               │  Différence
 (32-96 px)  │  Réaliste   : 42.4% (369/870)                               │  -22.5 pts (196 VÉHICULES PERDUS !)
             └─────────────────────────────────────────────────────────────┘
             ┌─────────────────────────────────────────────────────────────┐
  Petites    │  Synthétique: 16.4% (9/55)                                  │  Différence
  (< 32 px)  │  Réaliste   :  9.1% (5/55)                                  │   -7.3 pts
             └─────────────────────────────────────────────────────────────┘
```

### Constats majeurs :
1. **Sur les gros véhicules en plan rapproché (> 96 px)** : Le modèle Réaliste fait jeu égal avec le Synthétique (70.6% vs 72.2%). Sur les gros plans, le photoréalisme de Flux est bien exploité.
2. **Sur les véhicules de taille moyenne (32 à 96 px)** : C'est **le cœur du jeu de données (63.3% des véhicules)**, et c'est précisément là qu'a lieu **l'effondrement (-22.5 points de rappel, 196 véhicules manqués)**.

---

## 3. Bilan Comparatif des Types d'Erreurs

| Indicateur | Synthétique (3D) | Réaliste (Flux) | Curriculum (Synth $\to$ Flux) | Analyse |
| :--- | :---: | :---: | :---: | :--- |
| **Vrais Positifs (TP)** | **898** | 691 | 792 *(+101 vs Réaliste)* | Le Synthétique détecte **207 véhicules de plus**. |
| **Faux Négatifs (FN)** | **476** | 683 | 582 | Le Réaliste rate la moitié des cibles du jeu de test. |
| **Faux Positifs (FP)** | 701 | **318** | 560 | Le Réaliste génère **2× moins de fausses alertes** (Précision 68.5%). |
| **Décalage moyen du centre ($\Delta$ px)** | **9.04 px** | 11.79 px | 10.29 px | **+30% d'erreur de centrage** sur les boîtes du Réaliste. |
| **IoU moyen sur les TP** | **0.768** | 0.754 | 0.766 | Les boîtes du Synthétique collent plus précisément au GT. |

---

## 4. Les 3 Causes Fondamentales de la Dégradation

### A. Le "Biais de Texture" (*Texture Bias*) vs Silhouettes Éloignées
* **Modèle Synthétique** : En 3D pure, même un char de 40 pixels possède des **arêtes droites, des angles polygonaux tranchés et des contrastes nets** avec le terrain. Le réseau apprend un détecteur de *formes géométriques universelles*.
* **Modèle Réaliste** : Flux génère des micro-textures complexes (peinture écaillée, reflets, boue, dégradés d'ombrage). Le modèle Réaliste apprend que pour être un véhicule militaire, l'objet *doit présenter ces micro-textures*.
* **Échec sur le test réel** : À moyenne distance (40-80 px), un vrai véhicule militaire apparaît sur la photo comme une silhouette sombre avec très peu de résolution de texture. Le modèle Réaliste l'ignore par manque de micro-détails, tandis que le modèle Synthétique le détecte immédiatement grâce à sa silhouette géométrique.

### B. L'effet de "Camouflage Artificiel" induit par la Diffusion
* Lors du processus `img2img` avec Flux, le modèle de diffusion harmonise l'éclairage et les teintes de l'objet avec l'arrière-plan (végétation, boue, poussière).
* Sur les cibles moyennes/petites, le véhicule est souvent partiellement "fondu" dans le décor, ce qui supprime le contraste des contours. Le détecteur ne parvient plus à segmenter l'objet du fond.

### C. La Dérive Géométrique des Boîtes (*Center Shift* +30%)
* Les annotations `txt` proviennent de la caméra 3D mathématique.
* Quand Flux génère l'image réaliste, le processus de diffusion déforme légèrement la silhouette (chenilles modifiées, tourelle arrondie, canon déplacé).
* Le modèle Réaliste est donc entraîné avec un **bruit d'étiquetage spatial** :
  * Le décalage moyen de centrage passe de **9.04 px** à **11.79 px**.
  * Cela pénalise directement la fonction de perte `val/box_loss` et abaisse le mAP50-95.

---

## 5. L'Apport du Curriculum Learning

L'entraînement séquentiel en deux étapes (**Étape 1 : Synthétique $\to$ Étape 2 : Réaliste à bas LR**) a permis de :
* Récupérer **+101 vrais véhicules détectés** par rapport au Réaliste seul (Rappel moyen : 42.4% $\to$ 53.9%).
* Réduire l'erreur de centrage des boîtes (11.79 px $\to$ 10.29 px).
* Faire progresser le mAP50 de **0.5340 à 0.5679 (+3.39 points)**.

> Cela prouve que **l'ancrage géométrique initial 3D empêche le modèle de perdre la notion de forme** lors de l'adaptation aux textures réalistes.

---

## 6. Plan d'Action Recommandé pour Débloquer le Réalisme Flux

Pour que les images issues de Flux surpassent définitivement le synthétique 3D pur :

1. **Conditionnement strict par ControlNet (Canny / LineArt / Depth)** :
   * Forcer Flux à préserver 100% de la géométrie et des contours extérieurs du véhicule, quelle que soit la taille de l'objet.
2. **Denoising adaptatif selon la surface de l'objet** :
   * Gros plans (>96 px) : `denoise = 0.55 - 0.65` (génération de textures réalistes riches).
   * Cibles moyennes/petites (<96 px) : `denoise = 0.30 - 0.40` max (conserver la netteté et le contraste des silhouettes pour éviter le camouflage).
3. **Réalignement automatique des boîtes d'annotation (SAM / Grounding DINO)** :
   * Re-segmenter les images générées pour éliminer l'erreur de centrage (+30%) et fournir des labels parfaits à YOLO.
