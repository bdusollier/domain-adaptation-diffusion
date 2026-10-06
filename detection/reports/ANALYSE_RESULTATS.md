# Analyse des Résultats et Comportement d'Entraînement

Ce document résume les observations, analyses théoriques et constatations empiriques concernant la comparaison entre les entraînements **Synthétiques (3D)** et **Réalistes (Flux/ComfyUI)** sur le jeu de données de détection.

---

## 1. Le Paradoxe des Métriques : Réalisme Perceptuel vs Détection d'Objets

Si l'on compare les résultats du dossier `measures/` (FID, GMMD) et ceux de la détection YOLO :

* **Mesures de Réalisme (`measures/`)** :
  * **FID** : 122.28 (Synthétique) -> **95.62** (Réaliste) : amélioration de 26.6 points.
  * **GMMD** : 0.2528 (Synthétique) -> **0.2087** (Réaliste) : amélioration des textures.
  * *Interprétation* : Le pipeline Flux/ComfyUI génère des images dont les textures globales, le grain et la colorimétrie sont beaucoup plus proches de vraies photos.

* **Performances YOLO sur le Test Réel** :
  * **Synthétique** : mAP50 = **0.3775** | Recall = **0.3770**
  * **Réaliste** : mAP50 = **0.2753** | Recall = **0.2879**
  * *Paradoxe* : Bien que plus "réaliste" à l'œil et au sens du FID, le modèle entraîné sur ces images est moins performant en détection pure.

---

## 2. Pourquoi le Modèle Réaliste est-il moins performant en détection ?

### A. Désalignement des Bounding Boxes (Label Drift)
* Les étiquettes (`labels/*.txt`) ont été créées sur les images 3D de synthèse où la position de chaque véhicule est mathématiquement exacte.
* L'étape de diffusion (img2img / ComfyUI) altère subtilement la géométrie de l'objet : déformation d'un blindage, déplacement d'une roue, flou sur les contours du canon.
* La boîte englobante d'origine ne cadre plus parfaitement le véhicule sur l'image réaliste, ce qui pénalise fortement la régression de boîte (`box_loss`) et le calcul de l'IoU.

### B. Primitives géométriques rigides vs "Hallucinations" de texture
* En 3D synthétique, même si les textures sont simples, **les arêtes, les angles et la silhouette sont parfaitement tranchés et stables**.
* Le réseau de neurones YOLO apprend alors des filtres géométriques universels qui se transfèrent très bien aux vrais véhicules.
* Les images générées par diffusion contiennent des micro-textures et des hallucinations propres au modèle génératif sur lesquelles le détecteur se focalise à tort.

---

## 3. Analyse du comportement d'entraînement : L'effondrement après l'Epoch 1

L'examen attentif du fichier `results.csv` pour l'entraînement réaliste met en évidence un phénomène critique :

### Évolution epoch par epoch (Modèle Réaliste, YOLOv8s, `val_real`) :
* **Epoch 1** : `train/cls_loss` = 3.94 | `val/cls_loss` = **2.75** (meilleur) | **mAP50 = 0.2868** (meilleur)
* **Epoch 2** : `train/cls_loss` = 1.74 | `val/cls_loss` = 3.54 | mAP50 = 0.1562
* **Epoch 6** : `train/cls_loss` = 1.48 | `val/cls_loss` = 5.17 | mAP50 = 0.0363
* **Epoch 12** : `train/cls_loss` = 1.17 | `val/cls_loss` = 5.13 | **mAP50 = 0.0083** (quasi nul)
* **Epoch 21** : Arrêt anticipé (*EarlyStopping* avec patience 20 car aucun progrès depuis l'epoch 1).

### Explication du phénomène :
1. **Surapprentissage sur la signature du générateur (Generator Fingerprint)** :
   Le modèle commence à apprendre les artefacts de fréquence spécifiques à Flux plutôt que la structure réelle du véhicule militaire.
2. **Oubli catastrophique (Catastrophic Forgetting)** :
   Partant des poids pré-entraînés sur COCO (`yolov8s.pt`), l'epoch 1 bénéficie encore des filtres réels appris sur COCO. Dès les epochs suivantes, un taux d'apprentissage trop élevé réécrit ces couches extractrices et détruit la capacité de généralisation sur des photos réelles.

---

## 4. Solutions Appliquées pour Corriger l'Entraînement

Pour permettre au modèle réaliste d'apprendre sans détruire ses représentations réelles, les mesures suivantes sont implémentées :

1. **Gel du Backbone (`--freeze 10`)** :
   * Les 10 premières couches de YOLOv8 (l'extracteur de features pré-entraîné sur COCO) sont verrouillées.
   * Seules les couches de détection (Neck & Head) sont mises à jour.
   * Empêche l'oubli catastrophique des caractéristiques réelles de base.

2. **Réduction du Taux d'Apprentissage (`--lr0 0.0001`)** :
   * Un learning rate plus doux (1e-4 au lieu de 1e-2 / 1e-3) pour stabiliser la convergence et éviter d'osciller sur les textures génératives.

3. **Curriculum Learning Adapté** :
   * Pré-entraînement sur la géométrie synthétique nette.
   * Fine-tuning à bas learning rate sur les textures réalistes.
