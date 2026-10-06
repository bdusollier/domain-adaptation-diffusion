# Adaptation de Domaine par Modèles Génératifs (FLUX.1)

Ce dépôt regroupe les travaux et outils développés pour l'adaptation de domaine d'images par modèles génératifs de diffusion (**FLUX.1 [dev]**), incluant l'adaptation fine, le conditionnement spatial, l'évaluation du réalisme et l'impact sur des tâches avales de détection d'objets.

---

## Vue d'Ensemble des Composants

Le projet est articulé autour de plusieurs modules clés :

- **Adaptation/** : Pipeline multimodal d'adaptation de domaine combinant guidage par cartes de profondeur, segmentation sémantique et masques d'in-painting via ComfyUI.
- **ControlNet/** : Implémentation et entraînement d'un adaptateur de contrôle léger basé sur **MobileNetV3-Small** conditionné par des masques de segmentation pour FLUX.1.
- **LoRa/** : Scripts et configurations pour l'entraînement d'adaptateurs LoRA (Text-to-Image) sur FLUX.1 avec quantification FP8.
- **ComfyUI/** : Intégrations ComfyUI, incluant le nœud personnalisé pour l'inférence du ControlNet MobileNet (`custom_nodes/comfyui_mobilenet_controlnet`).
- **Workflows/** : Graphes de workflows ComfyUI (fichiers JSON) pour l'inférence interactive et les démonstrateurs.
- **detection/** : Évaluation de l'utilité des images générées via des modèles de détection d'objets YOLOv8 (comparaisons synthétique / réel / combiné).
- **metrics/** : Évaluation quantitative et qualitative du réalisme des images générées (FID, Gram-MMD, projections ACP Inception / Gram / DINOv2).

---

## Données et Modèles Volumineux

Les checkpoints et datasets volumineux ne sont pas versionnés sur Git.
 
  
