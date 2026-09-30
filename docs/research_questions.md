# Problem Statement and Research Questions (draft v1, 2026-09-29)

## Working title

**Why Do Plant Disease Classifiers Fail in the Field? A Factor-Level Diagnosis of the Lab-to-Field Gap Across CNNs and Vision Foundation Models**

## Problem statement

Plant disease classifiers trained on laboratory images (PlantVillage) reach about 99% accuracy but fall to roughly 30–55% on field photographs (PlantDoc). Prior work has *measured* this gap and tried generic fixes such as domain adaptation, calibration and ensembles, with limited success (Xiang et al. 2026). No study has *decomposed* the gap into its causes (background clutter, multiple leaves and scale, illumination and colour, blur and resolution) or asked whether modern foundation models (DINOv2, CLIP) are robust to some of these causes and not others. Without that diagnosis, practitioners cannot tell whether to collect more field data, segment leaves, or simply switch backbones.

## Research questions

**RQ1: How large is the gap, and does it depend on the backbone?**
Train or probe on PlantVillage, then test on the overlapping classes of two field datasets (PlantDoc and FieldPlant). Compare four families: a CNN (ResNet-50, EfficientNet-B0), a supervised ViT (ViT-B/16), a self-supervised model (DINOv2 ViT-S/B) and a vision-language model (CLIP ViT-B/16, also zero-shot).

**RQ2: Which factors cause the gap? (main novelty)**
Use controlled interventions that change one factor at a time:
- *Forward (lab → field-like):* paste PlantVillage leaves onto real field backgrounds; add multiple leaves and scale changes; shift illumination and colour; add blur, noise and JPEG compression.
- *Backward (field → lab-like):* isolate leaves in PlantDoc/FieldPlant images with SAM, so the background is removed.
For each backbone, the accuracy drop caused by each factor gives the **gap attribution**.

**RQ3: Can a diagnosis-driven fix close the gap cheaply?**
Apply augmentation that targets the factor that matters most for each backbone (e.g. background compositing), then add k = 0, 1, 5, 10 or 20 labelled field images per class. This produces a **label-efficiency curve** per backbone: how many field labels you save by choosing the right backbone and augmentation.

## Expected contributions

1. The first factor-level attribution of the lab-to-field gap, across four backbone families and two field datasets.
2. A clean, leakage-aware evaluation protocol with a class mapping between PlantVillage, PlantDoc and FieldPlant (released on GitHub).
3. Practical guidance: which factor to fix first, and how many field labels each backbone needs.

## Differentiation from the closest work

| | Xiang et al. 2026 | Ahmad et al. 2023 | **Ours** |
|---|---|---|---|
| Backbones | ResNet-50 + frozen FMs | 5 CNNs | CNN, ViT, DINOv2, CLIP |
| Field targets | PlantDoc | PlantDoc, CD&S (corn only) | PlantDoc + FieldPlant, all overlapping crops |
| Cause analysis | Image statistics (correlational) | Background removal only | Controlled one-factor interventions (causal) |
| Label efficiency | No | No | k-shot curves per backbone |

## Scope guardrails (so we finish in 12 weeks)

- Classification only, no detection. Use leaf crops from the bounding boxes.
- Only the classes shared between the datasets: up to 27 for PlantVillage↔PlantDoc (see `configs/class_map.yaml`), fewer for FieldPlant.
- Three seeds for every result; report mean ± std and paired significance tests.
- Everything must run on a Kaggle T4/P100 within about 30 GPU-hours a week. Use frozen features plus linear probes where possible; fully fine-tune only ResNet-50, EfficientNet-B0 and ViT-B.

## Target venues (decide in week 9)

- *Computers and Electronics in Agriculture* (Elsevier, Q1): best fit
- *Frontiers in Plant Science* / *Plant Phenomics*: where the closest work appeared
- *IEEE Access*: faster decisions
- arXiv preprint on the day we submit
