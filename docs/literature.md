# Related Work — Lab-to-Field Gap in Plant Disease Recognition

Last updated: 2026-09-29. Status column: `to-read` / `read` / `cite`.

## A. Closest to our paper (must read first, must cite and position against)

| # | Paper | Venue / Year | What they did | What they did NOT do (our opening) | Status |
|---|-------|--------------|---------------|-------------------------------------|--------|
| 1 | [Quantifying the reliability gap in cross-domain plant disease classification](https://consensus.app/papers/details/36d06ae4fec55c9a8327d3b6ad7cc5cc/) — Xiang et al. | Frontiers in Plant Science, 2026 | PV→PlantDoc; ResNet-50 drops 67.7 pts; calibration, selective prediction, AdaBN, DANN; frozen-backbone comparison (incl. DINOv2); measured saturation/edge/foreground statistics; Grad-CAM | Measured *correlates* of shift but did not *intervene* factor-by-factor; no label-efficiency curve; single target dataset | to-read |
| 2 | [Backbone diversity beats text supervision: frozen multi-foundation-model fusion for in-the-wild plant disease recognition](https://pmc.ncbi.nlm.nih.gov/articles/PMC13503617/) | PMC, 2026 | Fuses DINOv2 + DINOv3 + CLIP (ViT-L), 80.23% on PlantWild | In-domain field training; not a lab→field transfer study | to-read |
| 3 | [CLIP in the Field: Plant Disease Detection via Zero-Shot and Fine-Tuning](https://link.springer.com/chapter/10.1007/978-3-032-05925-3_2) | Springer LNCS, 2025 | CLIP zero-shot vs fine-tuning on field images | Only CLIP; no factor diagnosis | to-read |
| 4 | [Robust Plant Disease Diagnosis with Few Target-Domain Samples](https://arxiv.org/pdf/2510.12909) | arXiv, 2025 | Few target samples to adapt | Check which backbones and datasets they use | to-read |
| 5 | [From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild (MSUN)](https://consensus.app/papers/details/b5a9629da2cc5cb692250134eb96e230/) — Wu et al. | Plant Phenomics, 2023 (89 cit.) | UDA method; 56.06% on PlantDoc | Standard UDA baseline to compare against | to-read |
| 6 | [Toward Generalization of DL-Based Plant Disease Identification Under Controlled and Field Conditions](https://consensus.app/papers/details/808737f2fa275054b4be3e9cec47f546/) — Ahmad et al. | IEEE Access, 2023 (80 cit.) | Corn only, 5 datasets; background removal (RGBA) helps | CNNs only, one crop; background effect not isolated from other factors | to-read |

## B. Datasets

| Dataset | Type | Notes |
|---------|------|-------|
| PlantVillage | Lab, ~54k images, 38 classes | Source domain |
| PlantDoc | Web-scraped field, 2,569 images, 27 classes | Target 1; known label noise and some lab images; duplicates, so use a parent-image-aware split (as in #1) |
| [FieldPlant](https://consensus.app/papers/details/64eb9609ffec5a20836919034b3a8a08/) — Moupojou et al., IEEE Access 2023 (226 cit.) | Real plantation, 5,170 images / 8,629 leaves, 27 classes, pathologist-verified | Target 2: makes our results stronger than a PlantDoc-only study |
| PlantWild / LeafNet ([arXiv 2602.13662](https://arxiv.org/html/2602.13662)) | Large in-the-wild / VLM benchmark | Optional extra target |

## C. Mitigation methods (baselines and ideas)

- [Cross-domain few-shot learning for crop disease identification](https://consensus.app/papers/details/edf37e9f772551d4a6039047aa10cf9d/) — Yang et al., Frontiers Plant Sci 2024
- [Multi-task learning + CORAL alignment, PV→PlantDoc](https://consensus.app/papers/details/290ac061c2ec5dc499ae1510486d61d4/) — Rustamova et al., ICECCO 2026
- [DANN-CORAL hybrid](https://consensus.app/papers/details/f2ef97d7d64a5e39bab267cf1e7830a7/) — Prasad et al., 2024
- [Dataset merging + ensemble of lightweight CNNs, 67.45% on PlantDoc](https://consensus.app/papers/details/2be93625f8ab5a70baa33f98984e6565/) — Aarizou et al., J. Phytopathology 2026
- [SAM + FCDD leaf isolation, +10% on PlantDoc](https://consensus.app/papers/details/e5668984753f5434887a418df37614bc/) — Moupojou et al., IEEE Access 2024
- [Unified self-supervised framework (BYOL + MIM + contrastive), 77.82% on PlantDoc](https://consensus.app/papers/details/5adafc5619595df89bb86afc7b306af8/) — Huan et al., Electronics 2025
- [Lightweight CLIP adaptation for few-shot plant disease (PD-Adapter)](https://link.springer.com/article/10.1007/s00530-026-02298-7) — Multimedia Systems 2026
- [QLoRA adapters on DINOv2 for few-shot plant disease](https://www.researchsquare.com/article/rs-9014055/v1) — Research Square preprint
- [Towards automatic field plant disease recognition (background replacement augmentation)](https://www.sciencedirect.com/science/article/abs/pii/S0168169921005408) — Comput. Electron. Agric. 2021
- [Diffusion models for plant disease augmentation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1280496/full) — Frontiers Plant Sci 2023

## D. Still to search (week 1)

- [ ] Original PlantVillage (Hughes & Salathé 2015; Mohanty et al. 2016) and PlantDoc (Singh et al., CODS-COMAD 2020) papers: required citations
- [ ] Papers on PlantVillage background/capture bias (e.g. "background alone predicts the class")
- [ ] General domain-generalization background: ImageNet-C (Hendrycks), shortcut learning (Geirhos 2020), DINOv2 (Oquab 2023), CLIP (Radford 2021)
- [ ] Full-text check of #1 and #4 so we don't duplicate their experiments
