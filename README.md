# plant-lab2field

Why do plant disease classifiers fail in the field? This repo analyses the lab-to-field gap (PlantVillage → PlantDoc / FieldPlant) factor by factor, across CNNs, ViTs, DINOv2 and CLIP.

- Problem statement and research questions: [docs/research_questions.md](docs/research_questions.md)
- Related work: [docs/literature.md](docs/literature.md)
- Unified label space: [configs/class_map.yaml](configs/class_map.yaml)

## Layout

```
configs/    class maps, experiment configs
src/        reusable code (data, models, interventions, evaluation)
notebooks/  one Kaggle notebook per experiment
results/    CSVs of metrics (the source of truth for every table in the paper)
docs/       paper planning
```

## 12-week plan

| Week | Milestone |
|------|-----------|
| 1–2 | Literature review, verify the class map, data pipeline, leakage-aware PlantDoc split |
| 3–4 | RQ1 baselines: 4 backbone families × 2 field targets |
| 5–7 | RQ2 factor interventions (forward and backward) |
| 8 | RQ3 targeted augmentation + k-shot curves, 3 seeds, significance tests |
| 9–10 | Write the paper |
| 11 | Internal review, figures, clean repo |
| 12 | Submit + arXiv |
