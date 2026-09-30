# Running on Kaggle (free tier)

1. **New Notebook** → Settings: *Accelerator* = GPU T4 x2 (or P100), *Internet* = On.
   Internet needs a phone-verified Kaggle account.
2. **Add Input → Datasets tab** (not Notebooks!) → search `plantvillage dataset` and add the one by
   **abdallahalidev**. It contains a `color/` folder with 38 class folders.
   Notebooks added as input only hold HTML and outputs, not images.
3. PlantDoc comes straight from its official GitHub repo, so no Kaggle dataset is needed.
   First cell:

```python
!git clone -q https://github.com/23f1000642/plant-lab2field.git
!git clone -q --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git /kaggle/working/PlantDoc
%cd /kaggle/working/plant-lab2field
!pip install -q open_clip_torch
# locate the PlantVillage colour folder (the input path layout varies)
!find /kaggle/input -maxdepth 5 -type d -name color
!ls /kaggle/working/PlantDoc
```

4. Put the `color` path printed above into `PV`, then check the class map
   (fix `configs/class_map.yaml` if any folder shows as MISSING):

```python
PV = "/kaggle/input/.../plantvillage dataset/color"   # <- paste the path from step 3
PD = "/kaggle/working/PlantDoc"
!python src/check_class_map.py --plantvillage "{PV}" --plantdoc {PD}/train
!python src/check_class_map.py --plantvillage "{PV}" --plantdoc {PD}/test
```

5. Run RQ1 (about 30–40 min on a T4 for all 5 backbones):

```python
!python src/rq1_frozen_probe.py --plantvillage "{PV}" --plantdoc {PD}
```

6. Download `results/rq1_frozen_probe.csv`, copy it into the repo's `results/` and commit it.

GPU budget: about 30 h a week. Log every run's hours in `results/gpu_log.md` so we don't run out in weeks 5–8.
