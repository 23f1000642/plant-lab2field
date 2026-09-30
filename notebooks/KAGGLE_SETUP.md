# Running on Kaggle (free tier)

1. **New Notebook** → Settings: *Accelerator* = GPU T4 x2 (or P100), *Internet* = On.
   Internet needs a phone-verified Kaggle account.
2. **Add Input** → search for and add:
   - `plantvillage-dataset` (by abdallahalidev), which contains `plantvillage dataset/color/<class>/`
   - a PlantDoc classification copy with `train/<class>/` and `test/<class>/` folders
     (search "PlantDoc"; check the folder layout in the right-hand panel).
3. First cell:

```python
!git clone https://github.com/<your-username>/plant-lab2field.git
%cd plant-lab2field
!pip install -q open_clip_torch
!ls /kaggle/input   # confirm the exact input paths
```

4. Check the class map (fix `configs/class_map.yaml` if any folder shows as MISSING):

```python
!python src/check_class_map.py --plantvillage "/kaggle/input/plantvillage-dataset/plantvillage dataset/color" --plantdoc /kaggle/input/<plantdoc-folder>/train
```

5. Run RQ1 (about 30–40 min on a T4 for all 5 backbones):

```python
!python src/rq1_frozen_probe.py --plantvillage "/kaggle/input/plantvillage-dataset/plantvillage dataset/color" --plantdoc /kaggle/input/<plantdoc-folder>
```

6. Download `results/rq1_frozen_probe.csv`, copy it into the repo's `results/` and commit it.

GPU budget: about 30 h a week. Log every run's hours in `results/gpu_log.md` so we don't run out in weeks 5–8.
