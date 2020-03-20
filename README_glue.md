
ALBERT STUDIES ON THAI GLUE DATASETS
====================================
This code repository was modified from original Google ALBERT (https://github.com/google-research/ALBERT).

Reproducing the Experiments
===========================

Download GLUE dataset
```
python3 download_glue_data.py --data_dir glue_data --tasks all
```

Run experiments on GLUE dataset
```
./run_glue.sh
```