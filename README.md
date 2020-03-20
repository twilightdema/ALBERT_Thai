ALBERT STUDIES ON THAI LANGUAGE DATASETS
========================================

This code repository was modified from original Google ALBERT (https://github.com/google-research/ALBERT) to support experimenting ALBERT model on main Thai language datasets below:

Pretraining Language Model Datasets
-----------------------------------
- BEST2010 (NECTEC)
- THWIKI (Thailand Wikipedia dump file)

Downstream Finetune Datasets
----------------------------
- BEST2010 (Topic Classification)
- TRUEVOICE (Intention Classification)
- WISESIGHT (Sentiment Analysis)
- WONGNAI (Review Rating Prediction)

The default experiments use ALBERT Base V1 model, but it is easy to change to use other ALBERT configurations.

Release Notes
=============

- Initial release: 20 Mar 2020

Results
=======

The results of experiments are shown below:

ALBERT Base V1 - Pretrained with BEST2010 Language Model Dataset

|                | Accuracy  |
|----------------|-----------|
|BEST2010        |54.30      |
|TRUEVOICE       |88.30      |
|WISESIGHT       |70.42      |
|WONGNAI         |55.00      |

ALBERT Base V1 - Pretrained with THWIKI Language Model Dataset

|                | Accuracy  |
|----------------|-----------|
|BEST2010        |TBA        |
|TRUEVOICE       |TBA        |
|WISESIGHT       |TBA        |
|WONGNAI         |TBA        |


Reproducing the Experiments
===========================

Install prerequisites

```
pip3 install -r requirements.txt
```

Download Thai datasets

```
./download_thai_dataset.sh
```

Train Sentencepiece coding model for encoding/decoding text data BEST2010 corpus

```
./0_best2010_train_spm.sh
```

Create training dataset for BEST2010 language model

```
./1_best2010_create_lm_data.sh
```

Pretrain language model on BEST2010 language model dataset

```
./2_best2010_train_lm.sh
```

Finetune pretrained model on all 4 Thai finetune dataset

```
./3_best2010_finetune_best2010.sh
./3_best2010_finetune_truevoice.sh
./3_best2010_finetune_wisesight.sh
./3_best2010_finetune_wongnai.sh
```

Explore experiment results using Tensorboard (Run each command and view Tensorboard at: http://localhost:6006)
```
tensorboard --logdir exports_best2010_best2010
tensorboard --logdir exports_best2010_truevoice
tensorboard --logdir exports_best2010_wisesight
tensorboard --logdir exports_best2010_wongnai
```

The results are also saved in log file at ```eval_results.txt``` in each result directory. The best evaluation accuracy is logged at ```best_trial.txt```

Notes
=====
```
This code repository is a part of research topic: "Overcoming data shortage problem in natural language processing tasks with semi-suprevised learning" - Chulayuth A., Institute of Field Robotics, King Mongkut’s University of Technology Thonburi, 2020.
```
