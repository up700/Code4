## Requirements
- python==3.7.4
- pytorch==1.8.1
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

## Overview
```
├── root
│   └── dataset
│       ├── ar.train.sst.json  (source-to-target oriented sentence pairs)
│       ├── ar.train.stts.json
│       ├── ar.train.tst.json
│       ├── en_train.json      (labeled source language training data)
│       ├── ar_train.json      (unlabeled target language training data)
│       ├── ar_dev.json
│       ├── ar_test.json
│       ├── ar_tag_to_id.json
│       └── ...
│   └── models
│       ├── __init__.py
│       ├── modeling_span.py
│       ├── modeling_type.py
│       └── modeling_interaction.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── data_utils_span.py
│       ├── eval_span.py
│       └── ...
│   └── pro_contras_bri.py
│   └── span_extraction_transfer.py
│   └── type_prediction_transfer.py
│   └── subtask_combination.py
│   └── run_bash.sh
```

## How to run
```console
sh run_bash.sh <GPU ID> <SOURCE LANGUAGE NAME> <TARGET LANGUAGE NAME> <IF TRAIN> <IF PREDICT>
```
For example, English to Arabic on WikiAnn dataset
```console
sh run_bash.sh 6 Enen ar True False
```
