# vlm-tamp

## Installation

### Set up the environment

```
cd vlm-tamp/
conda env create -f env.yml
conda activate vlm-tamp
```

### Download datasets

Download the `tpvqa_dalle_v0` dataset from [here](https://drive.google.com/drive/folders/1EsJXV42pMpn6uN50y-Hrh9BT9JAPMs80?usp=sharing), and put `tpvqa_dalle_v0/` in the `datasets/` folder.

### Run one of the task

```
python src/clean_dishes.py
```