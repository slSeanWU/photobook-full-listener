# Listener Model for PhotoBook Game


## Installation

```bash
conda create -n photobook python=3.8.10
conda activate photobook
pip install -r requirements.txt

python
>>> import nltk
>>> nltk.download('punkt')
```

* get `logs.zip` and `images.zip` at [photobook_dataset](https://github.com/dmg-photobook/photobook_dataset/), unzip and save inside `data/`

## Data Preprocessing

1. Read `../data/data_splits.json` and save processed log data to `../data/{split}_sections.pickle`

     ```bash
     cd preprocess
     python dialogue_segmentation.py
     ```

2. Generate CLIP score

* Read `../data/{split}_sections.pickle` and save data to `../data/{split}_clean_sections.pickle`

  ```bash
  python process_section.py
  ```

3. Extract image features with Segformer

* Save features at `../data/image_feats.pickle`, the saved data is a dictionary (key: image path, value: hidden features)

  ```bash
  python process_image.py
  ```
  
## Training and Inference

* Edit hyperparams in `model/variables.py`
* Training (with the best "CLIPScore to all layers" configuration)

  ```zsh
  python3 train.py config_paper/vlscore_all.json exp/vlscore_all
  ```
* Inference

  ```zsh
  python3 inference.py exp/vlscore_all
  ```

## Takmaz Baseline

* To run the Takmaz baseline

  ```zsh
  python3 takmaz_baseline/train.py
  ```

## Reference Chain Extraction

* To reproduce the whole extraction and evaluation procedure described in the
  paper, run these commands in `chain-extraction`.

  ```zsh
  python src/extract_segments.py out/all_segments.dict --stopwords --meteor --from_first_common --utterances_as_captions
  python src/make_chains.py out/all_segments.dict out/all_chains.json --score f1
  python src/make_gold_chains.py out/gold_chains.json --from_first_common --first_reference_only
  python src/make_dataset.py out/all_chains.json out/gold_chains.json out/dataset

  python src/extract_segments.py out/eval_segments.dict --path_game_logs data/logs/test_logs.dict --stopwords --meteor --from_first_common --utterances_as_captions
  python src/eval_chains.py out/eval_segments.dict
  ```

* To alternatively use CLIPScore as part of the scoring in extraction, just
  add the `--clipscore` option whenever running `extract_segments.py` above.
