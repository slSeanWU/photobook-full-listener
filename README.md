# photobook-full-listener

Assignment 4 (Final Project), 11711 Advanced NLP, Fall '22

## Installation

```bash
conda create -n photobook python=3.8.10
conda activate photobook
pip install -r requirements.txt

python
>>> import nltk
>>> nltk.download('punkt')
```

* get logs.zip and images.zip at [photobook_dataset](https://github.com/dmg-photobook/photobook_dataset/)

## Preprocess

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
