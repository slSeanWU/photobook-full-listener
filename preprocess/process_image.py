import glob
import pickle
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerModel


def get_image_features(model, feature_extractor, image_path):
    image = Image.open(image_path)
    # forward
    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0]


def process_images(model, feat_extractor, image_dir):
    # mapping (path -> hidden state)
    image_feats = dict()

    # traverse all images
    files = glob.glob(f'{image_dir}/*/*.jpg')
    for i, path in enumerate(files):
        hidden_state = get_image_features(model, feat_extractor, path)
        # ../../images/person_motorcycle/COCO_train2014_000000279373.jpg
        path_list = path.split('/')
        category, name = path_list[-2], path_list[-1]
        clean_path = f'{category}/{name}'
        image_feats[clean_path] = hidden_state
        print(f'[{i+1}/{len(files)}]', clean_path, hidden_state.shape)

    # write to pickle
    data_dir = "../data"
    with open(f"{data_dir}/image_feats.pickle", 'wb') as f:
        pickle.dump(image_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

if __name__ == '__main__':
    # model
    model_type = "nvidia/segformer-b4-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_type)
    model = SegformerModel.from_pretrained(model_type)

    # image
    img_dir = "../../images"

    # process
    process_images(model, feature_extractor, img_dir)
