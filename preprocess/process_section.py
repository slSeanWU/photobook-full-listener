import pickle
import os
import glob
import numpy as np
from clipscore import get_clip_mdl, extract_all_images, get_clip_score
 

def read_pickle(filename):
    x = pickle.load(open(filename, 'rb'))
    return x

def process_images(image_dir, model, device):
    image_paths = glob.glob(f'{image_dir}/*/*.jpg')

    # a dictionary
    image_feats_lookup = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)
    
    return image_feats_lookup

def calc_clip(segments, image_set, image_feats_lookup, model, device):
    # image features of shape (6, 512)
    img_feats = []
    for img in image_set:
        path = img.split('/')
        img_path = f'{path[-2]}/{path[-1]}'
        img_feats.append(image_feats_lookup[img_path])
    img_feats = np.array(img_feats)

    # clip score for each utterance
    clips = []
    for spk, utt in segments:
        utts = [utt] * 6

        # get image-text clipscore, where len(per_instance_image_text) = 6
        _, per_instance_image_text, candidate_feats = get_clip_score(
            model, img_feats, utts, device)

        clips.append(per_instance_image_text)

    clips = np.array(clips)     # [N_segments, 6]
    return clips

def process_section(sections, image_feats_lookup, model, device):
    """
    1 INPUT section
        segments: list of (spk_id, sentence) -> no modification
        image_set: {spk_A: [...], spk_B: [...]}
    
    2 MODIFIED sections
        segments: [(spk_id, sentence), ...]
        spk_id:
        image_set: [O,O,O,O,O,O]
        clip_scores: [(x,x,x,x,x,x), ...]
        # where len(segments) == len(clip_scores), and len(clip_scores[i]) == len(image_set) == 6
    """
    ret = []
    for sec in sections:
        for spk in sec["image_set"]:
            new_sec = dict()
            new_sec["agent_id"] = spk
            new_sec["segments"] = sec["segments"]
            new_sec["image_set"] = sec["image_set"][spk]
            new_sec["clip_scores"] = calc_clip(sec["segments"], sec["image_set"][spk], image_feats_lookup, model, device)

            ret.append(new_sec)
    return ret

def process(filename, image_feats_lookup, model, device, split):
    # read pickle
    game_sections = read_pickle(filename)
    print("Number of game sections:", len(game_sections))
    data_dir = os.path.dirname(filename)

    # iter through dialogues
    res = []
    for gid, sections in game_sections:
        # each game has N rounds, N = len(sections)
        print('-'*50, "game id:", gid, f'[{len(sections)} rounds]', '-'*50)
        res.append((gid, process_section(sections, image_feats_lookup, model, device)))
    
    # write to file
    with open(os.path.join(data_dir, f"{split}_clean_sections.pickle"), 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    model, device = get_clip_mdl()
    image_dir = "../../images/"
    image_feats_lookup = process_images(image_dir, model, device)

    for split in ["train", "val", "test"]:
        process(f"../data/{split}_sections.pickle", image_feats_lookup, model, device, split)
