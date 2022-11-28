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

def mark_labeling_actions(self_spk_id, sec_utt_and_act, img_idx_dict):
    """ Added by Shih-Lun
    """
    utt_cnt = 0
    labeled_imgs = set()
    sec_labeling_acts = []
    for subsec in sec_utt_and_act:
        subsec_utts, subsec_acts = subsec
        utt_cnt += len(subsec_utts)
        for label_act in subsec_acts:
            assert isinstance(label_act, tuple) and len(label_act) == 3
            act_spk_id, com_or_dif, label_img_id = label_act
            assert label_img_id in img_idx_dict

            # answer from the player itself, add to labeling actions
            if act_spk_id == self_spk_id:
                sec_labeling_acts.append(
                    (utt_cnt, com_or_dif, img_idx_dict[label_img_id])
                ) 
                # tuple format --
                # (
                #   num of utts before this action, 
                #   "<com> or "<dif>",
                #   idx of image being labeled (possible values: 0, 1, 2, 3, 4, 5)
                # )

                labeled_imgs.add(label_img_id)

    # print (sec_labeling_acts)
    assert len(sec_labeling_acts) == 3 # photobook game setting

    return sec_labeling_acts, labeled_imgs


def process_section(sections, image_feats_lookup, model, device):
    """
    1 INPUT section
        segments: list of (spk_id, sentence) -> no modification
        image_set: {spk_A: [...], spk_B: [...]}
    
    2 MODIFIED sections
        segments: [(spk_id, sentence), ...]
        label_actions: [
            (# utts before action, "<com> or "<dif>", idx of img being labeled (in {0, 1, 2, 3, 4, 5}), 
            ...
        ]
        image_pred_ids: e.g., [0, 1, 0, 2, 0, 3] if 2nd, 4th, 6th imgs are highlighted
        spk_id:
        image_set: [O,O,O,O,O,O]
        clip_scores: [(x,x,x,x,x,x), ...]
        # where len(segments) == len(clip_scores), and len(clip_scores[i]) == len(image_set) == 6
    """
    ret = []
    for sec in sections:
        # print (sec["other"])
        
        assert sum([len(sec["other"][i][0]) for i in range(len(sec["other"]))]) == len(sec["segments"])
        for spk in sec["image_set"]:
            assert isinstance(sec["image_set"][spk], list)
            img_idx_dict = {img : i for i, img in enumerate(sec["image_set"][spk])}
            new_sec = dict()
            new_sec["agent_id"] = spk
            new_sec["segments"] = sec["segments"]
            new_sec["image_set"] = sec["image_set"][spk]
            new_sec["label_actions"], highlighted_imgs = mark_labeling_actions(spk, sec["other"], img_idx_dict)

            # NOTE (Shih-Lun): added to conform to model inputs, 
            #                  e.g., [0, 1, 0, 2, 0, 3] if 2nd, 4th, 6th imgs are highlighted
            new_sec["image_pred_ids"] = [0] * 6
            highlighted_cnt = 0
            for i, img in enumerate(sec["image_set"][spk]):
                if img in highlighted_imgs:
                    highlighted_cnt += 1
                    new_sec["image_pred_ids"][i] = highlighted_cnt

            # print (new_sec["label_actions"])
            # print (new_sec["image_pred_ids"], '\n')

            # new_sec["clip_scores"] = calc_clip(sec["segments"], sec["image_set"][spk], image_feats_lookup, model, device)

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
    image_dir = "../images/"
    image_feats_lookup = process_images(image_dir, model, device)

    # for split in ["train", "val", "test"]:
    for split in ["dev"]:
        process(f"../data/{split}_sections.pickle", image_feats_lookup, model, device, split)
