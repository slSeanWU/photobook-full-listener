import itertools
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaTokenizer

import copy

from print_round import print_round


class roundataset(Dataset):
    def __init__(self, picklefile, image_feats_path, image_dir='../data/images'):
        self.examples = []
        self.image_dir = image_dir
        self.image_feats_dict = pickle.load(open(image_feats_path, 'rb'))    # path to image features

        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        sections = pd.read_pickle(picklefile)
        for gameid, game in sections:
            for rounddict in game:
                print (rounddict.keys())
                self.examples.append(self.round2dict(
                    rounddict['round_data'], tokenizer, 'A', gameid,
                    rounddict['roundnr'], rounddict['clip_scores'], rounddict['image_set']))
                self.examples.append(self.round2dict(
                    rounddict['round_data'], tokenizer, 'B', gameid,
                    rounddict['roundnr'], rounddict['clip_scores'], rounddict['image_set']))

    def round2dict(self, gameround, tokenizer, player, gameid, roundnr, clip_scores, image_paths):
        input_ids = []
        labels = []
        clips = []
        c = 0

        images = [x for i, x in enumerate(
            gameround.images[player]) if gameround.highlighted[player][i]]
        # 0 = undecided, 1 = common, 2 = different
        image_status = [[0] for _ in images]
        for i, m in enumerate(gameround.messages):
            if m.type == "text":
                msgtxt = m.text
                # '[CLS] for self, [SEP] for other'
                if m.speaker == player:
                    msgtxt = '[CLS] ' + msgtxt
                else:
                    msgtxt = '[SEP] ' + msgtxt
                tokenized_msg = tokenizer(msgtxt, padding=False, truncation=True)[
                    'input_ids']

                # NOTE (Shih-Lun): to strip automatically added [CLS] and [SEP], add <eos> at the end
                tokenized_msg = tokenized_msg[1:-1] + [tokenizer.convert_tokens_to_ids('<|endoftext|>')]

                input_ids.append(tokenized_msg)
                labels.append([x * len(tokenized_msg) for x in image_status])
                clips.append(np.tile(clip_scores[c], (len(tokenized_msg), 1)))
                c += 1

            if m.type == "selection" and m.speaker == player:
                img = m.text.split()[2]
                img_index = images.index(img)
                label = 1 if m.text.split()[1] == "<com>" else 2
                image_status[img_index] = [label]
                # retroactively update last timestep's label at last token
                if len(labels) > 0:
                    for picid in range(len(labels[-1])):
                        labels[-1][picid][-1] = image_status[picid][0]
                else:
                    print('suspicious case')

        # Reshpe input ids and labels to be as long as num of tokens in round
        input_ids = list(itertools.chain(*input_ids))
        for turnnum, turn in enumerate(labels):
            labels[turnnum] = list(np.transpose(np.array(turn)))
        labels = list(itertools.chain(*labels))
        clips = np.vstack(clips)
        ret = {'gameid': gameid, 'roundnr': roundnr, 'input_ids': input_ids, 'labels': labels, 'vlscores': clips, 'image_paths': image_paths}
        return ret

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_feats = []
        for img in self.examples[idx]['image_paths']:
            image_feats.append(self.image_feats_dict[img])

        image_feats = torch.stack(image_feats).numpy()     # (6, 512, 16, 16)

        item = self.examples[idx]
        item["visual_inputs"] = copy.deepcopy(image_feats)

        del image_feats

        return item


if __name__ == '__main__':
    split = 'valid'
    image_feats_dict = '../data/image_feats.pickle'
    dset = roundataset(f'../data/{split}_clean_sections.pickle', image_feats_dict)
    print(dset)
