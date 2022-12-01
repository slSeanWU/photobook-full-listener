import itertools
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaTokenizer
from random import sample

import copy


class roundataset(Dataset):
    def __init__(self, picklefile, image_feats_path, image_dir='../data/images'):
        self.examples = []
        self.image_dir = image_dir
        self.image_feats_dict = pickle.load(
            open(image_feats_path, 'rb'))    # path to image features
        self.maxseqlen = 460

        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        sections = pd.read_pickle(picklefile)

        # sections = sections[:10]

        for gameid, game in sections:
            for rounddict in game:
                agent_id = rounddict['agent_id']
                self.examples.append(
                    self.round2dict(
                        rounddict['round_data'], tokenizer, agent_id, gameid,
                        rounddict['roundnr'], rounddict['clip_scores'], rounddict['image_set'], rounddict['image_pred_ids']
                    )
                )

    def round2dict(self, gameround, tokenizer, player, gameid, roundnr, clip_scores, image_paths, image_pred_ids):
        input_ids = []
        labels = []
        clips = []
        c = 0

        images = [x for i, x in enumerate(image_paths) if image_pred_ids[i]]

        # 0 = undecided, 1 = common, 2 = different
        image_status = [[0] for _ in images]
        for i, m in enumerate(gameround.messages):
            if m.type == "text":
                msgtxt = m.text
                # '[CLS] for self, [SEP] for other'
                if m.agent_id == player:
                    msgtxt = '[CLS] ' + msgtxt
                else:
                    msgtxt = '[SEP] ' + msgtxt
                tokenized_msg = tokenizer(msgtxt, padding=False, truncation=True)[
                    'input_ids']

                # NOTE (Shih-Lun): to strip automatically added [CLS] and [SEP], add <eos> at the end
                tokenized_msg = tokenized_msg[1:-1] + \
                    [tokenizer.convert_tokens_to_ids('<|endoftext|>')]

                input_ids.append(tokenized_msg)
                labels.append([x * len(tokenized_msg) for x in image_status])
                clips.append(np.tile(clip_scores[c], (len(tokenized_msg), 1)))
                c += 1

            if m.type == "selection" and m.agent_id == player:
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
        image_pred_ids = np.array(image_pred_ids)

        ret = {
            'gameid': gameid, 'roundnr': roundnr, 'input_ids': np.array(input_ids),
            'labels': np.array(labels), 'vlscores': clips, 'image_paths': image_paths,
            'img_pred_ids': image_pred_ids,
        }

        return ret

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_feats = []
        for img in self.examples[idx]['image_paths']:
            image_feats.append(self.image_feats_dict[img])

        image_feats = torch.stack(image_feats).float()    # (6, 512, 16, 16)

        item = self.examples[idx]

        # Pad ararys to uniform length
        if len(item['input_ids']) < self.maxseqlen:
            item['input_ids'] = np.pad(item['input_ids'], (0, self.maxseqlen -
                                                        len(item['input_ids'])),
                                    'constant', constant_values=(0,))
            item['labels'] = np.pad(item['labels'], ((
                0, self.maxseqlen - item['labels'].shape[0]), (0, 0)), 'constant', constant_values=(-100,))
            item['vlscores'] = np.pad(item['vlscores'], ((
                0, self.maxseqlen - item['vlscores'].shape[0]), (0, 0)), 'constant', constant_values=(0.,))
        
        elif len(item['input_ids']) > self.maxseqlen:
            item['input_ids'] = item['input_ids'][:self.maxseqlen]
            item['labels'] = item['labels'][:self.maxseqlen, :]
            item['vlscores'] = item['vlscores'][:self.maxseqlen, :]

        if isinstance(item['input_ids'], np.ndarray):
            item['img_pred_ids'] = torch.LongTensor(item['img_pred_ids'])
            item['input_ids'] = torch.LongTensor(item['input_ids'])
            item['labels'] = torch.LongTensor(item['labels'])
            item['vlscores'] = torch.tensor(item['vlscores']).float()

        _ret_item = copy.deepcopy(item)
        _ret_item['visual_inputs'] = copy.deepcopy(image_feats)

        del image_feats

        return _ret_item


if __name__ == '__main__':
    split = 'test'
    image_feats_dict = '../data/image_feats.pickle'
    dset = roundataset(
        f'../data/{split}_clean_sections.pickle', image_feats_dict)

    print (len(dset))
    for i in range(10):
        samp = dset[i]

        print(f'Example {i}')
        print(f'img_pred_ids shape = {samp["img_pred_ids"].shape}')
        print(f'input_ids shape = {samp["input_ids"].shape}')
        print(f'labels shape = {samp["labels"].shape}')
        print(f'vlscores shape = {samp["vlscores"].shape}')
        print(f'visual_inputs shape = {samp["visual_inputs"].shape}')
        print('')
