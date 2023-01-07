import itertools
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from random import sample

import copy


class roundataset(Dataset):
    def __init__(self, picklefile, image_feats_path, ref_chain_path='ref_chain_img.pickle', image_dir='../data/images'):
        self.examples = []
        self.image_dir = image_dir
        self.image_feats_dict = pickle.load(
            open(image_feats_path, 'rb'))    # path to image features: segformer (512, 16, 16) or default (512,)
        self.ref_chains = pickle.load(open(ref_chain_path, 'rb')) 

        self.maxseqlen = 460

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # section
        sections = pd.read_pickle(picklefile)
        # sections = sections[:10]

        for gameid, game in sections:
            for rounddict in game:
                agent_id = rounddict['agent_id']
                self.examples.extend(
                    self.round2dict(
                        rounddict['round_data'], agent_id, gameid,
                        rounddict['roundnr'], rounddict['image_set'], rounddict['image_pred_ids']
                    )
                )

    def round2dict(self, gameround, player, gameid, roundnr, image_paths, image_pred_ids):
        # print(f'Appending game {gameid} round {roundnr}')
        input_ids = []
        labels = []
        c = 0

        images = [x for i, x in enumerate(image_paths) if image_pred_ids[i]]
        # ['couch_dining_table/COCO_train2014_000000560969.jpg', 'couch_dining_table/COCO_train2014_000000275014.jpg', 'couch_dining_table/COCO_train2014_000000180606.jpg']

        # 0 = different, 1 = common
        image_status = [[0] for _ in images]
        for i, m in enumerate(gameround.messages):
            if m.type == "text":
                msgtxt = m.text
                # '[CLS] for self, [SEP] for other'
                if m.agent_id == player:
                    msgtxt = '[CLS] ' + msgtxt
                else:
                    msgtxt = '[SEP] ' + msgtxt

                tokenized_msg = self.tokenizer(msgtxt, padding=False, truncation=True)[
                    'input_ids']
                tokenized_msg = tokenized_msg[1:-1] + \
                    [self.tokenizer.convert_tokens_to_ids('<|endoftext|>')]

                input_ids.append(tokenized_msg)
                labels.append([x * len(tokenized_msg) for x in image_status])
                c += 1

            if m.type == "selection" and m.agent_id == player:
                img = m.text.split()[2]
                img_index = images.index(img)
                label = 1 if m.text.split()[1] == "<com>" else 0 
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
        image_pred_ids = np.array(image_pred_ids)
        labels = list(itertools.chain(*labels))

        # NOTE: add ref chain
        ref_chain = self.get_tokenized_ref_chain(gameid, roundnr, image_paths)

        assert len(images) == 3 and len(labels[-1]) == 3
        ret = []
        for i in range(3):
            # image path to feature
            img_feat = self.get_img_feat(images[i])
            item = {
                'gameid': gameid, 'roundnr': roundnr, 'input_text': np.array(input_ids),
                'label': np.array([labels[-1][i]]), 'image_paths': image_paths,
                'img_pred': np.array(img_feat), 'prev_hist': ref_chain, 
            }
            ret.append(item)

        return ret

    def get_img_feat(self, path):
        feat = self.image_feats_dict[path]
        feat = feat.reshape(512,-1)
        feat = feat.mean(1)
        return feat

    def get_img2ind(self, image_paths):
        img2ind = dict()
        for i in range(6):
            # 'couch_dining_table/COCO_train2014_000000109895.jpg'
            ID = image_paths[i].split('/')[-1].split('.')[0].split('_')[-1].lstrip('0')
            img2ind[ID] = i
        return img2ind
    
    def get_tokenized_ref_chain(self, gid, rid, image_paths):
        # https://github.com/slSeanWU/photobook-full-listener/blob/08f338eb35fb404e1f67d5f6a344fc2b2089b730/preprocess/processor.py#L72
        rid -= 1
        
        if (str(gid), rid) not in self.ref_chains:
            #print(f'({gid}, {rid}) not in ref chains')
            return []
    
        ret = [[], [], [], [], [], []]
        img2ind = self.get_img2ind(image_paths)
        chains = self.ref_chains[(str(gid), rid)]
        

        for utt, hist in chains:
            for img_key, msg in hist.items():
                #print(msg)
                if img_key not in img2ind:
                    continue
                ind = img2ind[img_key]
                tokenized_msg = self.tokenizer(msg[0], padding=False, truncation=True)[
                    'input_ids']
                tokenized_msg = tokenized_msg[1:-1] + \
                    [self.tokenizer.convert_tokens_to_ids('<|endoftext|>')]
                ret[ind].extend(tokenized_msg)

        # tensor cannot have varied length
        for i in range(6):
            ret[i] = self.pad(ret[i])
        ret = np.array(ret)
        return ret

    def pad(self, msg):
        if len(msg) < self.maxseqlen:
            msg = np.pad(msg, (0, self.maxseqlen-len(msg)), 
                                   'constant', constant_values=(0,))
        elif len(msg) > self.maxseqlen:
            msg = msg[:self.maxseqlen]
        return msg

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_feats = []
        for img in self.examples[idx]['image_paths']:
            image_feats.append(self.get_img_feat(img))

        image_feats = torch.stack(image_feats).float()    # (6, 512)

        item = self.examples[idx]

        # Pad ararys to uniform length
        if len(item['input_text']) < self.maxseqlen:
            item['input_text'] = np.pad(item['input_text'], (0, self.maxseqlen -
                                                           len(item['input_text'])),
                                       'constant', constant_values=(0,))

        elif len(item['input_text']) > self.maxseqlen:
            item['input_text'] = item['input_text'][:self.maxseqlen]

        if isinstance(item['input_text'], np.ndarray):
            item['img_pred'] = torch.LongTensor(item['img_pred'])
            item['input_text'] = torch.LongTensor(item['input_text'])
            item['prev_hist'] = torch.LongTensor(item['prev_hist'])
            item['label'] = torch.LongTensor(item['label'])

        _ret_item = copy.deepcopy(item)
        _ret_item['separate_images'] = copy.deepcopy(image_feats)

        del image_feats

        return _ret_item


if __name__ == '__main__':
    dset = roundataset(
        '../data/valid_clean_sections.pickle',
        '../data/image_feats.pickle'
    )

    print(len(dset))
    for i in range(10):
        samp = dset[i]

        print(f'Example {i}')
        print(f'img_pred shape = {samp["img_pred"].shape}')
        print(f'separate_images shape = {samp["separate_images"].shape}')
        print(f'input_text shape = {samp["input_text"].shape}')
        print(f'label = {samp["label"]}')
        print(f'prev_hist shape = {samp["prev_hist"].shape}')
        print(samp["prev_hist"])
        print('')
