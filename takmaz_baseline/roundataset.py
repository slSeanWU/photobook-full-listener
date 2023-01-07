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
        ref_chain, ref_chain_len = self.get_tokenized_ref_chain(gameid, roundnr, image_paths)

        assert len(images) == 3 and len(labels[-1]) == 3
        ret = []
        for i in range(3):
            # image path to feature
            img_feat = self.get_img_feat(images[i])
            item = {
                #'gameid': gameid, 'roundnr': roundnr, 
                'input_text': np.array(input_ids), 'labels': np.array([labels[-1][i]]), 'image_paths': image_paths,
                'img_pred': np.array(img_feat), 'prev_hist': np.array(ref_chain), 'prev_hist_len': np.array(ref_chain_len), 
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
        
        ret = [[], [], [], [], [], []]
        ret_len = [0]*6

        if (str(gid), rid) not in self.ref_chains:
            #print(f'({gid}, {rid}) not in ref chains')
            for i in range(6):
                ret[i], _ = self.pad(ret[i])
            return ret, ret_len 
    
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

        for ind in range(6):
            msg, length = self.pad(ret[ind])
            ret[ind] = msg
            ret_len[ind] = length

        return ret, ret_len

    def pad(self, msg):
        length = len(msg)
        if len(msg) < self.maxseqlen:
            msg = np.pad(msg, (0, self.maxseqlen-len(msg)), 
                                   'constant', constant_values=(0,))
        elif len(msg) > self.maxseqlen:
            msg = msg[:self.maxseqlen]
            length = self.maxseqlen
        return msg, length

    def get_mask(self, input_len, maxseqlen):
        # items to be masked are TRUE
        mask = [False] * input_len + [True] * (maxseqlen - input_len)
        return np.array(mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_feats = []
        for img in self.examples[idx]['image_paths']:
            image_feats.append(self.get_img_feat(img))

        image_feats = torch.stack(image_feats).float()    # (6, 512)

        item = self.examples[idx]
        input_len = len(item['input_text'])
        
        # Pad ararys to uniform length
        if len(item['input_text']) < self.maxseqlen:
            item['input_text'] = np.pad(item['input_text'], (0, self.maxseqlen -
                                                           len(item['input_text'])),
                                       'constant', constant_values=(0,))

        elif len(item['input_text']) > self.maxseqlen:
            item['input_text'] = item['input_text'][:self.maxseqlen]
            input_len = self.maxseqlen 

        masks = self.get_mask(input_len, self.maxseqlen)

        if isinstance(item['input_text'], np.ndarray):
            item['img_pred'] = torch.tensor(item['img_pred']).float()
            item['input_text'] = torch.LongTensor(item['input_text'])
            item['masks'] = torch.tensor(masks)
            item['prev_hist'] = torch.LongTensor(item['prev_hist'])
            item['prev_hist_len'] = torch.LongTensor(item['prev_hist_len'])
            item['labels'] = torch.LongTensor(item['labels'])

        _ret_item = copy.deepcopy(item)
        _ret_item['separate_images'] = copy.deepcopy(image_feats)

        del image_feats
        #print(_ret_item['input_text'].shape, _ret_item['label'], _ret_item['img_pred'].shape, _ret_item.keys())

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
        print(f'masks shape = {samp["masks"].shape}')
        #print(f'masks = {samp["masks"]}')
        print(f'labels = {samp["labels"]}')
        print(f'prev_hist shape = {samp["prev_hist"].shape}')
        print(f'prev_hist = {samp["prev_hist"]}')
        print(f'prev_hist_len = {samp["prev_hist_len"]}')
        print('')
