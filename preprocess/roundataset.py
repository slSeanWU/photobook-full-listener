import itertools

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaTokenizer

from print_round import print_round


class roundataset(Dataset):
    def __init__(self, picklefile):
        self.examples = []
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        sections = pd.read_pickle(picklefile)
        for gameid, game in sections:
            for rounddict in game:
                self.examples.append(self.round2dict(
                    rounddict['round_data'], tokenizer, 'A', gameid,
                    rounddict['roundnr']))
                self.examples.append(self.round2dict(
                    rounddict['round_data'], tokenizer, 'B', gameid,
                    rounddict['roundnr']))

    def round2dict(self, gameround, tokenizer, player, gameid, roundnr):
        input_ids = []
        labels = []

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
                input_ids.append(tokenized_msg)
                labels.append([x * len(tokenized_msg) for x in image_status])

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

        return {'gameid': gameid, 'roundnr': roundnr, 'input_ids': input_ids, 'labels': labels}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == '__main__':
    dset = roundataset('../data/test_sections.pickle')
    print(dset)
