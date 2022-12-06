import json
import glob
from processor import Log
from os import walk

data_path = '../data/logs'
files = glob.glob(f'{data_path}/*.json')


def game2theme(g):
    return '_'.join(g.domains)


def game2imgs(g):
    return list(walk('../images/' + game2theme(g)))[0][2]


def file2game(f):
    with open(f) as file:
        log = Log(json.load(file))
    return log


def game2dist(g):
    return [str({player: sorted([game2imgs(g).index(img.split('/')[1])+1 for img in r.images[player]]) for player in ('A', 'B')}) for r in g.rounds]


test_cls = {'person_refrigerator', 'chair_couch', 'car_motorcycle',
            'cake_dining_table', 'cup_dining_table', 'bus_truck'}
data_split = {'train': [], 'valid': [], 'test': []}
val_mapping = dict()

for f in files:
    g = file2game(f)
    if game2theme(g) in test_cls:
        data_split['test'].append({g.game_id: [0, 1, 2, 3, 4]})
    else:
        if game2theme(g) not in val_mapping:
            val_mapping[game2theme(g)] = game2dist(g)[0]
        val_round = game2dist(g).index(val_mapping[game2theme(g)])
        data_split['valid'].append({g.game_id: [val_round]})
        data_split['train'].append(
            {g.game_id: [i for i in range(5) if i != val_round]})

with open("../data/analysis_splits.json", 'w') as f:
    json.dump(data_split, f)
