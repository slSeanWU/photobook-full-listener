import json
import os
import glob
import random

# ../data/logs/2332_70_71_8.json 
# train [2332, 1259, ...]

def process(logfile):
    with open(logfile, 'r') as f:
        data = json.load(f)
    imgfile = data['rounds'][0]['images']['A'][0]
    # 'person_refrigerator/COCO_train2014_000000217700.jpg'
    cls = imgfile.split('/')[0]

    return cls

def info(mapping):
    total = 0
    for k, v in mapping.items():
        print(k, len(v))
        total += len(v)
    print('total:', total)
    return

if __name__ == '__main__':
    mapping = dict()    # key: category, value: img_domain_id

    data_path = '../data/logs'
    files = glob.glob(f'{data_path}/*.json')


    for logfile in files:
        gid = logfile.split('/')[-1].split('_')[0]
        cls = process(logfile)
        if cls not in mapping:
            mapping[cls] = []
        mapping[cls].append(int(gid))

    info(mapping)
    # split to 70:10:20
    '''
    [TEST]
    person_refrigerator 71
    chair_couch 63
    car_motorcycle 104
    cake_dining_table 85
    cup_dining_table 86
    bus_truck 90
    --> total: 499

    [VALID]
    couch_dining_table 87
    person_motorcycle 96
    person_boat 60
    --> total: 243
    '''

    test_cls = {'person_refrigerator', 'chair_couch', 'car_motorcycle', 'cake_dining_table', 'cup_dining_table', 'bus_truck'}
    valid_cls = {'couch_dining_table', 'person_motorcycle', 'person_boat'}

    data_split = {'train': [], 'valid': [], 'test': []}

    train_val = []
    for key, val in mapping.items():
        if key in test_cls:
            data_split['test'] += val
        elif key in valid_cls:
            data_split['valid'] += val
        else:
            data_split['train'] += val

    with open("../data/data_splits.json", 'w') as f:
        json.dump(data_split, f)
