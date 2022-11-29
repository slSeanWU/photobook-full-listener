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
    '''
    test_cls = {'person_refrigerator', 'chair_couch', 'car_motorcycle', 'cake_dining_table', 'cup_dining_table', 'bus_truck'}

    data_split = {'train': [], 'valid': [], 'test': []}

    train_val = []
    for key, val in mapping.items():
        if key in test_cls:
            data_split['test'] += val
        else:
            train_val += val

    random.shuffle(train_val)
    val_cnt = int(len(train_val)*0.13)
    data_split['valid'] = train_val[:val_cnt]
    data_split['train'] = train_val[val_cnt:]


    with open("../data/data_splits.json", 'w') as f:
        json.dump(data_split, f)
