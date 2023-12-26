from PIL import Image
from tqdm import tqdm
import json
import numpy as np

train_path = 'Yoga-82\yoga_train.txt'
test_path = 'Yoga-82\yoga_test.txt'

def data_verifier(key):
    data_dict = {'train':train_path, 'test':test_path}
    path = data_dict[key]

    img_count = {'6': [0]*6,
                '20': [0]*20,
                '82': [0]*82}
    
    img_paths = {'6': {},
                '20': {},
                '82': {}}
    
    for i in img_paths.keys():
        for k in range(int(i)):
            img_paths[i][k] = []

    with open(path,'+r') as file:
        lines = file.readlines()
        for i in tqdm(range(len(lines))):
            split_line = lines[i].split(',')
            try:
                img = Image.open('Images/'+split_line[0])
                img.verify()
                img_paths['6'][int(split_line[1])] += [split_line[0]]
                img_paths['20'][int(split_line[2])] += [split_line[0]]
                img_paths['82'][int(split_line[3])] += [split_line[0]]
                img_count['6'][int(split_line[1])] += 1
                img_count['20'][int(split_line[2])] += 1
                img_count['82'][int(split_line[3])] += 1
            except:
                continue
    with open(f'Data_Paths/{key}.json', 'w') as output_file:
        json.dump(img_paths, output_file, indent=2)

    with open(f'Data_Paths/{key}_class_count.json', 'w') as json_file:
        json.dump(img_count, json_file, indent=2)

if __name__ == '__main__':
    data_verifier('train')
    data_verifier('test')


