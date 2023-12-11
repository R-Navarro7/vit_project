import sys
import os
from PIL import Image
from tqdm import tqdm


file_list = os.listdir("Yoga-82/yoga_dataset_links")

if  not os.path.isdir('Images'):
    os.mkdir("Images")


for i in tqdm(range(len(file_list))):
    f = open("Yoga-82/yoga_dataset_links/" + file_list[i], 'r')
    lines = f.readlines()
    f.close()
    
    for j in tqdm(range(len(lines))):
        #print(lines[j])
        splits = lines[j][:len(lines[j])-1].split()
        #print('splits')
        img_path = splits[0]
        link = splits[1]
        
        folder_name, img_name = img_path.split("/")
        
        if(j == 0):
            if(not os.path.isdir("Images/" + folder_name)):
                os.mkdir("Images/" + folder_name)
        os.system("wget --quiet --tries=3 -O " + "Images/" + folder_name + "/" + img_name + " " + link)
        try:
            img = Image.open("Images/" + folder_name + "/" + img_name)
            img.verify()     # to veify if its an img
            img.close()     #to close img and free memory space
        except (IOError, SyntaxError) as e:
            print(e)
