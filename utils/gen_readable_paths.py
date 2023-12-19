from PIL import Image

train_path = 'Yoga-82\yoga_train.txt'
test_path = 'Yoga-82\yoga_test.txt'

def data_verifier(key):
    data_dict = {'train':train_path, 'test':test_path}
    path = data_dict[key]
    readable_images = []
    with open(path,'+r') as file:
        for line in file.readlines():
            split_line = line.split(',')
            try:
                img = Image.open('Images/'+split_line[0])
                img.verify()
                readable_images.append(line)
            except:
                continue
    with open(f'Data_Paths/{key}.txt', 'w') as output_file:
        output_file.writelines(readable_images)

if __name__ == '__main__':
    data_verifier('train')
    data_verifier('test')


