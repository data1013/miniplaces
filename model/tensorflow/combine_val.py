import os
import shutil

root_dir = '../../data/images/train/'

label_dict = {}
val_label_dict = {}

def rename_val_files():
    data = os.path.abspath("../../data/images/val/")
    for i, f in enumerate(os.listdir(data)):
        src = os.path.join(data, f)
        dst = os.path.join(data, "val_" + f)
        os.rename(src, dst)

def copy_val_to_train():
    data = os.path.abspath("../../data/images/val/")

    train_txt = open("../../data/train.txt", "a")

    for i, f in enumerate(os.listdir(data)):
        src = os.path.join(data, f)

        file_label = val_label_dict[f]
        file_dst = label_dict[file_label]

        train_txt_line = os.path.join(file_dst, f) + ' ' + file_label + '\n'

        train_txt.write(train_txt_line)

        file_dst = os.path.abspath('../../data/images/' + label_dict[file_label])

        shutil.copy(src, file_dst)

    train_txt.close()

with open('../../data/categories.txt') as f:
    for line in f:
        line = line.split()
        label_path = line[0]
        label_num = line[1]

        label_dict[label_num] = 'train' +  label_path

with open('../../data/val.txt') as f:
    for line in f:
        line = line.split()
        file_name = line[0].split('/')[1]
        file_label = line[1]

        val_label_dict[file_name] = file_label

rename_val_files()
copy_val_to_train()
