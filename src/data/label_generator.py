import os

def generate_labels(img_dir, label_flag):
    if label_flag:
        label = 1
    else:
        label = 0
    files_list = os.listdir(img_dir)
    label_dicts = []
    for img in files_list:
        if img[0] == '.':
            continue
        label_dicts.append({'img_file' : img, 'label': label})
    return label_dicts
