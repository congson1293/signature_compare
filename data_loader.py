import os
import cv2 as cv
import utils
import random
import numpy as np
import joblib

# load data to pairs
max_height = 407
max_width = 1215

final_height = 150
final_width = 200

def load_data_from_file(dataset):
    data = {}
    stack = os.listdir(dataset)
    max_h, max_w = 0, 0
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            img = cv.imread(file_path)
            loan_id = file_name.split('_')[2].split('.')[0]
            data[loan_id] = img
            if img.shape[0] > max_h:
                max_h = img.shape[0]
            if img.shape[1] > max_w:
                max_w = img.shape[1]
    # print(f'{max_h} {max_w}')
    return data

def padding_img(img, w, h):
    rate = img.shape[0] / img.shape[1]
    img_ = cv.resize(img, (int(w * rate), h), interpolation=cv.INTER_AREA)
    new_img = np.full((h, w, 3), 0, dtype=np.uint8)
    try:
        new_img[:img_.shape[0], :img_.shape[1], :] = img_
    except:
        pass
    return new_img

def build_data(ttsdtb, tttc):
    global max_width, max_height, final_width, final_height

    result = {0: [], 1: []}
    loand_id_ttsdtb = list(ttsdtb.keys())
    count = 0
    for loanid, img_tttc in tttc.items():
        count += 1
        print(f'\rprocess image {count}-th', end='', flush=True)
        # cv.imshow('xxx', img_tttc)
        # cv.waitKey()
        img_tttc_ = padding_img(img_tttc, max_width, max_height)
        try:
            img_ttsdtb = padding_img(ttsdtb[loanid], max_width, max_height)
            img = np.concatenate((img_tttc_, img_ttsdtb))
            img = padding_img(img, final_width, final_height)
            result[1].append(img)
        except:
            # random img from ttsdtb
            for _ in range(2):
                ind = random.randint(0, len(loand_id_ttsdtb))
                loandid_ = loand_id_ttsdtb[ind]
                img_ttsdtb = padding_img(ttsdtb[loandid_], max_width, max_height)
                img = np.concatenate((img_tttc_, img_ttsdtb))
                img = padding_img(img, final_width, final_height)
                result[0].append(img)
    utils.mkdir('data')
    joblib.dump(result, 'data/data.pkl')


if __name__ == '__main__':
    tttc = load_data_from_file('signature/tttc')
    ttsdtb = load_data_from_file('signature/ttsdtb')
    build_data(ttsdtb, tttc)