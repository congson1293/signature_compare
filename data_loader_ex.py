import os
import cv2 as cv
import utils
import random
import numpy as np
import joblib

# load data to pairs
max_height = 407
max_width = 1215

final_height = 67
final_width = 200

def load_data_from_file(dataset):
    data = {}
    stack = os.listdir(dataset)
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            img = cv.imread(file_path)
            loan_id = file_name.split('_')[2].split('.')[0]
            img = padding_img(img, max_width, max_height)
            img = padding_img(img, final_width, final_height)
            data[loan_id] = img
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
    X = np.zeros((len(tttc) * 2, 2, final_height, final_width, 3), dtype=np.float32)
    y = np.zeros(len(tttc) * 2)
    loand_id_ttsdtb = list(ttsdtb.keys())
    idx = 0
    for loanid, img_tttc in tttc.items():
        v = np.zeros((2, final_height, final_width, 3), dtype=np.float32)
        print(f'\rprocess image {idx}-th', end='', flush=True)
        v[0] = img_tttc
        try:
            img_ttsdtb = ttsdtb[loanid]
            v[1] = img_ttsdtb
            X[idx] = v
            idx += 1
        except:
            pass
        X[idx] = v
        y[idx] = 1
        # random img from ttsdtb
        for _ in range(1):
            v = np.zeros((2, final_height, final_width, 3), dtype=np.float32)
            v[0] = img_tttc
            try:
                ind = random.randint(0, len(loand_id_ttsdtb))
                loandid_ = loand_id_ttsdtb[ind]
                while loanid == loandid_:
                    ind = random.randint(0, len(loand_id_ttsdtb))
                    loandid_ = loand_id_ttsdtb[ind]
                img_ttsdtb = ttsdtb[loandid_]
                v[1] = img_ttsdtb
            except:
                pass
            X[idx] = v
            idx += 1
    X = X[:idx]
    y = y[:idx]
    utils.mkdir('data')
    joblib.dump(X, 'data/X.pkl')
    joblib.dump(y, 'data/y.pkl')


if __name__ == '__main__':
    # tttc = load_data_from_file('signature/tttc')
    # ttsdtb = load_data_from_file('signature/ttsdtb')
    # utils.mkdir('data')
    # joblib.dump({'tttc': tttc, 'ttsdtb': ttsdtb}, 'data/data.pkl')
    data = joblib.load('data/data.pkl')
    tttc = data['tttc']
    ttsdtb = data['ttsdtb']
    build_data(tttc, ttsdtb)