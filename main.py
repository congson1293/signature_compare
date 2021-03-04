import os
import cv2
import utils
import random
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt

# load data to pairs
def load_data_from_file(dataset):
    data = {}
    stack = os.listdir(dataset)
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            loan_id = file_name.split('_')[2].split('.')[0]
            data[loan_id] = gray
    return data

def cal_num_match_point(flann, detector, img1, img2):
    n_match = 0
    kpts_tttc, descs_tttc = detector.detectAndCompute(img1, None)
    kpts_ttsdtb, descs_ttsdtb = detector.detectAndCompute(img2, None)

    flann = flann.knnMatch(descs_tttc, descs_ttsdtb, k=2)
    for (m, n) in flann:
        if m.distance < 0.8 * n.distance:
            n_match += 1

    return n_match

def run():
    n_match_threshold = 21

    tttc = load_data_from_file('signature/tttc')
    ttsdtb = load_data_from_file('signature/ttsdtb')

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 8}, {'checks': 100})
    detector = cv2.SIFT_create()

    statistic = []
    count = 0
    loand_id_ttsdtb = list(ttsdtb.keys())
    for loanid, img_tttc in tttc.items():
        count += 1
        print(f'\rprocess image {count}-th', end='', flush=True)
        try:
            img_ttsdtb = ttsdtb[loanid]
            n_match = cal_num_match_point(flann, detector, img_tttc, img_ttsdtb)
            statistic.append(f'{n_match} 1')
        except:
            # random img from ttsdtb
            ind = random.randint(0, len(loand_id_ttsdtb))
            loandid_ = loand_id_ttsdtb[ind]
            img_ttsdtb = ttsdtb[loandid_]
            n_match = cal_num_match_point(flann, detector, img_tttc, img_ttsdtb)
            statistic.append(f'{n_match} 0')
    print('\n')

    with open('statistic.txt', 'w') as fp:
        fp.write('\n'.join(statistic))

    visualize(15)

def visualize(threshold):
    with open('statistic.txt') as fp:
        lines = fp.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    # bins = np.linspace(0, 180, 181)
    bins = np.linspace(0, 100, 100)

    plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
    plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))
    print('mu_0: ' + str(mu_0))
    print('sigma_0: ' + str(sigma_0))
    print('mu_1: ' + str(mu_1))
    print('sigma_1: ' + str(sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    plt.savefig('theta_dist.png')
    plt.show()



if __name__ == '__main__':
    run()
    # visualize(15)