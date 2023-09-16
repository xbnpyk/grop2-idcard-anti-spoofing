import os
import random
from utils import *

# DATA_ROOT = r'{path-to-dataset}/CASIA-SURF/phase1'
# DATA_ROOT = r'data/CASIA-SURF/phase1'
DATA_ROOT = r'data/0907'
TRN_IMGS_DIR = DATA_ROOT + '/Training/'
TST_IMGS_DIR = DATA_ROOT + '/Val/'
# RESIZE_SIZE = 640
RESIZE_SIZE = 112 #96
# RESIZE_SIZE = 320 #224

def load_train_list(dataroot=None):
    list = []
    print(f"load_train_list: {dataroot}")
    if dataroot == None:
        dataroot = DATA_ROOT
    f = open(dataroot + '/train_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list(dataroot=None):
    list = []
    if dataroot == None:
        dataroot = DATA_ROOT
    f = open(dataroot + '/val_private_list.txt')
    # f = open(dataroot + '/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list(dataroot=None):
    list = []
    if dataroot == None:
        dataroot = DATA_ROOT
    # f = open(dataroot + '/test_public_list.txt')
    f = open(dataroot + '/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

def transform_balance(train_list):
    pos_list = []
    neg_list = []
    for tmp in train_list:
        if tmp[3]=='1':
            pos_list.append(tmp)
        elif tmp[3][0]=='0' and len(tmp[3])>1:#'0000000'
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print(len(pos_list))
    print(len(neg_list))
    return [pos_list,neg_list]

def submission(probs,accs, outname, mode='valid', dataroot=None):
    if dataroot == None:
        dataroot = DATA_ROOT
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    acc0,acc1,acc2,acc3 = accs["acc"],accs["acc_repalys"],accs["acc_prints"],accs["acc_faces"]
    out = str(acc0) + ' ' + str(acc1) + ' ' + str(acc2) + ' ' + str(acc3)
    f.write(out+'\n')
    for line, prob0,prob1,prob2,prob3 in zip(lines, probs[0], probs[1], probs[2], probs[3]):
        # prob0,prob1,prob2,prob3 = prob
        out = line + ' ' + str(prob0) + ' ' + str(prob1) + ' ' + str(prob2)+ ' ' + str(prob3)
        f.write(out+'\n')
    f.close()
    return list



