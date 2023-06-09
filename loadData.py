import torch.utils.data as D
import cv2
import numpy as np
import torch
import htrAugmentor
from Config import Configs

cfg = Configs().parse()

baseDir = cfg.data_path
OUTPUT_MAX_LEN  = cfg.max_text_len
IMG_HEIGHT = cfg.img_height
IMG_WIDTH = cfg.img_width
TRAINTYPE = cfg.train_type


def labelDictionary():
    labels = ['1', '2', '9', '4', '3', '0', '7', '5', '8', '6', '', '2+', '0-', ',', '8+', '0^.', '.', '3^.', '6_.', '6__', '2__', '7__', '3__', '9__', '4__', '5__', '1__', '6_', '5_', '2^.', "9^'", "0^'", "1^'", "7^'", '5^.', '1^.', "5^'", '7^,', '5^,', "2^'", '3^’', '2^`', '0^`', '9^.', '1^`', '8^`', '5^`', '7^`', '8^_', '6^_', '9^`', '3^`', '0_.', '~', ':1', '.1', '9_.', '5_.', '4_.', '1_.', '8_.', '2_.', '7_.', '6^.', '7^.', '2/', '4^.', '8^.', '5/', '3_.', '2^./', ':', '9^_', '5^._.', '8^’', "8^'"]
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class Get_words(D.Dataset):
    def __init__(self, file_label, augmentation=True):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation

        self.transformer = htrAugmentor.augmentor

    def __getitem__(self, index):
        word = self.file_label[index]
        img, img_width = self.readImage_keepRatio(word[0])
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens)
        return word[0], img, img_width, label
        
    def __len__(self):
        return len(self.file_label)

    def readImage_keepRatio(self, file_name):
        
        url = baseDir + 'words/' + file_name.replace(',128','') + '.jpg'
        img = cv2.imread(url, 0)
        
        try:
            img.any()
        except:
            print('###!Cannot find image: ' + url)

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) 
        
        if self.augmentation: # augmentation for training data used by the htrAugmentor
            img_new = self.transformer(255-img)
            if img_new.shape[0] != 0 and img_new.shape[1] != 0:
                rate = float(IMG_HEIGHT) / img_new.shape[0]
                img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) 
        img_width = img.shape[-1]


        # if image superior than specified width then resize. If inferior, add zero padding to the image. 
        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg[:, :img_width] = img
        outImg = outImg/255. #float64
        outImg = outImg.astype('float32')
        
        # Transform to 3 channels and normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        outImgFinal = np.zeros([3, *outImg.shape])
        for i in range(3):
            outImgFinal[i] = (outImg - mean[i]) / std[i]
        return outImgFinal, img_width

    def label_padding(self, labels, num_tokens):
        # labels = labels.replace("?","")
        new_label_len = []
        ll =  [int(i) for i in labels.split(' ')]
        num = self.output_max_len - len(ll) - 2
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens   
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) 

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

def loadData():
    
    gt_tr = 'train.txt'
    gt_va = 'valid.txt'
    gt_te = 'test.txt'
    
    with open(baseDir+gt_tr, 'r') as f_tr:
        data_tr = f_tr.readlines()
        file_label_tr = [i[:-1].split(' ') for i in data_tr]

    with open(baseDir+gt_va, 'r') as f_va:
        data_va = f_va.readlines()
        file_label_va = [i[:-1].split(' ') for i in data_va]

    with open(baseDir+gt_te, 'r') as f_te:
        data_te = f_te.readlines()
        file_label_te = [i[:-1].split(' ') for i in data_te]


    np.random.shuffle(file_label_tr)
    data_train = Get_words(file_label_tr, augmentation= TRAINTYPE=='htr_Augm')
    data_valid = Get_words(file_label_va, augmentation=False)
    data_test = Get_words(file_label_te, augmentation=False)
    
    return data_train, data_valid, data_test


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        idx, img, img_width, label = batch[i]

        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)
    
    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    return train_index, train_in, train_in_len, train_out


def all_data_loader(batch_size):
    data_train, data_valid, data_test = loadData()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

