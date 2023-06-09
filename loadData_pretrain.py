import torch
from random import random
import torch.utils.data as D
import cv2
import numpy as np
import os
import random
from imgaug import augmenters as iaa
from Config import Configs
import htrAugmentor
cfg = Configs().parse()



WORD_LEVEL = True
VGG_NORMAL = True
RM_BACKGROUND = False
FLIP = False 



baseDir = cfg.data_path
OUTPUT_MAX_LEN  = cfg.max_text_len
IMG_HEIGHT = cfg.img_height
IMG_WIDTH = cfg.img_width
TRAINTYPE = cfg.train_type


def labelDictionary():
    # # English text
    # labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/','[',']','@','<','>','|','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    # Vatican
    labels = ['1', '2', '9', '4', '3', '0', '7', '5', '8', '6', '', '2+', '0-', ',', '8+', '0^.', '.', '3^.', '6_.', '6__', '2__', '7__', '3__', '9__', '4__', '5__', '1__', '6_', '5_', '2^.', "9^'", "0^'", "1^'", "7^'", '5^.', '1^.', "5^'", '7^,', '5^,', "2^'", '3^’', '2^`', '0^`', '9^.', '1^`', '8^`', '5^`', '7^`', '8^_', '6^_', '9^`', '3^`', '0_.', '~', ':1', '.1', '9_.', '5_.', '4_.', '1_.', '8_.', '2_.', '7_.', '6^.', '7^.', '2/', '4^.', '8^.', '5/', '3_.', '2^./', ':', '9^_', '5^._.', '8^’', "8^'"]
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class Get_words(D.Dataset):
    def __init__(self, file_label, set_data, augmentation=True):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation
        self.set_data = set_data
        # # scene text augmentor
        # self.transformer = self.transformer = iaa.Sequential([iaa.SomeOf((1, 5),
        #                    [iaa.LinearContrast((0.5, 1.0)),
        #                     iaa.GaussianBlur((0.5, 1.5)),
        #                     iaa.Crop(percent=((0, 0.4),
        #                                       (0, 0),
        #                                       (0, 0.4),
        #                                       (0, 0.0)),
        #                              keep_size=True),
        #                     iaa.Crop(percent=((0, 0.0),
        #                                       (0, 0.02),
        #                                       (0, 0),
        #                                       (0, 0.02)),
        #                              keep_size=True),
        #                     iaa.Sharpen(alpha=(0.0, 0.5),
        #                                 lightness=(0.0, 0.5)),
        #                     iaa.PiecewiseAffine(scale=(0.02, 0.03),
        #                                         mode='edge'),
        #                     iaa.PerspectiveTransform(
        #                         scale=(0.01, 0.02)),
        #                    ],
        #                    random_order=True)])

        # htr augmentor
        self.transformer = htrAugmentor.augmentor
    def __getitem__(self, index):
        word = self.file_label[index]
        img,img_bg,img_blur, img_width = self.readImage_keepRatio(word[0], flip=FLIP)
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens)
        return word[0], img,img_bg,img_blur, img_width, label
        #return {'index_sa': file_name, 'input_sa': in_data, 'output_sa': out_data, 'in_len_sa': in_len, 'out_len_sa': out_data_mask}

    def __len__(self):
        if self.set_data =='train':
            return len(self.file_label) # 5000
        else:
            return len(self.file_label)

    def addDist(self,img):
    
        if True:
            backgrounds = os.listdir(baseDir+'backgroundIAM/')
            ch=random.choice(backgrounds)

            bg = cv2.imread(baseDir+'backgroundIAM/'+ch)
            
            
            size_a = (img.shape[1]*2)
            size_bg = bg.shape[1]
            
            while size_a> size_bg:
                
                bg = np.concatenate((bg, bg), axis=1)
                size_bg = size_bg*2

            size_a = (img.shape[0]*2)
            size_bg = bg.shape[0]
            
            while size_a> size_bg:
                
                bg = np.concatenate((bg, bg), axis=0)
                size_bg = size_bg*2
            

            p = random.randint(1,100)
            p2 = random.randint(1,50)
                
                    
            bg = bg[p:p+img.shape[0],p2:p2+img.shape[1]]



            param1 = random.randint(1,9)/10

            param2 = random.randint(1,9)/10
            ww=random.randint(-60,30)
            img = img.astype(np.uint8)
            img_bg = cv2.addWeighted(bg,param1,img,param2,ww)

        
            kernel1=random.randint(1,15)
            kernel2=random.randint(1,15)
            img_blur = cv2.blur(img,(kernel1,kernel2), cv2.BORDER_DEFAULT) 
        
        return img_bg, img_blur
    def readImage_keepRatio(self, file_name, flip):
        if RM_BACKGROUND:
            thresh = int(file_name.split(',')[-1])
            # thresh = 128## int(thresh)
        if WORD_LEVEL:
            subdir = 'words/'
        else:
            subdir = 'lines/'
        file_name = file_name.split(',')[0]
        url = baseDir + subdir + file_name + '.jpg'
        img = cv2.imread(url)

        
        
        try:
            img.any()
        except:
            print('###!Cannot find image: ' + url)

        
        if RM_BACKGROUND:
            img[img>thresh] = 255
        #img = 255 - img
        #img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        #size = img.shape[0] * img.shape[1]

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        # c04-066-01-08.png 4*3, for too small images do not augment
        if self.augmentation: # augmentation for training data
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_new = self.transformer(255-gray)
            
            im_bg,im_blur = self.addDist(img)
            im_bg = cv2.cvtColor(im_bg, cv2.COLOR_BGR2GRAY)
            im_blur = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)
            

            if img_new.shape[0] != 0 and img_new.shape[1] != 0:
                rate = float(IMG_HEIGHT) / img_new.shape[0]
                img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error

                rate = float(IMG_HEIGHT) / im_bg.shape[0]
                im_bg = cv2.resize(im_bg, (int(im_bg.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error

                rate = float(IMG_HEIGHT) / im_blur.shape[0]
                im_blur = cv2.resize(im_blur, (int(im_blur.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error


            # else:
            #     img = 255 - img
            #     im_bg = 255 - im_bg
            #     im_blur = 255 - im_blur
                
        else:
            im_bg,im_blur = self.addDist(img)
            img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im_bg = 255 - cv2.cvtColor(im_bg, cv2.COLOR_BGR2GRAY)
            im_blur = 255 - cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)

        img_width = img.shape[1]
        img_bg_width = im_bg.shape[1]
        

        if flip: # because of using pack_padded_sequence, first flip, then pad it
            img = np.flip(img, 1)

            im_bg = np.flip(im_bg, 1)
            im_blur = np.flip(im_blur, 1)

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            #outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg[:, :img_width] = img
        
        if img_bg_width > IMG_WIDTH:
            outImg_bg = cv2.resize(im_bg, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            outImg_blur = cv2.resize(im_blur, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_bg_width = IMG_WIDTH
        else:   
            outImg_bg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg_bg[:, :img_bg_width] = im_bg

            outImg_blur = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg_blur[:, :img_bg_width] = im_blur


        outImg = outImg/255. #float64
        outImg = outImg.astype('float32')

        outImg_bg = outImg_bg/255. #float64
        outImg_bg = outImg_bg.astype('float32')
        
        outImg_blur = outImg_blur/255. #float64
        outImg_blur = outImg_blur.astype('float32')
        
        if VGG_NORMAL:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]


            outImgFinal = np.zeros([3, *outImg.shape])
            for i in range(3):
                outImgFinal[i] = (outImg - mean[i]) / std[i]
            
            outImg_bgFinal = np.zeros([3, *outImg_bg.shape])
            for i in range(3):
                outImg_bgFinal[i] = (outImg_bg - mean[i]) / std[i]
            
            outImg_blurFinal = np.zeros([3, *outImg_blur.shape])
            for i in range(3):
                outImg_blurFinal[i] = (outImg_blur - mean[i]) / std[i]
            
            return outImgFinal,outImg_bgFinal,outImg_blurFinal,  img_width


    

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
    data_train = Get_words(file_label_tr, "train", augmentation= TRAINTYPE=='htr_Augm')
    data_valid = Get_words(file_label_va, "valid" , augmentation=False)
    data_test = Get_words(file_label_te, "test",  augmentation=False)
    
    return data_train, data_valid, data_test


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    train_in_dist_bg = []
    train_in_dist_blur = []

    for i in range(n_batch):
        idx, img, dist_img_bg, dist_img_blur, img_width, label = batch[i]

        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)
        train_in_dist_bg.append(dist_img_bg)
        train_in_dist_blur.append(dist_img_blur)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')
    train_in_dist_bg = np.array(train_in_dist_bg, dtype='float32')
    train_in_dist_blur = np.array(train_in_dist_blur, dtype='float32')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)
    train_in_dist_bg = torch.from_numpy(train_in_dist_bg)
    train_in_dist_blur = torch.from_numpy(train_in_dist_blur)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_in_dist_bg = train_in_dist_bg[idx]
    train_in_dist_blur = train_in_dist_blur[idx]
    
    train_out = train_out[idx]
    train_index = train_index[idx]


    return train_index, train_in, train_in_dist_bg, train_in_dist_blur, train_in_len, train_out




def all_data_loader(batch_size):
    data_train, data_valid, data_test = loadData()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader