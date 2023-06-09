import torch
import torch.nn as nn
from models.ocr import Seq2SeqTransformer
from models.vit import ViT
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from einops import rearrange
import os
import loadData
from timeit import default_timer as timer
import utils 
from tqdm import tqdm
from Config import Configs




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get your configuration here:
cfg = Configs().parse()
train_type = cfg.train_type
batch_size = cfg.batch_size
patch_size = cfg.vit_patch_size
image_size =  (cfg.img_height,cfg.img_width)
test_model = cfg.test_model

# here the name of the current eperiment
EXPERIMENT = 'test_'+train_type + '_' + str(image_size[0])+'_'+str(image_size[1])+'_'+str(patch_size)
os.system('rm -r  pred_logs/'+EXPERIMENT)


# get the utils functions and variables
labelDictionary = loadData.labelDictionary
num_classes, letter2index, index2letter = labelDictionary()
tokens = loadData.tokens
num_tokens = loadData.num_tokens
SRC_VOCAB_SIZE = 1
TGT_VOCAB_SIZE = num_classes + num_tokens
count_cer = utils.count_cer
writePrediction = utils.writePrediction
load_data_func = loadData.loadData
all_data_loader = loadData.all_data_loader


# Build the dataloader for testing
_, _ , testloader = all_data_loader(batch_size)


PAD_IDX = tokens['PAD_TOKEN']
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



# Variables for the models size, those are for a "base" architecture

NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
EMB_SIZE = 768
NHEAD = 8
FFN_HID_DIM = 768



# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)


    memory = transformer.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = transformer.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = transformer.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 1:
            break
    return ys


# function to recognize images into texts 
def test_image(model: torch.nn.Module, src_imgs: str , tgt_size = 38):
    transformer.eval()
    batch_y=[]
    for src in src_imgs:
        src = src.view(1,3,image_size[0],image_size[1])
        num_tokens = tgt_size
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            transformer,  src, src_mask, max_len=num_tokens + 5, start_symbol=0).flatten()
        batch_y.append(tgt_tokens)
    return batch_y 



# Define the ViT encoder
vit_encoder = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = EMB_SIZE,  #1024
    depth = NUM_ENCODER_LAYERS,  #6
    heads = NHEAD,  #8
    mlp_dim = 2048
)


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,custom_encoder=vit_encoder,device=DEVICE,use_stn = train_type == 'stn')

transformer = transformer.to(DEVICE)


def test():
    epoch = 0
    transformer.eval()
    losses = 0
    folder_name = 'pred_logs/'+EXPERIMENT
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+'test'+'_predict_seq.'
    f = open(file_prefix+str(epoch)+'.log', 'a')
    print('will iterate for ',len(testloader))
    for i, (test_index, test_in, test_in_len, test_out) in tqdm(enumerate(testloader)):
        
        batch_y_pred = test_image(transformer,test_in)
        
        for ib in range(len(test_index)):
            w = ''
            y_pred = batch_y_pred[ib]
            y_index = test_index[ib]
            for c in y_pred[1:-1]:
                w +=  ' '+str(int(c.item())-num_tokens)
            f.write(y_index+' '+w+'\n')
    f.close()

    test_cer, test_wacc = count_cer('test',epoch,"pred_logs/"+EXPERIMENT)
    print("Test CER: ",test_cer)
    print("Test WACC: ",test_wacc)


if __name__ == "__main__":

    transformer.load_state_dict(torch.load(test_model))
    test()