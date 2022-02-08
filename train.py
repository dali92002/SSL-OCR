from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List
from models.models import PositionalEncoding, TokenEmbedding, Seq2SeqTransformer
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from einops import rearrange
import os
import loadData2_vgg as loadData
from timeit import default_timer as timer
import utils 
import Config as C

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




torch.manual_seed(0)

TRAINTYPE = C.TRAINTYPE
MODELSIZE = C.SETTING
EMB_SIZE = C.EMB_SIZE
NHEAD = C.NHEAD
FFN_HID_DIM = C.FFN_HID_DIM

BATCH_SIZE = C.batch_size
NUM_ENCODER_LAYERS = C.NUM_ENCODER_LAYERS
NUM_DECODER_LAYERS = C.NUM_DECODER_LAYERS   ## Base 6 layers  512 dim and 8 heads ## 

patch_size = C.patch_size      ### 16 ?
image_size =  C.image_size



DATATYPE = C.DATATYPE


EXPERIMENT = DATATYPE+ '_' +TRAINTYPE + '_' +MODELSIZE+ '_' + str(image_size[0])+'_'+str(image_size[1])+'_'+str(patch_size)

def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter






def writePredict(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs/'+EXPERIMENT
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+flag+'_predict_seq.'
    
    # pred = rearrange(pred, 't b s-> b t s')
    pred = pred.data
    pred2 = pred.topk(1)[1].squeeze(2) # (15, 32)
    pred2 = pred2.transpose(0, 1) # (32, 15)
    pred2 = pred2.cpu().numpy()

    batch_count_n = []
    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        for n, seq in zip(index, pred2):
            f.write(n+' ')
            count_n = 0
            for i in seq:
                if i ==tokens['END_TOKEN']:
                    break
                else:
                    if i ==tokens['GO_TOKEN']:
                        f.write('<GO>')
                    elif i ==tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        f.write(index2letter[i-num_tokens])
                    count_n += 1
            batch_count_n.append(count_n)
            f.write('\n')
    return batch_count_n


num_classes, letter2index, index2letter = labelDictionary()

tokens = loadData.tokens
num_tokens = loadData.num_tokens


SRC_VOCAB_SIZE = 1
TGT_VOCAB_SIZE = num_classes + num_tokens


count_cer = utils.count_cer
load_data_func = loadData.loadData
transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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
    
    # for t in (train_out):
    #     print((t))
    
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


def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

trainloader, validloader, testloader = all_data_loader()


# Define special symbols and indices
PAD_IDX = 2
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask






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
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,custom_encoder=vit_encoder,device=DEVICE)



for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

if TRAINTYPE == 'pretrain0':
    transformer.transformer.encoder.load_state_dict(torch.load('./weights/best-encoder-word_base_64_256_8_100.pt'))
    # for param in transformer.transformer.encoder.parameters():
    #     param.requires_grad = False
    print('Encoder loaded !')

if TRAINTYPE == 'fine_tune_IAM':
    transformer.load_state_dict(torch.load('./weights/best-seq2seq_No_pretrain_base_64_512_8.pt'))
    print(' model loaded fine tuning on synthetic !!') 


if TRAINTYPE == 'half_fine_tune_IAM':
    transformer.load_state_dict(torch.load('./weights/best-seq2seq_No_pretrain_base_64_512_16.pt'))
    for param in transformer.transformer.decoder.parameters():
        param.requires_grad = False

    print('model loaded (half tuning) !!')


transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# optimizer = optim.AdamW(transformer.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)  ### <-- Pretraining

# optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  ## <-- original seq2seq

# optimizer = torch.optim.SGD(transformer.parameters(), lr=0.01, momentum=0.9)   ## linear prob

optimizer = optim.Adam(transformer.parameters(),lr=1e-6, betas=(0.9, 0.95), eps=1e-6)  ### <-- fine-tune whole layers




def train_epoch(model, optimizer):
    model.train()
    losses = 0
    running_loss = 0.0
    for i, (train_index, train_in, train_in_len, train_out) in enumerate(trainloader):
        src = train_in.to(DEVICE)
        tgt = train_out.to(DEVICE)

        tgt = rearrange(tgt, 'b t -> t b')

        tgt_input = tgt[:-1, :]    ############## ???????  why is this ?

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(tgt_input, tgt_input)
        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        logits = model(src, tgt_input, None, None, None, None, None)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        running_loss += loss.item()
        
        writePredict(epoch, train_index, logits, 'train')

        show_every = int(len(trainloader) / 10)
        if i % show_every == show_every-1:    # print every 500 mini-batches
            print('[epoch: %d, iter: %5d] Train. loss: %.3f' % (epoch, i + 1, running_loss / show_every))
            running_loss = 0.0
       

    return losses / len(trainloader)




def test():
    transformer.eval()
    losses = 0

    for i, (test_index, test_in, test_in_len, test_out) in enumerate(testloader):

        src = test_in.to(DEVICE)
        tgt = test_out.to(DEVICE)
        
        tgt = rearrange(tgt, 'b t -> t b')

        tgt_input = tgt[:-1, :]

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(torch.rand(int(image_size[0]/patch_size*image_size[1]/patch_size),BATCH_SIZE), tgt_input)
        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = transformer(src, tgt_input, None, None, None, None, None)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        
        writePredict(epoch, test_index, logits, 'test')



def evaluate():
    transformer.eval()
    losses = 0

    for i, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(validloader):

        src = valid_in.to(DEVICE)
        tgt = valid_out.to(DEVICE)
        
        tgt = rearrange(tgt, 'b t -> t b')

        tgt_input = tgt[:-1, :]

        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(torch.rand(int(image_size[0]/patch_size*image_size[1]/patch_size),BATCH_SIZE), tgt_input)
        # logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        logits = transformer(src, tgt_input, None, None, None, None, None)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        
        writePredict(epoch, valid_index, logits, 'valid')

    
    cer,wacc = count_cer('valid',epoch,"pred_logs/"+EXPERIMENT)
    
    

    print("Valid CER: ",cer)
    print("Valid WACC: ",wacc)
    print("Last Best Valid CER: ",best_cer[0], "Epoch: ",best_cer[1])
    
    
    if cer < best_cer[0]:
        test(model)
        test_cer, test_wacc = count_cer('test',epoch,"pred_logs/"+EXPERIMENT)
        print("Test CER: ",test_cer)
        print("Test WACC: ",test_wacc)

        best_cer[0] = cer
        best_cer[1] = epoch
        torch.save(transformer.state_dict(), './weights/best-seq2seq_'+EXPERIMENT+'.pt')
            

    return losses / len(validloader)


st_epoch = 1
NUM_EPOCHS = 600

best_cer = [1,0]

continue_train = C.continue_train

if continue_train:
    epoch = 601
    transformer.load_state_dict(torch.load('./weights/best-seq2seq_'+EXPERIMENT+'.pt'))
    print('continue training ! ')
    val_loss = evaluate(transformer)



for epoch in range(st_epoch, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    f=41
a=41


