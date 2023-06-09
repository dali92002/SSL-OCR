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
pretrained_encoder_path = cfg.pretrained_encoder_path

# here the name of the current eperiment
EXPERIMENT = train_type + '_' + str(image_size[0])+'_'+str(image_size[1])+'_'+str(patch_size)
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
PAD_IDX = tokens['PAD_TOKEN']


# Build the dataloaders
trainloader, validloader, _ = all_data_loader(batch_size)

# Variables for the models size, those are for a "base" architecture
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
EMB_SIZE = 768
NHEAD = 8
FFN_HID_DIM = 768

# Define the ViT encoder
vit_encoder = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = EMB_SIZE,  
    depth = NUM_ENCODER_LAYERS,  #6
    heads = NHEAD,  #8
    mlp_dim = 2048
)

# Define the Full Transformer with the previous ViT as encoder
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,custom_encoder=vit_encoder,device=DEVICE,use_stn = train_type == 'stn')


# Model weight initializations
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer.transformer.encoder.load_state_dict(torch.load(pretrained_encoder_path))

transformer = transformer.to(DEVICE)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Define the optimizer and scheduler
optimizer = optim.Adam(transformer.parameters(),lr=1.5e-5, betas=(0.9, 0.95))  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))



# Function to train one epoch
def train_epoch(optimizer,sch=False):
    transformer.train()
    losses = 0
    running_loss = 0.0
    iters = 0
    for i, (train_index, train_in, train_in_len, train_out) in enumerate(trainloader):
        src = train_in.to(DEVICE)
        tgt = train_out.to(DEVICE)


        tgt = rearrange(tgt, 'b t -> t b')
        tgt_input = tgt[:-1, :]  
        tgt_out = tgt[1:, :]
        
        logits = transformer(src, tgt_input, None, None, None, None, None)

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        iters+=1
        optimizer.step()
        
        if sch:
            scheduler.step()
	
        losses += loss.item()
        running_loss += loss.item()
        
        writePrediction(epoch, train_index, logits, 'train' , 'pred_logs/'+EXPERIMENT)

        show_every = int(len(trainloader) / 10) # Specify at which number of iterations to show the loss, you can save it to visualize also
        if i % show_every == show_every-1:    
            print('[epoch: %d, iter: %5d] Train. loss: %.3f' % (epoch, i + 1, running_loss / show_every))
            running_loss = 0.0
        
    return losses / len(trainloader)


# Evaluate model on the validation set, count the CER, ACC  each epoch and save the best weights.
def evaluate():
    transformer.eval()
    losses = 0

    for i, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(validloader):

        src = valid_in.to(DEVICE)
        tgt = valid_out.to(DEVICE)
        
        tgt = rearrange(tgt, 'b t -> t b')

        tgt_input = tgt[:-1, :]
        
        with torch.no_grad():
            logits = transformer(src, tgt_input, None, None, None, None, None)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        
        writePrediction(epoch, valid_index, logits, 'valid','pred_logs/'+EXPERIMENT)

    
    cer,wacc = count_cer('valid',epoch,"pred_logs/"+EXPERIMENT)
    
    print("Valid CER: ",cer)
    print("Valid WACC: ",wacc)
    print("Last Best Valid CER: ",best_cer[0], "Epoch: ",best_cer[1])
    
    
    if cer<=best_cer[0]:
        best_cer[0] = cer
        best_cer[1] = epoch
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
        torch.save(transformer.state_dict(), './weights/best-seq2seq_'+EXPERIMENT+'.pt')
    
    return losses / len(validloader)


if __name__ == "__main__":
    st_epoch = 1
    NUM_EPOCHS = 600
    best_cer = [10,0] # this list is representing  [best_cer, best_epoch]
    schd = False
    
    for epoch in range(st_epoch, NUM_EPOCHS+1):
        start_time = timer()
        
        if epoch == 10: # 10 epochs for warmup
            print('start scheduler !')
            schd = True
        train_loss = train_epoch(optimizer,schd)
        end_time = timer()
        val_loss = evaluate()
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))