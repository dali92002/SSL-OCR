import sched
import torch
from vit_pytorch import ViT
from models.diae import DIAE
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from einops import rearrange
import os
import loadData_pretrain as loadData
from Config import Configs
import cv2

all_data_loader = loadData.all_data_loader
device = torch.device('cuda:0')
load_data_func = loadData.loadData
transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

C = Configs().parse()
batch_size = C.batch_size
patch_size = C.vit_patch_size
image_size =  (C.img_height,C.img_width)
MASKINGRATIO = 0.60
vis_results = C.vis_results
baseDir = C.data_path
weightDir = C.weights_path

NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
EMB_SIZE = 768
NHEAD = 8
FFN_HID_DIM = 768


EXPERIMENT = "pretrain"+'_' + str(image_size[0])+'_'+str(image_size[1])+'_'+str(patch_size)


trainloader, validloader, _ = all_data_loader(batch_size)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('foo.png')
    # plt.show()



def imvisualize(immask,imgt,impred,ind,epoch='0',iter='0'):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    immask = immask.numpy()
    imgt = imgt.numpy()
    impred = impred.numpy()
    immask = np.transpose(immask, (1, 2, 0))
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))
    
    
    for ch in range(3):
        immask[:,:,ch] = (immask[:,:,ch] *std[ch]) + mean[ch]
        imgt[:,:,ch] = (imgt[:,:,ch] *std[ch]) + mean[ch]
        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

    if not os.path.exists('vis_'+EXPERIMENT+'/epoch'+epoch):
        os.makedirs('vis_'+EXPERIMENT+'/epoch'+epoch)
    if not os.path.exists('vis_'+EXPERIMENT+'/epoch'+epoch+'/'+'iter'+iter):
        os.makedirs('vis_'+EXPERIMENT+'/epoch'+epoch+'/'+'iter'+iter)
    
    

    cv2.imwrite('vis_'+EXPERIMENT+'/epoch'+epoch+'/'+'iter'+iter+'/'+str(ind)+'masked.jpg',(immask*255))
    cv2.imwrite('vis_'+EXPERIMENT+'/epoch'+epoch+'/'+'iter'+iter+'/'+str(ind)+'gt.jpg',(imgt*255))
    cv2.imwrite('vis_'+EXPERIMENT+'/epoch'+epoch+'/'+'iter'+iter+'/'+str(ind)+'pred.jpg',(impred*255))
    



v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = EMB_SIZE,
    depth = NUM_ENCODER_LAYERS,
    heads = NHEAD,
    mlp_dim = 2048
)



diae = DIAE(
    encoder = v,
    masking_ratio = MASKINGRATIO,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6 ,
    image_size = image_size,
    patch_size = patch_size,
    dim = FFN_HID_DIM, 
)



diae = diae.to(device)

optimizer = optim.AdamW(diae.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))

def visualize(epoch,iter):
    diae.eval()
    VIS_NUMBER=100
    # for i, data in enumerate(testloader, 0):
    for i, (valid_index, valid_in, valid_in_bg, valid_in_bl, valid_in_len, valid_out) in enumerate(validloader):
        # inputs, labels = data
        inputs = valid_in.to(device)
        inputs_bg = valid_in_bg.to(device)
        inputs_bl = valid_in_bl.to(device)
        labels = valid_out.to(device)

        with torch.no_grad():
            rec_loss,en_loss,deb_loss,patches, batch_range, masked_indices, pred_pixel_values, _ = diae(inputs, inputs_bg, inputs_bl)
            
            rec_patches = patches.clone().detach()
            
            rec_patches[batch_range, masked_indices] = pred_pixel_values
            
            maskes = torch.zeros(pred_pixel_values.size())+0.5
            maskes = maskes.to(device)
            masked_patches = patches.clone().detach()
            masked_patches[batch_range, masked_indices]= maskes

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            masked_images = rearrange(masked_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            for j in range (0,len(valid_index)):
                imvisualize(masked_images[j].cpu(), inputs[j].cpu(),rec_images[j].cpu(),valid_index[j],epoch,iter)
                VIS_NUMBER -= 1
        
        if VIS_NUMBER <0:
            break



best_valid_loss = 99999999



def valid_model(best_loss):
	
    losses = 0
    diae.eval()
    for i, (valid_index, valid_in, valid_in_bg, valid_in_bl,valid_in_len, valid_out) in enumerate(validloader):
        
        inputs = valid_in.to(device)
        inputs_bg = valid_in_bg.to(device)
        inputs_bl = valid_in_bl.to(device)
        
        with torch.no_grad():
            loss_rec,loss_enh,loss_blur,_, _, _, _ ,_= diae(inputs, inputs_bg, inputs_bl)
            loss = loss_rec + loss_enh + loss_blur
            losses += loss.item()
    
    losses = losses / len(validloader)
    if losses < best_loss:
        best_loss = losses
        if not os.path.exists(weightDir+ 'weights/'):
            os.makedirs(weightDir+ 'weights/')
        torch.save(v.state_dict(), weightDir+ 'weights/best-encoder-'+EXPERIMENT+'.pt')
        torch.save(diae.state_dict(), weightDir+ 'weights/best-diae-'+EXPERIMENT+'.pt')
    return best_loss, losses
    




schd = False
for epoch in range(100): 

    running_loss = 0.0
    running_loss_r = 0.0
    running_loss_e = 0.0
    running_loss_b = 0.0
    
    # for i, data in enumerate(trainloader, 0):
    for i, (train_index, train_in,train_in_bg,train_in_bl, train_in_len, train_out) in enumerate(trainloader):
        

        # inputs, labels = data
        inputs = train_in.to(device)
        inputs_bg = train_in_bg.to(device)
        inputs_bl = train_in_bl.to(device)

        labels = train_out.to(device)


        optimizer.zero_grad()

        loss_rec,loss_enh,loss_blur,_, _, _, _,_= diae(inputs,inputs_bg,inputs_bl)
        
        loss_rec = loss_rec
        loss_enh = loss_enh
        loss_blur = loss_blur

        loss = loss_rec + loss_enh + loss_blur
        
        running_loss_r += loss_rec.item()
        running_loss_e += loss_enh.item()
        running_loss_b += loss_blur.item()

        loss.backward()
        
        if i == 50000:
            print('start scheduler')
            schd = True
        
        optimizer.step()
        if schd:
            scheduler.step()

        

        running_loss += loss.item()
        
        if i % 5000 ==0:
            if not os.path.exists(weightDir+ 'weights/'):
                os.makedirs(weightDir+ 'weights/')
            torch.save(v.state_dict(), weightDir+ 'weights/checkpoint-encoder-'+EXPERIMENT+'_pretrain.pt')
            torch.save(diae.state_dict(), weightDir+ 'weights/checkpoint-diae-'+EXPERIMENT+'_pretrain.pt')


        show_every = int(len(trainloader) / 10)

        if i % show_every == show_every-1:    # print every 20 mini-batches
            if vis_results and epoch%1 ==0:
                visualize(str(epoch),str(i))
                diae.train()
            
            print('[Epoch: %d, Iter: %5d] Train: Reconst. loss: %.3f, Enh. loss: %.3f, Deblur. loss: %.3f, Tot. Loss: %.3f' % (epoch, i + 1, running_loss_r / show_every, running_loss_e / show_every,running_loss_b / show_every,running_loss / show_every))
            running_loss = 0.0
            running_loss_r = 0.0
            running_loss_e = 0.0
            running_loss_b = 0.0
        
    best_valid_loss,valid_loss = valid_model(best_valid_loss)
    diae.train()
    print('Valid loss: ',valid_loss)
    print('Best valid loss: ',best_valid_loss)
