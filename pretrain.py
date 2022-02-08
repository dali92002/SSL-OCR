import torch
from vit_pytorch import ViT
from models.mae import MAE
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from einops import rearrange
import os
import loadData2_vgg as loadData
import Config as C

device = torch.device('cuda:0')
load_data_func = loadData.loadData
transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

MODELSIZE = C.SETTING
DATATYPE = C.DATATYPE
batch_size = C.batch_size


patch_size = C.patch_size
image_size =  C.image_size
MASKINGRATIO = C.MASKINGRATIO
VIS_RESULTS = C.VIS_RESULTS


if MODELSIZE == 'base':
    ENCODERLAYERS = 6
    ENCODERHEADS = 8
    ENCODERDIM = 768
if MODELSIZE == 'small':
    ENCODERLAYERS = 3
    ENCODERHEADS = 4
    ENCODERDIM = 512

EXPERIMENT = DATATYPE+'_'+MODELSIZE+ '_' + str(image_size[0])+'_'+str(image_size[1])+'_'+str(patch_size)

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
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

trainloader, validloader, testloader = all_data_loader()



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('foo.png')
    # plt.show()



def imvisualize(imgt,impred,ind,epoch='0',iter='0'):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    
    
    
    # imgt = imgt / 2 + 0.5     # unnormalize
    # impred = impred / 2 + 0.5     # unnormalize
    


    imgt = imgt.numpy()
    impred = impred.numpy()
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))
    
    for ch in range(3):
        imgt[:,:,ch] = (imgt[:,:,ch] *std[ch]) + mean[ch]
        impred[:,:,ch] = (impred[:,:,ch] *std[ch]) + mean[ch]

    impred[np.where(impred>1)] = 1
    impred[np.where(impred<0)] = 0

    if not os.path.exists('vis/epoch'+epoch):
        os.makedirs('vis/epoch'+epoch)
    if not os.path.exists('vis/epoch'+epoch+'/'+'iter'+iter):
        os.makedirs('vis/epoch'+epoch+'/'+'iter'+iter)
    
    

    plt.imsave('vis/epoch'+epoch+'/'+'iter'+iter+'/'+str(ind)+'gt.jpg',imgt)
    plt.imsave('vis/epoch'+epoch+'/'+'iter'+iter+'/'+str(ind)+'pred.jpg',impred)
    


# dataiter = iter(trainloader)
# images, labels = dataiter.next()





v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = ENCODERDIM,
    depth = ENCODERLAYERS,
    heads = ENCODERHEADS,
    mlp_dim = 2048
)



mae = MAE(
    encoder = v,
    masking_ratio = MASKINGRATIO,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

# images = torch.randn(8, 3, 256, 256)


mae = mae.to(device)

# loss = mae(images)
# loss.backward()

optimizer = optim.AdamW(mae.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)


def visualize(epoch,iter):
    a=45
    # for i, data in enumerate(testloader, 0):
    for i, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(validloader):
        # inputs, labels = data
        inputs = valid_in.to(device)
        labels = valid_out.to(device)

        with torch.no_grad():
            loss,patches, batch_range, masked_indices, pred_pixel_values, masked_pixels = mae(inputs)
            
            rec_patches = patches
            rec_patches[batch_range, masked_indices] = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
            
            for i in range (0,batch_size):
                imvisualize(inputs[i].cpu(),rec_images[i].cpu(),i,epoch,iter)
        break



best_valid_loss = 99999999



def valid_model(best_loss):
    losses = 0
    for i, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(validloader):
        
        inputs = valid_in.to(device)
        labels = valid_out.to(device)

        optimizer.zero_grad()

        loss,_, _, _, _ ,_= mae(inputs)

        loss.backward()
        optimizer.step()

        

        losses += loss.item()
    losses = losses / len(validloader)
    if losses < best_loss:
        best_loss = losses
        torch.save(v.state_dict(), './weights/best-encoder-'+EXPERIMENT+'.pt')
    
    return best_loss, losses
    





for epoch in range(101): 

    running_loss = 0.0
    # for i, data in enumerate(trainloader, 0):
    for i, (train_index, train_in, train_in_len, train_out) in enumerate(trainloader):
        

        # inputs, labels = data
        inputs = train_in.to(device)
        labels = train_out.to(device)

        optimizer.zero_grad()

        loss,_, _, _, _,_= mae(inputs)

        loss.backward()
        optimizer.step()

        

        running_loss += loss.item()
        
        show_every = int(len(trainloader) / 7)

        if i % show_every == show_every-1:    # print every 20 mini-batches
            if VIS_RESULTS and epoch%5 ==0:
                visualize(str(epoch),str(i))
            
            print('[Epoch: %d, Iter: %5d] Train loss: %.3f' % (epoch + 1, i + 1, running_loss / show_every))
            running_loss = 0.0
        
    best_valid_loss,valid_loss = valid_model(best_valid_loss)
    print('Valid loss: ',valid_loss)
    print('Best valid loss: ',best_valid_loss)

torch.save(v.state_dict(), './weights/best-encoder-'+EXPERIMENT+'_100.pt')
