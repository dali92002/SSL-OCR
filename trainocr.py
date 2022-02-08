from json import encoder
import torch
from vit_pytorch import ViT, MAE
from models.ocr import OCR
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from einops import rearrange
import os
import loadData2_vgg as loadData

device = torch.device('cuda:0')
load_data_func = loadData.loadData
transform = transforms.Compose([transforms.RandomResizedCrop(256),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

patch_size = 16
image_size =  (128,512)


v = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

ocrmodel = OCR(encoder = v, n_trg_vocab=80)

# mae = MAE(
#     encoder = v,
#     masking_ratio = 0.75,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )

# images = torch.randn(8, 3, 256, 256)


ocrmodel = ocrmodel.to(device)

# loss = mae(images)
# loss.backward()

optimizer = optim.AdamW(ocrmodel.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)


def visualize(epoch,iter):
    a=45
    # for i, data in enumerate(testloader, 0):
    for i, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(validloader):
        # inputs, labels = data
        inputs = valid_in.to(device)
        labels = valid_out.to(device)

        with torch.no_grad():
            loss,patches, batch_range, masked_indices, pred_pixel_values = mae(inputs)
            
            rec_patches = patches
            rec_patches[batch_range, masked_indices] = pred_pixel_values

            rec_images = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,  h=image_size[0]//16)
            
            for i in range (0,40):
                imvisualize(inputs[i].cpu(),rec_images[i].cpu(),i,epoch,iter)
        break


for epoch in range(10): 

    running_loss = 0.0
    # for i, data in enumerate(trainloader, 0):
    for i, (train_index, train_in, train_in_len, train_out) in enumerate(validloader):   ### train loader
        

        # inputs, labels = data
        inputs = train_in.to(device)
        labels = train_out.to(device)

        optimizer.zero_grad()

        outputs = ocrmodel(inputs,labels)

        loss.backward()
        optimizer.step()

        

        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            
            visualize(str(epoch),str(i))
            
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0


p=415