import editdistance
from tqdm import tqdm
from Config import Configs
import loadData
import os
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt


cfg = Configs().parse()

labelDictionary = loadData.labelDictionary
num_classes, letter2index, index2letter = labelDictionary()
tokens = loadData.tokens
num_tokens = loadData.num_tokens
baseDir = cfg.data_path


# Function to count the CER and Word Accuracy (W_ACC) 
def count_cer(set, epoch,path):

    names=[]
    texts=[]
    with open (baseDir+"/"+set+".txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        name = line.split(' ')[0]
        nums = line.replace(name+' ','')
        nums = nums.split(' ')
        text = ''
        for n in nums:
            text+= chr(ord('A')+int(n))

        names.append(name)
        texts.append(text)
    indextoline = {label: t for label, t in zip(names,texts)}

    with open(path+"/"+set+"_predict_seq."+str(epoch)+".log") as f:
        preds = f.readlines()


    cer=0
    w_acc=0
    ed1 = 0
    qo=0
    for  p in preds:
        p = p.split('\n')[0]
        p_name = p.split(' ')[0]
        p_nums = p.replace(p_name+' ','')
        p_nums = p_nums.split(' ')
        p_text = ''
        while '' in p_nums:
            p_nums.remove('')
        for pn in p_nums:
            if int(pn) >=0:
                p_text+=chr(ord('A')+int(pn))

        text = indextoline[p_name]
        cer += (editdistance.eval(p_text,text))/ len(text) 
        w_acc += 1*(p_text == text)
        qo+=1
    final_cer = cer/qo
    final_w_acc = (w_acc /qo)
    return final_cer,final_w_acc



def writePrediction(epoch, index, pred, flag, folder_name = ''): 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+flag+'_predict_seq.'
    
    pred = pred.data
    pred2 = pred.topk(1)[1].squeeze(2) 
    pred2 = pred2.transpose(0, 1) 
    pred2 = pred2.cpu().numpy()

    batch_count_n = []
    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        for n, seq in zip(index, pred2):
            text = n+''
            count_n = 0
            for i in seq:
                if i ==tokens['END_TOKEN']:
                    break
                else:
                    text+=' '+str(i-num_tokens)
                    count_n += 1
            batch_count_n.append(count_n)
            if n==text:
                text+=' '
            f.write(text+'\n')
    return batch_count_n




def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(transformer,loader,DEVICE):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(loader))[1].to(DEVICE)

        input_tensor = data.cpu()
        transformed_input_tensor = transformer.transformer.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

