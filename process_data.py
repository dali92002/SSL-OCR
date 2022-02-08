import os 
from  shutil import copy
from tqdm import tqdm
import shutil


from torch.utils import data





# # labels = ['\n',' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

with open ('data/iamwords/RWTH.iam_line_gt_final.train.thresh') as f:
    lines  = f.readlines()
# fc=open('data/syntdata/RWTH.iam_line_gt_final.valid.thresh',"w")
max_c =  0
for line in tqdm(lines[:]):
    max_c = max(max_c,len(line.replace(line.split(' ')[0]+' ','')))
fc.close()    


# gt_tr = 'RWTH.iam_'+subname+'_gt_final.train.thresh'
# gt_va = 'RWTH.iam_'+subname+'_gt_final.valid.thresh'
# gt_te = 'RWTH.iam_'+subname+'_gt_final.test.thresh'


# s=0
# fc=open('data/syntdata/clean_annonation.txt',"w")
# for line in tqdm(lines):
#     id = line.split('128 ')[0]
#     text = line.split('128 ')[-1]
    
#     clean = True
#     for t in text:
#         if t not in labels:
#             s+=1
#             clean = False
#             break
#     if clean:
#         fc.write(line)
# fc.close()      


# h=5


# division = 20


# with open ("data/iamwords/RWTH.iam_word_gt_final.train.thresh0") as f:
#     lines = f.readlines()

# f1 = open("data/iamwords/RWTH.iam_word_gt_final.train.thresh","w")
# lines  = lines[:int(len(lines)*(division/100))]

# for line in lines:
#     f1.write(line)
# f1.close()


### lines

folders = os.listdir("/home/mohamed/vit/iam/linesorig")
for f in tqdm(folders):
    subfolders = os.listdir("/home/mohamed/vit/iam/linesorig/"+f)
    for sub in subfolders:
        images = os.listdir("/home/mohamed/vit/iam/linesorig/"+f+"/"+sub)
        for im in images:
            shutil.copy("/home/mohamed/vit/iam/linesorig/"+f+"/"+sub+"/"+im,"data/iamwords/lines/"+im)



#### words
folders = os.listdir("data/iamwords/wordsorig")
for f in tqdm(folders):
    subfolders = os.listdir("data/iamwords/wordsorig/"+f)
    for sub in subfolders:
        images = os.listdir("data/iamwords/wordsorig/"+f+"/"+sub)
        for im in images:
            shutil.copy("data/iamwords/wordsorig/"+f+"/"+sub+"/"+im,"data/iamwords/words/"+im)




p=78