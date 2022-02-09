import editdistance
from tqdm import tqdm
import Config as C


def count_cer(set, epoch,path):

    names=[]
    texts=[]
    with open (C.baseDir_line+set+"_words.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        name = line.split(' ')[0]
        text = line.replace(name+' ','')
        names.append(name)
        texts.append(text)
    indextoline = {label: t for label, t in zip(names,texts)}

    with open(path+"/"+set+"_predict_seq."+str(epoch)+".log") as f:
        preds = f.readlines()


    cer=0
    w_acc=0
    qo=0
    for  p in preds:
        p = p.split('\n')[0]
        p_name = p.split(' ')[0]
        p_text = p.replace(p_name+' ','')
        text = indextoline[p_name]
        cer += (editdistance.eval(p_text,text))/ len(text) #max(len(text),len(p_text))
        w_acc += 1*(p_text == text)
        qo+=1
    final_cer = cer/qo
    final_w_acc = w_acc /qo
    return final_cer,final_w_acc



