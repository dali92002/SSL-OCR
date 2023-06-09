import cv2
from tqdm import tqdm
vocab = []
labels = ['1', '2', '9', '4', '3', '0', '7', '5', '8', '6', '', '2+', '0-', ',', '8+', '0^.', '.', '3^.', '6_.', '6__', '2__', '7__', '3__', '9__', '4__', '5__', '1__', '6_', '5_', '2^.', "9^'", "0^'", "1^'", "7^'", '5^.', '1^.', "5^'", '7^,', '5^,', "2^'", '3^’', '2^`', '0^`', '9^.', '1^`', '8^`', '5^`', '7^`', '8^_', '6^_', '9^`', '3^`', '0_.', '~', ':1', '.1', '9_.', '5_.', '4_.', '1_.', '8_.', '2_.', '7_.', '6^.', '7^.', '2/', '4^.', '8^.', '5/', '3_.', '2^./', ':', '9^_', '5^._.', '8^’', "8^'"]
letter2index = {label: n for n, label in enumerate(labels)}


f = open("data/valid0.txt")
f2 = open("data/valid.txt","w")

lines  = f.readlines()


for l in tqdm(lines):
    line = l.split(",128 ")[1].split("\n")[0]
    line = line.replace("?","")
    img = l.split(" ")[0]
    chars = [letter2index[i] for i in line.split(' ')]
    l2 = str(chars[0]) + " "
    for c in chars[1:]:
        l2 += str(c)+" "
    l2 = l2[:-1]

    f2.write(img+" "+l2+"\n")
f.close()
f2.close()



for line in lines:
    line = line.split(",128 ")[1].split("\n")[0]
    line = line.replace("?","")
    line = line.split(" ")
    for l in line:
        if l not in vocab:
            vocab.append(l)


f = open("data/test.txt")

lines  = f.readlines()

for line in lines:
    line = line.split(",128 ")[1].split("\n")[0]
    line = line.replace("?","")
    line = line.split(" ")
    for l in line:
        if l not in vocab:
            vocab.append(l)

f = open("data/valid.txt")

lines  = f.readlines()

for line in lines:
    line = line.split(",128 ")[1].split("\n")[0]
    line = line.replace("?","")
    line = line.split(" ")
    for l in line:
        if l not in vocab:
            vocab.append(l)

a=41