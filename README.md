# OCR-TR
## Description
A simple pytorch implementation for an OCR model. The model is using a Transformer encoder-decoder architecture in a seq2seq fashion.

<img src="./imgs/OCR-TR.png"  alt="1">

## Download Code
clone the repository
```bash
git clone https://github.com/dali92002/OCR-TR
cd OCR-TR
```
## Setup your environment
Create the following environment named vit with Anaconda. Then, Activate it.
```bash
conda env create -f environment.yml
conda activate vit
```

## Prepare Data
For this task we will create a synthetic data that simulate the handwritten text. I choosed to create the dataset from the EMNIST dataset (digits+characters). I created 80000 images for training, 10000 for validation and 10000 for testing. The images are composed of randomly concatenated characters with a size between 3 and 10.

The code of preparing the dataset can be found in the file prepare_data.py , to execute it, use the following command:

```bash
python prepare_data.py --train_words 100000 --valid_words 10000 --test_words 10000 
```

This will generate random words in the folder ./data/words/ and the transcription of each word in the files ./data/train.txt, ./data/valid.txt and ./data/test.txt  

In each txt file, there will be in each line the image name and its transcription, separated by a space. 

## Pretraining

Specify the data path and the folder to save your model --weights_path

```bash
python pretrain.py --data_path /data/ --weights_path /data/users/msouibgui/weights/vatican/  --img_height 64 --img_width 256 --train_type htr_Augm --batch_size 48 --vit_patch_size 8
```

## Train the model
After creating the data, we can train the model using this command

```bash
python train.py --data_path ./data/ --img_height 64 --img_width 256 --train_type htr_Augm --batch_size 64 --vit_patch_size 8 
```

Here I specified to use data augmentation for a better training, also I set the image sizes and the vit patch size to be 8x8. You can however use your custom configurations, check Config.py.

During training there will be a validation in each epoch, the best weights will be saved in a folder named ./weights/ and the predictions will be saved in a folder named ./pred_logs/


## Test the model

To test the model, run the following command. It will recognize the testing data using the trained model, you should specify which model you want to use by profiding its path (here I am using ./weights/best-seq2seq_htr_Augm_32_256_8.pt that will be created if you launched the training already):



```bash
python test.py --data_path ./data/ --img_height 64 --img_width 256 --train_type htr_Augm --batch_size 64 --vit_patch_size 8 --test_model ./weights/best-seq2seq_htr_Augm_32_256_8.pt
```

I trained a model already, you can dowload the weights from [here](https://drive.google.com/file/d/1wnPAZJXmYm5jLsefT2C3yFRvvGeNpWL5/view?usp=sharing) and use it directly to test.

After rnning the testing you will get the predictions in the folder ./pred_logs/ as well as the CER and WER.

## Authors
- [Mohamed Ali Souibgui](https://github.com/dali92002)
