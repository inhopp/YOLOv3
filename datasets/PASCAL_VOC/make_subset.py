import copy
import os
import shutil
import pandas as pd

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

train.columns = ['img_path', 'anno_path']
val.columns = ['img_path', 'anno_path']
test.columns = ['img_path', 'anno_path']

os.makedirs(os.path.join('./train'), exist_ok=True)
os.makedirs(os.path.join('./val'), exist_ok=True)
os.makedirs(os.path.join('./test'), exist_ok=True)

for i in range(len(train)):
    origin_img_path = './images/' + train.iloc[i]['img_path']
    copy_img_path = './train/' + train.iloc[i]['img_path']

    origin_anno_path = './labels/' + train.iloc[i]['anno_path']
    copy_anno_path = './train/' + train.iloc[i]['anno_path']

    shutil.copy(origin_img_path, copy_img_path)
    shutil.copy(origin_anno_path, copy_anno_path)


for i in range(len(val)):
    origin_img_path = './images/' + val.iloc[i]['img_path']
    copy_img_path = './val/' + val.iloc[i]['img_path']

    origin_anno_path = './labels/' + val.iloc[i]['anno_path']
    copy_anno_path = './val/' + val.iloc[i]['anno_path']

    shutil.copy(origin_img_path, copy_img_path)
    shutil.copy(origin_anno_path, copy_anno_path)


for i in range(len(test)):
    origin_img_path = './images/' + test.iloc[i]['img_path']
    copy_img_path = './test/' + test.iloc[i]['img_path']

    origin_anno_path = './labels/' + test.iloc[i]['anno_path']
    copy_anno_path = './test/' + test.iloc[i]['anno_path']

    shutil.copy(origin_img_path, copy_img_path)
    shutil.copy(origin_anno_path, copy_anno_path)


print("########### Folder Split Ended ###########")