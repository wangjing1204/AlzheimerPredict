import os,torch

from common import iterbrowse

#实现一个函数，将audio-embedding中，所有文件名带‘-’的文件全部删除```
ad_path = 'audio-embedding/AD2021_PAR+INV/10sec/train/adrso024.pt'
new_ad_path = 'audio-embedding/AD2021_PAR+INV/10sec_new/train/adrso024.pt'
ad24 = 'audio-embedding/AD2021_PAR+INV/10sec_new/train/adrso024-4.pt'

ad = torch.load(ad_path)
new_ad = torch.load(new_ad_path)
ad24 = torch.load(ad24)
print(ad[4,:10])
print(new_ad[4,:10])
print(ad24[:10])

if torch.allclose(ad,new_ad):
    print('true')


