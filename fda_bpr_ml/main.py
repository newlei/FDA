# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys
import random

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2022)

 
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.autograd as autograd 

from sklearn import metrics
from sklearn.metrics import f1_score
import pdb
import copy 
from collections import defaultdict
import time
import data_utils 
from shutil import copyfile    

dataset_base_path='../data/ml/' 

##movieLens-1M
user_num=6040#user_size
item_num=3952#item_size 
factor_num=64
batch_size=2048*100#10
top_k=20
num_negative_test_val=-1##all

run_id="fda_bpr_ml"
print('Model:',run_id)
dataset='ml'

path_save_model_base='./best_model/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)



train_dict,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)
test_dict,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True) 
users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]#gender

np.seterr(divide='ignore',invalid='ignore')

class FairData(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,gcn_user_embs=None,gcn_item_embs=None):
        super(FairData, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors. 
        """ 
        self.users_features = torch.cuda.LongTensor(users_features) 
        self.user_num = user_num
        self.factor_num =factor_num
        
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        self.noise_item = nn.Embedding(item_num, factor_num)    
        nn.init.normal_(self.noise_item.weight, std=0.01)  
        
#         self.zero_noise_item = torch.zeros(item_num, factor_num*2).cuda()
        
        self.min_clamp=-1
        self.max_clamp=1
   
    def fake_pos(self,male_noise_i_emb, female_noise_i_emb): 
        male_len = male_noise_i_emb.shape[0]
        female_len = female_noise_i_emb.shape[0]    
 
        avg_len = 1
        male_end_idx = male_len%avg_len+avg_len
        male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1,avg_len, self.factor_num)
        male_noise_i_mean = torch.mean(male_noise_i_reshape,axis=1)
        male_noise_len = male_noise_i_mean.shape[0]
        if male_noise_len > female_len:
            female_like = male_noise_i_mean[:female_len]
        else:
            expand_len = int(female_len/male_noise_len)+1 
            female_like = male_noise_i_mean.repeat(expand_len,1)[:female_len]
            
        
        female_end_idx = female_len%avg_len+avg_len 
        female_noise_i_emb_reshape = female_noise_i_emb[:-female_end_idx].reshape(-1,avg_len,self.factor_num)
        female_noise_i_mean = torch.mean(female_noise_i_emb_reshape,axis=1)
        female_noise_len = female_noise_i_mean.shape[0]
        if female_noise_len > male_len:
            male_like = female_noise_i_mean[:male_len]
        else:
            expand_len = int(male_len/female_noise_len)+1
            male_like = female_noise_i_mean.repeat(expand_len,1)[:male_len]
            
        return male_like,female_like
        
    def forward(self, adj_pos,u_batch,i_batch,j_batch):

        # filter gender
        user_emb = self.embed_user.weight 
        item_emb = self.embed_item.weight  
        noise_emb =  self.embed_item.weight 
        noise_emb = torch.clamp(noise_emb, min=self.min_clamp, max=self.max_clamp) 
        noise_emb += self.noise_item.weight
        
        #get gender attribute
        gender = F.embedding(u_batch,self.users_features)
        male_gender = gender.type(torch.BoolTensor)
        female_gender = (1-gender).type(torch.BoolTensor)
        
        u_emb = F.embedding(u_batch,user_emb)
        i_emb = F.embedding(i_batch,item_emb)  
        j_emb = F.embedding(j_batch,item_emb) 
        noise_i_emb2 = F.embedding(i_batch,noise_emb)
        len_noise = int(i_emb.size()[0]*0.4)
        add_emb = torch.cat((i_emb[:-len_noise],noise_i_emb2[-len_noise:]),0)

        noise_j_emb2 = F.embedding(j_batch,noise_emb)
        len_noise = int(j_emb.size()[0]*0.4)
        add_emb_j = torch.cat((noise_j_emb2[-len_noise:],j_emb[:-len_noise]),0)
        
        #according gender attribute, selecting embebdding
        male_i_batch = torch.masked_select(i_batch, male_gender)
        female_i_batch = torch.masked_select(i_batch, female_gender) 
        male_noise_i_emb = F.embedding(male_i_batch,noise_emb) 
        female_noise_i_emb = F.embedding(female_i_batch,noise_emb)   
        male_like_emb, female_like_emb = self.fake_pos(male_noise_i_emb,female_noise_i_emb)
        
        male_j_batch = torch.masked_select(j_batch, male_gender)
        female_j_batch = torch.masked_select(j_batch, female_gender) 
        male_j_emb = F.embedding(male_j_batch,item_emb) 
        female_j_emb = F.embedding(female_j_batch,item_emb)  
        
        male_u_batch = torch.masked_select(u_batch, male_gender)
        female_u_batch = torch.masked_select(u_batch, female_gender) 
        male_u_emb = F.embedding(male_u_batch,user_emb) 
        female_u_emb = F.embedding(female_u_batch,user_emb)  
        
        
        prediction_neg = (u_emb * add_emb_j).sum(dim=-1) 
        prediction_add = (u_emb * add_emb).sum(dim=-1)
        loss_add = -((prediction_add - prediction_neg).sigmoid().log().mean()) 
        l2_regulization = 0.01*(u_emb**2+add_emb**2+j_emb**2).sum(dim=-1).mean() 
        
        prediction_neg_male = (male_u_emb * male_j_emb).sum(dim=-1) 
        prediction_pos_male = (male_u_emb * male_like_emb).sum(dim=-1)
        loss_fake_male = -((prediction_pos_male - prediction_neg_male).sigmoid().log().mean()) 
        prediction_neg_female = (female_u_emb * female_j_emb).sum(dim=-1) 
        prediction_pos_female = (female_u_emb * female_like_emb).sum(dim=-1)
        loss_fake_female = -((prediction_pos_female - prediction_neg_female).sigmoid().log().mean()) 
        loss_fake = loss_fake_male + loss_fake_female
        l2_regulization2 = 0.01*(male_like_emb**2).sum(dim=-1).mean()+ 0.01*(female_like_emb**2).sum(dim=-1).mean() 
        
        
        loss_task = 1*loss_add + l2_regulization
        loss_add_item = loss_fake + l2_regulization2
        all_loss = [loss_task, l2_regulization,loss_add_item]

        return all_loss
    # Detach the return variables
    def embed(self, adj_pos): 
        u_emb = self.embed_user.weight 
        i_emb = self.embed_item.weight  
        return u_emb.detach(),i_emb.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))
      
    
############ Model #############
model = FairData(user_num, item_num, factor_num,users_features)
model=model.to('cuda') 

# task_optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

task_optimizer = torch.optim.Adam(list(model.embed_user.parameters()) + \
                            list(model.embed_item.parameters()) ,lr=0.001)
noise_optimizer = torch.optim.Adam(list(model.noise_item.parameters()),lr=0.001) 

############ dataset #############
train_dataset = data_utils.BPRData(
        train_dict=train_dict,num_item=item_num, num_ng=5 ,is_training=0, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4)

testing_dataset_loss = data_utils.BPRData(
        train_dict=test_dict,num_item=item_num, num_ng=5 ,is_training=1, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)

######################################################## TRAINING #####################################


print('--------training processing-------')
count, best_hr = 0, 0  
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
trained_end = 0 

for epoch in range(150):
    model.train()
    start_time = time.time()
    print('negative samping, need some minute')
    train_loader.dataset.ng_sample()
    print('negative samping, end')


    loss_current = [[],[],[],[]]
        
    for user_batch,  itemi_batch,itemj_batch, in train_loader: 
        user_batch = user_batch.cuda()
        itemi_batch = itemi_batch.cuda()
        itemj_batch = itemj_batch.cuda() 
        get_loss =  model(1,user_batch,itemi_batch,itemj_batch)
        task_loss,relu_loss,noise_loss = get_loss
        loss_current[0].append(task_loss.item()) 
        loss_current[1].append(relu_loss.item())
        task_optimizer.zero_grad()
        task_loss.backward()
        task_optimizer.step()
        
    for user_batch,  itemi_batch,itemj_batch, in train_loader: 
        user_batch = user_batch.cuda()
        itemi_batch = itemi_batch.cuda()
        itemj_batch = itemj_batch.cuda() 
        get_loss =  model(1,user_batch,itemi_batch,itemj_batch)
        task_loss,relu_loss,noise_loss = get_loss 
        loss_current[2].append(noise_loss.item()) 
        noise_optimizer.zero_grad()
        noise_loss.backward()
        noise_optimizer.step()              
    loss_current=np.array(loss_current)
    elapsed_time = time.time() - start_time
    train_loss_task = round(np.mean(loss_current[0]),4) 
    train_loss_sample = round(np.mean(loss_current[1]),4) 
    train_loss_noise = round(np.mean(loss_current[2]),4) 
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1)) 
    if epoch==135:
        trained_end=1

    loss_str='loss' 
    loss_str+=' task:'+str(train_loss_task) 
    str_print_train +=loss_str
    print(run_id+' '+str_print_train)
    
    
    model.eval()

    f1_u_embedding,f1_i_emb= model.embed(1)
    user_e_f1 = f1_u_embedding.cpu().numpy() 
    item_e_f1 = f1_i_emb.cpu().numpy()  
    if trained_end == 1:#epoch==135:
        PATH_model=path_save_model_base+'/best_model.pt'
        torch.save(model.state_dict(), PATH_model)
        
        PATH_model_u_f1=path_save_model_base+'/user_emb.npy'
        np.save(PATH_model_u_f1,user_e_f1) 
        PATH_model_i_f1=path_save_model_base+'/item_emb.npy'
        np.save(PATH_model_i_f1,item_e_f1)

        print("Training end")
        os.system("python ./test.py --runid=\'"+run_id+"\'")
        exit()
