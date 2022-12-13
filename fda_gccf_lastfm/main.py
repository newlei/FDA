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
setup_seed(2024)
# print('0000') 
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
# import error_parity

dataset_base_path='../data/lastfm/' 
user_reid, item_reid= np.load(dataset_base_path+'/u_i_reid.npy',allow_pickle=True)


##lastfm
user_num=len(user_reid)#130230  
item_num=len(item_reid)#138676
factor_num=64
batch_size=2048*100
top_k=20
num_negative_test_val=-1##all
 
run_id="fda_gccf_lastfm"
print('Model:',run_id)
dataset='lastfm'

path_save_model_base='./best_model/'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)


train_dict,train_dict_count = np.load(dataset_base_path+'/train.npy',allow_pickle=True)  
test_dict,test_dict_count = np.load(dataset_base_path+'/test.npy',allow_pickle=True) 
val_dict,val_dict_count = np.load(dataset_base_path+'/val.npy',allow_pickle=True)   
users_features=np.load(dataset_base_path+'/users_features.npy')
users_features = users_features[:,0]#gender

for i in train_dict:
    len_one = len(train_dict[i])
    if len_one<7:
        print(len_one,i)
# pdb.set_trace()
# print(train_dict_count,test_dict_count,val_dict_count)


class GCN(nn.Module):
    def __init__(self, user_num, item_num, factor_num,users_features,adj_matrix):
        super(GCN, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors. 
        """ 
        self.users_features = torch.cuda.LongTensor(users_features) 
        self.user_num = user_num
        self.factor_num = factor_num 
        
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        user_item_matrix,item_user_matrix,d_i_train,d_j_train = adj_matrix
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        
        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)
        
        self.noise_item = nn.Embedding(item_num, factor_num)    
        nn.init.normal_(self.noise_item.weight, std=0.01)
        
        self.min_clamp=-1
        self.max_clamp=1
        
    def gcn_layer(self,users_embedding,items_embedding):

        
        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding

        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding
          
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding)
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding)
        
        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding),-1)#+gcn4_items_embedding#
        return gcn_users_embedding,gcn_items_embedding

       
    def fake_pos(self,male_noise_i_emb, female_noise_i_emb): 
        male_len = male_noise_i_emb.shape[0]
        female_len = female_noise_i_emb.shape[0]    
# #         #打乱一下，方面后面的求和计算，这样可以保证更具有随机性。
#         male_indexes = torch.randperm(male_i_emb.shape[0])
#         male_i_emb = male_i_emb[male_indexes]
#         male_noise_i_emb = male_noise_i_emb[male_indexes]
        
#         female_indexes = torch.randperm(female_i_emb.shape[0])
#         female_i_emb = female_i_emb[female_indexes] 
#         female_noise_i_emb = female_noise_i_emb[female_indexes] 

        avg_len = 1
        male_end_idx = male_len%avg_len+avg_len#最后的一点去掉，方面每100个求和。不然最后有零头，没法统一处理 
        male_noise_i_reshape = male_noise_i_emb[:-male_end_idx].reshape(-1,avg_len, self.factor_num*3)
        male_noise_i_mean = torch.mean(male_noise_i_reshape,axis=1)
        male_noise_len = male_noise_i_mean.shape[0]
        if male_noise_len > female_len:
            female_like = male_noise_i_mean[:female_len]
        else:
            expand_len = int(female_len/male_noise_len)+1 
            female_like = male_noise_i_mean.repeat(expand_len,1)[:female_len]
            
        
        female_end_idx = female_len%avg_len+avg_len 
        female_noise_i_emb_reshape = female_noise_i_emb[:-female_end_idx].reshape(-1,avg_len,self.factor_num*3)
        female_noise_i_mean = torch.mean(female_noise_i_emb_reshape,axis=1)
        female_noise_len = female_noise_i_mean.shape[0]
        if female_noise_len > male_len:
            male_like = female_noise_i_mean[:male_len]
        else:
            expand_len = int(male_len/female_noise_len)+1
            male_like = female_noise_i_mean.repeat(expand_len,1)[:male_len]
            
        return male_like,female_like
    
    def forward(self, u_batch,i_batch,j_batch):    
        users_embedding = self.embed_user.weight 
        items_embedding = self.embed_item.weight  
        user_emb,item_emb = self.gcn_layer(users_embedding,items_embedding)
        
#         user_emb,item_emb = self.gcn_layer()
        noise_emb_based = self.noise_item.weight 
#         noise_emb = torch.clamp(noise_emb, min=self.min_clamp, max=self.max_clamp) 
        noise_emb_based =noise_emb_based+items_embedding # noise_emb+item_emb 
        _,noise_emb = self.gcn_layer(users_embedding,noise_emb_based)
        
        #get gender attribute
        gender = F.embedding(u_batch,self.users_features)
        male_gender = gender.type(torch.BoolTensor)
        female_gender = (1-gender).type(torch.BoolTensor)
        
        
        u_emb = F.embedding(u_batch,user_emb)
        i_emb = F.embedding(i_batch,item_emb)  
        j_emb = F.embedding(j_batch,item_emb)
        #这里得到虚假的样本，并且要保证推荐性能
        noise_i_emb2 = F.embedding(i_batch,noise_emb)
        len_noise = int(i_emb.size()[0]*0.4)
        add_emb = torch.cat((i_emb[:-len_noise],noise_i_emb2[-len_noise:]),0)
        
        noise_j_emb2 = F.embedding(j_batch,noise_emb)
        len_noise = int(j_emb.size()[0]*0.4)
        add_emb_j = torch.cat((noise_j_emb2[-len_noise:],j_emb[:-len_noise]),0)
#         add_emb_j = j_emb
        
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
    def embed(self): 
        users_embedding = self.embed_user.weight 
        items_embedding = self.embed_item.weight  
        gcn_users_embedding,gcn_items_embedding = self.gcn_layer(users_embedding,items_embedding)
        return gcn_users_embedding.detach(),gcn_items_embedding.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


train_item_dict=dict()
for u_id in train_dict:
    one_data = train_dict[u_id]
    for v_id in one_data:
        if v_id not in train_item_dict:
            train_item_dict[v_id]=[]
        train_item_dict[v_id].append(u_id)

# count=0
# for i in range(item_num): 
#     if i not in train_item_dict:
#         print('item',i,count)
#         count+=1
    
g_adj= data_utils.generate_adj(train_dict,train_item_dict,user_num,item_num)
pos_adj=g_adj.generate_pos()

############ Model #############
model = GCN(user_num, item_num, factor_num,users_features,pos_adj)
model=model.to('cuda') 

task_optimizer = torch.optim.Adam(list(model.embed_user.parameters()) + \
                            list(model.embed_item.parameters()) ,lr=0.001)
noise_optimizer = torch.optim.Adam(list(model.noise_item.parameters()),lr=0.001) 

# task_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

############ dataset #############
# train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):

train_dataset = data_utils.BPRData(
        train_dict=train_dict,num_item=item_num, num_ng=5 ,is_training=0, data_set_count=train_dict_count)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)

testing_dataset_loss = data_utils.BPRData(
        train_dict=test_dict,num_item=item_num, num_ng=5 ,is_training=1, data_set_count=test_dict_count)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=test_dict_count, shuffle=False, num_workers=0)

val_dataset_loss = data_utils.BPRData(
        train_dict=val_dict,num_item=item_num, num_ng=5 ,is_training=2, data_set_count=val_dict_count)
val_loader_loss = DataLoader(val_dataset_loss,
        batch_size=val_dict_count, shuffle=False, num_workers=0)

######################################################## TRAINING #####################################


print('--------training processing-------')
count, best_hr = 0, 0  
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

for epoch in range(150):
    model.train()  
    start_time = time.time()  
    print('negative samping, about 3 minute')
    train_loader.dataset.ng_sample()
    
    loss_current = [[],[],[],[]] 
        
    for user_batch,  itemi_batch,itemj_batch, in train_loader: 
        user_batch = user_batch.cuda()
        itemi_batch = itemi_batch.cuda()
        itemj_batch = itemj_batch.cuda() 
        get_loss =  model(user_batch,itemi_batch,itemj_batch)
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
        get_loss =  model(user_batch,itemi_batch,itemj_batch)
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
     

    loss_str='loss' 
    loss_str+=' task:'+str(train_loss_task)
    str_print_train +=loss_str
    print(run_id+' '+str_print_train)
    
    
    model.eval()

    f1_u_embedding,f1_i_emb= model.embed()
    user_e_f1 = f1_u_embedding.cpu().numpy() 
    item_e_f1 = f1_i_emb.cpu().numpy()
    
    if epoch==48:
        PATH_model=path_save_model_base+'/best_model.pt'
        torch.save(model.state_dict(), PATH_model)
        
        PATH_model_u_f1=path_save_model_base+'/user_emb.npy'
        np.save(PATH_model_u_f1,user_e_f1) 
        PATH_model_i_f1=path_save_model_base+'/item_emb.npy'
        np.save(PATH_model_i_f1,item_e_f1)

        print("Training end")
        os.system("python ./test.py --runid=\'"+run_id+"\'")
        exit()

#     user_e = user_e_f1
#     item_e = item_e_f1