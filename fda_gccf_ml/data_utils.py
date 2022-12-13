# -- coding:UTF-8
import numpy as np 
# import pandas as pd 
import scipy.sparse as sp 

import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random
import os

class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0):
        super(BPRData, self).__init__()

        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count 
        self.set_all_item=set(range(num_item)) 
        
        self.base_path = ''
        if self.is_training==0:
            self.base_path = '../data/ml_reload/train/'
        elif self.is_training==1:
            self.base_path = '../data/ml_reload/test/'
        else:
            self.base_path = '../data/ml_reload/val/'
        self.save_dataid = self.countPath(self.base_path) 
        
        self.seed = 2022
    def ng_sample(self):

        if self.save_dataid>(self.seed-2022):
            rand_id = self.seed-2022
            save_path = self.base_path+str(rand_id)+'.npy'
            self.features_fill = np.load(save_path)
            self.seed = self.seed + 1
            return

        self.features_fill = []
        np.random.seed(self.seed)
        self.seed = self.seed + 1 


        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id] 
            #item_i: positive item ,,item_j:negative item   
            for item_i in positive_list:   
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j]) 
        
        # save_path = self.base_path+str(self.save_dataid)+'.npy'
        # print(save_path)
        # np.save(save_path,self.features_fill)
        # self.save_dataid+=1
        
    def countPath(self,base_path):
        tmp =0 
        for item in os.listdir(base_path):
            if os.path.isfile(os.path.join(base_path,item)):
                tmp+=1
        return tmp
           
    def __len__(self):  
        return self.num_ng*self.data_set_count

    def __getitem__(self, idx):
        features = self.features_fill  
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] 
        return user, item_i, item_j


class generate_adj():
    def __init__(self,training_user_set,training_item_set,user_num,item_num):
        self.training_user_set=training_user_set
        self.training_item_set=training_item_set
        self.user_num=user_num
        self.item_num=item_num 

    def readD(self, set_matrix, num_):
        user_d=[] 
        for i in range(num_):
            if i not in set_matrix:
                len_set=1.0#item set少了一些在train里面 
            else:
                len_set=1.0/(len(set_matrix[i])+1)  
            user_d.append(len_set)
        return user_d

    #user-item  to user-item matrix and item-user matrix
    def readTrainSparseMatrix(self,set_matrix,is_user,u_d,i_d):
        user_items_matrix_i=[]
        user_items_matrix_v=[]  
        if is_user:
            d_i=u_d
            d_j=i_d
            user_items_matrix_i.append([self.user_num-1,self.item_num-1])
            user_items_matrix_v.append(0)
        else:
            user_items_matrix_i.append([self.item_num-1,self.user_num-1])
            user_items_matrix_v.append(0)
            d_i=i_d
            d_j=u_d
        for i in set_matrix: 
            len_set=len(set_matrix[i])#+1
            for j in set_matrix[i]:
                user_items_matrix_i.append([i,j])
#                 d_i_j=np.sqrt(d_i[i]*d_j[j])
                d_i_j=(d_i[i]*d_j[j])
                #1/sqrt((d_i+1)(d_j+1)) 
                user_items_matrix_v.append(d_i_j)#(1./len_set) 
        user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
        user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v) 
    
    def generate_pos(self): 
        u_d=self.readD(self.training_user_set,self.user_num)
        i_d=self.readD(self.training_item_set,self.item_num)
        #1/(d_i+1)
        d_i_train=u_d
        d_j_train=i_d
        sparse_u_i=self.readTrainSparseMatrix(self.training_user_set,True,u_d,i_d)
        sparse_i_u=self.readTrainSparseMatrix(self.training_item_set,False,u_d,i_d)
        #user_item_matrix,item_user_matrix,d_i_train,d_j_train  
        return sparse_u_i,sparse_i_u,d_i_train,d_j_train
    
 


 
