import numpy as np
import pandas as pd

import pickle
import scipy.io as sio
import os

#距离
import scipy.spatial.distance as ssd


#用户和item的索引
users_index=pickle.load(open('users_index.pkl','rb'))
items_index=pickle.load(open('items_index.pkl','rb'))
#print(items_index.values())
n_users=len(users_index)
n_items=len(items_index)#item的id

#加载倒排表
#每个用户打分过的电影
user_items=pickle.load(open('user_items.pkl','rb'))
#print(user_items.values())
#对该电影打分的用户
item_users=pickle.load(open('item_users.pkl','rb'))

#用户-物品关系矩阵R
user_item_scores=sio.mmread('user_item_scores.mtx')
user_item_scores=user_item_scores.tocsr()
#print(user_item_scores)

#计算每个用户的平均打分
users_mu=np.zeros(n_users)
for u in range(n_users):
    n_user_items=0
    r_acc=0

    for i in user_items[u]:
        r_acc+=user_item_scores[u,i]
        n_user_items+=1

    #某一用户u对打过分的电影的平均打分
    users_mu[u]=r_acc/n_user_items

#返回相似性结果
def user_similarity(uid1,uid2):
    si={}   #有效item（两个用户均有打分的item）的集合
    for item in user_items[uid1]:#uid1对该item打过分
        if item in user_items[uid2]:#uid2也对该item打过分
            si[item]=1#item为一个有效item

    n=len(si)   #有效item数，uid1对item打过分，uid2也对item打过分
    if n==0:
        similarity=0.0
        return similarity

    #用户uid1的有效打分（对某一item的打分减去平均打分）
    s1=np.array(user_item_scores[uid1,item]-users_mu[uid1] for item in si)

    #用户uid2的有效打分
    s2=np.array(user_item_scores[uid2,item]-users_mu[uid2] for item in si)

    #相似性
    similarity=1-ssd.cosine(s1,s2)

    if np.isnan(similarity):
        similarity=0.0
    return similarity


#生成一个两用户之间的相似性矩阵
users_similarity_matrix=np.matrix(np.zeros(shape=(n_users,n_users)),float)

for ui in range(n_users):
    users_similarity_matrix[ui,ui]=1.0

    #打印进度条
    if(ui%100==0):
        print('ui=%d'%ui)

    #矩阵对称轴之外的数值
    for uj in range(ui+1,n_users):
        users_similarity_matrix[ui,uj]=user_similarity(ui,uj)
        users_similarity_matrix[uj,ui]=users_similarity_matrix[ui,uj]

#将用户之间的相似性矩阵保存起来
pickle.dump(users_similarity_matrix,open('users_similarity.pkl','wb'))

#返回相似性矩阵
def users_similarity(n_users):
    users_similarity_matrix=np.array(np.zeros(shape=(n_users,n_users)),float)
    for ui in range(n_users):
        users_similarity_matrix[ui,ui]=1.0

        #打印进度条
        if(ui%100==0):
            print('users_similarity(n_users),ui=%d'%ui)
        for uj in range(ui+1,n_users):
            users_similarity_matrix[ui,uj]=user_similarity(ui,uj)
            users_similarity_matrix[uj,ui]=users_similarity_matrix[ui,uj]
    pickle.dump(users_similarity_matrix,open('users_similarity.pkl','wb'))
    return users_similarity_matrix


#预测用户对item的打分
def User_CF_pred(uid,iid):
    sim_accumulate=0.0
    rat_acc=0.0
    #对iid对应的item打过分的用户
    for user_id in item_users[iid]:
        #计算当前用户与给该item打过分的用户之间的相关性
        sim=users_similarity_matrix[user_id,uid]

        if sim !=0:
            rat_acc+=sim*(user_item_scores[user_id,iid]-users_mu[user_id])
            sim_accumulate+=np.abs(sim)

    if sim_accumulate!=0:
        score=users_mu[uid]+rat_acc/sim_accumulate
    else:
        score=users_mu[uid]
    return score

#根据用户id返回推荐的item，以及对应的打分
def recommend(user):
    cur_user_id=users_index[user]

    ##取出训练集中用户打过分的item
    cur_user_items=user_items[cur_user_id]

    user_items_scores=np.zeros(n_items)

    #预测打分
    for i in range(n_items):
        if i not in cur_user_items:
            user_item_scores[i]=User_CF_pred(cur_user_id,i)

    #推荐
    sort_index=sorted(((e,i) for i,e in enumerate(list(user_item_scores))),reverse=False)

    columns=['item_id','score']
    df=pd.DataFrame(columns=columns)


    for i in range(0,len(sort_index)):
        cur_item_index=sort_index[i][1]


