import pandas as pd
import numpy as np

import pickle
import scipy.io as sio
import os

import scipy.spatial.distance as ssd

#用户和item的索引
users_index=pickle.load(open('users_index.pkl','rb'))
items_index=pickle.load(open('items_index.pkl','rb'))

n_users=len(users_index)
n_items=len(items_index)

#倒排表
##每个用户打过分的电影
user_items=pickle.load(open('user_items.pkl','rb'))
#对该电影打过分的用户
item_users=pickle.load(open('item_users.pkl','rb'))


#用户-物品关系矩阵
user_item_scores=sio.mmread('user_item_scores')
user_item_scores=user_item_scores.tocsr()

users_mu=np.zeros(n_users)
for u in range(n_users):
    n_user_items=0
    r_acc=0.0

    for i in user_items[u]:
        r_acc+=user_item_scores[u,i]
        n_user_items+=1

    users_mu[u]=r_acc/n_user_items

def item_similarity(iid1,iid2):
    su={} #有效item的集合
    for user in item_users[iid1]:
        if user in item_users[iid2]:
            su[user]=1#该用户为一有效user

    n=len(su) #有效item数，
    if(n==0):
        similarity=0
        return similarity

    #iid1的有效打分（减去用户的平均打分）
    s1=np.array([user_item_scores[user,iid1]-users_mu[user] for user in su])

    #iid2的有效打分
    s2=np.array([user_item_scores[user,iid2]-users_mu[user] for user in su])

    similarity=1-ssd.cosine(s1,s2)

    if(np.isnan(similarity)):
        similarity=0.0
    return similarity


items_similarity_matrix=np.matrix(np.zeros(shape=(n_items,n_items)),float)

for i in range(n_items):
    items_similarity_matrix[i,i]=1.0

    #进度
    if(i%100==0):
        print('i=%d'%i)
    #相似矩阵的右上角和左下角
    for j in range(i+1,n_items):
        items_similarity_matrix[j,i]=item_similarity(i,j)
        items_similarity_matrix[i,j]=items_similarity_matrix[j,i]

pickle.dump(items_similarity_matrix,open('items_similarity.pkl','wb'))

def items_similarity(n_items):
    items_similarity_matrix=np.matrix(np.zeros(shape=(n_items,n_items)))

    for i in range(n_items):
        items_similarity_matrix[i,i]=1.0

        #进度
        if(i%100==0):
            print('i=%d'%i)

        for j in range(i+1,n_items):
            items_similarity_matrix[j,i]=item_similarity(i,j)
            items_similarity_matrix[i,j]=items_similarity_matrix[j,i]

    pickle.dump(open('items_similarity.pkl','wb'))
    return items_similarity_matrix

#预测用户对item的打分
def Item_CF_pred1(uid,iid):
    sim_accumulate=0.0
    rat_acc=0.0

    for item_id in user_items[uid]:
        sim=items_similarity_matrix[item_id,iid]

        #由于相似性可能为负，而用户打过分的item又不多预测为负
        if sim !=0:
            rat_acc+=sim*(user_item_scores[uid,item_id])
            sim_accumulate+=np.abs(sim)

    if sim_accumulate!=0:
        score=rat_acc/sim_accumulate
    else:
        score=users_mu[uid]

    if score<0:
        score=0.0

    return score

#预测用户对item的打分，取所有item中n_Knns最相似的物品
def Item_CF_pred2(uid,iid,n_Knns):
    sim_accumulate=0.0
    rat_acc=0.0
    n_nn_items=0

    #显示度排序
    cur_items_similarity=np.array(items_similarity_matrix[iid,:])
    cur_items_similarity=cur_items_similarity.flatten()
    sort_index=sorted(((e,i) for i,e in enumerate(list(cur_items_similarity))),reverse=True)

    for i in range(0,len(sort_index)):
        cur_item_index=sort_index[i][1]

        if n_nn_items>=n_Knns: #相似的items已经足够多（>n_Knns)
            break
        if cur_item_index in user_items[uid]:
            #计算当前用户打过分的item与其他item之间的相似度
            sim=items_similarity_matrix[iid,cur_item_index]

            if sim!=0:
                rat_acc+=sim*(user_item_scores[uid,cur_item_index])
                sim_accumulate+=np.abs(sim)

            n_nn_items+=1

    if sim_accumulate!=0:
        score=rat_acc/sim_accumulate
    else:
        score=users_mu[uid]
    if score<0:
        score=0.0
    return score

def Item_CF_pred3(uid,iid,n_Knns):
    sim_accumulate=0.0
    rat_acc=0.0

    #相似度排序
    cur_items_similarity=np.array(items_similarity_matrix[iid,:])
    cur_items_similarity=cur_items_similarity.flatten()
    sort_index=sorted(((e,i) for i,e in enumerate(list(cur_items_similarity))),reverse=True)[0:n_Knns]

    for i in range(0,len(sort_index)):
        cur_item_index=sort_index[i][1]

        if cur_item_index in user_items[uid]:#用户打过分的item
            sim=items_similarity_matrix[iid,cur_item_index]

            if sim !=0:
                rat_acc+=sim*(user_item_scores[uid,cur_item_index])
                sim_accumulate+=np.abs(sim)
    if sim_accumulate!=0:
        score=rat_acc/sim_accumulate
    else:
        score=users_mu[uid]

    if score<0:
        score=0.0
    return score

#返回推荐items及其打分
def recommand(user):
    cur_user_id=users_index[user]

    #训练集中该用户打过分的item
    cur_user_items=user_items[cur_user_id]

    #对所有item的打分
    user_items_scores=np.zeros(n_items)

    #预测打分
    for i in range(n_items):
        if i not in cur_user_items:
            user_items_scores[i]=Item_CF_pred2(cur_user_id,i,10)

    #推荐
    sort_index=sorted(((e,i) for i,e in enumerate(list(user_items_scores))),reverse=True)

    columns=['item_id','score']
    df=pd.DataFrame(columns=columns)

    for i in range(0,len(sort_index)):
        cur_item_index=sort_index[i][1]
        cur_item=list(items_index.keys())[list(items_index.values()).index(cur_item_index)]

        if ~np.isnan(sort_index[i][0]) and cur_item_index not in cur_user_items:
            df.loc[len(df)]=[cur_item,sort_index[i][0]]

    return df

#读取测试数据
triplet_cols=['user_id','item_id','rating','timestamp']
df_triplet_test=pd.read_csv('u1.test',sep='\t',names=triplet_cols,encoding='latin-1')

#统计总的用户
unique_users_test=df_triplet_test['user_id']

n_rec_items=10

#性能评价参数初始化，用户计算precision 和recall
n_hits=0
n_total_rec_items=0
n_test_items=0

#所有被推荐商品的集合，用于计算覆盖度
all_rec_items=set()

#残差平方和，用于计算rmse
rss_test=0.0

#对每个测试用户
for user in unique_users_test:
    #测试集中该用户打分过的电影
    if user not in users_index:
        print(str(user)+'is a new user.\n')
        continue

    user_records_test=df_triplet_test[df_triplet_test.user_id==user]

    #对每个测试用户，计算该用户对训练集中未出现过的商品的打分，并基于该打分进行推荐
    rec_items=recommand(user)

    for i in range(n_rec_items):
        item=rec_items.iloc[i]['item_id']

        if item in user_records_test['item_id'].values:
            n_hits+=1
        all_rec_items.add(item)

    #计算rmse
    for i in range(user_records_test.shape[0]):
        item=user_records_test.iloc[i]['item_id']
        score=user_records_test.iloc[i]['rating']

        df1=rec_items[rec_items.item_id==item]
        if(df1.shape[0]==0):
            print(str(item)+'is a new item.\n')
            continue
        pred_score=df1['score'].values[0]
        rss_test+=(pred_score-score)**2 #残差平方和

    #推荐的item总数
    n_total_rec_items+=n_rec_items

    #真实的item总数
    n_test_items+=user_records_test.shape[0]#按行取出数据


#计算准确率和召回率
precision=n_hits/(1.0*n_total_rec_items)
recall=n_hits/(1.0*n_test_items)

#覆盖率
coverage=len(all_rec_items)/(1.0*n_items)

#均方误差
rmse=np.sqrt(rss_test/df_triplet_test.shape[0])

