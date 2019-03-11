import pandas as pd
import numpy as np

import pickle
import json

from numpy.random import random

#
users_index=pickle.load(open('users_index.pkl','rb'))
items_index=pickle.load(open('items_index.pkl','rb'))

n_users=len(users_index)
n_items=len(items_index)

#倒排表
user_items=pickle.load(open('user_items.pkl','rb'))
item_users=pickle.load(open('item_users.pkl','rb'))

#训练数据
triplet_cols=['user_id','item_id','rating','timestamp']

df_triplet=pd.read_csv('u1.base',sep='\t',names=triplet_cols,encoding='latin-1')
df_triplet=df_triplet.drop(['timestamp'],axis=1)

#隐含变量的维数
K=40

#item和用户的偏置项
bi=np.zeros((n_items,1))
bu=np.zeros((n_users,1))

#item和用户的隐含向量
qi=np.zeros((n_items,K))
pu=np.zeros((n_users,K))

for uid in range(n_users):
    pu[uid]=np.reshape(random((K,1))/10.0*(np.sqrt(K)),K)
for iid in range(n_items):
    qi[iid]=np.reshape(range((K,1))/10.0*(np.sqrt(K)),K)

#所有用户的平均打分
mu=df_triplet['rating'].mean()

def svd_pred(uid,iid):
    score=mu+bi[iid]+bu[uid]+np.sum(qi[iid]*pu[uid])
    return score

#gamma:学习率
#Lambda:正则参数
#steps：迭代次数

steps=50
gamma=0.04
Lambda=0.15

#总的打分次数
n_records=df_triplet.shape[0]

for step in range(steps):
    print('The '+str(step)+'-th step is running')
    rmse_sum=0.0

    #将训练样本打散顺序
    kk=np.random.permutation(n_records)
    for j in range(n_records):
        #每次训练一个样本
        line=kk[j]

        uid=users_index[df_triplet.iloc[line]['user_id']]
        iid=items_index[df_triplet.iloc[line]['item_id']]

        rating=df_triplet.iloc[line]['rating']

        #预测残差
        eui=rating-svd_pred(uid,iid)

        #残差平方和
        rmse_sum+=eui**2

        #随机梯度下降，更新
        bu[uid]+=gamma*(eui-Lambda*bu[uid])
        bi[iid]+=gamma*(eui-Lambda*bi[iid])

        temp=qi[iid]
        qi[iid]+=gamma*(eui*pu[uid]-Lambda*qi[iid])
        pu[uid]+=gamma*(eui*temp-Lambda*pu[uid])

    #学习率递减
    gamma=gamma*0.93
    print('the rmse of this step on train data is ',np.sqrt(rmse_sum/n_records))

#保存到json
def save_json(filepath):
    dict_={}
    dict_['mu']=mu
    dict_['K']=K

    dict_['bi']=bi.tolist()
    dict_['bu']=bu.tolist()

    dict_['qi']=qi.tolist()
    dict_['pu']=pu.tolist()

    json_txt=json.dumps(dict_)
    with open(filepath,'w') as file:
        file.write(json_txt)

#加载json文件
def load_json(filepath):
    with open(filepath,'r') as file:
        dict_=json.load(file)

        mu=dict_['mu']
        K=dict_['K']

        bi=np.asarray(dict_['bi'])
        bu=np.asarray(dict_['bu'])

        qi=np.asarray(dict_['qi'])
        pu=np.asarray(dict_['pu'])

save_json('svd_model.json')
load_json('svd_model.json')

#返回items及其打分
def svd_CV_Recommend(user):
    cur_user_id=users_index[user]

    #训练集中该用户打过分的item
    cur_user_items=user_items[cur_user_id]

    #该用户对所有item的打分
    user_items_scores=np.zeros(n_items)

    #预测打分
    for i in range(n_items):
        if i not in cur_user_items:#训练集中没打过分
            user_items_scores[i]=svd_pred(cur_user_id,i)

    #推荐
    sort_index=sorted(((e,i) for i,e in enumerate(list(user_items_scores))),reverse=True)

    #创建一个dataFrame
    columns=['item_id','score']
    df=pd.DataFrame(columns==columns)

    for i in range(0,len(sort_index)):
        cur_item_index=sort_index[i][1]
        cur_item=list(items_index.values())[list(items_index.values()).index(cur_item_index)]

        if ~np.isnan(sort_index[i][0]) and cur_item_index not in cur_user_items:
            df.loc[len(df)]=[cur_item,sort_index[i][0]]
    return df

#读取测试数据
triplet_cols=['user_id','item_id','rating','timestamp']

df_triplet_test=pd.read_csv('u1.test',sep='\t',names=triplet_cols,encoding='latin-1')

#统计总的用户
unique_users_test=df_triplet_test['user_id'].unique()

#为每个用户推荐的item数目
n_rec_items=10

#性能评价参数初始化，用户计算准确率和召回率
n_hits=0
n_total_rec_items=0
n_test_items=0

#所有被推荐商品的集合，用于计算覆盖度
all_rec_items=set()

#残差平方和
rss_test=0.0

#对每个测试用户
for user in unique_users_test:
    #测试集中用户打过分的电影
    if user not in users_index:#新用户不能用于推荐----不支持热启动
        print(str(user)+'is s new user.\n')
        continue
    user_records_test=df_triplet_test[df_triplet_test.user_id==user]

    #对每个测试用户，计算该用户对训练集中未出现过的商品的打分，并基于该打分进行用户推荐
    #返回结果为DataFrame
    rec_items=svd_CV_Recommend(user)
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
            print(str(item)+'is a new item or user'+str(user)+'already rated it.\n')
            continue
        pred_score=df1['score'].values[0]
        rss_test+=(pred_score-score)**2

    #推荐的item总数
    n_total_rec_items+=n_rec_items

    #真实item总数
    n_test_items+=user_records_test.shape[0]

precision=n_hits/(1.0*n_total_rec_items)
recall=n_hits/(1.0*n_rec_items)

coverage=len(all_rec_items)/(1.0*n_items)

#rmse
rmse=np.sqrt(rss_test/df_triplet_test.shape[0])

print('precision: ',precision)
print('recall: '+recall)
print('coverage: ',coverage)

