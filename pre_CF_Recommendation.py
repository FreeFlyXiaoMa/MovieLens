import pandas as pd
import numpy as np

#字典，用于建立用户和物品的索引
from collections import defaultdict

#稀疏矩阵，存储打分表
import scipy.io as sio
import scipy.sparse as ss

#数据存储
import pickle

#读取训练数据
triplet_cols=['user_id','item_id','rating','timestamp']

df_triplet=pd.read_csv('u1.base',sep='\t',names=triplet_cols,encoding=
                       'latin-1')
print(df_triplet.head())

#统计总的用户数目和物品数目
unique_users = df_triplet['user_id'].unique()
unique_items = df_triplet['item_id'].unique()

n_users = unique_users.shape[0]
n_items = unique_items.shape[0]

print(n_users)
print(n_items)

# 建立用户和物品的索引表
# 本数据集中user_id和item_id都已经是索引了,可以减1，将从1开始编码变成从0开始的编码
# 下面的代码更通用，可对任意编码的用户和物品重新索引
users_index = dict()
items_index = dict()

for j, u in enumerate(unique_users):
    users_index[u] = j

# 重新编码活动索引字典
for j, i in enumerate(unique_items):
    items_index[i] = j

# 保存用户索引表
pickle.dump(users_index, open("users_index.pkl", 'wb'))
# 保存活动索引表
pickle.dump(items_index, open("items_index.pkl", 'wb'))

# 倒排表
# 统计每个用户打过分的电影   / 每个电影被哪些用户打过分
user_items = defaultdict(set)
item_users = defaultdict(set)

# 用户-物品关系矩阵R, 稀疏矩阵，记录用户对每个电影的打分
user_item_scores = ss.dok_matrix((n_users, n_items))

# 扫描训练数据
for line in df_triplet.index:  # 对每条记录
    cur_user_index = users_index[df_triplet.iloc[line]['user_id']]
    cur_item_index = items_index[df_triplet.iloc[line]['item_id']]

    # 倒排表
    user_items[cur_user_index].add(cur_item_index)  # 该用户对这个电影进行了打分
    item_users[cur_item_index].add(cur_user_index)  # 该电影被该用户打分

    user_item_scores[cur_user_index, cur_item_index] = df_triplet.iloc[line]['rating']

##保存倒排表
# 每个用户打分的电影
pickle.dump(user_items, open("user_items.pkl", 'wb'))
##对每个电影打过分的用户
pickle.dump(item_users, open("item_users.pkl", 'wb'))

# 保存打分矩阵，在UserCF和ItemCF中用到
sio.mmwrite("user_item_scores", user_item_scores)