import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

triplet_cols=['user_id','item_id','rating','timestamp']
df_triplet=pd.read_csv('u.data',sep='\t',names=triplet_cols,
                       encoding='latin-1')
#print(df_triplet.head())
#print(df_triplet.info())

#timestamp格式为unicode,将其转为float格式
df_triplet['timestamp']=df_triplet['timestamp'].astype('float64')
#时间格式转换
df_triplet['timestamp']=df_triplet['timestamp'].map(datetime.
                                                    datetime.
                                                    fromtimestamp)
#print(df_triplet.head())
#统计用户数量
n_users=df_triplet['user_id'].unique().shape[0]
#统计电影数量
n_items=df_triplet['item_id'].unique().shape[0]
#print(n_items)

#统计每个用户评分次数
user_freq=df_triplet['user_id'].value_counts()
#print(user_freq.head(-1))
#plt.bar(user_freq.index,user_freq)
#plt.show()
#统计每部电影的评分人数，可以看出电影的流行程度，默认是降序排列
items_rating_times=df_triplet['item_id'].value_counts()
#print(items_rating_times.head(-1))
df_items_sorted_by_rating_times=pd.DataFrame({'item_id':items_rating_times.index,
                                'rating_times':items_rating_times})
#print(df_items_sorted_by_rating_times.head())

#plt.bar(items_rating_times.index,items_rating_times)
#plt.show()

#获取item信息
item_cols=['item_id','title','release_date','video_release_date',
           'imdb_url','unknown','Action','Adventure','Animation',
           'Children\'','Comedy','Crime','Documentary','Drama',
           'Fantasy','Film-Noir','Horror','Musical','Mystery',
           'Romance','Sci-Fi','Thriller','War','Western']
df_items=pd.read_csv('u.item',sep='|',names=item_cols,encoding='latin-1')
#print(df_items.head())

#根据频次大小依次获取电影信息
df_items_sorted_by_rating_times_merge=pd.merge(df_items_sorted_by_rating_times,
                                               df_items,how='left',
                                               left_on='item_id',
                                               right_on='item_id')
#加上排名，名次越小排在越前
df_items_sorted_by_rating_times_merge['ranking_rating_times']=range(
    items_rating_times.count()
)
#print(df_items_sorted_by_rating_times_merge.head())
#找出前20大流行电影
popular_items_count_top20=df_items_sorted_by_rating_times_merge.iloc[0:20][
    'rating_times'
]
popular_items_count_top20_titles=df_items_sorted_by_rating_times_merge.iloc[0:20][
    'title'
]
objects=(list(popular_items_count_top20_titles))
y_pos=np.arange(len(objects))
performance=list(popular_items_count_top20)

'''plt.rcdefaults()
plt.bar(y_pos,performance,align='center',alpha=0.5)
plt.xticks(y_pos,objects,rotation='vertical')
plt.ylabel('Rating Count')
plt.tight_layout()
plt.title('Most popular Movies')
plt.show()'''

items_mean_rating=df_triplet['rating'].groupby(df_triplet['item_id']).mean()
#items_mean_rating=items_mean_rating.sort_value(ascending=False)

df_items_sorted_by_mean_rating=pd.DataFrame({'item_id':items_mean_rating.index,
                                             'mean_rating':items_mean_rating})
print(df_items_sorted_by_mean_rating.head())

#根据频次大小依次取电影信息
df_items_sorted_by_mean_rating_merge=pd.merge(df_items_sorted_by_mean_rating,
                                              df_items_sorted_by_rating_times_merge,
                                              how='left',left_on='item_id',right_on='item_id')
#加上排名，数字越小，越靠前
df_items_sorted_by_mean_rating_merge['ranking_mean_rate']=range(items_mean_rating.count())
#print(df_items_sorted_by_mean_rating_merge.head())

#前二十大流行电影
popular_items_count_top20=df_items_sorted_by_mean_rating_merge.iloc[0:20]['mean_rating']
popular_items_count_top20_titles=df_items_sorted_by_mean_rating_merge.iloc[0:20]['title']

objects=(list(popular_items_count_top20_titles))
y_pos=np.arange(len(objects))
performance=list(popular_items_count_top20)

'''plt.rcdefaults()
plt.bar(y_pos,performance,align='center',alpha=0.5)
plt.xticks(y_pos,objects,rotation='vertical')
plt.ylabel('Mean Rating')
plt.title('Most popular Movies')
plt.tight_layout()
plt.show()'''

#去掉评分次数<20的电影
df_items_sorted_by_mean_rating_merge2=df_items_sorted_by_mean_rating_merge[
    df_items_sorted_by_mean_rating_merge.rating_times>20
]
#print('评分次数大于20:',df_items_sorted_by_mean_rating_merge2.head())
fig,ax=plt.subplots(1,1,figsize=(12,4))
ax.scatter(df_items_sorted_by_mean_rating_merge['rating_times'],
           df_items_sorted_by_mean_rating_merge['mean_rating'])

plt.show()
#使用正则表达式取出上映年份
df_items_sorted_by_mean_rating_merge['year']=df_items_sorted_by_mean_rating_merge.title.str.extract(
    '(\((\d{4})\))',expand=True
).iloc[:,1]
print(df_items_sorted_by_rating_times_merge.head())

#统计每年上映的电影数目，默认是降序排列
items_sorted_by_year=df_items_sorted_by_mean_rating_merge['year'].value_counts()
#print(items_sorted_by_year.head())


