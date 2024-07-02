import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(X, Y):
    return cosine_similarity([X], [Y])[0][0]

def pearson_correlation(X, Y):
    return np.corrcoef(X, Y)[0, 1]

Original_path = r'./Original/'

all_Original = pd.DataFrame()

for file in os.listdir(Original_path):
    fea_Original = pd.read_csv(Original_path + file, header=None,delimiter=' ')
    fea_Original = fea_Original.iloc[:, :-1] # 去掉最后一列
    all_Original = pd.concat([all_Original, fea_Original], axis=1)

# print(all_Original.shape)

LV_path = r'./LV/'

all_LV = pd.DataFrame()

for file in os.listdir(LV_path):
    fea_LV = pd.read_csv(LV_path + file, header=None,delimiter=' ')
    fea_LV = fea_LV.iloc[:, :-1] # 去掉最后一列
    all_LV = pd.concat([all_LV, fea_LV], axis=1)

# print(all_LV.shape)

SRM_path = r'./SRM/'

all_SRM = pd.DataFrame()

for file in os.listdir(SRM_path):
    fea_SRM = pd.read_csv(SRM_path + file, header=None,delimiter=' ')
    fea_SRM = fea_SRM.iloc[:, :-1] # 去掉最后一列
    all_SRM = pd.concat([all_SRM, fea_SRM], axis=1)

# print(all_SRM.shape)

all_SRM = all_SRM.to_numpy()
all_LV = all_LV.to_numpy()
all_Original = all_Original.to_numpy()

persion = []
consin = []
for i in range(0,(all_SRM.shape[0])):
    per = pearson_correlation(all_Original[i], all_LV[i])
    cos = cosine_sim(all_Original[i], all_LV[i])
    persion.append(per)
    consin.append(cos)

print("原图与使用局部方差嵌入的图片的皮尔逊相关系数:", persion)
print("原图与使用局部方差嵌入的图片的余弦相似度：", consin)

persion1 = []
consin1 = []
for i in range(0,(all_SRM.shape[0])):
    per = pearson_correlation(all_Original[i], all_SRM[i])
    cos = cosine_sim(all_Original[i], all_SRM[i])
    persion1.append(per)
    consin1.append(cos)

print("原图与使用SRM方法嵌入的图片的皮尔逊相关系数:", persion1)
print("原图与使用SRM方法嵌入的图片的余弦相似度:", consin1)


print("两组皮尔逊相关系数的相关系数", pearson_correlation(persion, persion1))
print("两组余弦相似度的余弦相似度：",cosine_sim(consin, consin1))