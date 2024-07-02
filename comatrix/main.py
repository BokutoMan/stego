from matplotlib import pyplot as plt
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def get_entropy(glcm):
    # 计算总的频数
    total = np.sum(glcm)
    
    # 归一化灰度共生矩阵以得到频率
    glcm_normalized = glcm / total
    
    # 初始化熵
    entropy = 0.0
    
    # 计算熵
    for i in range(glcm_normalized.shape[0]):
        for j in range(glcm_normalized.shape[1]):
            if glcm_normalized[i, j] != 0:  # 避免log(0)，因为它是未定义的
                entropy -= glcm_normalized[i, j] * np.log(glcm_normalized[i, j])
    return entropy[0][0]

def get_stats(I):
    # 将彩图转化为灰度图像
    image = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    def extract_features(glcm):
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        entropy = get_entropy(glcm)
        return contrast, energy, correlation, homogeneity, entropy
    
    # 0°的灰度共生矩阵
    glcm0 = graycomatrix(image, [1], [0], symmetric=True, normed=True)
    stats0 = extract_features(glcm0)
    
    # 45°的灰度共生矩阵
    glcm45 = graycomatrix(image, [1], [np.pi/4], symmetric=True, normed=True)
    stats45 = extract_features(glcm45)
    
    # 90°的灰度共生矩阵
    glcm90 = graycomatrix(image, [1], [np.pi/2], symmetric=True, normed=True)
    stats90 = extract_features(glcm90)
    
    # 135°的灰度共生矩阵
    glcm135 = graycomatrix(image, [1], [3*np.pi/4], symmetric=True, normed=True)
    stats135 = extract_features(glcm135)
    
    stats = np.array([
        list(stats0),
        list(stats45),
        list(stats90),
        list(stats135)
    ])
    
    # 求每项特征在四个方向上的均值
    per_stat = 0.25 * np.sum(stats, axis=0)
    
    return stats, per_stat

if __name__ == "__main__":
    for i in range(24,30):
        # 读取图像并计算特征
        I1 = cv2.imread(f'../img/steg/LV/{i}_LV_steg.pgm')
        stats1, per_stat1 = get_stats(I1)
        I2 = cv2.imread(f'../img/steg/SRM/{i}_SRM_steg.pgm')
        stats2, per_stat2 = get_stats(I2)
        I3 = cv2.imread(f'../img/original/{i}.pgm')
        stats3, per_stat3 = get_stats(I3)

        # 汇总四张图的平均特征
        four_per_stat = np.array([per_stat1, per_stat2, per_stat3])
        print(f"{i}.pgm")
        print(four_per_stat)
# 3D条形图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x_labels = ['LV', 'SRM', 'Orignal']
# y_labels = ['Contrast', 'Energy', 'Correlation', 'Homogeneity', 'Entropy']
# x = np.arange(len(x_labels))
# y = np.arange(len(y_labels))
# x, y = np.meshgrid(x, y)

# x = x.flatten()
# y = y.flatten()
# z = np.zeros_like(x)

# dx = dy = 0.5
# dz = four_per_stat.T.flatten()

# ax.bar3d(x, y, z, dx, dy, dz)

# ax.set_xticks(np.arange(len(x_labels)) + dx/2)
# ax.set_xticklabels(x_labels)
# ax.set_yticks(np.arange(len(y_labels)) + dy/2)
# ax.set_yticklabels(y_labels)
# ax.set_zlabel('Feature Value')

# plt.title('Comparison of Gray-Level Co-occurrence Matrix Features for Four Images')
# plt.legend(['Contrast', 'Energy', 'Correlation', 'Homogeneity', 'Entropy'], loc='upper left')

# plt.show()

