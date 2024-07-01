import numpy as np
import cv2
from scipy.stats import norm

def statistical_residual_method(image_path, threshold=0.05):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 计算图像的统计残差
    mean = np.mean(img)
    std_dev = np.std(img)
    residuals = (img - mean) / std_dev
    
    # 计算残差的正态分布累积概率
    p_values = norm.cdf(residuals)
    
    # 判断是否存在异常值
    anomaly_count = np.sum(p_values < threshold)
    
    # 输出结果
    if anomaly_count > 0:
        print(f"Image {image_path} contains potential steganographic information.")
    else:
        print(f"Image {image_path} does not appear to contain steganographic information.")
    return anomaly_count

# 测试代码
if __name__ == "__main__":
    img_dir = "D:\Download\database\BOSSbase_1.01"
    l = []
    for i in range(1, 100):
        image_path = f"{img_dir}\{i}.pgm"
        anomaly_count = statistical_residual_method(image_path) 
        l.append(anomaly_count)

    print(l)
