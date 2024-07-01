import cv2
import numpy as np

def calculate_srm_features(image, x, y, window_size=3):
    # 计算局部区域的SRM特征（均值和标准差）
    half_size = window_size // 2
    local_region = image[max(0, x-half_size):min(image.shape[0], x+half_size+1),
                         max(0, y-half_size):min(image.shape[1], y+half_size+1)]
    
    mean_value = np.mean(local_region)
    std_deviation = np.std(local_region)
    
    return mean_value, std_deviation

def choose_embed_positions(image, threshold=0.35):
    # 计算每个像素的局部区域的SRM特征
    rows, cols = image.shape
    srm_features = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            # 计算像素（i, j）的局部区域的SRM特征
            mean_value, std_deviation = calculate_srm_features(image, i, j)
            
            # 示例：将均值和标准差组合成一个分数
            srm_features[i, j] = mean_value + std_deviation
            
    # 将SRM特征归一化到[0, 1]
    srm_features = srm_features / np.max(srm_features)
    
    # 根据SRM特征分数选择嵌入位置
    embed_positions = []
    for i in range(rows):
        for j in range(cols):
            if srm_features[i, j] < threshold:
                embed_positions.append((i, j))
    
    return embed_positions

def read_data_to_embed(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip()
    # 将数据转换为二进制字符串
    # print(data)
    bit_string = ''.join(format(ord(char), '08b') for char in data)
    # print(bit_string)
    return bit_string

def embed_data(image, embed_positions, bit_string):
    img_copy = np.copy(image)
    embed_count = len(embed_positions)
    data_length = len(bit_string)
    
    if data_length > embed_count:
        raise ValueError("要嵌入的数据对选择的位置来说太大了")
    
    mask = np.zeros_like(image)
    
    embed_index = 0
    for pos in embed_positions:
        row, col = pos
        if embed_index < data_length:
            mask[pos] = 255
            # 获取像素值的最低有效位
            pixel_value = img_copy[row, col]
            
            # 将bit_string中的位嵌入像素值的最低有效位中
            new_lsb = int(bit_string[embed_index])
            if new_lsb:
                new_lsb = pixel_value | 1
            else:
                new_lsb = pixel_value & 254
            
            # 用嵌入的值更新图像副本
            img_copy[row, col] = new_lsb

            embed_index += 1
        else:
            break
    
    return img_copy, mask

def extract_embedded_data(embedded_image, embed_positions, embed_count):
    extracted_data = []
    bit_string = ''
    embed_index = 0
    for pos in embed_positions:
        row, col = pos
        if embed_index < embed_count:
            # 获取像素值的最低有效位
            pixel_value = embedded_image[row, col]
            
            lsb = pixel_value & 1

            bit_string += str(lsb)
            
            embed_index += 1
        else:
            break
    # print(bit_string)
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        extracted_data.append(chr(int(byte, 2)))
    
    extracted_text = ''.join(extracted_data)
    return extracted_text

# 示例用法
if __name__ == "__main__":
    # 加载一个示例PGM格式的图像（替换为你自己的图像路径）
    image_path = '1.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 根据SRM特征选择LSB嵌入位置
    embed_positions = choose_embed_positions(image)
    
    # 从date.txt读取要嵌入的数据
    data_to_embed = read_data_to_embed('date.txt')
    
    # 将数据嵌入图像的选择位置中
    embedded_image, mask = embed_data(image, embed_positions, data_to_embed)
    
    # 保存嵌入数据的图像为PNG格式（替换为你想要的输出路径）
    cv2.imwrite('embedded_image.png', embedded_image)
    print(f"数据嵌入成功。嵌入数据的图像已保存为'embedded_image.png'。")
    # 单独保存掩码图像
    cv2.imwrite('embedded_image_mask.png', mask)
    print(f"嵌入掩码图像已保存为'embedded_image_mask.png'。")
    
    # 示例：从嵌入数据的图像中提取嵌入的数据
    extracted_data = extract_embedded_data(embedded_image, embed_positions, len(data_to_embed))
    print(f"从嵌入图像中提取的数据：{extracted_data}")
