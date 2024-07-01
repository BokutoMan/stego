import os
import cv2
import numpy as np

def calculate_local_variance(image, x, y, window_size):
    # 计算图像中 (x, y) 周围区域的局部方差
    region = image[max(x-window_size//2, 0):min(x+window_size//2+1, image.shape[0]),
                   max(y-window_size//2, 0):min(y+window_size//2+1, image.shape[1])]
    local_variance = np.var(region)
    return local_variance

def choose_embed_positions(image, embed_length, window_size=3):
    # 计算每个像素的局部方差
    rows, cols = image.shape
    srm_features = []

    for i in range(rows):
        for j in range(cols):
            local_variance = calculate_local_variance(image, i, j, window_size)
            srm_features.append((local_variance, (i, j)))

    # 按局部方差排序并选择前 embed_length 个位置
    srm_features.sort(reverse=True, key=lambda x: x[0])
    embed_positions = [pos for _, pos in srm_features[:embed_length]]

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

def embed(img_path, data_path):
    # 加载图像
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 读取要嵌入的数据
    data_to_embed = read_data_to_embed(data_path)
    
    # 根据数据长度选择LSB嵌入位置
    embed_positions = choose_embed_positions(image, len(data_to_embed))
    
    # 将数据嵌入图像的选择位置中
    embedded_image, mask = embed_data(image, embed_positions, data_to_embed)
    
    # 保存嵌入数据的图像为PNG格式（替换为你想要的输出路径）
    img_name = os.path.basename(img_path).split('.')[0]
    cv2.imwrite(f'../img/steg/{img_name}_LV_steg.png', embedded_image)
    print(f"数据嵌入成功。嵌入数据的图像已保存为'img/steg/{img_name}_LV_steg.png'。")
    # 单独保存掩码图像
    cv2.imwrite(f'../img/mask/{img_name}_LV_mask.png', mask)
    print(f"嵌入掩码图像已保存为'img/mask/{img_name}_LV_mask.png'。")



if __name__ == "__main__":
    # 加载一个示例PGM格式的图像（替换为你自己的图像路径）
    image_path = '../1.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 从date.txt读取要嵌入的数据
    data_to_embed = read_data_to_embed('../date.txt')
    
    # 根据数据长度选择LSB嵌入位置
    embed_positions = choose_embed_positions(image, len(data_to_embed))
    
    # 将数据嵌入图像的选择位置中
    embedded_image, mask = embed_data(image, embed_positions, data_to_embed)
    
    # 保存嵌入数据的图像为PNG格式（替换为你想要的输出路径）
    cv2.imwrite('../embedded_image.png', embedded_image)
    print(f"数据嵌入成功。嵌入数据的图像已保存为'embedded_image.png'。")
    # 单独保存掩码图像
    cv2.imwrite('../embedded_image_mask.png', mask)
    print(f"嵌入掩码图像已保存为'embedded_image_mask.png'。")
    
    # 示例：从嵌入数据的图像中提取嵌入的数据
    extracted_data = extract_embedded_data(embedded_image, embed_positions, len(data_to_embed))
    print(f"从嵌入图像中提取的数据：{extracted_data}")