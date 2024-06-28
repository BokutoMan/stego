import cv2
import numpy as np



def calculate_local_variance(image, x, y, window_size):
    # Calculate local variance for a region around (x, y) in the image
    region = image[max(x-window_size//2, 0):min(x+window_size//2+1, image.shape[0]),
                   max(y-window_size//2, 0):min(y+window_size//2+1, image.shape[1])]
    local_variance = np.var(region)
    return local_variance

def choose_embed_positions(image, window_size=3, threshold=100.0):
    # Calculate local variance for each pixel
    rows, cols = image.shape
    embed_positions = []

    for i in range(rows):
        for j in range(cols):
            local_variance = calculate_local_variance(image, i, j, window_size)
            if local_variance > threshold:
                embed_positions.append((i, j))

    return embed_positions

def read_data_to_embed(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().strip()
    # Convert data to binary string
    print(data)
    bit_string = ''.join(format(ord(char), '08b') for char in data)
    print(bit_string)
    return bit_string

def embed_data(image, embed_positions, bit_string):
    img_copy = np.copy(image)
    embed_count = len(embed_positions)
    data_length = len(bit_string)
    
    if data_length > embed_count:
        raise ValueError("Data to embed is too large for selected positions")
    
    embed_index = 0
    for pos in embed_positions:
        row, col = pos
        if embed_index < data_length:
            # Get LSB of pixel value
            pixel_value = img_copy[row, col]
            
            # Embed the bit from bit_string into LSB of pixel_value
            new_lsb = int(bit_string[embed_index])
            if new_lsb:
                new_lsb = pixel_value | 1
            else:
                new_lsb = pixel_value & 126
            
            # Update the image copy with embedded value
            img_copy[row, col] = new_lsb

            embed_index += 1
        else:
            break
    
    return img_copy

def extract_embedded_data(embedded_image, embed_positions, embed_count):
    extracted_data = []
    bit_string = ''
    embed_index = 0
    for pos in embed_positions:
        row, col = pos
        if embed_index < embed_count:
            # Get LSB of pixel value
            pixel_value = embedded_image[row, col]
            
            lsb = pixel_value & 1

            bit_string += str(lsb)
            
            embed_index += 1
        else:
            break
    print(bit_string)
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        extracted_data.append(chr(int(byte, 2)))
    
    extracted_text = ''.join(extracted_data)
    return extracted_text

# Example usage
if __name__ == "__main__":
    # Load an example PGM format image (replace with your own image path)
    image_path = '1.pgm'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Choose embedding positions based on local variance
    embed_positions = choose_embed_positions(image, window_size=5, threshold=100.0)
    
    # Read data to embed from date.txt
    data_to_embed = read_data_to_embed('date.txt')
    
    # Embed data into image at selected positions
    embedded_image = embed_data(image, embed_positions, data_to_embed)
    
    # Save embedded image as PNG (replace with your desired output path)
    cv2.imwrite('embedded_image.png', embedded_image)
    print(f"Data embedded successfully. Embedded image saved as 'embedded_image.png'.")
    
    # Example: Extract embedded data from embedded_image
    extracted_data = extract_embedded_data(embedded_image, embed_positions, len(data_to_embed))
    print(f"Extracted data from embedded image: {extracted_data}")
