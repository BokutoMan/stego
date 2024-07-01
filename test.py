

import os

img_path = "./1.pgm"
img_name = os.path.basename(img_path).split('.')[0]
print(img_name)