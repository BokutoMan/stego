import SRM
import local_variance as LV

if __name__ == "__main__":
    image_path = '../1.pgm'
    data_path = '../data.txt'
    SRM.embed(img_path=image_path, data_path=data_path)
    LV.embed(img_path=image_path, data_path=data_path)