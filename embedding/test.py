import SRM
import local_variance as LV

if __name__ == "__main__":
    image_dir = 'D:\Download\database\BOSSbase_1.01'
    data_path = '../data.txt'
    for i in range(1,30):
        image_path = f"{image_dir}\{i}.pgm"
        SRM.embed(img_path=image_path, data_path=data_path, img_hz="pgm")
        LV.embed(img_path=image_path, data_path=data_path, img_hz="pgm")

    # import os
    # from PIL import Image

    # def convert_pgm_to_png(image_dir, output_dir):
    #     # Create output directory if it doesn't exist
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     # List all files in the image_dir
    #     for filename in os.listdir(image_dir):
    #         if filename.endswith(".pgm"):
    #             # Open the PGM image
    #             pgm_path = os.path.join(image_dir, filename)
    #             with Image.open(pgm_path) as img:
    #                 # Convert the image to PNG format
    #                 png_filename = filename.replace(".pgm", ".png")
    #                 png_path = os.path.join(output_dir, png_filename)
    #                 img.save(png_path, "PNG")
    #                 print(f"Converted {pgm_path} to {png_path}")

    # # Example usage
    # image_dir = "../img/mask"
    # output_dir = "../img/png"
    # convert_pgm_to_png(image_dir, output_dir)

    