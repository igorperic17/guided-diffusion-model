import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path
import os

from thispersondoesnotexist import Person, get_online_person, save_picture, get_checksum_from_picture

# params
NUM_OF_IMAGES = 10
IMAGE_FOLDER = "images/"
NUM_DIFFUSION_ITERATIONS = 4

VERBOSE = False

def download_face_images(image_count: int, clear_folder: bool = False):
    
    if clear_folder:
        pass
    
    downloaded_images = set({})
    
    while True:
        
        i = len(downloaded_images)
        if i == image_count:
            break;
        
        print("Downloading image {0}...".format(str(i)))
        
        face = get_online_person()
        
        md5 = get_checksum_from_picture(face)
        if md5 in downloaded_images:
            print("Downloaded duplicate, repeating.")
            continue
        downloaded_images.add(md5)
        
        file_name = os.path.join(IMAGE_FOLDER, str(i) + '_0.jpg')
        
        print("Saving image {0}...".format(str(i)))
        save_picture(face, file_name)
    
    

def generate_diffused_images():
    
    for dir in os.listdir(IMAGE_FOLDER):
        image_path = Path(IMAGE_FOLDER + dir)
        
        if image_path.suffix != ".jpg":
            continue
        image = Image.open(image_path)
        image = np.asarray(image) / 255.0
        
        if VERBOSE:
            plt.imshow(image)
            plt.show()
        
        # perform diffusion
        for i in range(NUM_DIFFUSION_ITERATIONS):
            
            # generate random noise
            image = image + np.random.normal(size=image.size).reshape((image.shape))
            image[image > 1.0] = 1.0
            image[image < 0.0] = 0.0
            if VERBOSE:
                plt.imshow(image)
                plt.show()
            
            # save noisy image
            new_image_path = "{0}/{1}{3}{2}".format(IMAGE_FOLDER, image_path.stem, image_path.suffix, str(i + 1))
            new_image = np.array(image * 255, dtype=np.uint8)
            new_image = Image.fromarray(new_image)
            new_image.save(new_image_path)

if __name__ == "__main__":
    
    download_face_images(NUM_OF_IMAGES)
    
    generate_diffused_images()