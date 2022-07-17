import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pathlib import Path
import os

# params
IMAGE_FOLDER = "images/"
NUM_DIFFUSION_ITERATIONS = 5

VERBOSE = False

if __name__ == "__main__":
    
    for dir in os.listdir(IMAGE_FOLDER):
        image_path = Path(IMAGE_FOLDER + dir)
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
            new_image_path = "{0}/{1}{3}{2}".format(IMAGE_FOLDER, image_path.stem, image_path.suffix, str(i))
            new_image = np.array(image * 255, dtype=np.uint8)
            new_image = Image.fromarray(new_image)
            new_image.save(new_image_path)
        
        