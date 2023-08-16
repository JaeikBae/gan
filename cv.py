# make video with fake images
import cv2 as cv
import numpy as np
import os

dir_name = 'GAN_results'
# dir_name = 'backup'

# Read fake images
fake_images = []
image_names = os.listdir(dir_name)
image_names = [image_name for image_name in image_names if 'GAN_fake_samples' in image_name]
image_names.sort(key=lambda x: int(x.split('GAN_fake_samples')[1].split('.')[0]))

for image_name in image_names:
    fake_images.append(cv.imread(os.path.join(dir_name, image_name)))

# Make video with fake images
height, width, layers = fake_images[0].shape
fourcc = cv.VideoWriter_fourcc(*'mp4v')
video = cv.VideoWriter(os.path.join(dir_name, 'GAN_fake_images.mp4'), fourcc, 5, (width, height))

for i in range(len(fake_images)):
    video.write(fake_images[i])

cv.destroyAllWindows()
video.release()