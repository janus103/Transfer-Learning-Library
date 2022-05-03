from PIL import Image
import numpy as np
#im = Image.open('/data/ImageNet_dataset/v100/jin/imageNet/train/n02071294/n02071294_2990.JPEG')
im = Image.open('/data/ImageNet_dataset/v100/jin/imageNet_gray/train/n02071294/n02071294_2990.JPEG')
print(im.mode)

x = np.array(im)

print(np.shape(x))


