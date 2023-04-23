import os
import cv2 as cv

import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from show_marks import show_marks

print(f"PID {os.getpid()}")


checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

bgr_img = cv.imread("Louvre.jpg")
rgb_img = bgr_img[:,:,::-1]

print('Before sam registry model')
sam = sam_model_registry[model_type](checkpoint=checkpoint)

mask_generator = SamAutomaticMaskGenerator(sam)
print('SamAutomaticMaskGenerator ...Done!')

print('Generaring masks...')
masks = mask_generator.generate(rgb_img)
print('Masks generated')

print(len(masks))
print(masks)


plt.imshow(rgb_img)
show_marks(masks)
plt.axis('off')
plt.show()