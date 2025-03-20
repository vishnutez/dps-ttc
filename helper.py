import matplotlib.pyplot as plt
import os

from facenet_pytorch import MTCNN



mtcnn = MTCNN(image_size=256, margin=70, min_face_size=20, )  # Keep the default values

from PIL import Image


for img_file in os.listdir('../guidance_images/'):

    img = Image.open('../guidance_images/' + img_file)
    img_cropped = mtcnn(img)

    img_reshaped = img_cropped.permute(1, 2, 0).numpy()
    # plt.imshow(img_cropped)
    # plt.show()

    # Conver to pil and save
    img_rescaled = (img_reshaped + 1) / 2
    
    plt.imsave('../guidance_images_cropped/' + img_file, img_rescaled)
    print('Image saved:', img_file)
    print('Image shape:', img_rescaled.shape)