import os
import numpy as np 

load_gan = os.listdir 
load_discriminator = os.listdir 


img_loader = None


XTRAINDIR = None
YTRAINDIR = None
N_BATCHES = 100
EPOCHS = 50


def preprocess_vgg_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def load_img(img):
    full_img = image.load_img(img)
    full_img = image.img_to_array(full_img)
    full_img = cv2.resize(full_img, (200, 200))
    full_img = np.expand_dims(full_img, axis=0)
    return full_img

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

files = os.listdir(XTRAINDIR)
model = model_loader()


for epoch in range(EPOCHS):

    np.random.shuffle(files)
    batches = chunks(files, N_BATCHES)

    for batch in batches:
        
        xtrain = [] 
        ytrain = []
        for fp in batch:
            xtrain.append(load_img(XTRAINDIR+fp))
            ytrain.append(load_img(YTRAINDIR+fp))



        








        
















