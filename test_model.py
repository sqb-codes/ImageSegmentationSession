import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np

model = load_model("unet_model.h5")

def do_pred(img):
    # img = "NuclieDataset/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png"

    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    dtype=np.uint8)

    img = imread(img)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                mode="constant", preserve_range=True)
    X_test[0] = img

    pred_mask = model.predict(X_test)
    pred_mask = pred_mask.reshape(128,128)
    return pred_mask
