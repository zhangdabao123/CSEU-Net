import time
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from unet import Unet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    unet = Unet()
    mode = "predict"
    count           = False
    name_classes    = ["background","ICE"]
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            img_name = os.path.splitext(img)[0]
            try:
                image = Image.open(img).convert('L')
                image_cv = cv2.imread(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                start_time = time.time()
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                stop_time = time.time()
                ss_time = stop_time-start_time
                print('%.3f秒' % ss_time)
                r_image_cv = np.array(r_image)
                r_image_cv = r_image_cv[:, :, ::-1].copy()
                cv2.imwrite(img_name + '_result.png', r_image_cv)
                r_image_mix = cv2.addWeighted(image_cv, 0.5, r_image_cv, 0.5, 0)
                cv2.imwrite(img_name + '_mix.png', r_image_mix)

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict' or 'fps'.")
