import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image

#import tensorflow as tf

def check_images(imagepath):

    broken_Images=[]

    path = Path(imagepath).rglob("*.jpg")
    for img_p in path:
        print(img_p)
        try:
            img = Image.open(img_p)
            print("success")
            
            """
            fn='/Users/ijiwon/workspace/python_code/EasyLabeling/python_ver/images/21_12_10_14847.jpg'

            with tf.Graph().as_default():
                image_contents = tf.io.read_file(fn)
                image = tf.image.decode_jpeg(image_contents, channels=3)
                init_op = tf.compat.v1.initialize_all_tables()
                with tf.compat.v1.Session() as sess:
                    sess.run(init_op)
                    #sess.run(tf.global_variables_initialier())
                    tmp = sess.run(image)

            """
        except PIL.UnidentifiedImageError:
            print(img_p)
            broken_Images.append(img_p)
    
    if len(broken_Images)==0:
        return True
    else:
        return broken_Images