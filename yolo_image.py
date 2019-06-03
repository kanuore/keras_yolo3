# -*- coding: utf-8 -*-
"""
change code for image inference on a image from yolo.py,
and new style using keras mode just like caffe
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os

from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_loss
from yolo3.utils import get_random_data
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model():
    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw


    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    return model_body


def draw_output(image,out_boxes, out_scores, out_classes):


    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    classes_path = 'model_data/coco_classes.txt'
    class_names = get_classes(classes_path)
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=(0,0,255)
            )
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(0,0,255))
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image


if __name__ == '__main__':

    '''
    13*13:(116x90)，(156x198)，(373x326)
    26*26:(30x61)，(62x45)，(59x119)
    52*52:(10x13)，(16x30)，(33x23)，
    '''
    img= '/home/lbk/keras_study/yolo2-master/images/person.jpg'
    image = Image.open(img)
    input_size = (416,416)
    boxed_image = letterbox_image(image, input_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    # 根据代码构造网络结构
    model = create_model()    
    # 从参数模型中加载模型参数
    model.load_weights('model_test/my_model.h5')
    model.summary()
   

    classes_path = 'model_data/coco_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_image_shape = K.placeholder(shape=(2, ))

    # 根据网络输出矩阵进行挑选
    boxes, scores, classes = yolo_eval(model.output, anchors,
            num_classes, input_image_shape,
            score_threshold= 0.3, iou_threshold=0.45)


    #os._exit(0)
    # 开启会话sess，要执行预测
    sess = K.get_session()
    '''
    sess.run();model.fit()能够交叉使用，应该是都与底层做了相应的适配
    '''
    '''
    在一个sess.run()环境中，一个大图里面的节点只会计算一次，
    哪怕有c->a->b这种依赖关系，运行sess.run([a,b])这种时候，
    先查看图中的依赖关系，然后进行一次计算，返回计算的a,b的值。
    '''
    #  the value "1" (training mode) or "0" (test mode) to feed_dict:
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
                model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
        }
    )

    sess.close()
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    result = draw_output(image,out_boxes, out_scores, out_classes)
    result.show()