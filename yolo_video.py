import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, default = 'model_data/yolo.h5',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    '''
    13*13:(116x90)，(156x198)，(373x326)
    26*26:(30x61)，(62x45)，(59x119)
    52*52:(10x13)，(16x30)，(33x23)，
    '''
    parser.add_argument(
        '--anchors', type=str, default = 'model_data/yolo_anchors.txt',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, default = 'model_data/coco_classes.txt',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int, default = 1,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    '''
    只要运行使用该变量，就将该变量设为True
    '''
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        # YOLO是个类，这里相当于实例化一个对象
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
