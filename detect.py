import os

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
# from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# framework = 'tflite'
# weights = './checkpoints/custom-416.tflite'
# size = 416
# tiny = False
# model = 'yolov4'
# image_path = './data/images/2.jpg'
# output = './detections/'
# iou = 0.45
# score = 0.50
# count = False
# dont_show = True
# info = False
# crop = True
# ocr = False
# plate = True

# python detect.py --weights ./checkpoints/custom-416.tflite --size 416 /
# --model yolov4 --images ./data/images/3.jpg --framework tflite

def platedetect(image_path):
    dont_show = True
    info = False
    crop = True
    pocr = False
    plate = True
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    # images = images

    # load model
    interpreter = tf.lite.Interpreter(model_path='./checkpoints/custom-416.tflite')

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.


    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                    input_shape=tf.constant([input_size, input_size]))


    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]


    # class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    # allowed_classes = list(class_names.values())


    allowed_classes = ['license_plate']

    if crop:
        crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)


    if pocr:
        ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

    image, plate_num = utils.draw_bbox(original_image, pred_bbox, info, allowed_classes=allowed_classes,
                                       read_plate=plate)

    image = Image.fromarray(image.astype(np.uint8))
    # if dont_show:
    #     image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    new_path = './detections/' + 'detection' + image_name + '.png'
    cv2.imwrite(new_path, image)
    crop_path = crop_path+'\license_plate.png'
    return new_path, plate_num, crop_path

# platedetect('./data/images/5.jpg')
