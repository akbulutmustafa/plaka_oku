import os
import cv2
import re
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'




def recognize_plate(img, coords):
    xmin, ymin, xmax, ymax = coords

    box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu Threshold", thresh)
    #cv2.waitKey(0)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    im2 = gray.copy()
    plate_num = ""

    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        if height / float(h) > 6: continue

        ratio = h / float(w)
        if ratio < 1.5: continue

        if width / float(w) > 15: continue

        area = h * w
        if area < 100: continue

        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)#black txt white
        roi = cv2.medianBlur(roi, 5)
        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except:
            text = None
    # if plate_num != None:
        # print("License Plate #: ", plate_num)
    #cv2.imshow("Character's Segmented", im2)
    #cv2.waitKey(0)
    return plate_num


def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes



def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_name = 'license_plate'
    if class_name in allowed_classes:
        xmin, ymin, xmax, ymax = boxes[0]

        cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

        img_name = class_name + '.png'
        img_path = os.path.join(path, img_name)

        gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        blur = cv2.bitwise_not(thresh)  # cv2.GaussianBlur(gray, (5,5), 0)

        cv2.imwrite(img_path, cropped_img)
        img_path = img_path.split('.')[0]
        cv2.imwrite(img_path+'gray.png', gray)
        cv2.imwrite(img_path+'blur.png', blur)
        cv2.imwrite(img_path+'thresh.png', thresh)


def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_name = 'license_plate'

    xmin, ymin, xmax, ymax = boxes[0]

    box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    blur = cv2.medianBlur(thresh, 3)

    blur = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
        print("Class: {}, Text Extracted: {}".format(class_name, text))
    except:
        text = None