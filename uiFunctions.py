from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QStyleFactory
from plateUi import Ui_MainWindow
from detect import platedetect
import cv2
from PyQt5.QtGui import QPixmap, QImage


class myapp(QMainWindow):
    def __init__(self):
        super(myapp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.timer = QtCore.QTimer()

        self.ui.imchoose.clicked.connect(self.openDialog)
        self.ui.plateFind.clicked.connect(self.plateBox)
        self.ui.readPlate.clicked.connect(self.ocr)
        self.ui.rgb.clicked.connect(self.cropped)
        self.ui.gray.clicked.connect(self.gray_img)
        self.ui.blur.clicked.connect(self.blur_img)
        self.ui.thresh.clicked.connect(self.thresh_img)

        self.timer.timeout.connect(self.view_cam)
        self.ui.opencam.clicked.connect(self.control_timer)
        self.ui.savecam.clicked.connect(self.save)


    image_path = ''
    crop_path = ''
    plate_num = ''
    # save_img = None

    def openDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(directory='./data/images/',
                                                  filter="All Files (*);;Image Files (*.jpg, *.png)")
        if fileName:
            # fileName = fileName.split('/')[-1]
            # print(fileName)
            # self.photo.setPixmap(QtGui.QPixmap('data/images/'+fileName))
            self.ui.img.setPixmap(QtGui.QPixmap(fileName))
            self.image_path = fileName

    def plateBox(self):
        image, self.plate_num, self.crop_path = platedetect(self.image_path)
        # image = './detections/detection1.png'
        self.ui.img.setPixmap(QtGui.QPixmap(image))

    def ocr(self):
        self.ui.platetxt.setText(self.plate_num)

    def cropped(self):
        self.ui.label.setPixmap(QtGui.QPixmap(self.crop_path))

    def gray_img(self):
        gray_path = self.crop_path.split('.')[0]
        gray_path = gray_path+'gray.png'
        self.ui.label.setPixmap(QtGui.QPixmap(gray_path))

    def blur_img(self):
        blur_path = self.crop_path.split('.')[0]
        blur_path = blur_path+'blur.png'
        self.ui.label.setPixmap(QtGui.QPixmap(blur_path))

    def thresh_img(self):
        thresh_path = self.crop_path.split('.')[0]
        thresh_path = thresh_path+'thresh.png'
        self.ui.label.setPixmap(QtGui.QPixmap(thresh_path))

    def view_cam(self):
        ret, image = self.cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./data/images/camCapture.png', image)

        height, width, channel = image.shape
        step = channel * width

        qImg = QImage(image.data, width, height, step, QImage.Format_BGR888)
        self.ui.img.setPixmap(QPixmap.fromImage(qImg))

    def control_timer(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)
            self.ui.opencam.setText("Kamerayı Kapat")
        else:
            self.timer.stop()
            self.cap.release()
            self.ui.opencam.setText("Kamerayı Aç")

    def save(self):
        if self.timer.isActive():
            self.ui.opencam.click()
        image = cv2.imread('./data/images/camCapture.png')

        height, width, channel = image.shape
        step = channel * width
        self.image_path = './data/images/camCapture.png'

        qImg = QImage(image.data, width, height, step, QImage.Format_BGR888)
        self.ui.img.setPixmap(QPixmap.fromImage(qImg))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('GTK+'))
    win = myapp()
    win.show()
    sys.exit(app.exec_())
