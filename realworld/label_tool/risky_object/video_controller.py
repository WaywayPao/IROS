from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from opencv_engine import opencv_engine
from collections import OrderedDict
import time

# videoplayer_state_dict = {
#  "stop":0,
#  "play":1,
#  "pause":2
# }


class video_controller(object):
    def __init__(self, video_path, ui, main_contriller, FPS=2):
        self.video_path = video_path
        self.ui = ui
        self.qpixmap_fix_width = 640
        self.qpixmap_fix_height = 256
        self.FPS = FPS
        self.current_frame_no = 1
        self.videoplayer_state = "stop"
        self.main_contriller = main_contriller
        self.init_video_info()
        self.set_video_player()

    def update_video_path(self, video_path):

        self.video_path = video_path
        self.init_video_info()

    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)

        self.vc = videoinfo["vc"]
        self.video_fps = videoinfo["fps"]
        self.video_total_frame_count = videoinfo["frame_count"]
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"]
        self.ui.slider_videoframe.setRange(1, self.video_total_frame_count)
        self.ui.slider_videoframe.valueChanged.connect(self.getslidervalue)


    def getslidervalue(self):
        self.current_frame_no = self.ui.slider_videoframe.value()



    def setslidervalue(self, frame_no):
        self.current_frame_no = frame_no
        self.ui.slider_videoframe.setValue(frame_no)

    def set_video_player(self):
        self.timer = QTimer()  # init QTimer
        # when timeout, do run one
        self.timer.timeout.connect(self.timer_timeout_job)
        # self.timer.start(1000//self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        # but if CPU can not decode as fast as fps, we set 1 (need decode time)
        self.timer.start(1)

    def __get_frame_from_frame_no(self, frame_no):
        self.vc.set(1, frame_no-1)
        ret, frame = self.vc.read()
        self.ui.label_framecnt.setText(
            f"Frame No. {frame_no:3d}/{self.video_total_frame_count:3d}")
        self.setslidervalue(frame_no)

        return frame

    def __update_label_frame(self, frame):
        bytesPerline = 3 * self.video_width
        qimg = QImage(frame, self.video_width, self.video_height,
                      bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)

        # like 1600/16 > 90/9, height is shorter, align width
        if self.qpixmap.width()/16 >= self.qpixmap.height()/9:
            self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        else:  # like 1600/16 < 9000/9, width is shorter, align height
            self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.label_videoframe.setPixmap(self.qpixmap)
        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.label_videoframe.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # Center

    def play(self):
        self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"

    def timer_timeout_job(self):
        frame = self.__get_frame_from_frame_no(self.current_frame_no)
        self.__update_label_frame(frame)

        if (self.videoplayer_state == "play"):

            time.sleep(1/self.FPS)
            if self.current_frame_no > self.video_total_frame_count:
                self.videoplayer_state = "stop"
            else:
                self.current_frame_no += 1
                self.setslidervalue(self.current_frame_no)

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 1

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
