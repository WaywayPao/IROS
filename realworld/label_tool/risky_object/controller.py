# from PyQt5 import QtCore
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from pathlib import Path
import time
import os
from collections import OrderedDict
from UI import Ui_MainWindow
from video_controller import video_controller
import json


class MainWindow_controller(QMainWindow):
    def __init__(self):

        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.video_root = f"../videos"
        self.behavio_root = f"../behavior/annotation"
        self.risky_object_root = f"./annotation"
        self.phases = ["train", "test"][:]
        self.actor_id = None
        self.ro_dict = OrderedDict()
        
        self.init_phase()

        self.ui.pushButton_next.clicked.connect(self.frame_next)
        self.ui.pushButton_back.clicked.connect(self.frame_back)
        self.ui.pushButton_next_3.clicked.connect(self.basic_next)
        self.ui.pushButton_back_3.clicked.connect(self.basic_back)

        self.ui.start_frame_bottom.clicked.connect(self.set_start_frame)
        self.ui.end_frame_bottom.clicked.connect(self.set_end_frame)
        self.ui.save_non_risky_bottom.clicked.connect(self.save_non_risky)

        self.ui.save_RO_bottom.clicked.connect(self.save_RO)
        self.ui.add_RO_bottom.clicked.connect(self.add_RO)
        self.ui.clean_all_bottom.clicked.connect(self.clean_all)

        self.ui.comboBox_phase.currentIndexChanged.connect(
            self.init_scenarios)
        self.ui.comboBox_scenario.currentIndexChanged.connect(
            self.update_scenario)


    def mousePressEvent(self, event):
        x = event.pos().x()-220
        y = event.pos().y()-80

        # TODO
        self.actor_id = int(x+y)//2

        # if not self.actor_id in gt_id
        #   self.ui.label_selectedID.setText(f"Select ID : None")
        #   return

        if event.button() == Qt.LeftButton:
            # add specific id in buffer
            start_frame = int(self.ui.start_frame_bottom.text().split()[1])
            end_frame = int(self.ui.end_frame_bottom.text().split()[1])
            self.ro_dict[str(self.actor_id)] = [start_frame, end_frame]
            self.ui.label_selectedID.setText(f"Select ID : {int(self.actor_id):3d}")

        elif event.button() == Qt.RightButton:
            # delete specific id in all frame
            if str(self.actor_id) in self.ro_dict:
                del self.ro_dict[str(self.actor_id)]
            self.ui.label_selectedID.setText(f"Delete ID : {int(self.actor_id):3d}")

        self.show_ro_list()


    def init_phase(self):

        self.ui.comboBox_phase.clear()
        for phase in self.phases:
            self.ui.comboBox_phase.addItem(phase)

        self.init_video = True
        self.init_scenarios()
        self.init_video = False

    def init_scenarios(self):

        self.phase = self.ui.comboBox_phase.currentText()

        self.video_list = list()
        for video in sorted(os.listdir(self.video_root+f"/{self.phase}")):
            scenario = video.split('.')[0]

            json_path = os.path.join(self.behavio_root, self.phase, scenario+'.json')
            data = json.load(open(json_path))
            if list(data.values())[0] == -1:                
                continue

            self.video_list.append(scenario)

        print("Risky scenario :", len(self.video_list))

        self.scenario_clear = True
        self.ui.comboBox_scenario.clear()
        N = len(self.video_list)
        for idx, basic in enumerate(self.video_list, 1):
            self.ui.comboBox_scenario.addItem(basic)
        self.scenario_clear = False

        self.update_scenario()

    def update_scenario(self):

        if not self.scenario_clear:
            self.current_scenario = self.ui.comboBox_scenario.currentText()
            self.update_video()
            self.load_behavior_json()
            self.load_RO_json()
            
    def update_video(self):

        self.video_path = f"{self.video_root}/{self.phase}/{self.current_scenario}.avi"
        assert os.path.exists(self.video_path), f"{self.video_path} not found"

        if self.init_video:
            self.video_controller = video_controller(
                video_path=self.video_path, ui=self.ui, main_contriller=self)

            self.ui.button_play.clicked.connect(self.video_controller.play)
            self.ui.button_stop.clicked.connect(self.video_controller.stop)
            self.ui.button_pause.clicked.connect(self.video_controller.pause)

        self.video_controller.update_video_path(self.video_path)


    def load_behavior_json(self):

        self.ui.label_11.setText(f"Behavior Stop")
        file_path = f"{self.behavio_root}/{self.phase}/{self.current_scenario}.json"
        if os.path.exists(file_path):
            data = json.load(open(file_path))
        else:
            data = {
                "start_frame": -1,
                "end_frame": -1
            }

        self.ui.start_frame_bottom.setText(
            f"Start: {(data['start_frame']):3d}")
        self.ui.end_frame_bottom.setText(f"End: {(data['end_frame']):3d}")

        return data

    def write_json(self, non_risky=False):

        if non_risky:
            start_frame = -1
            end_frame = -1
            self.ui.start_frame_bottom.setText(f"Start: {-1:3d}")
            self.ui.end_frame_bottom.setText(f"End: {-1:3d}")
        else:
            start_frame = int(self.ui.start_frame_bottom.text().split()[1])
            end_frame = int(self.ui.end_frame_bottom.text().split()[1])
        data = {
            "start_frame": start_frame,
            "end_frame": end_frame
        }
        with open(f"{self.behavio_root}/{self.phase}/{self.current_scenario}.json", "w") as f:
            json.dump(data, f, indent=4)

        behavior_dict = OrderedDict()
        for scene_json in sorted(os.listdir(f"{self.behavio_root}/{self.phase}")):
            labeled_frame = json.load(open(f"{self.behavio_root}/{self.phase}/{scene_json}"))
            scene = scene_json.split('.')[0]
            behavior_dict[scene] = labeled_frame

        with open(f"{self.behavio_root}/{self.phase}.json", "w") as f:
            json.dump(behavior_dict, f, indent=4)
        
        
        # print(set(self.video_list)-set(behavior_dict.keys()))


    def load_RO_json(self):

        self.actor_id = None
        self.ui.label_selectedID.setText(f"Select ID : {self.actor_id}")

        file_path = f"{self.risky_object_root}/{self.phase}/{self.current_scenario}.json"
        if os.path.exists(file_path):
            self.ro_dict = json.load(open(file_path))
        else:
            self.ro_dict = OrderedDict()

        self.show_ro_list()
        self.ui.start_lineEdit.setText("")
        self.ui.end_lineEdit.setText("")

    def show_ro_list(self):

        text = "" if len(self.ro_dict)>0 else "None"
        for id in self.ro_dict:
            text += f"{int(id):3d} : {self.ro_dict[id]}\n"
        self.ui.label_saved_id.setText(text)

    def add_RO(self):

        start_text = self.ui.start_lineEdit.text()
        end_text = self.ui.end_lineEdit.text()

        if start_text == "" or end_text == "":
            print("No input frame no. !!!")
            return
        elif self.actor_id == None:
            print("No input actor id !!!")
            return

        start_frame = int(start_text)
        end_frame = int(end_text)
        self.ro_dict[str(self.actor_id)] = [start_frame, end_frame]
        self.show_ro_list()

    def save_RO(self):
        with open(f"{self.risky_object_root}/{self.phase}/{self.current_scenario}.json", "w") as f:
            json.dump(self.ro_dict, f, indent=4)
    
    def clean_all(self):

        self.actor_id = None
        self.ro_dict = OrderedDict()
        self.show_ro_list()
        self.ui.label_selectedID.setText(f"Select ID : {self.actor_id}")


    def frame_next(self):
        # frame no. start with 1
        current_frame_no = self.video_controller.current_frame_no
        current_frame_no = current_frame_no % self.video_controller.video_total_frame_count + 1

        self.video_controller.setslidervalue(current_frame_no)

    def frame_back(self):
        # frame no. start with 1
        current_frame_no = self.video_controller.current_frame_no
        current_frame_no = (
            current_frame_no-2) % self.video_controller.video_total_frame_count + 1

        self.video_controller.setslidervalue(current_frame_no)


    def basic_next(self):

        index = self.ui.comboBox_scenario.currentIndex()
        index = (index+1) % self.ui.comboBox_scenario.count()

        self.ui.comboBox_scenario.setCurrentIndex(index)
        self.update_scenario()

    def basic_back(self):

        index = self.ui.comboBox_scenario.currentIndex()
        index = (index-1) % self.ui.comboBox_scenario.count()

        self.ui.comboBox_scenario.setCurrentIndex(index)
        self.update_scenario()

    def set_start_frame(self):
        current_frame_no = self.video_controller.current_frame_no
        self.ui.start_frame_bottom.setText(f"Start: {current_frame_no:3d}")
        self.write_json()

    def set_end_frame(self):
        current_frame_no = self.video_controller.current_frame_no
        self.ui.end_frame_bottom.setText(f"End: {current_frame_no:3d}")
        self.write_json()

    def save_non_risky(self):
        self.write_json(non_risky=True)
