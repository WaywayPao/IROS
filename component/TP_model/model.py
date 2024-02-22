import torch
import torch.nn as nn

     
class TP_MODEL(nn.Module):
    def __init__(self, out_dim=32, time_step=5):
        super(TP_MODEL, self).__init__()

        self.time_step = time_step
        self.fusion_size = 8*16
        self.hidden_size = 512

        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 16)),
            nn.Flatten()
        )

        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)

        self.tp_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )


    def forward(self, seg_inputs):

        B, T, C, H, W = seg_inputs.shape

        # initialize LSTM
        hx = torch.zeros((B, self.hidden_size))
        cx = torch.zeros((B, self.hidden_size))
        
        seg_inputs = seg_inputs.reshape(-1, C, H, W)
        global_feature = self.img_encoder(seg_inputs)
        seg_inputs = seg_inputs.reshape(B, T, C, H, W)

        for t in range(0, self.time_step):
            global_feature = global_feature[:, t].clone()
            global_feature = global_feature.reshape(B, -1)

            # LSTM
            hx, cx = self.lstm(self.drop(global_feature), (hx, cx))

        pred_tp = self.tp_predictor(hx)

        return pred_tp
