import torch
import torch.nn as nn
import os
import numpy as np
import torch
from torchvision.transforms import transforms
import cv2 as cv

class MyNet (nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()

        self.conv =nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(196, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=(1,1))

        self.age_fc = nn.Sequential(
            nn.Linear(in_features=196,out_features=25,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=25,out_features=1,bias=True),
            nn.Sigmoid()
        )

        self.gender_fc = nn.Sequential(
            nn.Linear(in_features=196,out_features=25,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=25,out_features=2,bias=True)
        )

    def forward (self,x):
        x = self.conv(x)
        x = self.global_max_pool(x)
        x=x.view(-1,196)
        age = self.age_fc(x)
        gender = self.gender_fc(x)
        return age, gender
model_bin = "../model/face_detector/opencv_face_detector_uint8.pb"
config_text = "../model/face_detector/opencv_face_detector.pbtxt"
net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)

def video_landmark_demo():
    cnn_model = torch.load("./age_gender_model.pt")
    print(cnn_model)
    # capture = cv.VideoCapture(0)
    capture = cv.VideoCapture("D:/images/video/example_dsh.mp4")

    # load tensorflow model
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3]*w
                top = detection[4]*h
                right = detection[5]*w
                bottom = detection[6]*h

                # roi and detect landmark
                roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
                rw = right - left
                rh = bottom - top
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                age_, gender_ = cnn_model(x_input.cuda())
                predict_gender = torch.max(gender_, 1)[1].cpu().detach().numpy()[0]
                gender = "Male"
                if predict_gender == 1:
                    gender = "Female"
                predict_age = age_.cpu().detach().numpy()*116.0
                print(predict_gender, predict_age)

                # 绘制
                cv.putText(frame, ("gender: %s, age:%d"%(gender, int(predict_age[0][0]))), (int(left), int(top)-15), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                # cv.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                c = cv.waitKey(10)
                if c == 27:
                    break
                cv.imshow("face detection + landmark", frame)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    video_landmark_demo()