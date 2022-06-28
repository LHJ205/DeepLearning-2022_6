import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import Cnn_Net
import Data_clean

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

train_on_gpu=1
net = torch.load('age_gender_model2.pt')
ds = Data_clean.AgeGenderDataset("./data/train")
num_train_samples = ds.num_of_samples()
bs=1
dataloader = DataLoader(ds,batch_size=bs,shuffle=True)

for i_batch, sample_batched in enumerate(dataloader):
    images_batch, age_batch, gender_batch = \
        sample_batched['image'], sample_batched['age'], sample_batched['gender']
    if train_on_gpu:
        images_batch, age_batch, gender_batch = \
            images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()

    age_, gender_ = net(images_batch)
    predict_gender = torch.max(gender_, 1)[1].cpu().detach().numpy()[0]
    gender = "Male"
    if predict_gender == 1:
        gender = "Female"
    predict_age = age_

    print("预测：年龄{}，性别{}".format(predict_age.item(),gender))
    if gender_batch.item()==0:
        gender_real="Male"
    else:
        gender_real="Female"
    print("实际：年龄{}，性别{}".format(age_batch.item(),gender_real))
    print("------------------------------------------------------------------------")

