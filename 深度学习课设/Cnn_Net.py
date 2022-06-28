import torch
import torch.nn as nn

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
            # nn.Sigmoid()
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

# model = MyNet()
# print (model)



