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

model = torch.load('age_gender_model2.pt')
train_on_gpu=True

if train_on_gpu:
    model.cuda()

ds = Data_clean.AgeGenderDataset("./data/UTKFace/UTKFace")
num_train_samples = ds.num_of_samples()
bs=16
dataloader = DataLoader(ds,batch_size=bs,shuffle=True)

num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
mse_loss = nn.MSELoss()
cross_loss = nn.CrossEntropyLoss()
index = 0

for epoch in range(num_epochs):
    train_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        images_batch,age_batch,gender_batch = \
            sample_batched['image'],sample_batched['age'],sample_batched['gender']
        if train_on_gpu:
            images_batch,age_batch,gender_batch=\
                images_batch.cuda(),age_batch.cuda(),gender_batch.cuda()

        optimizer.zero_grad()

        m_age_out,m_gender_out=model(images_batch)
        age_batch=age_batch.view(-1,1)
        gender_batch = gender_batch.long()

        loss = mse_loss(m_age_out,age_batch) +\
            cross_loss(m_gender_out,gender_batch)

        loss.backward()
        optimizer.step()

        train_loss +=loss.item()
        if index%100 == 0:
            print('step: {} \t train_loss:{:.6f}'.format(index,loss.item()))
        index +=1

    train_loss = train_loss / num_train_samples

    print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

# save model
model.eval()
torch.save(model, 'age_gender_model3.pt')

