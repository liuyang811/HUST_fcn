import torch
import os
import torch.nn as nn
from torch import utils
import numpy as np
import cv2
from torch.utils.data import TensorDataset,DataLoader,random_split,Dataset
from torchvision import transforms,models
import visdom
from mIOU import IOUMetric


class FCN32(nn.Module):
    def __init__(self,pretrained_net):
        super(FCN32, self).__init__()
        # 基础网络
        self.pretrained_net=pretrained_net
        # 全卷积
        self.model=nn.Sequential(

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self,x):
        x=self.pretrained_net(x)
        x=self.model(x)
        return x

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class hhhorse(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('weizmann_horse_db/horse'))

    def __getitem__(self, idx):
        img_name = os.listdir('weizmann_horse_db/horse')[idx]
        imgA = cv2.imread('weizmann_horse_db/horse/' + img_name)
        imgA = cv2.resize(imgA, (224, 224))
        imgB = cv2.imread('weizmann_horse_db/mask/' + img_name, 0)
        imgB = cv2.resize(imgB, (224, 224))
        # imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB

# 读取图片时进行的预处理操作
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                     [0.229,0.224,0.225])  ##图像标准化处理
])

horse=hhhorse(transform)

train_length=int(0.85*len(horse))
test_length=len(horse)-train_length

train_data,test_data=torch.utils.data.random_split(dataset=horse,lengths=[train_length,test_length])
train_data_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_data_loader=DataLoader(test_data,batch_size=4,shuffle=True)



# 网络初始化
vgg16 = models.vgg16(pretrained=True)
net = nn.Sequential(*list(vgg16.children())[:-1])
fcn32=FCN32(net)

# 一些参数的定义及gpu、cpu的选择，可视化
epoch=30
learning_rate=1e-3
optimizer=torch.optim.Adam(fcn32.parameters(),lr=learning_rate,weight_decay=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fcn32 = fcn32.to(device)
criterion = nn.BCELoss().to(device)
vis=visdom.Visdom()

all_train_loss=[]
all_test_loss=[]
# 训练,i为计数，data为数据
for i in range(epoch):
    print(print("----------第{}轮训练开始-----------".format(i+1)))
    train_loss_epoch=0

    for j,(img,target) in enumerate(train_data_loader):
        img=img.to(device)
        target=target.to(device)
        outputs=fcn32(img)
        outputs=torch.sigmoid(outputs)

        loss=criterion(outputs,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_train_loss.append(loss.item())
        train_loss_epoch=train_loss_epoch+loss.item()


        outputs_np = outputs.cpu().detach().numpy().copy()
        outputs_np = np.argmin(outputs_np, axis=1)
        target_np = target.cpu().detach().numpy().copy()
        target_np = np.argmin(target_np, axis=1)

        ioumetric = IOUMetric(2)
        miou = ioumetric.evaluate(outputs_np, target_np)
        print("mIOU", miou)


        if np.mod(j, 15) == 0:
            print('epoch {}, {}/{},train loss is {}'.format(i, j, len(train_data_loader), loss.item()))
            # vis.close()
            vis.images(outputs_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
            vis.images(target_np[:, None, :, :], win='train_label', opts=dict(title='label'))
            vis.line(all_train_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

        if i%5==0:
            torch.save(fcn32.state_dict(),"FCN32{}.pth".format(i))
    print("loss:",train_loss_epoch)

    fcn32.eval()
    test_loss=0
    with torch.no_grad():
        for ii, (img, target) in enumerate(test_data_loader):
            img = img.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = fcn32(img)
            outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, target)
            test_loss=test_loss+loss.item()
            all_test_loss.append(loss.item())
            print("test loss:", test_loss)

            outputs_np = outputs.cpu().detach().numpy().copy()
            outputs_np = np.argmin(outputs_np, axis=1)
            target_np = target.cpu().detach().numpy().copy()
            target_np = np.argmin(target_np, axis=1)
            if np.mod(ii, 15) == 0:
                print(r'Testing... Open http://localhost:8097/ to see test result.')
                # vis.close()
                vis.images(outputs_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                vis.images(target_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                vis.line(all_test_loss, win='test_iter_loss', opts=dict(title='test iter loss'))



