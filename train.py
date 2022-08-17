import torch
import os
import torch.nn as nn
from torch import utils
import numpy as np
import cv2
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torchvision import transforms,models
import visdom
from mIOU import IOUMetric
from torchvision.models.vgg import VGG
from FCN8 import FCN8
from FCN16 import FCN16
from FCN32 import FCN32

#  构建vgg网络
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# 改变mask维度，使其与预测输出size相同
def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

'''
读入数据，并构造迭代器
'''
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

# 训练与测试
def train(fcn):
    # 一些参数的定义及gpu、cpu的选择，可视化
    epoch=30
    learning_rate=1e-3
    optimizer=torch.optim.Adam(fcn.parameters(),lr=learning_rate,weight_decay=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcn = fcn.to(device)
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
            outputs=fcn(img)
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

            if np.mod(j, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(i, j, len(train_data_loader), loss.item()))
                # vis.close()
                vis.images(outputs_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                vis.images(target_np[:, None, :, :], win='train_label', opts=dict(title='train_label'))
                vis.line(all_train_loss, win='train_epoch_loss', opts=dict(title='train epoch loss'))

            ioumetric = IOUMetric(2)
            miou = ioumetric.evaluate(outputs_np, target_np)
            print("mIOU", miou)

            '''
            每五个epoch保存一次模型
            保存方式为 只保存模型参数
            保存目录为根文件夹
            '''
            if i%5==0:
                if choice==8:
                    torch.save(fcn.state_dict(),"FCN8{}.pth".format(i))
                elif choice==16:
                    torch.save(fcn.state_dict(), "FCN16{}.pth".format(i))
                elif choice==32:
                    torch.save(fcn.state_dict(), "FCN32{}.pth".format(i))

        print("loss:",train_loss_epoch)

        # 测试
        fcn.eval()
        test_loss=0
        with torch.no_grad():
            for ii, (img, target) in enumerate(test_data_loader):
                img = img.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = fcn(img)
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
                    vis.images(target_np[:, None, :, :], win='test_label', opts=dict(title='test_label'))
                    vis.line(all_test_loss, win='test_epoch_loss', opts=dict(title='test epoch loss'))

                ioumetric = IOUMetric(2)
                miou = ioumetric.evaluate(outputs_np, target_np)
                print("测试集上的mIOU", miou)

if __name__ == '__main__':
    # 网络初始化
    choice=input("FCN8s网络请输入8，FCN16s网络请输入16，FCN32s网络请输入32：")

    # 网络选择
    if choice=='8':
        print("开始运行FCN8s网络\n")
        vgg16 = models.vgg16(pretrained=True)
        net = vgg_model = VGGNet(requires_grad=True, show_params=False)
        fcn=FCN8(net, 2)
        train(fcn)
    elif choice=='16':
        print("开始运行FCN16s网络\n")
        vgg16 = models.vgg16(pretrained=True)
        net = vgg_model = VGGNet(requires_grad=True, show_params=False)
        fcn=FCN16(net,2)
        train(fcn)
    elif choice=='32':
        print("开始运行FCN32s网络\n")
        vgg16 = models.vgg16(pretrained=True)
        net = nn.Sequential(*list(vgg16.children())[:-1])
        fcn = FCN32(net)
        train(fcn)
    else:
        print("输入错误，请重新运行")
