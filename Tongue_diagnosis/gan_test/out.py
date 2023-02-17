import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

#结构：参数配置，网络搭建，数据载入，训练，验证

#--------------------------------1.参数配置---------------------------------#
class Config():

    result_save_path = 'out256/' #生成图像保存的路径
    d_net_path = './1/snapshots2/dnet.pth' #判别网络权重文件保存的路径
    g_net_path = './1/snapshots2/gnet.pth' #生成网络权重文件保存的路径
    img_path = 'dataset/' #源图像文件路径

    img_size = 256 #图像裁剪尺寸 #256
    batch_size = 64 #批数量
    max_epoch = 300 #循环轮次
    noise_dim = 100 #初始噪声的通道维度
    feats_channel = 64 #中间特征图维度

opt = Config() #类实例化

if not os.path.exists('out256'):
    os.mkdir('out256')  #生成results文件夹
if not os.path.exists('snapshots1'):
    os.mkdir('snapshots1') #生成snapshots文件夹

#---------------------------------2.生成网络设计----------------------------------#
class Gnet(nn.Module):
    def __init__(self, opt):
        super(Gnet, self).__init__()
        self.feats = opt.feats_channel
        self.generate = nn.Sequential(

           #input = (n, c, h, w = 256, 100, 1, 1)
            nn.ConvTranspose2d(in_channels=opt.noise_dim, out_channels=self.feats * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.ReLU(inplace=True),
            # deconv = (input - 1 ) * stride + k -2* padding = (1- 1)*1  +4-0 = 4
            #output = (256, 800 ,4, 4)

            nn.ConvTranspose2d(in_channels=self.feats * 8, out_channels=self.feats * 4, kernel_size=4, stride=4, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (4-1)*2 + 4-2 = 8

            nn.ConvTranspose2d(in_channels=self.feats * 4, out_channels=self.feats * 2, kernel_size=4, stride=4, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (8-1)*2 + 4-2 = 16


            nn.ConvTranspose2d(in_channels=self.feats * 2, out_channels=self.feats, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (16-1)*2 + 4-2 = 32

            nn.ConvTranspose2d(in_channels=self.feats, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh(),
            # decon = (input - 1)*stride + k - 2*padding = (32-1)*3 + 5-2 = 96
            #output = (n, c, h, w = 256, 3, 96, 96)

    )

    def forward(self, x):
        return self.generate(x)

def Out(epoch):
    g_net = Gnet(opt)  # 类实例化

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 调用cpu或者cuda

    g_net.to(device)

    optimize_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))


    criterions = nn.BCELoss().to(device)  # BCEloss， 求二分类概率


    noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

    # 用于测试
    test_noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

    try:
        g_net.load_state_dict(torch.load(opt.g_net_path))  # 载入权重文件
        print('加载成功')
    except:
        print('加载失败')

    optimize_g.zero_grad()  # 生成网络优化器梯度清零
    noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1))  # 复制一份噪声数据，区别于训练判别网络

    # fake_image = g_net(noises) #生成图像
    # output = d_net(fake_image) #生成图像输入判别网络
    # g_loss = criterions(output, true_labels) #计算生成图像是真的概率
    # g_loss.backward() #生成网络loss反向传播

    vid_fake_image = g_net(test_noises)  # 验证，随机生成256个噪声
    torchvision.utils.save_image(vid_fake_image.data[1], "%s/%s.png" % (opt.result_save_path, epoch),
                                 normalize=True)  # 保存前16幅图像
    print('epoch:', epoch)  # loss可视化
    optimize_g.step()  # 生成网络优化器梯度更新

if __name__ == '__main__':
    for epoch in range(opt.max_epoch):
        Out(epoch)