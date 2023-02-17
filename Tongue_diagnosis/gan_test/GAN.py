import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#结构：参数配置，网络搭建，数据载入，训练，验证

#--------------------------------1.参数配置---------------------------------#
class Config():

    result_save_path = 'results1/' #生成图像保存的路径
    d_net_path = 'snapshots1/dnet.pth' #判别网络权重文件保存的路径
    g_net_path = 'snapshots1/gnet.pth' #生成网络权重文件保存的路径
    img_path = '../../data/tongue_crack' #源图像文件路径

    img_size = 96 #图像裁剪尺寸 #256
    batch_size = 128 #批数量
    max_epoch = 300 #循环轮次
    noise_dim = 100 #初始噪声的通道维度
    feats_channel = 64 #中间特征图维度

opt = Config() #类实例化
tb_writer = SummaryWriter()
if not os.path.exists('results1'):
    os.mkdir('results1')  #生成results文件夹
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

            nn.ConvTranspose2d(in_channels=self.feats * 8, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (4-1)*2 + 4-2 = 8

            nn.ConvTranspose2d(in_channels=self.feats * 4, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (8-1)*2 + 4-2 = 16


            nn.ConvTranspose2d(in_channels=self.feats * 2, out_channels=self.feats, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # decon = (input - 1)*stride + k - 2*padding = (16-1)*2 + 4-2 = 32

            nn.ConvTranspose2d(in_channels=self.feats, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),

            nn.Tanh(),
            # decon = (input - 1)*stride + k - 2*padding = (32-1)*3 + 5-2 = 96
            #output = (n, c, h, w = 256, 3, 96, 96)

    )

    def forward(self, x):
        return self.generate(x)

#----------------------------------3.判别网络设计-----------------------------#
class Dnet(nn.Module):
    def __init__(self, opt):
        super(Dnet, self).__init__()
        self.feats = opt.feats_channel
        self.discrim = nn.Sequential(

            #input = （n, c, h, w = 256, 3, 96, 96)

            nn.Conv2d(in_channels=3, out_channels= self.feats, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),
            #con = (input - k + 2 * padding ) / stride +  1 = (256 - 5 + 2) / 3 + 1 = 128



            nn.Conv2d(in_channels= self.feats, out_channels= self.feats * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.feats* 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels= self.feats * 2, out_channels= self.feats * 4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.LeakyReLU(0.2, True),


            nn.Conv2d(in_channels= self.feats * 4, out_channels= self.feats * 8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.feats *8),
            nn.LeakyReLU(0.2, True),


            nn.Conv2d(in_channels= self.feats * 8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            nn.Sigmoid()

            #output = ( n, c, h, w = 256, 1, 1, 1)
        )

    def forward(self, x):
        return self.discrim(x).view(-1) #拉成一列

g_net, d_net = Gnet(opt), Dnet(opt) #类实例化

#---------------------------------4.数据载入准备---------------------------------#
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.img_size), #resize图像尺寸
    torchvision.transforms.CenterCrop(opt.img_size), #中心裁剪图像尺寸
    torchvision.transforms.ToTensor() #array变tensor.float(), 适应torch框架的数据格式
    # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder(root=opt.img_path, transform=transforms)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size, #批数量
    num_workers = 0, #多线程，一般设置为0
    drop_last = True #batch不能整除图片总数，drop掉最后一个批次
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #调用cpu或者cuda

g_net.to(device)
d_net.to(device)

optimize_g = torch.optim.Adam(g_net.parameters(), lr= 2e-4, betas=(0.5, 0.999))
optimize_d = torch.optim.Adam(d_net.parameters(), lr= 2e-4, betas=(0.5, 0.999))
# optimize_d = torch.optim.SGD(d_net.parameters(), lr= 2e-4)

criterions = nn.BCELoss().to(device) #BCEloss， 求二分类概率

# 定义标签，并且开始注入生成器的输入noise
true_labels = torch.ones(opt.batch_size).to(device) #正确样本标签为1
fake_labels = torch.zeros(opt.batch_size).to(device) #错误样本标签为0

# 生成满足N(1,1)标准正态分布，100维，256个数的随机噪声
noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

# 用于测试
test_noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device)

#-----------------------------------5.开始训练----------------------------------#
try:
    g_net.load_state_dict(torch.load(opt.g_net_path)) #载入权重文件
    d_net.load_state_dict(torch.load(opt.d_net_path))
    print('加载成功，继续训练')
except:
    print('加载失败，重新训练')

for epoch in range(opt.max_epoch):  #总循环轮次

    for itertion, (img, _) in tqdm((enumerate(dataloader))): #遍历数据集
        real_img = img.to(device)

        if itertion % 5 == 0: #判别网络训练5次，生成网络训练1次

            optimize_d.zero_grad() #判别网络梯度清零

            output = d_net(real_img) #真实数据输入判别网络
            d_real_loss = criterions(output, true_labels) #希望判别器将真实图像判别为正样本，标签为1
            fake_image = g_net(noises.detach()).detach() #截断梯度，防止判别器的梯度传入生成器
            output = d_net(fake_image) #生成数据输入判别网络
            d_fake_loss = criterions(output, fake_labels) #希望判别器将生成图像判别为负样本，标签为0
            d_loss = (d_fake_loss + d_real_loss) / 2 #loss融合计算

            d_loss.backward() #判别网络loss反向传播
            optimize_d.step() #判别网路优化器梯度更新

        if itertion % 1 == 0:
            optimize_g.zero_grad() #生成网络优化器梯度清零
            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1)) #复制一份噪声数据，区别于训练判别网络

            fake_image = g_net(noises) #生成图像
            output = d_net(fake_image) #生成图像输入判别网络
            g_loss = criterions(output, true_labels) #计算生成图像是真的概率

            g_loss.backward() #生成网络loss反向传播
            optimize_g.step() #生成网络优化器梯度更新

    vid_fake_image = g_net(test_noises) #验证，随机生成256个噪声
    torchvision.utils.save_image(vid_fake_image.data[:64], "%s/%s.png" % (opt.result_save_path, epoch), normalize=True) #保存前16幅图像
    torch.save(d_net.state_dict(),  opt.d_net_path) #保存判别网络权重文件
    torch.save(g_net.state_dict(),  opt.g_net_path) #保存生成网络权重文件
    if epoch % 10 == 0:
        torch.save(d_net.state_dict(), "./weights/d_net_model-{}.pth".format(epoch))
        torch.save(g_net.state_dict(), "./weights/g_net_model-{}.pth".format(epoch))
    tags = ["d_loss", "g_loss", "optimize_d", "optimize_g"]
    tb_writer.add_scalar(tags[0], d_loss.item(), epoch)
    tb_writer.add_scalar(tags[1],  g_loss.item(), epoch)
    tb_writer.add_scalar(tags[2], optimize_d.param_groups[0]["lr"], epoch)
    tb_writer.add_scalar(tags[3], optimize_g.param_groups[0]["lr"], epoch)
    print('epoch:', epoch, '---D-loss:---', d_loss.item(), '---G-loss:---', g_loss.item()) #loss可视化

