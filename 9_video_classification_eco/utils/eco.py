# 第9章 動画分類（ECO：Efficient 3DCNN）
# 9.2、9.3	ECOの2D Netモジュールと3D Netモジュールの実装

# 必要なパッケージのimport
import torch
from torch import nn


class BasicConv(nn.Module):
    '''ECOの2D Netモジュールの最初のモジュール'''

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_3x3_reduce = nn.Conv2d(
            64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(64, 192, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(
            192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

    def forward(self, x):
        out = self.conv1_7x7_s2(x)
        out = self.conv1_7x7_s2_bn(out)
        out = self.conv1_relu_7x7(out)
        out = self.pool1_3x3_s2(out)
        out = self.conv2_3x3_reduce(out)
        out = self.conv2_3x3_reduce_bn(out)
        out = self.conv2_relu_3x3_reduce(out)
        out = self.conv2_3x3(out)
        out = self.conv2_3x3_bn(out)
        out = self.conv2_relu_3x3(out)
        out = self.pool2_3x3_s2(out)
        return out


class InceptionA(nn.Module):
    '''InceptionA'''

    def __init__(self):
        super(InceptionA, self).__init__()

        self.inception_3a_1x1 = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace=True)

        self.inception_3a_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.inception_3a_3x3 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace=True)

        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            192, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        self.inception_3a_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace=True)
        self.inception_3a_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace=True)

        self.inception_3a_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3a_pool_proj = nn.Conv2d(
            192, 32, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace=True)

    def forward(self, x):

        out1 = self.inception_3a_1x1(x)
        out1 = self.inception_3a_1x1_bn(out1)
        out1 = self.inception_3a_relu_1x1(out1)

        out2 = self.inception_3a_3x3_reduce(x)
        out2 = self.inception_3a_3x3_reduce_bn(out2)
        out2 = self.inception_3a_relu_3x3_reduce(out2)
        out2 = self.inception_3a_3x3(out2)
        out2 = self.inception_3a_3x3_bn(out2)
        out2 = self.inception_3a_relu_3x3(out2)

        out3 = self.inception_3a_double_3x3_reduce(x)
        out3 = self.inception_3a_double_3x3_reduce_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_reduce(out3)
        out3 = self.inception_3a_double_3x3_1(out3)
        out3 = self.inception_3a_double_3x3_1_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_1(out3)
        out3 = self.inception_3a_double_3x3_2(out3)
        out3 = self.inception_3a_double_3x3_2_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_2(out3)

        out4 = self.inception_3a_pool(x)
        out4 = self.inception_3a_pool_proj(out4)
        out4 = self.inception_3a_pool_proj_bn(out4)
        out4 = self.inception_3a_relu_pool_proj(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    '''InceptionB'''

    def __init__(self):
        super(InceptionB, self).__init__()
        
        self.inception_3b_1x1 = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace=True)

        self.inception_3b_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace=True)
        self.inception_3b_3x3 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace=True)

        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        self.inception_3b_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace=True)
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace=True)

        self.inception_3b_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3b_pool_proj = nn.Conv2d(
            256, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out1 = self.inception_3b_1x1(x)
        out1 = self.inception_3b_1x1_bn(out1)
        out1 = self.inception_3b_relu_1x1(out1)

        out2 = self.inception_3b_3x3_reduce(x)
        out2 = self.inception_3b_3x3_reduce_bn(out2)
        out2 = self.inception_3b_relu_3x3_reduce(out2)
        out2 = self.inception_3b_3x3(out2)
        out2 = self.inception_3b_3x3_bn(out2)
        out2 = self.inception_3b_relu_3x3(out2)

        out3 = self.inception_3b_double_3x3_reduce(x)
        out3 = self.inception_3b_double_3x3_reduce_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_reduce(out3)
        out3 = self.inception_3b_double_3x3_1(out3)
        out3 = self.inception_3b_double_3x3_1_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_1(out3)
        out3 = self.inception_3b_double_3x3_2(out3)
        out3 = self.inception_3b_double_3x3_2_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_2(out3)

        out4 = self.inception_3b_pool(x)
        out4 = self.inception_3b_pool_proj(out4)
        out4 = self.inception_3b_pool_proj_bn(out4)
        out4 = self.inception_3b_relu_pool_proj(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    '''InceptionC'''

    def __init__(self):
        super(InceptionC, self).__init__()

        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            320, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        self.inception_3c_double_3x3_1 = nn.Conv2d(
            64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.inception_3c_double_3x3_reduce(x)
        out = self.inception_3c_double_3x3_reduce_bn(out)
        out = self.inception_3c_relu_double_3x3_reduce(out)
        out = self.inception_3c_double_3x3_1(out)
        out = self.inception_3c_double_3x3_1_bn(out)
        out = self.inception_3c_relu_double_3x3_1(out)

        return out


class ECO_2D(nn.Module):
    def __init__(self):
        super(ECO_2D, self).__init__()

        # BasicConvモジュール
        self.basic_conv = BasicConv()

        # Inceptionモジュール
        self.inception_a = InceptionA()
        self.inception_b = InceptionB()
        self.inception_c = InceptionC()

    def forward(self, x):
        '''
        入力xのサイズtorch.Size([batch_num, 3, 224, 224]))
        '''
        out = self.basic_conv(x)
        out = self.inception_a(out)
        out = self.inception_b(out)
        out = self.inception_c(out)

        return out


class Resnet_3D_3(nn.Module):
    '''Resnet_3D_3'''

    def __init__(self):
        super(Resnet_3D_3, self).__init__()
        
        self.res3a_2 = nn.Conv3d(96, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3a_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3b_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.res3a_2(x)
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_relu(out)
        out = self.res3b_2(out)

        out += residual

        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out


class Resnet_3D_4(nn.Module):
    '''Resnet_3D_4'''

    def __init__(self):
        super(Resnet_3D_4, self).__init__()

        self.res4a_1 = nn.Conv3d(128, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res4a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_relu = nn.ReLU(inplace=True)
        self.res4a_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res4a_down = nn.Conv3d(128, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.res4a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_relu = nn.ReLU(inplace=True)
        
        self.res4b_1 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res4b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res4b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res4a_down(x)

        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_relu(out)

        out = self.res4a_2(out)

        out += residual

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_relu(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_relu(out)

        out = self.res4b_2(out)

        out += residual2

        out = self.res4b_bn(out)
        out = self.res4b_relu(out)

        return out


class Resnet_3D_5(nn.Module):
    '''Resnet_3D_5'''

    def __init__(self):
        super(Resnet_3D_5, self).__init__()
        
        self.res5a_1 = nn.Conv3d(256, 512, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res5a_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_1_relu = nn.ReLU(inplace=True)
        self.res5a_2 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5a_down = nn.Conv3d(256, 512, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.res5a_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_relu = nn.ReLU(inplace=True)
        
        self.res5b_1 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res5b_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_1_relu = nn.ReLU(inplace=True)
        self.res5b_2 = nn.Conv3d(512, 512, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5b_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res5a_down(x)

        out = self.res5a_1(x)
        out = self.res5a_1_bn(out)
        out = self.res5a_1_relu(out)

        out = self.res5a_2(out)

        out += residual  # res5a

        residual2 = out

        out = self.res5a_bn(out)
        out = self.res5a_relu(out)

        out = self.res5b_1(out)

        out = self.res5b_1_bn(out)
        out = self.res5b_1_relu(out)

        out = self.res5b_2(out)

        out += residual2  # res5b

        out = self.res5b_bn(out)
        out = self.res5b_relu(out)

        return out


class ECO_3D(nn.Module):
    def __init__(self):
        super(ECO_3D, self).__init__()

        # 3D_Resnetジュール
        self.res_3d_3 = Resnet_3D_3()
        self.res_3d_4 = Resnet_3D_4()
        self.res_3d_5 = Resnet_3D_5()

        # Global Average Pooling
        self.global_pool = nn.AvgPool3d(
            kernel_size=(4, 7, 7), stride=1, padding=0)

    def forward(self, x):
        '''
        入力xのサイズtorch.Size([batch_num,frames, 96, 28, 28]))
        '''
        out = torch.transpose(x, 1, 2)  # テンソルの順番入れ替え
        out = self.res_3d_3(out)
        out = self.res_3d_4(out)
        out = self.res_3d_5(out)
        out = self.global_pool(out)
        
        # テンソルサイズを変更
        # torch.Size([batch_num, 512, 1, 1, 1])からtorch.Size([batch_num, 512])へ
        out =out.view(out.size()[0], out.size()[1])
        
        return out

