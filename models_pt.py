from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from ConvLSTM import ConvLSTM

def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5,
          1e-5, 1e-6, 1e-6, 1e-7, 1e-7]
    return lr[epoch]

class ACLLoss(nn.Module):
    def __init__(self):
        super(ACLLoss, self).__init__()
        return

    def forward(self, y_pred, y_sal, y_fix):
        y_pred = F.normalize(y_pred, p=1, dim=[2, 3])
        y_sal = F.normalize(y_sal, p=1, dim=[2, 3])
        loss_kl = self.kl_divergence(y_sal, y_pred)
        loss_cc = self.correlation_coefficient(y_sal, y_pred)
        loss_nss = self.nss(y_sal, y_fix)
        loss = 10 * loss_kl + loss_cc + loss_nss
        return loss, loss_kl, loss_cc, loss_nss

    def kl_divergence(self, y_sal, y_pred):
        loss = torch.sum(y_sal * torch.log((y_sal + 1e-7) / (y_pred + 1e-7)))
        return loss

    def correlation_coefficient(self, y_sal, y_pred):
        N = y_pred.size()[2] * y_pred.size()[3]
        sum_prod = torch.sum(y_sal * y_pred, dim=[2, 3])
        sum_x = torch.sum(y_sal, dim=[2, 3])
        sum_y = torch.sum(y_pred, dim=[2, 3])
        sum_x_square = torch.sum(y_sal**2, dim=[2, 3]) + 1e-7
        sum_y_square = torch.sum(y_pred**2, dim=[2, 3]) + 1e-7
        num = sum_prod - ((sum_x * sum_y) / N)
        den = torch.sqrt((sum_x_square - sum_x**2 / N) * (sum_y_square - sum_y**2 / N))
        loss = torch.sum(-2 * num / den)  #
        return loss

    def nss(self, y_fix, y_pred):
        y_pred = F.layer_norm(y_pred, normalized_shape=y_pred.size()[2:])
        loss = -torch.sum(((torch.sum(y_fix * y_pred, dim=[2, 3])) / (torch.sum(y_fix, dim=[2, 3]))))
        return loss

class acl_net_pt(nn.Module):

    def __init__(self):
        super(acl_net_pt, self).__init__()
        self.dcn_vgg()
        self.acl_vgg()

    def forward(self, x):
        out = self.dcn_vgg_fwd(x)
        outs = self.acl_vgg_fwd(out)
        return outs

    def dcn_vgg(self):
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.padding_one_side = nn.ConstantPad2d(padding=(1,0,1,0), value=0)
        self.maxpool4 = nn.MaxPool2d((2, 2), stride=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))

    def dcn_vgg_fwd(self, x):
        out = self.conv1_1(x)
        out = F.relu(out)
        out = self.conv1_2(out)
        out = F.relu(out)
        out = self.maxpool1(out)

        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2_2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out)
        out = F.relu(out)
        out = self.conv3_3(out)
        out = F.relu(out)
        out = self.maxpool3(out)

        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4_2(out)
        out = F.relu(out)
        out = self.conv4_3(out)
        out = F.relu(out)
        out = self.padding_one_side(out)
        out = self.maxpool4(out)

        out = self.conv5_1(out)
        out = F.relu(out)
        out = self.conv5_2(out)
        out = F.relu(out)
        out = self.conv5_3(out)
        out = F.relu(out)
        return out

    def acl_vgg(self):
        self.maxpool_atn1_1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_1 = nn.Conv2d(512, 64, 1)
        self.conv_atn1_2 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.maxpool_atn1_2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_3 = nn.Conv2d(128, 64, 1)
        self.conv_atn1_4 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv_atn1_5 = nn.Conv2d(128, 1, 1)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=4)

        self.convLSTM = ConvLSTM(input_channels=512, hidden_channels=[256], kernel_size=3, step=num_frames,
                        effective_step=[4]).cuda()
        self.conv_atn2_1 = nn.Conv2d(256, 1, 1)
        self.upsampling2_1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upsampling2_2 = nn.UpsamplingNearest2d(scale_factor=2)

    def acl_vgg_fwd(self, x):
        outs = x
        attention = self.maxpool_atn1_1(outs)
        attention = F.relu(self.conv_atn1_1(attention))
        attention = F.relu(self.conv_atn1_2(attention))
        attention = self.maxpool_atn1_2(attention)
        attention = F.relu(self.conv_atn1_3(attention))
        attention = F.relu(self.conv_atn1_4(attention))
        attention = torch.sigmoid(self.conv_atn1_5(attention))
        attention = self.upsampling1(attention)

        f_attention = attention.repeat(1, 512, 1, 1)
        m_outs = f_attention * outs
        outs = outs + m_outs

        outs = self.convLSTM(outs.cuda())[0][0]
        outs = torch.sigmoid(self.conv_atn2_1(outs))
        outs = self.upsampling2_1(outs)
        # attention = self.upsampling2_2(attention)
        # return [outs, attention]
        return outs
