import torch
import torch.nn as nn
from example_nn.net import PConv2d


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # 64 x 128
        self.conv1 = PConv2d(2, 64, 3, 2, 1)
        # 32 x 64
        self.conv2 = PConv2d(64, 128, 3, 2, 1)
        # 16 x 32
        self.conv3 = PConv2d(128, 256, 3, 2, 1)
        # 8 x 16
        self.conv4 = PConv2d(256, 512, 3, 2, 1)
        # 4 x 8
        self.conv5 = PConv2d(512, 512, 3, 2, 1)
        # 2 x 4

        #keep it the same
        self.conv6 = PConv2d(512, 512, 3, 1, 1)
        self.conv7 = PConv2d(768, 256, 3, 1, 1)
        self.conv8 = PConv2d(384, 128, 3, 1, 1)
        self.conv9 = PConv2d(192, 64, 3, 1, 1)
        self.conv10 = PConv2d(66, 2, 3, 1, 1)

        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.batchNorm4 = nn.BatchNorm2d(512)

        self.batchNorm6 = nn.BatchNorm2d(512)
        self.batchNorm7 = nn.BatchNorm2d(256)
        self.batchNorm8 = nn.BatchNorm2d(128)
        self.batchNorm9 = nn.BatchNorm2d(64)

        self.Relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input, input_mask):
        #input: 64 x 128
        #print(input.shape)
        conv1_output, conv1_output_mask = self.conv1(input, input_mask)
        batchNorm1_output = self.batchNorm1(conv1_output)
        #print(batchNorm1_output.shape)
        relu1_output =  self.Relu(batchNorm1_output)
        # 32x64
        conv2_output, conv2_output_mask =  self.conv2(relu1_output, conv1_output_mask)
        batchNorm2_output =  self.batchNorm2(conv2_output)
        #print(conv2_output.shape)
        relu2_output =  self.Relu(batchNorm2_output)
        #16x32
        conv3_output, conv3_output_mask =  self.conv3(relu2_output, conv2_output_mask)
        batchNorm3_output =  self.batchNorm3(conv3_output)
        #print(conv3_output.shape)
        relu3_output =  self.Relu(batchNorm3_output)
        # 8x16
        conv4_output, conv4_output_mask =  self.conv4(relu3_output, conv3_output_mask)
        batchNorm4_output =  self.batchNorm4(conv4_output)
        #print(conv4_output.shape)
        relu4_output =  self.Relu(batchNorm4_output)
        # 4x8
        conv5_output, conv5_output_mask =  self.conv5(relu4_output, conv4_output_mask)
        batchNorm5_output =  self.batchNorm4(conv5_output)
        #print(conv5_output.shape)
        relu5_output =  self.Relu(batchNorm5_output)
        # 2x4

        upsample1 =  self.upsample(relu5_output)
        upsample1_mask =  self.upsample(conv5_output_mask)
        conv6_output, conv6_output_mask =  self.conv6(upsample1, upsample1_mask)
        batchNorm6_output =  self.batchNorm4(conv6_output)
        leakyRelu1 =  self.leakyRelu(batchNorm6_output)
        #print(conv6_output.shape)
        # 4x8

        upsample2 =  self.upsample(leakyRelu1)
        upsample2_mask =  self.upsample(conv6_output_mask)
        concat2 = torch.cat((upsample2, relu3_output), 1)
        concat2_mask = torch.cat((upsample2_mask, conv3_output_mask), 1)
        conv7_output, conv7_output_mask =  self.conv7(concat2, concat2_mask)
        batchNorm7_output =  self.batchNorm7(conv7_output)
        leakyRelu2 =  self.leakyRelu(batchNorm7_output)
        #print(conv7_output.shape)
        #8x16

        upsample3 =  self.upsample(leakyRelu2)
        upsample3_mask =  self.upsample(conv7_output_mask)
        concat3 = torch.cat((upsample3, relu2_output), 1)
        concat3_mask = torch.cat((upsample3_mask, conv2_output_mask), 1)
        conv8_output, conv8_output_mask =  self.conv8(concat3, concat3_mask)
        batchNorm8_output =  self.batchNorm8(conv8_output)
        leakyRelu3 =  self.leakyRelu(batchNorm8_output)
        #print(conv8_output.shape)
        #16x32

        upsample4 =  self.upsample(leakyRelu3)
        upsample4_mask =  self.upsample(conv8_output_mask)
        concat4 = torch.cat((upsample4, relu1_output), 1)
        concat4_mask = torch.cat((upsample4_mask, conv1_output_mask), 1)
        conv9_output, conv9_output_mask =  self.conv9(concat4, concat4_mask)
        batchNorm9_output =  self.batchNorm9(conv9_output)
        leakyRelu4 =  self.leakyRelu(batchNorm9_output)
        #print(conv9_output.shape)
        #32x64

        upsample5 =  self.upsample(leakyRelu4)
        upsample5_mask =  self.upsample(conv9_output_mask)
        concat5 = torch.cat((upsample5, input), 1)
        concat5_mask = torch.cat((upsample5_mask, input_mask), 1)
        conv10_output, conv10_output_mask =  self.conv10(concat5, concat5_mask)
        leakyRelu5 =  self.leakyRelu(conv10_output)
        #print(conv10_output.shape)
        return leakyRelu5
        #64x128

