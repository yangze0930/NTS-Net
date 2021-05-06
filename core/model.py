from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM, NUM_CLASS


# Navigator
class ProposalNet(nn.Module): 
    def __init__(self):
        super(ProposalNet, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, NUM_CLASS)
        self.proposal_net = ProposalNet()
        self.topN = topN # proposal number
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), NUM_CLASS)
        self.partcls_net = nn.Linear(512 * 4, NUM_CLASS)
        #其中每行的4个值[公式]表示矩形左上角和右下角点坐标。9个矩形共有3种形状，3种长宽比
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        # RPN 的输入为 backbone (VGG16, ResNet, etc) 的输出（简称 feature maps）。
        # pretrained_model 是resnet50

        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        """
        RPN 包括以下部分：
        生成 anchor boxes
        判断每个 anchor box 为 foreground(包含物体) 或者 background(背景) ，二分类
        边界框回归(bounding box regression) 对 anchor box 进行微调，使得 positive anchor 和真实框(Ground Truth Box)更加接近
        """
        # imformation score for Navigator
        rpn_score = self.proposal_net(rpn_feature.detach())

        # all candidates img box (proposal number)
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        # batch size 内所有照片的top proposal number 的index,probability
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        # 数据进行下采样，抽取重要信息，减少计算量
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        # fed into feature extractor to generate those K regions’ feature vector, each with length 2048. 
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        # TOP K candidates's feature
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        # 全连接层，输出200
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        # Teacher's output: confidence
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        # top_n = Top K
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]

# Teacher 的 loss function
def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)

# Navigator 的 loss function
def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size
