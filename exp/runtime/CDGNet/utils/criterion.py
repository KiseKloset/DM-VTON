import torch.nn as nn
import torch
import numpy as np
import utils.lovasz_losses as L
from torch.nn import functional as F
from torch.nn import Parameter
from .loss import OhemCrossEntropy2d
from dataset.target_generation import generate_edge


# class ConsistencyLoss(nn.Module):
#     def __init__(self, ignore_index=255):
#         super(ConsistencyLoss, self).__init__()
#         self.ignore_index=ignore_index

#     def forward(self, parsing, edge, label):
#         parsing_pre = torch.argmax(parsing, dim=1)
#         parsing_pre[label==self.ignore_index]=self.ignore_index
#         generated_edge = generate_edge(parsing_pre)
#         edge_pre = torch.argmax(edge, dim=1)
#         v_generate_edge = generated_edge[label!=255]
#         v_edge_pre = edge_pre[label!=255]
#         v_edge_pre = v_edge_pre.type(torch.cuda.FloatTensor)
#         positive_union = (v_generate_edge==1)&(v_edge_pre==1) # only the positive values count
#         return F.smooth_l1_loss(v_generate_edge[positive_union].squeeze(0), v_edge_pre[positive_union].squeeze(0))

class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.ConsEdge = ConsistencyLoss(ignore_index=ignore_index)
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)  
        # self.l2Loss = torch.nn.MSELoss(reduction='mean')
        self.l2loss = torch.nn.MSELoss()
           
    def parsing_loss(self, preds, target, hwgt ):
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        loss = 0      

        # loss for parsing
        pws = [0.4,1,1,1]
        preds_parsing = preds[0]
        ind = 0
        tmpLoss = 0
        if isinstance(preds_parsing, list):
            for pred_parsing in preds_parsing:
                scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                tmpLoss = self.criterion(scale_pred, target[0])
                scale_pred = F.softmax( scale_pred, dim = 1 )
                tmpLoss += L.lovasz_softmax( scale_pred, target[0], ignore = self.ignore_index )          
                tmpLoss *= pws[ind]
                loss += tmpLoss
                ind+=1
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target[0])
            # scale_pred = F.softmax( scale_pred, dim = 1 )
            # loss += L.lovasz_softmax( scale_pred, target[0], ignore = self.ignore_index )

        # loss for edge
        tmpLoss = 0
        preds_edge = preds[1]
        if isinstance(preds_edge, list):
            for pred_edge in preds_edge:
                scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                           mode='bilinear', align_corners=True)
                tmpLoss += F.cross_entropy(scale_pred, target[1],
                                        weights.cuda(), ignore_index=self.ignore_index)
        else:
            scale_pred = F.interpolate(input=preds_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            tmpLoss += F.cross_entropy(scale_pred, target[1],
                                    weights.cuda(), ignore_index=self.ignore_index)
        loss += tmpLoss       
        # loss for height and width attention
         #loss for hwattention        
        hwLoss = 0
        hgt = hwgt[0]
        wgt = hwgt[1]
        n,c,h = hgt.size()
        w = wgt.size()[2]
        hpred = preds[2][0]                         #fea_h...
        scale_hpred = hpred.unsqueeze(3)            #n,c,h,1
        scale_hpred = F.interpolate(input=scale_hpred, size=(h,1),mode='bilinear', align_corners=True)
        scale_hpred = scale_hpred.squeeze(3)        #n,c,h
        # hgt = hgt[:,1:,:]
        # scale_hpred=scale_hpred[:,1:,:]
        hloss = torch.mean( ( hgt - scale_hpred ) * ( hgt - scale_hpred ) )
        wpred = preds[2][1]                         #fea_w...
        scale_wpred = wpred.unsqueeze(2)            #n,c,1,w
        scale_wpred = F.interpolate(input=scale_wpred, size=(1,w),mode='bilinear', align_corners=True)
        scale_wpred = scale_wpred.squeeze(2)        #n,c,w    
        # wgt=wgt[:,1:,:]   
        # scale_wpred = scale_wpred[:,1:,:]
        wloss = torch.mean( ( wgt - scale_wpred ) * ( wgt - scale_wpred ) ) 
        hwLoss =  ( hloss + wloss ) * 45
        loss += hwLoss               
        return loss

    def forward(self, preds, target, hwgt ):
          
        loss = self.parsing_loss(preds, target, hwgt  ) 
        return loss
    