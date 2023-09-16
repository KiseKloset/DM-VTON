
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import functools

from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm1d as BatchNorm1d

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

def conv1d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)


class CDGAttention(nn.Module):
    def  __init__(self, feat_in=512, feat_out=256, num_classes=20, size=[384//16,384//16], kernel_size =7 ):
        super(CDGAttention, self).__init__()   
        h,w = size[0],size[1]
        kSize = kernel_size
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.rowpool = nn.AdaptiveAvgPool2d((h,1))
        self.colpool = nn.AdaptiveAvgPool2d((1,w))
        self.conv_hgt1 =conv1d(feat_in,feat_out,3)
        self.conv_hgt2 =conv1d(feat_in,feat_out,3)
        self.conv_hwPred1 = nn.Sequential(
            nn.Conv1d(feat_out,num_classes,3,stride=1,padding=1,bias=True),
            nn.Sigmoid(),   
        )
        self.conv_hwPred2 = nn.Sequential(
            nn.Conv1d(feat_out,num_classes,3,stride=1,padding=1,bias=True),
            nn.Sigmoid(),                                                            
         )         
        self.conv_upDim1 = nn.Sequential(
            nn.Conv1d(feat_out,feat_in,kSize,stride=1,padding=kSize//2,bias=True),  
            nn.Sigmoid(),                                                                              
        )
        self.conv_upDim2 = nn.Sequential( 
            nn.Conv1d(feat_out,feat_in,kSize,stride=1,padding=kSize//2,bias=True),  
            nn.Sigmoid(),                                                                            
        )
        self.cmbFea = conv2d( feat_in*3,feat_in,3)        
    def forward(self,fea):
        n,c,h,w = fea.size()       
        fea_h = self.rowpool(fea).squeeze(3)      #n,c,h
        fea_w = self.colpool(fea).squeeze(2)      #n,c,w
        fea_h = self.conv_hgt1(fea_h)             #n,c,h
        fea_w = self.conv_hgt2(fea_w) 
        #===========================================================               
        fea_hp = self.conv_hwPred1(fea_h)            #n,class_num,h
        fea_wp = self.conv_hwPred2(fea_w)            #n,class_num,w 
        #===========================================================
        fea_h = self.conv_upDim1(fea_h)                    
        fea_w = self.conv_upDim2(fea_w) 
        fea_hup = fea_h.unsqueeze(3)
        fea_wup = fea_w.unsqueeze(2)
        fea_hup = F.interpolate( fea_hup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w
        fea_wup = F.interpolate( fea_wup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w       
        fea_hw = self.beta*fea_wup + self.gamma*fea_hup        
        fea_hw_aug = fea * fea_hw        
        #===============================================================      
        fea = torch.cat([fea, fea_hw_aug, fea_hw], dim = 1 )
        fea = self.cmbFea( fea )       
        return fea, fea_hp, fea_wp
        
class C2CAttention(nn.Module):
    def  __init__(self, in_fea, out_fea, num_class ):
        super(C2CAttention, self).__init__()
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.num_class = num_class
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.bias1 = Parameter( torch.FloatTensor( num_class, num_class ))
        self.bias2 = Parameter( torch.FloatTensor( num_class, num_class ))
        self.convDwn1 = conv2d( in_fea, out_fea, 1 )
        self.convDwn2 = conv2d( in_fea, out_fea, 1 )
        self.convUp1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            conv2d( num_class, out_fea, 1 ),
            nn.Conv2d(out_fea,in_fea,1,stride=1,padding=0,bias=True),                       
        )
        self.toClass = nn.Sequential(
            nn.Conv2d( out_fea, num_class, 1, stride=1, padding = 0, bias = True ),          
        )       
        self.convUp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            conv2d( num_class, out_fea, 1 ),
            nn.Conv2d(out_fea,in_fea,1,stride=1,padding=0,bias=True),                        
        )        
        self.fea_fuse = conv2d( in_fea*2, in_fea, 1 )
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
    def reset_parameters(self):        
        torch.nn.init.xavier_uniform_(self.bias1)  
        torch.nn.init.xavier_uniform_(self.bias2)   
    def forward(self,input_fea):  
        n, c, h, w = input_fea.size()        
        fea_ha = self.convDwn1( input_fea )
        fea_wa = self.convDwn2( input_fea )
        cls_ha = self.toClass( fea_ha )
        cls_ha = F.softmax(cls_ha, dim=1)
        cls_wa = self.toClass( fea_wa )
        cls_wa = F.softmax(cls_wa, dim=1)
        cls_ha = cls_ha.view( n, self.num_class, h*w )
        cls_wa = cls_wa.view( n, self.num_class, h*w )
        cch = F.relu(torch.matmul( cls_ha, cls_ha.transpose( 1, 2 ) ))  #class*class
        cch = cch 
        cch = self.sigmoid( cch ) + self.bias1                           
        ccw = F.relu(torch.matmul( cls_wa, cls_wa. transpose( 1, 2 ) )) #class*class
        ccw = ccw 
        ccw = self.sigmoid( ccw )+ self.bias2                            
        cls_ha = torch.matmul( cls_ha.transpose(1,2), cch.transpose(1,2) )
        cls_ha = cls_ha.transpose( 1,2).contiguous().view( n, self.num_class, h, w )
        cls_wa = torch.matmul( cls_wa.transpose(1,2), ccw.transpose(1,2) )
        cls_wa = cls_wa.transpose(1,2).contiguous().view( n, self.num_class, h, w )        
        fea_ha = self.convUp1( cls_ha )
        fea_wa = self.convUp2( cls_wa )
        fea_hwa = self.gamma*fea_ha + self.beta*fea_wa
        fea_hwa_aug = input_fea * fea_hwa                   #*
        fea_fuse = torch.cat( [fea_hwa_aug, input_fea], dim = 1 )
        fea_fuse = self.fea_fuse( fea_fuse )
        return fea_fuse, cch, ccw  

class StatisticAttention(nn.Module):
    def  __init__(self,fea_in, fea_out, num_classes ):
       super(StatisticAttention, self).__init__()
    #    self.gamma = Parameter(torch.ones(1))
       self.conv_1 = conv2d( fea_in, fea_in//2, 1)      #kernel size 3
       self.conv_2 = conv2d( fea_in//2, num_classes, 3 )
       self.conv_pred = nn.Sequential(
           nn.Conv2d( num_classes, 1, 3, stride=1, padding=1, bias=True),   #kernel size 1
           nn.Sigmoid()
       )
       self.conv_fuse = conv2d( fea_in * 2, fea_out, 3 )
    def forward(self,fea):
        fea_att = self.conv_1( fea )
        fea_cls = self.conv_2( fea_att )
        fea_stat = self.conv_pred( fea_cls )     
        fea_aug = fea * ( 1 - fea_stat ) 
        fea_fuse = torch.cat( [fea, fea_aug], dim = 1 )
        fea_res = self.conv_fuse( fea_fuse )
        return fea_res, fea_stat        
        
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 7, 11), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
        
class PCM(Module):
    def __init__(self, feat_channels=[256,1024]):
        super().__init__()
        feat1, feat2 = feat_channels
        self.conv_x2 = conv2d( feat1, 256, 1 )
        self.conv_x4 = conv2d( feat2, 256, 1 )
        self.conv_cmb = conv2d( 256+256+3, 256, 1 )
        self.softmax = Softmax(dim=-1)
        self.psp = PSPModule()
        self.addCAM = conv2d( 512, 256, 1)    
    def forward(self, xOrg, stg2, stg4, cam ):
        n,c,h,w = stg2.size()
        stg2 = self.conv_x2( stg2 )
        stg4 = self.conv_x4( stg4 )
        stg4 = F.interpolate( stg4, (h,w), mode='bilinear', align_corners= True)
        stg0 = F.interpolate( xOrg, (h,w), mode='bilinear', align_corners= True)
        stgSum = torch.cat([stg0,stg2,stg4],dim=1)
        stgSum = self.conv_cmb( stgSum )
        stgPool = self.psp( stgSum )                            #(N,c,s)
        stgSum = stgSum.view( n, -1, h*w ).transpose(1,2)       #(N,h*w,c)
        stg_aff = torch.matmul( stgSum, stgPool ) #(N,h*w,c)*(N,c,s)=(N,h*w,s)
        stg_aff = ( c ** -0.5 ) * stg_aff
        stg_aff = F.softmax( stg_aff, dim = -1 ) #(N,h*w,s)
        with torch.no_grad():
            cam_d = F.relu( cam.detach() ) 
        cam_d = F.interpolate( cam_d, (h,w), mode='bilinear', align_corners= True)
        cam_pool = self.psp( cam_d ).transpose(1,2) #(N,s,c)
        cam_rv = torch.matmul( stg_aff, cam_pool ).transpose(1,2)
        cam_rv=cam_rv.view(n, -1, h, w )
        out = torch.cat([cam, cam_rv], dim=1 )
        out = self.addCAM( out )
        return out
        
class GCM(Module):
    def __init__(self, feat_channels=512):
        super().__init__()

        chHig = feat_channels
        self.gamma = Parameter(torch.ones(1))
        self.higC = conv2d( chHig, 256, 3 )
        self.coe = nn.Sequential(                  
            conv2d( 256, 256, 3 ),
            nn.AdaptiveAvgPool2d((1,1)) 
        )       

    def forward(self, fea ):
        n,_,h, w = fea.size() 
        stgHig = self.higC( fea )
        coeHig = self.coe( stgHig )      
        sim = stgHig - coeHig
        # print( sim.size() )
        simDis = torch.norm( sim, 2, 1, keepdim = True )
        # print( simDis.size() )
        simDimMin = simDis.view( n, -1 )
        simDisMin = torch.min( simDimMin, 1, keepdim = True )[0]        
        # print( simDisMin.size() )
        simDis = simDis.view( n, -1 )
        weightHig = torch.exp( -( simDis - simDisMin ) / 5 )
        weightHig = weightHig.view(n, -1, h, w )
        upFea = F.interpolate( coeHig, (h,w), mode='bilinear', align_corners=True)
        upFea = upFea * weightHig
        stgHig = stgHig + self.gamma * upFea

        return weightHig, stgHig

class LCM(Module):
    def __init__(self, feat_channels=[256, 256, 512]):
        super().__init__()
        
        chHig, chLow1, chLow2 = feat_channels
        self.beta = Parameter(torch.ones(1)) 
        self.lowC1 = conv2d( chLow1, 48,3)
        self.lowC2 = conv2d( chLow2,128,3)
        self.cat1 = conv2d( 256+48, 256, 1 )
        self.cat2 = conv2d( 256+128, 256, 1 )     

    def forward(self, feaHig, feaCeo, feaLow1, feaLow2 ):        
        n,c,h,w = feaLow1.size()
        stgHig = F.interpolate( feaHig, (h,w), mode='bilinear', align_corners=True)
        weightLow = F.interpolate( feaCeo, (h,w), mode='bilinear', align_corners=True )
        coeLow = 1 - weightLow  
        stgLow1 = self.lowC1(feaLow1)
        stgLow2 = self.lowC2(feaLow2)   
        stgLow2 = F.interpolate( stgLow2, (h,w), mode='bilinear', align_corners=True )

        stgLow1 = self.beta * coeLow * stgLow1
        stgCat = torch.cat( [stgHig, stgLow1], dim = 1 )
        stgCat = self.cat1( stgCat )
        stgLow2 = self.beta * coeLow * stgLow2
        stgCat = torch.cat( [stgCat, stgLow2], dim = 1 )
        stgCat = self.cat2( stgCat )       
        return stgCat
