import numpy as np

from models.base_network import BaseNetwork
from models.encoder import AttrEncoder, AttrDilatedEncoder
from models.aad import AADGenerator


class RMGNGenerator(BaseNetwork):
    def __init__(self, multilevel=False, predmask=True):
        super().__init__()
        nf = 64
        in_nc_clothes = 4
        in_nc_person = 3
        out_nc = 4

        SR_scale = 1
        aei_encoder_head = False
        head_layers = int(np.log2(SR_scale)) + 1 if aei_encoder_head or SR_scale > 1 else 0
        
        self.inp_encoder = AttrEncoder(nf=nf, in_nc=in_nc_person, head_layers=head_layers)
        self.ref_encoder = AttrEncoder(nf=nf, in_nc=in_nc_clothes, head_layers=head_layers)
        self.generator = AADGenerator(nf=nf, out_nc=out_nc, SR_scale=SR_scale, multilevel=multilevel, predmask=predmask)
        
        self.init_weights()
        
    def get_inp_attr(self, inp):
        inp_attr_list = self.inp_encoder(inp)
        return inp_attr_list

    def get_ref_attr(self, ref):
        ref_attr_list = self.ref_encoder(ref)
        return ref_attr_list

    def get_gen(self, inp_attr_list, ref_attr_list):
        out = self.generator(inp_attr_list, ref_attr_list)
        return out

    def forward(self, inp, ref):
        inp_attr_list = self.get_inp_attr(inp)
        ref_attr_list = self.get_ref_attr(ref)
        out, out_L1, out_L2, M_list = self.get_gen(inp_attr_list, ref_attr_list)
        return out, out_L1, out_L2, M_list

