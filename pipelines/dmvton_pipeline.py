import torch
import torch.nn.functional as F

from models.generators.mobile_unet import MobileNetV2_unet
from models.warp_modules.mobile_afwm import MobileAFWM as AFWM
from pipelines.base_pipeline import BaseVTONPipeline
from utils.torch_utils import get_ckpt, load_ckpt


class DMVTONPipeline(BaseVTONPipeline):
    """
    DM-VTON inference pipeline
    """

    def __init__(self, align_corners=True, checkpoints=None):
        super().__init__()
        self.align_corners = align_corners
        self.warp_model = AFWM(3, align_corners)
        self.gen_model = MobileNetV2_unet(7, 4)

        if checkpoints is not None:
            self._load_pretrained(checkpoints)

    def _load_pretrained(self, checkpoints):
        if checkpoints.get('warp') is not None:
            warp_ckpt = get_ckpt(checkpoints['warp'])
            load_ckpt(self.warp_model, warp_ckpt)
        if checkpoints.get('gen') is not None:
            gen_ckpt = get_ckpt(checkpoints['gen'])
            load_ckpt(self.gen_model, gen_ckpt)

    def forward(self, person, clothes, clothes_edge, phase="test"):
        clothes_edge = (clothes_edge > 0.5).float()
        clothes = clothes * clothes_edge

        # Warp
        flow_out = self.warp_model(person, clothes, phase=phase)
        (
            warped_cloth,
            last_flow,
        ) = flow_out
        warped_edge = F.grid_sample(
            clothes_edge,
            last_flow.permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=self.align_corners,
        )

        # Gen
        gen_inputs = torch.cat([person, warped_cloth, warped_edge], 1)
        gen_outputs = self.gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

        return p_tryon, warped_cloth
