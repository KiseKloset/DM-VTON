import json

import cupy
import torch
from thop import profile as ops_profile
from tqdm import tqdm

# from ACGPN.raw import ACGPN
# from WUTON.raw import WUTON
# from RMGN.raw import RMGN
# from PFAFN.raw import PFAFN
# from FlowStyle.raw import FlowStyle
from SRMGN.raw import SRMGN
# from CDGNet.raw import CDGNet
# from ShineOn.raw import ShineOn
# from SDAFN.raw import SDAFN
# from CPVTON.raw import CPVTON
# from CVTON.raw import CVTON
# from ClothFlow.raw import ClothFlow
# from SRMGNLastHope.raw import SRMGNLastHope
# from ViTPose.configs.ViTPose_base_coco_256x192 import model as model_cfg
# from ViTPose.models.model import ViTPose
# from DensePose.raw import DensePose
from utils import Profile

DEVICE_ID = 0
DEVICE = f"cuda:{DEVICE_ID}"

WUTON_ID = 1
RMGN_ID = 2
PFAFN_ID = 3
FLOW_STYLE_ID = 4
SRMGN_ID = 5
CDGNET_ID = 6
SHINE_ON_ID = 7
SDAFN_ID = 8
VITPOSE_ID = 9
ACGPN_ID = 10
CPVTON_ID = 11
CLOTHFLOW_ID = 12
CVTON_ID = 13
DENSEPOSE_ID = 14
LAST_HOPE_ID = 15

MODELS = {
    # WUTON_ID: WUTON(),
    # RMGN_ID: RMGN(),
    # PFAFN_ID: PFAFN(),
    # FLOW_STYLE_ID: FlowStyle(),
    SRMGN_ID: SRMGN(),
    # CDGNET_ID: CDGNet(20, (473, 473)),
    # SHINE_ON_ID: ShineOn(),
    # SDAFN_ID: SDAFN(ref_in_channel=6),
    # VITPOSE_ID: ViTPose(model_cfg),
    # ACGPN_ID: ACGPN(),
    # CPVTON_ID: CPVTON(DEVICE),
    # CLOTHFLOW_ID: ClothFlow(),
    # CVTON_ID: CVTON(DEVICE),
    # DENSEPOSE_ID: DensePose(),
    # LAST_HOPE_ID: SRMGNLastHope()
}

PROFILES = {i: Profile() for i in MODELS}
RESULTS = {i: {} for i in MODELS}


def gen_input(model_id, device):
    if model_id == WUTON_ID:
        gan_product_image_batch = torch.rand(1, 3, 224, 224).to(device)
        model_agnostic_image_batch = torch.rand(1, 6, 224, 224).to(device)
        return [gan_product_image_batch, model_agnostic_image_batch]

    elif model_id in (RMGN_ID, PFAFN_ID, FLOW_STYLE_ID, SRMGN_ID, LAST_HOPE_ID):
        person = torch.rand(1, 3, 256, 192).to(device)
        cloth = torch.rand(1, 3, 256, 192).to(device)
        cloth_edge = torch.rand(1, 1, 256, 192).to(device)
        return [person, cloth, cloth_edge]

    elif model_id == CDGNET_ID:
        return [torch.rand(1, 3, 473, 473).to(device)]

    elif model_id == SHINE_ON_ID:
        return [torch.rand(1, 3, 256, 192).to(device), torch.rand(1, 7, 256, 192).to(device)]

    elif model_id == SDAFN_ID:
        ref_input = torch.rand(1, 6, 256, 192).to(device)
        source_image = torch.rand(1, 3, 256, 192).to(device)
        ref_image = torch.rand(1, 3, 256, 192).to(device)
        return [ref_input, source_image, ref_image]

    elif model_id == VITPOSE_ID:
        return [torch.rand(1, 3, 256, 192).to(device)]

    elif model_id == ACGPN_ID:
        # person, cloth, cloth_edge, pose, parse
        inputA = torch.rand(1, 3, 256, 192).to(device)
        inputB = torch.rand(1, 1, 256, 192).to(device)
        inputC = torch.rand(1, 18, 256, 192).to(device)
        return [inputA, inputA, inputB, inputC, inputB]

    elif model_id == CPVTON_ID:
        inputA = torch.rand(1, 22, 256, 192).to(device)
        inputB = torch.rand(1, 3, 256, 192).to(device)
        return [inputA, inputB]

    elif model_id == CLOTHFLOW_ID:
        inputA = torch.rand(1, 3, 256, 192).to(device)
        inputB = torch.rand(1, 1, 256, 192).to(device)
        inputC = torch.rand(1, 1, 256, 192).to(device)
        return [inputA, inputB, inputC]

    elif model_id == CVTON_ID:
        image = torch.rand(1, 3, 256, 192).to(device)
        cloth_image = torch.rand(1, 3, 256, 192).to(device)
        masked_image = torch.rand(1, 3, 256, 192).to(device)
        cloth_seg_transf = torch.rand(1, 1, 256, 192).to(device)
        body_seg_transf = torch.rand(1, 1, 256, 192).to(device)
        densepose_seg_transf = torch.rand(1, 1, 256, 192).to(device)

        return [
            {
                "image": {"I": image, "C_t": cloth_image, "I_m": masked_image},
                "cloth_label": cloth_seg_transf,
                "body_label": body_seg_transf,
                "densepose_label": densepose_seg_transf,
                "name": "abcxyz",
                "agnostic": "",
                "original_size": [256, 192],
                "label_centroid": None,
            }
        ]

    elif model_id == DENSEPOSE_ID:
        height = 256
        width = 192
        image = torch.rand(3, height, width).to(device)
        return [{"image": image, "height": height, "width": width}]


def run_once(model_id, device, measure_time=False):
    model = MODELS[model_id].eval().to(device)
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    time_profile = PROFILES[model_id]
    _input = gen_input(model_id, device)

    if measure_time:
        with time_profile:
            model(*_input)
    else:
        ops, params = ops_profile(model, (*_input,))
        RESULTS[model_id]["ops"] = ops
        RESULTS[model_id]["params"] = params
        RESULTS[model_id]["size"] = mem


if __name__ == "__main__":
    cupy.cuda.Device(DEVICE_ID).use()
    N = 1000
    for model_id in MODELS:
        with torch.no_grad():
            run_once(model_id, DEVICE, False)
            for i in tqdm(range(N)):
                run_once(model_id, DEVICE, True)

    for i, profile in PROFILES.items():
        RESULTS[i]["time"] = profile.t / N * 1e3

    with open("results.json", "w") as f:
        json.dump(RESULTS, f)
