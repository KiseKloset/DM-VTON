from FlowStyle.raw import FlowStyle
from SRMGN.raw import SRMGN

import torch
import cupy
from thop import profile


def main():
    DEVICE_ID = 2

    device = f"cuda:{DEVICE_ID}"
    MODELS = {
        "flow style": FlowStyle(),
        "srmgn": SRMGN(),
    }

    with cupy.cuda.Device(DEVICE_ID):
        for name in MODELS:
            model = MODELS[name].to(device)
            person = torch.rand(1, 3, 256, 192).to(device)
            cloth = torch.rand(1, 3, 256, 192).to(device)
            cloth_edge = torch.rand(1, 1, 256, 192).to(device)
            total_ops, total_params = profile(model, (person, cloth, cloth_edge, ))
            print(
                "%s | %.2f | %.2f" % (name, total_params , total_ops)
            )

if __name__ == "__main__":
    print("aa")
    main()