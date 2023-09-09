# 😎 Supported Models
| Methods | Source | Teacher | Student |
| :- | :- | :-:| :-: |
| [PF-AFN](#pf-afn) | [Parser-Free Virtual Try-on via Distilling Appearance Flows](https://arxiv.org/abs/2103.04559) | ✅ | ✅ |
| [FS-VTON](#fs-vton) | [Style-Based Global Appearance Flow for Virtual Try-On](https://arxiv.org/abs/2204.01046) | ✅ | ✅ |
| [RMGN](#rmgn) | [RMGN: A Regional Mask Guided Network for Parser-free Virtual Try-on](https://arxiv.org/abs/2204.11258) | ❌ | ✅ |
| [DM-VTON (Ours)](#dm-vton) | [DM-VTON: Distilled Mobile Real-time Virtual Try-On](https://arxiv.org/abs/2308.13798) | ✅ | ✅ |

### PF-AFN
```py
from models.warp_modules.afwm import AFWM
from models.generators.res_unet import ResUnetGenerator

# Teacher
warp_model = AFWM(45)
gen_model  = ResUnetGenerator(8, 4, 5)

# Student
warp_model = AFWM(3)
gen_model  = ResUnetGenerator(7, 4, 5)
```

### FS-VTON
```py
from models.warp_modules.style_afwm import StyleAFWM
from models.generators.res_unet import ResUnetGenerator

# Teacher
warp_model = StyleAFWM(45)
gen_model  = ResUnetGenerator(8, 4, 5)

# Student
warp_model = StyleAFWM(3)
gen_model  = ResUnetGenerator(7, 4, 5)
```


### RMGN
```py
from models.warp_modules.afwm import AFWM
from models.generators.rmgn_generator import RMGNGenerator

# Student
warp_model = AFWM(3)
gen_model  = RMGNGenerator(in_person_nc=3, in_clothes_nc=4)
```

*Note: [RMGN](https://github.com/jokerlc/RMGN-VITON) only release inference code for the Student network.*

### DM-VTON
```py
from models.warp_modules.mobile_afwm import MobileAFWM
from models.generators.mobile_unet import MobileNetV2_unet

# Teacher
warp_model = MobileAFWM(45)
gen_model  = MobileNetV2_unet(8, 4)

# Student
warp_model = MobileAFWM(3)
gen_model  = MobileNetV2_unet(7, 4)
```