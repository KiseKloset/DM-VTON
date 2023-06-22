from PIL import Image

import torchvision.transforms as T

FINE_SIZE = [256,192]


# TODO: check params: opt.netG, opt.n_downsample_global (num_downs generator), opt.n_local_enhancers
# TODO: Solve non-none resize_or_crop, flip
def get_transform(resize_or_crop, n_downsample_global, method=Image.BICUBIC, normalize=True):
    transform_list = []
    # if 'resize' in opt.resize_or_crop:
    #     osize = [opt.loadSize, opt.loadSize]
    #     transform_list.append(T.Scale(osize, method))   
    # elif 'scale_width' in opt.resize_or_crop:
    #     transform_list.append(ScaleWidth(load_size, method))
    #     transform_list.append(T.Scale(FINE_SIZE, method))  
    # if 'crop' in opt.resize_or_crop:
    #     transform_list.append(T.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if resize_or_crop == 'none':
        base = float(2 ** n_downsample_global)
        # if opt.netG == 'local':
        #     base *= (2 ** opt.n_local_enhancers)
        transform_list.append(MakePower2(base, method))

    # transform_list.append(T.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list.append(T.ToTensor())

    if normalize:
        transform_list.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(transform_list)

class MakePower2:
    def __init__(self, base, method=Image.BICUBIC) -> None:
        self.base = base
        self.method = method

    def __call__(self, img):
        ow, oh = img.size        
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), self.method)

# class ScaleWidth:
#     def __init__(self, width, method=Image.BICUBIC):
#         self.w = width
#         self.method = method
    
#     def __call__(self, img):
#         ow, oh = img.size
#         if ow == self.w:
#             return img    
#         h = int(self.w * oh / ow)    
#         return img.resize((self.w, h), self.method)

# def __scale_width(img, target_width, method=Image.BICUBIC):
#     ow, oh = img.size
#     if (ow == target_width):
#         return img    
#     w = target_width
#     h = int(target_width * oh / ow)    
#     return img.resize((w, h), method)

# def __crop(img, pos, size):
#     ow, oh = img.size
#     x1, y1 = pos
#     tw = th = size
#     if (ow > tw or oh > th):        
#         return img.crop((x1, y1, x1 + tw, y1 + th))
#     return img

# def __flip(img, flip):
#     if flip:
#         return img.transpose(Image.FLIP_LEFT_RIGHT)
#     return img
