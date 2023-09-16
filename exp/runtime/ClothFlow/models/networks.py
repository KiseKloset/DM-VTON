"""FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import torch.autograd as autograd
import functools

###############################################################################
# Blocks
######################### ######################################################


class ResBlock(nn.Module):
    def __init__(self, dim, norm="instance", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        ]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super().__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "instance":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlockDown(nn.Module):
    def __init__(self, dim_in, dim_out, norm="batch", activation="relu", pad_type="zero"):
        super(ResBlockDown, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim_in, dim_out, 3, 2, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [ResBlock(dim=dim_out, norm=norm, activation="none", pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


###############################################################################
# Networks
######################### ######################################################


class ResBlockFPN(nn.Module):
    def __init__(self, num_blocks=5, input_nc=3):
        super(ResBlockFPN, self).__init__()
        self.layer1 = ResBlockDown(input_nc, 64)
        self.layer2 = ResBlockDown(64, 128)
        self.layer3 = ResBlockDown(128, 256)
        self.layer4 = ResBlockDown(256, 256)
        self.layer5 = ResBlockDown(256, 256)

        self.upsample_dim1 = Conv2dBlock(
            input_dim=64, output_dim=256, kernel_size=1, stride=1, padding=0
        )
        self.upsample_dim2 = Conv2dBlock(
            input_dim=128, output_dim=256, kernel_size=1, stride=1, padding=0
        )
        self.upsample_dim3 = Conv2dBlock(
            input_dim=256, output_dim=256, kernel_size=1, stride=1, padding=0
        )

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="nearest") + y

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        s5 = self.layer5(c4)

        s4 = self._upsample_add(s5, self.upsample_dim3(c4))
        s3 = self._upsample_add(s4, self.upsample_dim3(c3))
        s2 = self._upsample_add(s3, self.upsample_dim2(c2))
        s1 = self._upsample_add(s2, self.upsample_dim1(c1))

        return s1, s2, s3, s4, s5


class FlowEstimator(nn.Module):
    def __init__(self, input_nc=256):  #! Should be FPN first channel * 4
        super(FlowEstimator, self).__init__()
        self.SourceFPN = ResBlockFPN(input_nc=4)
        self.TargetFPN = ResBlockFPN(input_nc=1)
        self.e5 = self.get_flow(input_nc)
        self.e4 = self.get_flow(input_nc)
        self.e3 = self.get_flow(input_nc)
        self.e2 = self.get_flow(input_nc)
        self.e1 = self.get_flow(input_nc)

    def forward(self, c_s, s_s, s_t):
        """[Forward pass of flow estimation network]

        Arguments:
            c_s {[torch Tensor]} -- [Source clothing item]
            s_s {[torch Tensor]} -- [Source segmentation]
            s_t {[torch Tensor]} -- [Target segmentation]

        Returns:
            [type] -- [description]
        """

        source_input = torch.cat([c_s, s_s], dim=1)
        s1, s2, s3, s4, s5 = self.SourceFPN(source_input)
        t1, t2, t3, t4, t5 = self.TargetFPN(s_t)

        f5 = self.e5(torch.cat([s5, t5], dim=1))
        f4 = self.upsample(f5) + self.e4(torch.cat([self.warp(s4, self.upsample(f5)), t4], dim=1))

        f3 = self.upsample(f4) + self.e3(torch.cat([self.warp(s3, self.upsample(f4)), t3], dim=1))

        f2 = self.upsample(f3) + self.e2(torch.cat([self.warp(s2, self.upsample(f3)), t2], dim=1))

        f1 = self.upsample(f2) + self.e1(torch.cat([self.warp(s1, self.upsample(f2)), t1], dim=1))

        # Warped clothing item
        c_s_prime = self.warp(c_s, self.upsample(f1))
        s_s_prime = self.warp(s_s, self.upsample(f1))

        return f5, f4, f3, f2, f1, c_s_prime, s_s_prime

    # Define convolutional encoder
    def get_flow(self, input_nc):
        """Given source and target feature maps, get flow

        Arguments:
            s {[torch Tensor]} -- [source feature map]
            t {[torch Tensor]} -- [target feature map]
        """
        # Concatenate source and target features

        flow_conv = nn.Conv2d(input_nc * 2, 2, kernel_size=3, stride=1, padding=1)

        return flow_conv

    def upsample(self, F):
        """[2x nearest neighbor upsampling]

        Arguments:
            F {[torch tensor]} -- [tensor to be upsampled, (B, C, H, W)]
        """
        upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        return upsample(F)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN([2, 2, 2, 2])


import torch
import torchvision

###############################################################################
# Losses
######################### ######################################################


##################################################################################
# VGG network definition
##################################################################################
from torchvision import models

# Source: https://github.com/NVIDIA/pix2pixHD
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Source: https://github.com/NVIDIA/pix2pixHD
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = Vgg19().cuda().eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    def __init__(self, loss_name="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss_name = loss_name
        if loss_name == "lsgan":
            self.loss = nn.MSELoss()
        elif loss_name == "wgan":
            self.loss = lambda x: torch.mean(x)
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.loss_name == "wgan":
            if target_is_real:  # Miminize value
                return self.loss(input)
            else:  # Maximize value
                return -self.loss(input)
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


###############################################################################
# Helper Functions
######################### ######################################################


def calc_gradient_penalty(opt, netD, real_data, fake_data):
    DIM = opt.data.img_size
    LAMBDA = 1
    nc = opt.dis.default.input_nc
    alpha = torch.rand(real_data.shape)
    # alpha = alpha.view(batch_size, nc, DIM, DIM)
    # alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()

    alpha = alpha.cuda()
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def get_norm_layer(norm_type="instance"):
    if not norm_type:
        print("norm_type is {}, defaulting to instance")
        norm_type = "instance"
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def define_D(opts):
    input_nc = opts.dis.params.input_nc
    ndf = opts.dis.params.ndf
    n_layers = opts.dis.params.n_layers
    norm_layer = get_norm_layer(opts.dis.params.norm)
    use_sigmoid = opts.dis.params.use_sigmoid
    kw = opts.dis.params.kw
    padw = opts.dis.params.padw
    nf_mult = opts.dis.params.nf_mult
    nf_mult_prev = opts.dis.params.nf_mult_prev

    init_type = opts.dis.params.init_type
    init_gain = opts.dis.params.init_gain

    net = None

    net = NLayerDiscriminator(
        input_nc, ndf, n_layers, norm_layer, use_sigmoid, kw, padw, nf_mult, nf_mult_prev
    )

    init_weights(net, init_type, init_gain)

    return net


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf, n_layers, norm_layer, use_sigmoid, kw, padw, nf_mult, nf_mult_prev,
    ):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # Use spectral normalization
                SpectralNorm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # Use spectral normalization
            SpectralNorm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                )
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Use spectral normalization
        sequence += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)