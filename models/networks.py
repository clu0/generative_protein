import functools

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import init
import pickle


device="cuda:0"

_conv_dict = {2: nn.Conv2d, 3: nn.Conv3d}
_max_pool_dict = {2: F.max_pool2d, 3: F.max_pool3d}
_batchnorm_dict = {2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
_convtranspose_dict = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}


###############################################################################
# Helper Functions
###############################################################################


def get_scheduler(optimizer, opt, lr_policy='linear'):
    epoch_count = opt.epoch_count
    n_epochs = opt.n_epochs
    n_epochs_decay = opt.n_epochs_decay
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    #elif lr_policy == 'plateau':
    #    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device='cuda'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if device=='cuda':
        assert(torch.cuda.is_available())
        net=net.to(device)
        # not going to do multi gpu training for now
        #net = nn.parallel.DistributedDataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

##############################################################################
# Classes
##############################################################################

class Losses():
    def __init__(self):
        self.losses = {}

    def load(self, load_dir):
        with open(load_dir, 'rb') as f:
            self.losses = pickle.load(f)

    def save(self, save_dir):
        with open(save_dir, 'wb') as f:
            pickle.dump(self.losses, f)

    def append(self, loss):
        if not self.losses:
            self.losses = loss
        else:
            for k in self.losses:
                self.losses[k] += loss[k]

    def get(self):
        return self.losses

    def get_len(self):
        l = next(iter(self.losses))
        return len(self.losses[l])

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

# a unet that is both used for the regression task and as the generator in GAN training
class Unet(nn.Module):
    def __init__(
        self,
        opt,
        in_channels=1,
        n_classes=1,
        depth=5,
        wf=4,
        in_kernel=3,
        in_padding=1,
        out_kernel=3,
        out_padding=1,
        down_kernel=[4, 4],
        up_mode='upconv',
        tf=4,
    ):
        super(Unet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        assert opt.dim in (2,3)
        self.is_gan = opt.is_gan
        use_leaky = opt.is_gan
        use_dropout = opt.is_gan
        self.dim = opt.dim
        self.down_kernel=down_kernel
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(opt, prev_channels, 2 ** (wf + i), in_kernel, in_padding, use_leaky)
            )
            prev_channels = 2 ** (wf + i)
        
        self.add_label = opt.add_label
        if not opt.is_gan:
            self.thetas = nn.Parameter(torch.normal(mean=0, 
                                                    std=1/(2 **(tf -1)), 
                                                    size=(opt.n_protein, 2**tf)),
                                                    requires_grad=True)

        prev_channels += 2 ** tf
        
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):                
            if i == depth-2:
                if self.add_label:
                    in_conv_size = prev_channels
                else:
                    in_conv_size = prev_channels - 2**tf
            else:
                if self.add_label:
                    in_conv_size = prev_channels + 2**tf
                else:
                    in_conv_size = prev_channels
            self.up_path.append(
                UNetUpBlock(opt, prev_channels, in_conv_size,  2 ** (wf + i), up_mode, down_kernel[0] if i==0 else down_kernel[1], out_kernel, out_padding, use_dropout=use_dropout if i < 3 else False)
            )
            prev_channels =  2 ** (wf + i)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    # when is_gan, the protein is the theta vector
    # if not, then it is just an int label
    def forward(self, x, protein):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, self.down_kernel[0] if i==0 else self.down_kernel[1])
                
        if self.is_gan:
            theta = protein.view(protein.size(0), protein.size(1), 1, 1)
            theta_bottom = theta.repeat(1, 1, x.size(2), x.size(3))
        else:
            theta = self.thetas[protein,:]
            if self.dim == 2:
                theta = theta.view(theta.size(0), theta.size(1), 1, 1)
                theta_bottom = theta.repeat(1, 1, x.size(2), x.size(3))
            else:
                theta = theta.view(theta.size(0), theta.size(1), 1, 1, 1)
                theta_bottom = theta.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        x = torch.cat((x, theta_bottom), 1)

        for i, up in enumerate(self.up_path):
            if self.add_label:
                x = up(x, blocks[-i - 1], theta)
            else:
                x = up(x, blocks[-i - 1])
        x = self.last(x)
        return x

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Discriminator(nn.Module):
    def __init__(self, input_nc=18,  ndf=16, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
        
class UNetConvBlock(nn.Module):
    def __init__(self, opt, in_size, out_size, kernel, padding, use_leaky=False, use_dropout=False):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel, padding=padding))
        block.append(nn.ReLU()) if not use_leaky else block.append(nn.LeakyReLU(0.2, True))
        if not opt.no_batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        if use_dropout:
            block.append(nn.Dropout(0.5))

        if opt.second_layer:
            block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel, padding=padding))
            block.append(nn.ReLU()) if not use_leaky else block.append(nn.LeakyReLU(0.2, True))
            if opt.no_batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            if use_dropout:
                block.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, opt, in_size, in_conv_size, out_size, up_mode, up_kernel, out_kernel, out_padding, use_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.dim = opt.dim
        if up_mode == 'upconv':
            if use_dropout:
                self.up = nn.Sequential(
                        nn.ConvTranspose2d(in_size, out_size, kernel_size=up_kernel, stride=up_kernel),
                        nn.Dropout(0.5)
                        )
            else:
                self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=up_kernel, stride=up_kernel)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=up_kernel),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(opt, in_conv_size, out_size, out_kernel, out_padding, use_dropout=use_dropout)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge, theta=None):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        if theta != None:
            if self.dim == 2:
                theta_repeat = theta.repeat(1, 1, up.size(2), up.size(3))
            else:
                theta_repeat = theta.repeat(1, 1, up.size(2), up.size(3), up.size(4))
            out = torch.cat([up, crop1, theta_repeat], 1)
        else:
            out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
