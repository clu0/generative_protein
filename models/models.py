import os
import torch
import numpy as np
from torch import nn
#from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

from models.networks import Unet 
from models.networks import UnetGenerator 
from models.networks import Discriminator 
from models.networks import init_net
from models.networks import GANLoss
from models.networks import get_scheduler
from models.networks import Losses

class BaseModel(ABC):
    def __init__(self, opt):
        self.is_train = not opt.is_test
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # not sure what the difference between torch.device, and simply specifying 'cuda'... 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if opt.is_gan:
            self.name = f'generative_{opt.save_suffix}'
        else:
            self.name = f'regression_{opt.save_suffix}'
        self.model_save_dir = os.path.join(opt.model_save_dir, self.name)
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        self.loss_save_dir = opt.loss_save_dir
        if not os.path.exists(self.loss_save_dir):
            os.mkdir(self.loss_save_dir)
        self.save_suffix = opt.save_suffix
        self.model_names = []
        self.loss_names = []
        self.losses = Losses()

    @abstractmethod
    def set_input(self, sample):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def load_networks(self, epoch):
        for name in self.model_names:
            load_fn = '%s_net_%s.pth' % (epoch, name)
            load_pth = os.path.join(self.model_save_dir,load_fn) 
            net = getattr(self, 'net'+name)
            print('loading the model') 
            state_dict = torch.load(load_pth, map_location=str(self.device))
            net.load_state_dict(state_dict)

    def load_losses(self):
        save_pth = os.path.join(self.loss_save_dir, f'losses_{self.save_suffix}.pkl')
        self.losses.load(save_pth)

    def get_current_losses(self):
        errors = {}#OrderedDict()
        for name in self.loss_names:
            errors[name] = [float(getattr(self, 'loss_'+name))]
        self.losses.append(errors)
        return errors

    def save_losses(self):
        save_pth = os.path.join(self.loss_save_dir, f'losses_{self.save_suffix}.pkl')
        self.losses.save(save_pth)

    def save_networks(self, epoch):
        for name in self.model_names:
            save_fn = '%s_net_%s.pth' % (epoch, name)
            save_pth = os.path.join(self.model_save_dir,save_fn) 

            net = getattr(self, 'net'+name)
            # need to do something different if we use multi gpus
            #if torch.cuda.is_available():
            #    torch.save(net.module.cpu().state_dict(),save_pth)
            #else:
            torch.save(net.state_dict(), save_pth)

class GANModel(BaseModel):
    def __init__(self, opt):
        super(GANModel, self).__init__(opt)
        #self.img_save_dir = os.path.join(opt.img_save_dir, self.name)
        self.pix2pix_gen = opt.pix2pix_gen
        if opt.pix2pix_gen:
            in_c = opt.label_dim + 1
            netG = UnetGenerator(in_c, 1, 8, use_dropout=True)
        else:
            netG = Unet(opt)
        self.netG = init_net(netG)
        if not opt.is_test:
            netD = Discriminator()
            self.netD = init_net(netD)
            self.model_names=['G', 'D']
        else:
            self.model_names=['G']
        self.loss_names = ['G_GAN', 'G_L1', 'G', 'D_real', 'D_fake', 'D']
        self.lambda_L1 = opt.lambda_L1
        self.gan_loss = GANLoss().to(self.device)
        self.l1_loss = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, sample):
        self.real_A = sample['dapi'].to(self.device)
        self.real_B = sample['gfp'].to(self.device)
        theta = sample['theta'].to(self.device)
        theta = theta.view(theta.size(0), theta.size(1), 1, 1)
        self.theta = theta.repeat(1, 1, self.real_A.size(2), self.real_A.size(3))
        self.label = sample['label'].to(self.device)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    #def save_current_imgs(self, epoch):
    #    with torch.no_grad():
    #        net = self.netG.to(self.device)
    #        if self.pix2pix_gen:
    #            img = torch.cat((self.theta, self.real_A), 1)
    #            fake_B = net(img).detach().cpu().numpy()
    #        else:
    #            fake_B = net(self.real_A, self.theta).detach().cpu().numpy()
    #        save_fn = '%s_imgs.npy' % epoch 
    #        save_dir = os.path.join(self.img_save_dir, save_fn)
    #        np.save(save_dir, fake_B)

    def get_protein_losses(self, dataloader, save_dir, opt):
        protein_gloss = np.zeros(opt.n_protein) - 1
        protein_l1loss = np.zeros(opt.n_protein) - 1
        protein_imgs = np.zeros((opt.n_protein, 3, 256, 256))
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                for j in range(opt.batch_size):
                    print(f"the original shape of theta is {data['theta'].size()}")
                    print(f"the size of dim 1 is {data['theta'].size(1)}")
                    theta = data['theta'][j,:].repeat(opt.batch_size, 1)
                    print(theta.size())
                    theta = theta.view(theta.size(0), theta.size(1), 1, 1)
                    print(theta.size())
                    theta = theta.repeat(1, 1, 256, 256).to(self.device)
                    print(f'shape of theta is {theta.size()}')
                    print(f'in batch {i} image {j}')
                    real_A = data['dapi'][j,...].repeat(opt.batch_size, 1, 1, 1).to(self.device) 
                    print(f'shape of real A is {real_A.size()}')
                    print('just loaded dapi')
                    real_B = data['gfp'][j,...].repeat(opt.batch_size, 1, 1, 1).to(self.device) 
                    print('just loaded gfp')
                    label = data['label'].repeat(opt.batch_size).to(self.device)
                    print('just loaded labels')
                    netG = self.netG.to(self.device)
                    print(f'loaded all images')
                    img = torch.cat((theta, real_A), 1).to(self.device)
                    fake_B = netG(img)
                    fake_AB = torch.cat((theta, real_A, fake_B), 1).to(self.device)
                    netD = self.netD.to(self.device)
                    gloss = self.gan_loss(netD(fake_AB), True).detach().cpu().numpy()
                    l1loss = self.l1_loss(fake_B, real_B).detach().cpu().numpy()
                    pred = fake_B.detach().cpu().numpy()
                    gfp = real_B.detach().cpu().numpy()
                    dapi = real_A.detach().cpu().numpy()
                    label = label.detach().cpu().numpy()
                    print(f'generated predictions')
                    pIndex = label[j]
                    if protein_gloss[pIndex] < 0:
                        protein_gloss[pIndex] = gloss
                        protein_l1loss[pIndex] = l1loss
                        protein_imgs[pIndex, 0, ...] = dapi[j, 0, ...]
                        protein_imgs[pIndex, 1, ...] = gfp[j, 0, ...]
                        protein_imgs[pIndex, 2, ...] = pred[j, 0, ...]
        np.savez(open(save_dir, 'wb'), protein_gloss=protein_gloss, protein_l1loss=protein_l1loss, protein_imgs=protein_imgs)

    def generate_imgs(self):
        with torch.no_grad():
            net = self.netG.to(self.device)
            if self.pix2pix_gen:
                img = torch.cat((self.theta, self.real_A), 1)
                fake_B = net(img).detach().cpu().numpy()
            else:
                fake_B = net(self.real_A, self.theta).detach().cpu().numpy()
            return {'pred':fake_B,'gfp':self.real_B.detach().cpu().numpy(), 'dapi':self.real_A.detach().cpu().numpy(), 'label':self.label.detach().cpu().numpy()}

    def forward(self):
        if self.pix2pix_gen:
            labelled_img = torch.cat((self.theta, self.real_A), 1)
            self.fake_B = self.netG(labelled_img)
        else:
            self.fake_B = self.netG(self.real_A, self.theta)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.theta, self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.gan_loss(pred_fake, False)
        # Real
        real_AB = torch.cat((self.theta, self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.gan_loss(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.theta, self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.gan_loss(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.l1_loss(self.fake_B, self.real_B) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

class RegressionModel(BaseModel):
    def __init__(self, opt):
        super(RegressionModel, self).__init__(opt)
        self.model_names = ['G']
        self.loss_names = ['G']
        self.netG = Unet(opt).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=3e-4)

    def set_input(self, sample):
        self.dapi = sample['dapi'].to(self.device)
        self.gfp = sample['gfp'].to(self.device)
        self.label = sample['label'].to(self.device)

    def forward(self):
        self.pred = self.netG(self.dapi, self.label)

    def backward(self):
        self.loss_G = self.loss(self.pred, self.gfp)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        # regression model robust enough without learning rate updates
        return
