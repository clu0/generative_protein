from models.models import GANModel 
from models.models import RegressionModel 
import numpy as np
import torch
import os

def create_model(opt):
    if not opt.is_gan:
        model = RegressionModel(opt)
        print("a regression model was created")
        return model
    else:
        model = GANModel(opt)
        print("a generative model was created")
        return model

def save_current_imgs(opt, dataloader, model, epoch):
    generated = {}
    if opt.is_gan:
        save_dir = os.path.join(opt.img_save_dir, f'generative_{opt.save_suffix}')
    else:
        save_dir = os.path.join(opt.img_save_dir, f'regression_{opt.save_suffix}')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            model.set_input(data)
            img_batch = model.generate_imgs()
            if not generated:
                generated = img_batch
            else:
                for k in generated:
                    generated[k] = np.concatenate((generated[k], img_batch[k]))
        for k in generated:
            save_loc = os.path.join(save_dir, f'{epoch}_imgs_{k}.npy')
            np.save(save_loc, generated[k])
