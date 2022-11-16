from data import create_dataset
from options.options import BaseOptions
from models import create_model
from models import save_current_imgs
import time
import pickle

if __name__ == '__main__':
    opt = BaseOptions().get_opt()
    dataloader = create_dataset(opt)
    model = create_model(opt)
    print(f'training options are {opt}')

    if opt.load_trained:
        model.load_networks(opt.load_epoch)
        model.load_losses()

    start_time = time.time()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate()
        for i, data in enumerate(dataloader):
            print(f"batch {i}")
            model.set_input(data)
            model.optimize_parameters()
            model.get_current_losses()
        print(f"finished epoch {epoch}")
        if epoch % opt.save_loss_freq == 0:
            model.save_losses()
            model.save_networks(epoch)
        if epoch % opt.save_img_freq == 0:
            samples = next(iter(dataloader))
            model.set_input(samples)
            #model.save_current_imgs(epoch)
            save_current_imgs(opt, dataloader, model, epoch)
    model.save_networks('latest')
    end_time = time.time()

    if opt.is_gan:
        opt_dir = f'train_opt/generative_{opt.save_suffix}.pkl'
    else:
        opt_dir = f'train_opt/regression_{opt.save_suffix}.pkl'
    with open(opt_dir, 'wb') as f:
        pickle.dump(opt, f)

    print("Done!")
    print(f"Used {(end_time - start_time)} seconds")
