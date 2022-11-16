import argparse

class BaseOptions():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--n_protein', type=int, default=100, help='number of proteins in this dataset')
        parser.add_argument('--dim', type=int, default=2, help='dimension of input images, 2 or 3')
        parser.add_argument('--is_test', action='store_true', help='default mode is training, if specified, do testing')
        parser.add_argument('--load_trained', action='store_true', help='if speficied, load pretrianed model and losses')
        parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--model_save_dir', type=str, default='./saved_models/', help='the directory to save models to, or load trained models from')
        parser.add_argument('--img_save_dir', type=str, default='./saved_imgs/', help='the directory to save images during training')
        parser.add_argument('--loss_save_dir', type=str, default='./saved_loss/', help='the directory to save losses to, or load previous training losses from')
        parser.add_argument('--save_suffix', type=str, default='', help='the suffix to add to save directories to identify experiment')
        parser.add_argument('--name', type=str, default='regression', help='the name of the model, used for saving')

        # arguments to build dataset
        parser.add_argument('--protein_list', type=str, default='./metadata/nuclear_protein_dict.pkl', help='file for dictionary of all proteins in dataset')
        parser.add_argument('--img_dir', type=str, default='./datasets/train_nuclear/', help='dir for images to load')
        parser.add_argument('--img_list', type=str, default='./metadata/nuclear_images_train.csv', help='file that lists all the images loaded')
        parser.add_argument('--theta_file', type=str, default='./metadata/thetas.pth', help='file with the learnt protein representations')
        parser.add_argument('--no_rand', action='store_true', help='if no_rand, data is not augmented')
        parser.add_argument('--no_std', action='store_true', help='if specified, data is not standardized')
        parser.add_argument('--no_rotate', action='store_true', help='if specified, data will not be random rotated')
        parser.add_argument('--no_shuffle', action='store_true', help='if specified, dataloader will not shuffle')
        parser.add_argument('--img_w', type=int, default=8, help='dimension of input to model will be 2 ** img_w')
        parser.add_argument('--img_d', type=int, default=5, help='if input images are 3D, then the depth of the images will be 2 ** img_d')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')

        # args to build model
        parser.add_argument('--is_gan', action='store_true', help='if specified, will use gan model')
        # params for unet
        parser.add_argument('--wf', type=int, default=4, help='number of filters in first layer of Unet will be 2 ** 4')
        parser.add_argument('--label_dim', type=int, default=16, help='dimension of the vector label for proteins')
        parser.add_argument('--add_label', action='store_true', help='if specified, will add labels to upward unet layers')
        parser.add_argument('--second_layer', action='store_true', help='if specified, will add additional layer for each layer of unet')
        parser.add_argument('--no_batch_norm', action='store_true', help='if specified, no batch norm')
        parser.add_argument('--pix2pix_gen', action='store_true', help='if specified, will use generator from pix2pix mode')

        # args for training
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--save_loss_freq', type=int, default=50, help='frequency of saving losses at the end of epochs')
        parser.add_argument('--save_img_freq', type=int, default=50, help='frequency of saving images at the end of epochs')
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lambda_L1', type=float, default=100., help='weight for L1 loss')
        # not used right now, dropout is applied automatically if training the generative model
        #parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        self.parser = parser

    def get_opt(self):
        opt, _ = self.parser.parse_known_args()
        return opt
