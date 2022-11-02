import os
import argparse
import copy
from tqdm import tqdm
import numpy as np

import utils
import models
from models import GITConfig,Generator
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn import functional as F


def exp_mov_avg(Gs, G, alpha = 0.999, global_step = 999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def getmaskedmodel(geomodel):
    maskcls = np.random.randint(6, 10)
    # maskcls=np.random.randint(10,16)
    mask = np.zeros(geomodel.shape)
    mask_near = np.zeros(geomodel.shape)
    mask_extendnear = np.zeros(geomodel.shape)

    for i in range(maskcls):
        r1, r2 = np.random.randint(geomodel.shape[1]), np.random.randint(geomodel.shape[1])
        mask[:, r1, r2] = 1
        xstrat = 0 if r1 - 1 < 0 else r1 - 1
        ystrat = 0 if r2 - 1 < 0 else r2 - 1
        xstop = geomodel.shape[0] - 1 if r1 + 2 > geomodel.shape[0] - 1 else r1 + 2
        ystop = geomodel.shape[0] - 1 if r2 + 2 > geomodel.shape[0] - 1 else r2 + 2
        mask_near[:, xstrat:xstop, ystrat:ystop] = 1
        xstrat = 0 if r1 - 2 < 0 else r1 - 2
        ystrat = 0 if r2 - 2 < 0 else r2 - 2
        xstop = geomodel.shape[0] - 1 if r1 + 3 > geomodel.shape[0] - 1 else r1 + 3
        ystop = geomodel.shape[0] - 1 if r2 + 3 > geomodel.shape[0] - 1 else r2 + 3
        mask_extendnear[:, xstrat:xstop, ystrat:ystop] = 1
    mask_tensor =[torch.Tensor(mask),torch.Tensor(mask_near),torch.Tensor(mask_extendnear)]
    return mask_tensor

def stratLoss(pre_,mask_,tar_):
        # t1=time.time()
        w1,w2,w3,w4=0,0.5,0.3,0.1
        b,block_size,cls=pre_.shape
        mask=mask_[0].reshape(-1)
        masknear=mask_[1].reshape(-1)
        masknearextend=mask_[2].reshape(-1)
        pre=pre_.reshape(-1,cls)
        tar=tar_.reshape(-1)
        holepre=pre[[mask!=0]]
        holereal=tar[[mask!=0]]

        nearpre=pre[[masknear!=0]]
        nearreal=tar[[masknear!=0]]

        nearextendpre=pre[[masknearextend!=0]]
        extendreal=tar[[masknearextend!=0]]

        loss1 = F.cross_entropy(pre_.view(-1, pre_.size(-1)), tar_.view(-1).long())
        losshole = F.cross_entropy(holepre.view(-1, pre_.size(-1)), holereal.view(-1).long())
        lossnear = F.cross_entropy(nearpre.view(-1, pre_.size(-1)), nearreal.view(-1).long())
        lossnearextend = F.cross_entropy(nearextendpre.view(-1, pre_.size(-1)), extendreal.view(-1).long())
        loss = w1*loss1+w2*losshole+w3*lossnear+w4*lossnearextend
        # loss = w1*loss1+w2*losshole
        # # t2 = time.time()
        # tar = tar_
        # b,h,w=tar.shape
        # pre = torch.argmax((pre_.view(-1, pre_.size(-1))), axis=1).view(tar.shape)
        # tar_strat=self.getstrat(tar)
        # pre_strat=self.getstrat(pre)
        # pre_onehot=torch.nn.functional.one_hot(pre_strat,self.config.stratNum).float()
        # # t3=time.time()
        # # l1time=t2-t1
        # # l2time=t3-t1
        # # crossentropyloss = torch.nn.MSELoss()
        # crossentropyloss = torch.nn.CrossEntropyLoss()
        # loss2 = crossentropyloss(pre_onehot.view(-1,self.config.stratNum), tar_strat.view(-1))
        # # loss2 = crossentropyloss(pre,tar_)
        # loss=loss1+loss2
        # # loss=loss2
        # loss.requires_grad_(True)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        # a=torch.unique(tar[0], sorted=False,dim=0)
        # b=np.array(a.cpu())
        # b
        return loss

def train(generator, generator_s, discriminator, optim_g, optim_d, data_loader, device):
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim))).to(device)
    for step in tqdm(range(args.steps + 1)):
        # Train Discriminator
        optim_d.zero_grad()

        # Forward + Backward with real images
        r_img = next(data_loader).to(device)
        mask = getmaskedmodel(r_img[0])[0].to(device)
        mask=torch.unsqueeze(mask,0)
        mask=mask.repeat(r_img.shape[0],1,1,1)
        maskedmodel = mask*r_img
        r_label = torch.ones(args.batch_size).to(device)
        r_logit = discriminator(r_img).flatten()
        lossD_real = criterion(r_logit, r_label)
        lossD_real.backward()

        # Forward + Backward with fake images

        # �������ݲ���
        #latent_vector = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device)
        f_img = generator(maskedmodel)
        f_label = torch.zeros(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossD_fake = criterion(f_logit, f_label)+ stratLoss(f_img, getmaskedmodel(r_img[0]),r_img)
        lossD_fake.backward()

        optim_d.step()

        # Train Generator
        optim_g.zero_grad()
        f_img = generator(torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device))
        r_label = torch.ones(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossG = criterion(f_logit, r_label)
        lossG.backward()
        optim_g.step()

        exp_mov_avg(generator_s, generator, global_step = step)

        if step % args.sample_interval == 0:
            generator.eval()
            vis = generator(fixed_noise).detach().cpu()
            vis = make_grid(vis, nrow = 4, padding = 5, normalize = True)
            vis = T.ToPILImage()(vis)
            vis.save('samples/vis{:05d}.jpg'.format(step))
            generator.train()
            print("Save sample to samples/vis{:05d}.jpg".format(step))

        if (step + 1) % args.sample_interval == 0 or step == 0:
            # Save the checkpoints.
            torch.save(generator.state_dict(), 'weights/Generator.pth')
            torch.save(generator_s.state_dict(), 'weights/Generator_ema.pth')
            torch.save(discriminator.state_dict(), 'weights/Discriminator.pth')
            print("Save model state.")

# img_size = opt.load_size
stratNum = 45
img_size = 32, 32, 32
patch = 8
block = 6
patch_size = (patch, patch, patch)
embed_dim = patch*patch*patch
block_size = block*block*block
grid = (block, block, block)

mconf = GITConfig(img_size,block_size,grid, 1,patch_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,stratNum=stratNum,
                  n_layer=16, n_head=8, embed_dim=embed_dim, ar_bert_loss=True)
model = Generator(mconf)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 100000,
                        help = "Number of steps for training (Default: 100000)")
    parser.add_argument("--batch-size", type = int, default = 128,
                        help = "Size of each batches (Default: 128)")
    parser.add_argument("--lr", type = float, default = 0.002,
                        help = "Learning rate (Default: 0.002)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--latent-dim", type = int, default = 1024,
                        help = "Dimension of the latent vector")
    parser.add_argument("--data-dir", type = str, default = "dataset/3D32",
                        help = "Data root dir of your training data")
    parser.add_argument("--sample-interval", type = int, default = 1000,
                        help = "Interval for sampling image from generator")
    parser.add_argument("--gpu-id", type = int, default = 1,
                        help = "Select the specific gpu to training")
    args = parser.parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    data_loader = utils.get_dataloader(args.data_dir,img_size=32, batch_size = args.batch_size)


    # Create the log folder
    os.makedirs("weights", exist_ok = True)
    os.makedirs("samples", exist_ok = True)

    # Initialize Generator and Discriminator
    netG = models.Generator(mconf).to(device)
    netG_s = copy.deepcopy(netG)
    netD = models.Discriminator(mconf).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer and lr_scheduler
    optimizer_g = torch.optim.Adam(netG.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(netD.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )

    # Start Training
    train(netG, netG_s, netD, optimizer_g, optimizer_d, data_loader, device)
