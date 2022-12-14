import os
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
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

        w1,w2,w3,w4=0,0.6,0.3,0.1
        b,block_size,cls=pre_.shape
        mask = mask_[0]
        mask = torch.unsqueeze(mask,0)
        mask=mask.repeat(b,1,1,1)
        mask = mask.reshape(-1)

        masknear = torch.unsqueeze(mask_[1],0)
        masknear = masknear.repeat(b,1,1,1).reshape(-1)

        masknearextend=torch.unsqueeze(mask_[2],0).repeat(b,1,1,1).reshape(-1)
        pre=pre_.reshape(-1,cls)
        tar=tar_.reshape(-1)
        holepre=pre[[mask!=0]]
        holereal=tar[[mask!=0]]

        nearpre=pre[[masknear!=0]]
        nearreal=tar[[masknear!=0]]

        nearextendpre=pre[[masknearextend!=0]]
        extendreal=tar[[masknearextend!=0]]

        # loss1 = F.cross_entropy(pre_.view(-1, pre_.size(-1)), tar_.view(-1).long())
        losshole = F.cross_entropy(holepre.view(-1, pre_.size(-1)), holereal.view(-1).long())
        lossnear = F.cross_entropy(nearpre.view(-1, pre_.size(-1)), nearreal.view(-1).long())
        lossnearextend = F.cross_entropy(nearextendpre.view(-1, pre_.size(-1)), extendreal.view(-1).long())
        # loss = w1*loss1+w2*losshole+w3*lossnear+w4*lossnearextend
        loss = w2*losshole+w3*lossnear+w4*lossnearextend
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



def get_trainning_result(gan_label,div_path,counter):#??????????????????????????????????????????????????????

    #???????????????????????????
    #print("?????????????????????????????? = ",trainning_3Dimage.shape,gan_label.shape,type(gan_label))
    #trainning_3Dimage = (trainning_3Dimage - 1.) * 5.
    #trainning_3Dimage = np.rint(trainning_3Dimage)
    #gan_label = (gan_label - 1.) * 5.
    #gan_label = np.rint(gan_label)
    gan_label = gan_label[0]
    #gan_label = torch.squeeze(gan_label,dim = 0)
    a = gan_label.reshape(-1)
    #b = trainning_3Dimage.reshape(-1)
    #acc = (b == a).sum() / float(a.shape[0])#??????
    #now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    #f = open('./train_out/acc.txt', 'a')
    #f.write(str(counter) + '\t' + str(acc)+ '\t' + str(now) + '\t' + '\n')#????????????
    #f.close()

    df = pd.read_csv("/mnt/data/hay/project3d/data/I_J_K.csv", low_memory=False)
    I = df["I"].values.reshape(128, 128, 128)[::4, ::4, ::4].reshape(-1)//4+1
    J = df["J"].values.reshape(128, 128, 128)[::4, ::4, ::4].reshape(-1)//4+1
    K = df["K"].values.reshape(128, 128, 128)[::4, ::4, ::4].reshape(-1)//4+1
    #dataframe = pd.DataFrame({'I': I, 'J': J, 'K': K, 'StratCode': trainning_3Dimage.reshape(-1)})
    dataframelable = pd.DataFrame({'I': I, 'J': J, 'K': K, 'StratCode': gan_label.reshape(-1)})
    #dataframe.to_csv(div_path + "/out_trainning_3Dimage" + str(counter) + ".csv", index=False)
    dataframelable.to_csv(div_path + "/out_trainning_3Dimagelable" + str(counter) + ".csv", index=False)


def train(generator, generator_s, discriminator, optim_g, optim_d, data_loader, device):

    # fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim))).to(device)
    for step in tqdm(range(start_step, args.steps + 1)):
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

        # ????????????????????????????????
        #latent_vector = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device)
        f_img,_ = generator(maskedmodel)
        f_label = torch.zeros(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossD_fake = criterion(f_logit, f_label)
        lossD_fake.backward()

        optim_d.step()

        # Train Generator
        optim_g.zero_grad()
        f_img,_logit = generator(maskedmodel)
        r_label = torch.ones(args.batch_size).to(device)
        f_logit = discriminator(f_img).flatten()
        lossG = criterion(f_logit, r_label)+ stratLoss(_logit, getmaskedmodel(r_img[0]),r_img)
        lossG.backward()
        optim_g.step()
        #print("???????????????loss??? = ", lossG, lossD_fake,lossD_real)
        exp_mov_avg(generator_s, generator, global_step = step)

        if step % args.sample_interval == 0:
            generator.eval()
            gan_label = generator(maskedmodel)[0].detach().cpu()
            get_trainning_result(gan_label,'output',step)

            vis = generator(maskedmodel)[0][:,15,:,:].detach().cpu().float()
            vis = torch.unsqueeze(vis,1)
            vis = make_grid(vis, nrow = 4, padding = 5, normalize = False)
            # vis = make_grid(vis, nrow = 4, padding = 5, normalize = True)

            # vis = vis.astype(np.uint8)
            vis = T.ToPILImage()(vis)
            vis.save('samples/vis{:05d}.jpg'.format(step))
            generator.train()
            print("Save sample to samples/vis{:05d}.jpg".format(step))

        if step % args.sample_interval == 0 or step == 0:
            # Save the checkpoints.
            print('step:', step)
            checkpoint = {
                'generator': generator.state_dict(),
                'generator_s': generator_s.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'step': step,
                'lossG': lossG,
                'lossDF': lossD_fake,
                'lossDR': lossD_real
            }
            torch.save(checkpoint, 'weights/ckpt_best1.pth' )
            print("Save model state.")

# img_size = opt.load_size
stratNum = 11
img_size = 32, 32, 32
# ??????????????????
patch = 4
# ????????????
block = 8
patch_size = (patch, patch, patch)
embed_dim = patch*patch*patch
block_size = block*block*block
grid = (block, block, block)

mconf = GITConfig(img_size,block_size,grid, 1,patch_size,
                  embd_pdrop=0.0, resid_pdrop=0.0, dropout=0.0,stratNum=stratNum,
                  blocks=8, num_heads=8, embed_dim=embed_dim, ar_bert_loss=True)
model = Generator(mconf)

start_step = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 100000,
                        help = "Number of steps for training (Default: 100000)")
    parser.add_argument("--batch-size", type = int, default = 4,
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
    optimizer_g = torch.optim.Adam(netG.parameters(), lr = 0.002,
        betas = (args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(netD.parameters(), lr = 0.00002,
        betas = (args.beta1, args.beta2)
    )

#??????checkpoint???????????????????????????
    if os.path.exists("weights/ckpt_best1.pth"):
        path_checkpoint = "weights/ckpt_best.pth"  # ????????????
        checkpoint = torch.load(path_checkpoint)  # ????????????

        netG.load_state_dict(checkpoint['generator'])  # ???????????????????????????
        netG_s.load_state_dict(checkpoint['generator_s'])
        netD.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])  # ?????????????????????
        start_step = checkpoint['step']  # ???????????????epoch
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

    # Start Training
    train(netG, netG_s, netD, optimizer_g, optimizer_d, data_loader, device)
