import copy
import torch
import utils
from main import mconf,train
from torch import device
import models


beta1 = 0.0
beta2 = 0.99
netG = models.Generator(mconf)

netD = models.Discriminator(mconf)

# Optimizer and lr_scheduler
optimizer_g = torch.optim.Adam(netG.parameters(), lr=0.002,
                               betas=(beta1, beta2)
                               )
optimizer_d = torch.optim.Adam(netD.parameters(), lr=0.0002,
                               betas=(beta1, beta2)
                               )

checkpoint = torch.load('weights/Model.pth')

netG.load_state_dict(checkpoint['generator.state_dict'])
netG_s = copy.deepcopy(netG)
netD.load_state_dict(checkpoint['discriminator.state_dict'])
#optimizer_g.load_state_dict(checkpoint['optimizer_g'])
#optimizer_d.load_state_dict(checkpoint['optimizer_d'])



netG.eval()
netD.eval()

data_loader = utils.get_dataloader("dataset/3D32",img_size=32, batch_size = 4)

train(netG, netG_s, netD, optimizer_g, optimizer_d, data_loader, device)
