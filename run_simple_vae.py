import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset.mnist_loader import MnistDataset
import numpy as np
from tqdm import tqdm
import cv2
import torchvision
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using device: {device}")


class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        
        self.common_fc = nn.Sequential(
            nn.Linear(28*28, 196),
            nn.Tanh(),
            nn.Linear(196, 48),
            nn.Tanh(),
        )
        self.mean_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
        self.log_var_fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
        self.decoder_fcs = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 48),
            nn.Tanh(),
            nn.Linear(48, 196),
            nn.Tanh(),
            nn.Linear(196, 28*28),
            nn.Tanh()
        )
        
    def forward(self, x):
        # B,C,H,W
        ## Encoder part
        mean, log_var = self.encode(x)
        ## Sampling
        z = self.sample(mean, log_var)
        ## Decoder part
        out = self.decode(z)
        return mean, log_var, out

    def encode(self, x):
        out = self.common_fc(torch.flatten(x, start_dim=1))
        mean = self.mean_fc(out)
        log_var = self.log_var_fc(out)
        return mean, log_var
    
    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        z = z * std + mean
        return z
    
    def decode(self, z):
        out = self.decoder_fcs(z)
        out = out.reshape((z.size(0), 1, 28, 28))
        return out
        
        

def train_vae():
    # Create the data set and the data loader
    mnist = MnistDataset('train', im_path='data/train/images')
    mnist_test = MnistDataset('test', im_path='data/test/images')
    
    # Increase batch size to utilize more GPU memory
    batch_size = 8192 if torch.cuda.device_count() > 1 else 4096
    num_workers = 8  # Increase workers for faster data loading
    
    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Instantiate the model
    model = VAEModel().to(device)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        print(f"Batch size increased to {batch_size} to utilize multiple GPUs")
    
    # Specify training parameters
    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()
    
    recon_losses = []
    kl_losses = []
    losses = []
    # Run training for 10 epochs
    for epoch_idx in range(num_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optimizer.zero_grad()
            mean, log_var, out = model(im)
            
            cv2.imwrite('input.jpeg', 255*((im+1)/2).detach().cpu().numpy()[0, 0])
            cv2.imwrite('output.jpeg', 255 * ((out + 1) / 2).detach().cpu().numpy()[0, 0])
            
            kl_loss = torch.mean(0.5* torch.sum(torch.exp(log_var) + mean**2 - 1 -log_var, dim=-1))
            recon_loss = criterion(out, im)
            loss = recon_loss + 0.00001 * kl_loss
            recon_losses.append(recon_loss.item())
            losses.append(loss.item())
            kl_losses.append(kl_loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Recon Loss : {:.4f} | KL Loss : {:4f}'.format(
            epoch_idx+1,
            np.mean(recon_losses),
            np.mean(kl_losses)
        ))
        
    print('Done Training ...')
    # Run a reconstruction for some sample test images
    idxs = torch.randint(0, len(mnist_test)-1, (100, ))
    ims = torch.cat([mnist_test[idx][0][None, :] for idx in idxs]).float()
    
    #_, _, generated_im = model(ims)
    _, _, generated_im = model(ims.to(device))
    
    ims = (ims + 1)/ 2
    generated_im = 1- (generated_im + 1) / 2
    out = torch.hstack([ims, generated_im.cpu()])

    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstruction.png')
    print('Done Reconstruction...')
    

if __name__ == '__main__':
    train_vae()
