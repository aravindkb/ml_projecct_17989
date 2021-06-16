
if __name__ == "__main__":


  import torch
  import torch.nn as nn
  import numpy as np
  from tqdm import tqdm
  from torchvision.utils import save_image, make_grid


  dataset_path = '~/datasets'

  #cuda = True
  #cuda = False
  cuda = torch.cuda.is_available()
  DEVICE = torch.device("cuda" if cuda else "cpu")


  batch_size = 100
  x_dim  = 784
  hidden_dim = 400
  latent_dim = 20
  lr = 1e-3
  epochs = 1

  from torchvision.datasets import MNIST
  import torchvision.transforms as transforms
  from torch.utils.data import DataLoader, RandomSampler 


  mnist_transform = transforms.Compose([
          transforms.ToTensor(),
  ])

  kwargs = {'num_workers': 1, 'pin_memory': True} 

  train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
  test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
  test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)

  rand_data = DataLoader(dataset = test_dataset,batch_size= 1,shuffle=True, **kwargs)


  class Encoder(nn.Module):
      
      def __init__(self, input_dim, hidden_dim, latent_dim):
          super(Encoder, self).__init__()

          self.FC_input = nn.Linear(input_dim, hidden_dim)
          self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
          self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
          self.FC_var   = nn.Linear (hidden_dim, latent_dim)
          
          self.LeakyReLU = nn.LeakyReLU(0.2)
          
          self.training = True
          
      def forward(self, x):
          h_       = self.LeakyReLU(self.FC_input(x))
          h_       = self.LeakyReLU(self.FC_input2(h_))
          mean     = self.FC_mean(h_)
          log_var  = self.FC_var(h_)                      
                                                        
          
          return mean, log_var

  class Decoder(nn.Module):
      def __init__(self, latent_dim, hidden_dim, output_dim):
          super(Decoder, self).__init__()
          self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
          self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
          self.FC_output = nn.Linear(hidden_dim, output_dim)
          
          self.LeakyReLU = nn.LeakyReLU(0.2)
          
      def forward(self, x):
          h     = self.LeakyReLU(self.FC_hidden(x))
          h     = self.LeakyReLU(self.FC_hidden2(h))
          
          x_hat = torch.sigmoid(self.FC_output(h))
          return x_hat

  class Model(nn.Module):
      def __init__(self, Encoder, Decoder):
          super(Model, self).__init__()
          self.Encoder = Encoder
          self.Decoder = Decoder
          
      def reparameterization(self, mean, var):
          epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
          z = mean + var*epsilon                          # reparameterization trick
          return z
          
                  
      def forward(self, x):
          mean, log_var = self.Encoder(x)
          z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
          x_hat            = self.Decoder(z)
          
          return x_hat, mean, log_var, z

  encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
  decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

  model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)



  from torch.optim import Adam

  BCE_loss = nn.BCELoss()

  beta = 5

  def loss_function(x, x_hat, mean, log_var):
      reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
      KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
      #beta value > 1 will cause the latent variables to learn disentangled representations
      return reproduction_loss + beta * KLD


  optimizer = Adam(model.parameters(), lr=lr)

 





  

  print("Start training b-VAE...")
  model.train()

  for epoch in range(epochs):
      overall_loss = 0
      for batch_idx, (x, _) in enumerate(train_loader):
          x = x.view(batch_size, x_dim)
          x = x.to(DEVICE)

          optimizer.zero_grad()

          x_hat, mean, log_var,z = model(x)
          loss = loss_function(x, x_hat, mean, log_var)
          
          overall_loss += loss.item()
          
          loss.backward()
          optimizer.step()
          
      print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
      
  print("Finish!!") 

  """### Step 5. Generate images from test dataset"""

  import matplotlib.pyplot as plt

  model.eval()

  with torch.no_grad():
      for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
          x = x.view(batch_size, x_dim)
          x = x.to(DEVICE)
          
          x_hat, i, j,z = model(x)


          break

  encoder1 = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
  decoder1 = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
  model1 = Model(Encoder=encoder1, Decoder=decoder1).to(DEVICE)
  model_save_name = 'bvaemodel.pt'
  path = model_save_name
  model1.load_state_dict(torch.load(path,map_location=torch.device('cpu')))



  def show_image(x, idx):
      x = x.view(batch_size, 28, 28)

      fig = plt.figure()
      plt.imshow(x[idx].cpu().numpy(), cmap ="binary_r")
      plt.title("trained model")
      plt.show()

  def show_single_img(x):
      x = x.view(28,28)
      fig = plt.figure()
      plt.imshow(x.cpu().numpy(), cmap ="binary_r")
      plt.title("pretrained model")
      plt.show()

  model1.eval()



  import random
  
  rc = z[0]
  with torch.no_grad():
    x_rand = decoder1(rc)
  show_single_img(x_rand)
  #here we are showing imgaes by changing the 3th latent variable, of the pre trained model
  k = 3
  with torch.no_grad():
    for i in range(-10,8):
      y = rc.detach().clone()
      pz = y[k]
      y[k] = y[k] + i*0.3
      img = decoder1(y)
      show_single_img(img)

  #here we are showing imgaes by changing the 11th latent variable, of the pre trained model
  k =11
  with torch.no_grad():
    for i in range(-10,8):
      y = rc.detach().clone()
      pz = y[k]
      y[k] = y[k] + i*0.3
      img = decoder1(y)
      show_single_img(img)





  with torch.no_grad():
    x_rand = decoder(z)
  x_rand = x_rand.cpu().detach().numpy()
  #show_single_img(x_rand)

  show_image(x, idx=0)
  show_image(x_hat, idx=0)


 #random generated sample from latent space, trained model

  with torch.no_grad():
      noise = torch.randn(batch_size, latent_dim).to(DEVICE)
      generated_images = decoder(noise)

  show_image(generated_images, idx=12)


