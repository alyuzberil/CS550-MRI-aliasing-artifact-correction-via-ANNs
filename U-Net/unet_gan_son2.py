
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim

import math
import numpy as np

class Block(nn.Module):
  def __init__(self,in_ch,out_ch,kernel_size,padding):
    super().__init__()
    self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size,padding=padding)
    self.relu=nn.ReLU()
    self.conv2=nn.Conv2d(out_ch,out_ch,kernel_size,padding=padding)
   
   
  def forward(self,x):
    return self.relu(self.conv2(self.relu(self.conv1(x))))

class Contracting_Path(nn.Module):
  def __init__(self,channels=(1,64,128,256)):
    super().__init__()
    self.blocks = nn.ModuleList([Block(channels[i],channels[i+1],3,1)  for i in range(3)])
    self.max_pool = nn.MaxPool2d(2,stride=2)

  def forward(self,x):
    features = []
    for block in self.blocks:
      x=block(x)
      features.append(x)
      x = self.max_pool(x)
    return features

class Middle_Block(nn.Module):
  def __init__(self,in_ch,out_ch,kernel_size,padding):
    super().__init__()
    self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size,padding=padding)
    self.relu=nn.ReLU()
    self.conv2=nn.Conv2d(out_ch,out_ch,kernel_size,padding=padding)

  def forward(self,x):
    return self.relu(self.conv2(self.relu(self.conv1(x))))

class Expansive_Block(nn.Module):
  def __init__(self,in_ch,out_ch):
    super().__init__()
    self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
    self.relu=nn.ReLU()
    self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
  def forward(self,x):
    return self.relu(self.conv2(self.relu(self.conv1(x))))

class Decoder(nn.Module):
  def __init__(self,channels=(256,128,64)):
    super().__init__()
    self.channels = channels
    self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1],2,2) for i in range(len(channels)-1)] )
    self.dec_blocks = nn.ModuleList([Expansive_Block(channels[i], channels[i+1]) for i in range(2)])

  def forward(self,x,encoder_features):
    for i in range(2):
     
      x= self.upconvs[i](x)
      enc_ftrs = self.crop(encoder_features[i],x)
      x= torch.cat([x,enc_ftrs],dim=1)
      x=self.dec_blocks[i](x)
    return x
 
 
  def crop(self,enc_ftrs,x):
    _,_,H,W = x.shape
    #enc_ftrs = torchvision.transforms.CenterCrop(enc_ftrs,H,W)
    #enc_ftrs = enc_ftrs[0:H,0:W]
	ww= int(round((w - W) / 2.))
    hh = int(round((h - H) / 2.))
	enc_ftrs[ww:ww+W,hh:hh+H]
    return enc_ftrs

class U_Net(nn.Module):
  def __init__(self,contracting_channels=(1,64,128,256),expansive_channels=(256,128,64),output_size=(256,256)):
    super().__init__()
    self.contracting = Contracting_Path(contracting_channels)
    self.decoder = Decoder(expansive_channels)
    self.final_conv =nn.Conv2d(64,1,1)
  def forward(self,x):
    features = self.contracting(x)
    u_net_output=self.decoder(features[::-1][0],features[::-1][1:])
    out = self.final_conv(u_net_output)
    return out


class Discriminator_Block(nn.Module):
  #patch GAN
  def __init__(self):
    super().__init__()
    self.conv1 =nn.Conv2d(1,64,4,stride=2,padding=1)
    self.relu=nn.LeakyReLU()
    self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=1)
    self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=1)
    self.conv4 = nn.Conv2d(256,512,4,stride=1,padding=1)
    self.conv5 = nn.Conv2d(512,1,4,stride=1,padding=1)
  def forward(self,x):
    return self.conv5(self.relu(self.conv4(self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))))))

import random
import h5py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#first_input = torch.cat((x,y),1)
generator = U_Net()
generator.to(device)
discriminator = Discriminator_Block()
discriminator.to(device)
#define the optimizers
optimizer_G = optim.Adam(generator.parameters(),lr=0.0002, betas=(0.1,0.999))
optimizer_D = optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.1,0.999))

#define loss functions
BCE_loss = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()

def c_psnr(img1,img2):
  # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0/ math.sqrt(mse))

def fft2c(im):
  return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im)))

def ifft2c(d):
  return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d)))

def data_consistency(image,y):
  image=image.reshape((256,256))

  k_space = fft2c(image)
  y = np.reshape(y,(256,256))
  full_k_space = fft2c(y)

  #replace the central calibration region first

  new_k_space=k_space
  for i in range(256,0,3):
    new_k_space[:,i] = full_k_space[:,i]

  new_k_space[97:157,97:157] = full_k_space[97:157,97:157]
  new_image = ifft2c(new_k_space)
  return new_image

def train(number_of_epoch,alpha):
  print('check')
  for epoch in range(number_of_epoch):
    print('Epoch: ' + str(epoch))
    for i in range(419):
      im_save = np.zeros((number_of_epoch,256,768))
      imname= '/auto/data/el/train/train_' + str(i+1) +'.png'
      image = mpimg.imread(imname)
      #undersampled part
      x = image[:,0:256]
      #fully sampled part
      y = image[:,256:512]
      x = torch.from_numpy(x)
      x=torch.reshape(x,(1,1,256,256))

      y = torch.from_numpy(y)
      y=torch.reshape(y,(1,1,256,256))
     
      x = x.to(device)
      y = y.to(device)    
     
      D_losses =[]
      G_losses =[]

      ssim_train=[]
      psnr_train=[]
      #train discriminator D with real source, real target
      optimizer_D.zero_grad()  
      discriminator_output =discriminator(y)
      discriminator_real_loss = BCE_loss(discriminator_output,Variable(torch.ones(discriminator_output.size()).cuda()))
      generator_output = generator(x)

      #train the discriminator with fake target
      #the_two_channel_input = torch.cat((x,G_result.detach()),1)
      discriminator_output = discriminator(generator_output.detach())
      discriminator_fake_loss = BCE_loss(discriminator_output,Variable(torch.zeros(discriminator_output.size()).cuda()))
      #discriminator_fake_loss = BCE_loss(discriminator_output,Variable(torch.zeros(discriminator_output.size())))
      #print('Discriminator Fake Loss'+str(D_fake_loss))
      disc_train_loss =(discriminator_real_loss + discriminator_fake_loss) * 0.5
      #print('Discriminator Train Loss'+str(D_train_loss))
      disc_train_loss.backward()
      optimizer_D.step()

      D_losses.append(disc_train_loss)

      #train generator
      optimizer_G.zero_grad()
      discriminator_output_fake =discriminator(generator_output)

      #y is the real target
      G_bce =BCE_loss(discriminator_output_fake, Variable(torch.ones(discriminator_output_fake.size()).cuda()))
      G_L1 =L1_loss(generator_output,y)
      G_train_loss = G_bce + alpha*G_L1

      G_train_loss.backward()
      optimizer_G.step()
      #print('Generator Train Loss' + str(G_train_loss))
      G_losses.append(G_train_loss)
     
      x = x.detach().cpu().numpy()
      x=x.reshape((256,256))

      y = y.detach().cpu().numpy()
      y=y.reshape((256,256))

      image = generator_output.detach().cpu().numpy()
      image=image.reshape((256,256))
     
      #map the values btw 0 and 1
      image = image / np.max(image)

      ssim_im = ssim(y, image, data_range=image.max() - image.min())

      psnr_im = c_psnr(image/np.max(image),y)

      ssim_train.append(ssim_im)
      psnr_train.append(psnr_im)

      im_to_print_index = random.randint(1,299)
      if i==30:
          img = np.zeros((256,768))
          #first source
          img[:,0:256] = x
          #fake
          img[:,256:512] = image
          img[:,512:768] = y
          #save the image
          mpimg.imsave(('/auto/data/el/results/epoch_'+str(epoch)+'alpha' +str(alpha)+'_output'+str(i)+'.png'),img,cmap = 'gray')
          im_save[epoch,:,:] = img
    if epoch%10 == 0:
        #validate on validation set
        ssim_array_val = []
        psnr_array_val =[]
        
        val_results = np.zeros((81,256,1024))
        for j in range(81):
        	imname= '/auto/data/el/val/val_' + str(j+1) +'.png'
        	image = mpimg.imread(imname)
        	#undersampled part
        	x = image[:,0:256]
        	#fully sampled part
        	y = image[:,256:512]
        	x = torch.from_numpy(x)
        	x=torch.reshape(x,(1,1,256,256)).to(device)
        	image = generator(x)
        
        	y = torch.from_numpy(y)
        	y=torch.reshape(y,(1,1,256,256))
        
        
        	x = x.detach().cpu().numpy()
        	x=x.reshape((256,256))
        	y = y.detach().cpu().numpy()
        	y=y.reshape((256,256))
        	image = image.detach().cpu().numpy()
        	image=image.reshape((256,256))
        
        	#map the values btw 0 and 1
        	image = image / np.max(image)
        
        	ssim_im = ssim(y, image, data_range=image.max() - image.min())
        	ssim_array_val.append(ssim_im)
        
        	psnr_im = c_psnr(image/np.max(image),y)
        	psnr_array_val.append(psnr_im)
        
        	#error image
        	error_img = abs(y-image)
        	 
        	img = np.zeros((256,1024));
        	#first source
        	img[:,0:256] = x
        	#fake
        	img[:,256:512] = image
        	img[:,512:768] = y
        	img[:,768:1024] = error_img
        
        	val_results[j,:,:] = img
        	#save the image
        	mpimg.imsave(('/auto/data/el/results/output_val'+str(j)+'alpha' +str(alpha[a])+'epoch'+str(epoch)+'.png'),img,cmap = 'gray')
        
        #print('ssim' + str(len(ssim_array_val)))
        #val_results_im = f2.create_dataset("val_results_im" + str(alpha), (81,256,1024), data=val_results)
        val_ssims = f2.create_dataset("val_ssims" + str(alpha) +'epoch'+str(epoch)+, (81,1),data=ssim_array_val)
        val_psnrs = f2.create_dataset("val_psnr" + str(alpha) + 'epoch'+str(epoch)+, (81,1), data=psnr_array_val)
        
        #test
        ssim_array_test = []
        psnr_array_test =[]
        results = np.zeros((81,256,1024))
        for j in range(81):
        	imname= '/auto/data/el/test/test_' + str(j+1) +'.png'
        	image = mpimg.imread(imname)
        	#undersampled part
        	x = image[:,0:256]
        	#fully sampled part
        	y = image[:,256:512]
        	x = torch.from_numpy(x)
        	x=torch.reshape(x,(1,1,256,256)).to(device)
        	image = generator(x)
        
        	y = torch.from_numpy(y)
        	y=torch.reshape(y,(1,1,256,256))
        
        
        	x = x.detach().cpu().numpy()
        	x=x.reshape((256,256))
        	y = y.detach().cpu().numpy()
        	y=y.reshape((256,256))
        	image = image.detach().cpu().numpy()
        	image = data_consistency(image,y)
        	image=image.reshape((256,256))
        	image = abs(image)
        	#map the values btw 0 and 1
        	image = image / np.max(image)
        
        	ssim_im = ssim(y, image, data_range=image.max() - image.min())
        	ssim_array_test.append(ssim_im)
        
        	psnr_im = c_psnr(image/np.max(image),y)
        	psnr_array_test.append(psnr_im)
        
        	#error image
        	error_img = abs(y-image)
        	 
        	img = np.zeros((256,1024));
        	#first source
        	img[:,0:256] = x
        	#fake
        	img[:,256:512] = image
        	img[:,512:768] = y
        	img[:,768:1024] = error_img
        
        	results[j,:,:] = img
        	#save the image
        	mpimg.imsave(('/auto/data/el/results/output_test'+str(j)+'alpha' +str(alpha)+'epoch'+str(epoch)+'.png'),img,cmap = 'gray')
        
        
        #test_results = f2.create_dataset("test_results" + str(alpha),(81,256,1024),data=results)
        test_ssims = f2.create_dataset("test_ssims"+ str(alpha)+'epoch'+str(epoch)+,data=ssim_array_test)
        test_psnrs = f2.create_dataset("test_psnr"+ str(alpha)+'epoch'+str(epoch)+, data=psnr_array_test)

    g_l = torch.mean(torch.stack(G_losses))
    d_l = torch.mean(torch.stack(D_losses))

    G_epoch.append(g_l)
    D_epoch.append(d_l)
    #compute ssim and psnr
    ssim_avg = np.mean(ssim_train)
    psnr_avg = np.mean(psnr_train)

    ssim_epoch.append(ssim_avg)
    psnr_epoch.append(psnr_avg)
  return im_save,G_epoch, D_epoch, ssim_epoch, psnr_epoch

#try alphas with 1 fold cross val
f2 = h5py.File("validation_res.hdf5","w")
alpha=[200,500,1000]
for a in range(len(alpha)):
    
    number_of_epoch=100
    #create a file to save training images and loss arrays
    G_epoch=[]
    D_epoch=[]
    ssim_epoch=[]
    psnr_epoch=[]
    #try it for the images dataset and compare the results
    #print('training')
    im_save,G_epoch,D_epoch, ssim_epoch, psnr_epoch=train(number_of_epoch,alpha[a])
    G_epoch = torch.stack(G_epoch)
    G_epoch = G_epoch.detach().cpu().numpy()
    
    D_epoch = torch.stack(D_epoch)
    D_epoch = D_epoch.detach().cpu().numpy()
    #save all these into a file4
    #im_save_file = f2.create_dataset("im_save_file" + str(alpha), (number_of_epoch,256,768), data=im_save)
    g_losses_train = f2.create_dataset("g_losses_train" + str(alpha[a]), (number_of_epoch,1), data =G_epoch)
    d_losses_train = f2.create_dataset("d_losses_train" + str(alpha[a]), (number_of_epoch,1), data =D_epoch)
    
    ssim_res_train = f2.create_dataset("ssim_res_train" + str(alpha[a]), (number_of_epoch,1), data =ssim_epoch)
    psnr_res_train = f2.create_dataset("psnr_res_train" + str(alpha[a]),(number_of_epoch,1), data=psnr_epoch)


#validate on validation set
ssim_array_val = []
psnr_array_val =[]

val_results = np.zeros((81,256,1024))
for j in range(81):
	imname= '/auto/data/el/val/val_' + str(j+1) +'.png'
	image = mpimg.imread(imname)
	#undersampled part
	x = image[:,0:256]
	#fully sampled part
	y = image[:,256:512]
	x = torch.from_numpy(x)
	x=torch.reshape(x,(1,1,256,256)).to(device)
	image = generator(x)

	y = torch.from_numpy(y)
	y=torch.reshape(y,(1,1,256,256))


	x = x.detach().cpu().numpy()
	x=x.reshape((256,256))
	y = y.detach().cpu().numpy()
	y=y.reshape((256,256))
	image = image.detach().cpu().numpy()
	image=image.reshape((256,256))

	#map the values btw 0 and 1
	image = image / np.max(image)

	ssim_im = ssim(y, image, data_range=image.max() - image.min())
	ssim_array_val.append(ssim_im)

	psnr_im = c_psnr(image/np.max(image),y)
	psnr_array_val.append(psnr_im)

	#error image
	error_img = abs(y-image)
	 
	img = np.zeros((256,1024));
	#first source
	img[:,0:256] = x
	#fake
	img[:,256:512] = image
	img[:,512:768] = y
	img[:,768:1024] = error_img

	val_results[j,:,:] = img
	#save the image
	mpimg.imsave(('/auto/data/el/results/output_val'+str(j)+'alpha' +str(alpha[a])+'.png'),img,cmap = 'gray')

#print('ssim' + str(len(ssim_array_val)))
#val_results_im = f2.create_dataset("val_results_im" + str(alpha), (81,256,1024), data=val_results)
val_ssims = f2.create_dataset("val_ssims" + str(alpha), (81,1),data=ssim_array_val)
val_psnrs = f2.create_dataset("val_psnr" + str(alpha), (81,1), data=psnr_array_val)

#test
ssim_array_test = []
psnr_array_test =[]
results = np.zeros((81,256,1024))
for j in range(81):
	imname= '/auto/data/el/test/test_' + str(j+1) +'.png'
	image = mpimg.imread(imname)
	#undersampled part
	x = image[:,0:256]
	#fully sampled part
	y = image[:,256:512]
	x = torch.from_numpy(x)
	x=torch.reshape(x,(1,1,256,256)).to(device)
	image = generator(x)

	y = torch.from_numpy(y)
	y=torch.reshape(y,(1,1,256,256))


	x = x.detach().cpu().numpy()
	x=x.reshape((256,256))
	y = y.detach().cpu().numpy()
	y=y.reshape((256,256))
	image = image.detach().cpu().numpy()
	image = data_consistency(image,y)
	image=image.reshape((256,256))
	image = abs(image)
	#map the values btw 0 and 1
	image = image / np.max(image)

	ssim_im = ssim(y, image, data_range=image.max() - image.min())
	ssim_array_test.append(ssim_im)

	psnr_im = c_psnr(image/np.max(image),y)
	psnr_array_test.append(psnr_im)

	#error image
	error_img = abs(y-image)
	 
	img = np.zeros((256,1024));
	#first source
	img[:,0:256] = x
	#fake
	img[:,256:512] = image
	img[:,512:768] = y
	img[:,768:1024] = error_img

	results[j,:,:] = img
	#save the image
	mpimg.imsave(('/auto/datael/results/output_test'+str(j)+'alpha' +str(alpha)+'.png'),img,cmap = 'gray')


#test_results = f2.create_dataset("test_results" + str(alpha),(81,256,1024),data=results)
test_ssims = f2.create_dataset("test_ssims"+ str(alpha),data=ssim_array_test)
test_psnrs = f2.create_dataset("test_psnr"+ str(alpha), data=psnr_array_test)

f2.close()