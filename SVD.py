# %% codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import os
import cv2 as cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import legendre
from scipy.optimize import leastsq
from scipy.interpolate import griddata
import scipy
import h5py
from scipy.spatial import KDTree
from pathlib import Path
import tifffile
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
#%load_ext memory_profiler
# %% codecell
Path.cwd()
allFiles=glob.glob('*.*')
allFiles
img0=cv2.imread(allFiles[0],-1)[20:-20,20:-20]
nx,ny=img0.shape
maty=img0.flatten().shape[0]
matx=np.size(allFiles)
data_mat=np.zeros((matx,maty))

k=-1
for file_ in allFiles:
    k+=1
    temp=cv2.imread(file_,-1)[20:-20,20:-20]
    data_mat[k,:]=temp.flatten()
# %% codecell
pca=PCA(n_components=50)
u,s,v=pca._fit(data_mat.T)

# %% codecell
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA

A=data_mat.T

transformerICA = FastICA(n_components=50,random_state=0)
ICA_img = transformerICA.fit_transform(A)


# %% codecell
fig,ax=plt.subplots(7,7,figsize=(14,12))
for i in range(0,7):
    for j in range(0,7):
        ax[i][j].imshow(data_mat.T[:,7*i+j].reshape(nx,ny),cmap='bwr')
        ax[i][j].axes.get_xaxis().set_ticks([])
        #temp=np.abs(ICA_img[:,7*i+j]).reshape(nx,ny)
        #ax[i][j].set_title(str(np.unravel_index(temp.argmax(), temp.shape)))
plt.suptitle('Original',fontsize=(22))

# %% codecell
fig,ax=plt.subplots(7,7,figsize=(14,12))
for i in range(0,7):
    for j in range(0,7):
        ax[i][j].imshow(ICA_img[:,7*i+j].reshape(nx,ny),cmap='bwr')
        ax[i][j].axes.get_xaxis().set_ticks([])
        temp=np.abs(ICA_img[:,7*i+j]).reshape(nx,ny)
        ax[i][j].set_title(str(np.unravel_index(temp.argmax(), temp.shape)))
plt.suptitle('ICA Analysis of Poke',fontsize=(22))
# %% codecell
fig,ax=plt.subplots(7,7,figsize=(14,12))
for i in range(0,7):
    for j in range(0,7):
        ax[i][j].imshow(u[:,7*i+j].reshape(nx,ny),cmap='bwr')
plt.suptitle('SVD Analysis of Poke',fontsize=(22))
# %% codecell
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
# %% codecell
A=data_mat.T
inputArr=A/(2**16-1)
inputArr=inputArr.astype(np.float32)
inputArr=inputArr.reshape(nx,ny,50)
inputArr=np.swapaxes(inputArr,0,-1)
inputArr=np.swapaxes(inputArr,1,2)
inputArr.shape
#nputArr=inputArr[np.newaxis,:,:,:]
#print(inputArr.shape)
#plt.imshow(inputArr[1,:,:])
#inputArr=tf.constant(inputArr,dtype=tf.float32)
#inputArr
# %% codecell
A=A/(2**16-1)
testA=tf.constant(A[np.newaxis,:,:].swapaxes(1,2),dtype=tf.float32)
testA.shape
# %% codecell
latent_dim = 50

class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      #layers.Flatten(),
      layers.Dense(latent_dim,input_shape=(8320,),activation='linear',kernel_regularizer='l2')
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(8320,activation='linear',kernel_regularizer='l2')
      #layers.Reshape((80, 104))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# %% codecell
cb=tf.keras.callbacks.TensorBoard(
    log_dir='logflat', histogram_freq=1, write_graph=True, write_images=False,
    update_freq='epoch')
estop=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10,min_delta=0.001)

autoencoder.fit(testA,testA,epochs=100,shuffle=True,callbacks=[cb,estop])
# %% codecell
x_test=inputArr
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
# %% codecell
autoencoder.summary()
# %% codecell
enc1=autoencoder.encoder(2*testA.numpy()[0,0:1,:]).numpy()[0,:]
enc2=autoencoder.encoder(testA.numpy()[0,1:2,:]).numpy()[0,:]
#np.dot(enc1.numpy().ravel(),enc2.numpy().ravel())
enc1
# %% codecell
a,b,c,d=autoencoder.get_weights()
print(a.shape,b.shape,c.shape,d.shape)
# %% codecell
plt.imshow(c[:,2].reshape(nx,ny))
# %% codecell
from sklearn.decomposition import PCA
u,s,v=np.linalg.svd(c.T,full_matrices=False)
new_u=np.reshape(u.T,[20,nx,ny])
new_u.shape
plt.imshow(new_u[0,:,:])
#test=PCA(n_components=20).fit(c.T)
#test.components_.shape
# %% codecell
fig,ax=plt.subplots(6,2,figsize=(15,15))
j=23
for row,j in enumerate([0,4,7,15,23,29]):
    ax[row][0].imshow(autoencoder(testA).numpy()[0,j,:].reshape(nx,ny))
    ax[row][1].imshow(testA.numpy()[0,j,:].reshape(nx,ny))
# %% codecell
test=np.zeros((1,20))
test[0,4]=1
test
plt.imshow(autoencoder.decoder(test).numpy().reshape(nx,ny))
# %% codecell
fig,ax=plt.subplots(1,3,figsize=(10,10))
j=32
ax[0].imshow(inputArr[j,:,:])
ax[1].imshow(autoencoder(inputArr[j:j+1,:,:]).numpy()[0,:,:])
ax[2].imshow(inputArr[j,:,:]-autoencoder(inputArr[j:j+1,:,:]).numpy()[0,:,:])
# %% codecell
test=np.expand_dims(np.zeros(50),1).T
test[0,3]=1
plt.imshow(autoencoder.decoder(test).numpy()[0,:,:])
# %% codecell
inputArr[20,:,:]-autoencoder(inputArr[20:21,:,:]).numpy()
# %% codecell
pca_score=np.zeros(49)
pca_score_std=np.zeros(49)
for n_ in range(1,50):

    pca=PCA(n_components=n_)
    u,s,v=pca._fit(data_mat.T)
    #pca_score[n_-1]=pca.score(data_mat.T)
    pca_score[n_-1]=np.mean(cross_val_score(pca,data_mat.T,cv=))
    pca_score_std[n_-1]=np.std(cross_val_score(pca,data_mat.T,cv=3))
plt.errorbar(range(1,50),pca_score,pca_score_std)

# %% codecell
allFiles=glob.glob(r'D:\L4\AustinPoke\ColdWF\*.tif')
img0=cv2.imread(allFiles[0],-1)[20:-20,20:-20]
nx,ny=img0.shape
maty=img0.flatten().shape[0]
matx=np.size(allFiles)
test_mat=np.zeros((matx,maty))
k=-1
for file_ in allFiles:
    k+=1
    temp=cv2.imread(file_,-1)[20:-20,20:-20]
    test_mat[k,:]=temp.flatten()

# %% codecell
print(cross_val_score(pca,data_mat.T,cv=5))
print(cross_val_score(pca,data_mat.T,y=test_mat.T,cv=5))
# %% codecell
sns.heatmap(data_mat.T)
# %% codecell
control_mat=scipy.linalg.pinv(data_mat.T)
# %% codecell
control_mat=v.T @ np.diag(1/s) @ u.T
#test-control_mat
# %% codecell
test_wf.shape, control_mat.shape
# %% codecell
test_wf=test_mat[0,:]
act_commands=-1*control_mat@test_wf
pred_wf_delta=data_mat.T@act_commands
resid=test_wf+pred_wf_delta
print (np.std(test_wf),np.std(pred_wf_delta),np.std(resid))

# %% codecell
fig,ax=plt.subplots(3,1,figsize=(5,10))

vmax=test_wf.max()
vmin=test_wf.min()

ax[0].imshow(test_wf.reshape(nx,ny),vmin=vmin,vmax=vmax)
ax[1].imshow(-1*pred_wf_delta.reshape(nx,ny),vmin=vmin,vmax=vmax)
ax[2].imshow(resid.reshape(nx,ny),vmin=vmin,vmax=vmax)


# %% codecell
