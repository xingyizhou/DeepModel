import numpy as np
import h5py
import cv2
import scipy.io as sio
import sys
import os
import math
paths = {}
with open('../path.config', 'r') as f:
  for line in f:
    name, path = line.split(': ')
    print name, path
    paths[name] = path
    
## This part of code is modified from [DeepPrior](https://cvarlab.icg.tugraz.at/projects/hand_detection/)
def CropImage(image, com):
  u, v, d = com
  zstart = d - cube_size / 2.
  zend = d + cube_size / 2.
  xstart = int(math.floor((u * d / fx - cube_size / 2.) / d * fx))
  xend = int(math.floor((u * d / fx + cube_size / 2.) / d * fx))
  ystart = int(math.floor((v * d / fy - cube_size / 2.) / d * fy))
  yend = int(math.floor((v * d / fy + cube_size / 2.) / d * fy))
  
  cropped = depth[max(ystart, 0):min(yend, depth.shape[0]), max(xstart, 0):min(xend, depth.shape[1])].copy()
  cropped = np.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, depth.shape[0])), 
                                (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, depth.shape[1]))), mode='constant', constant_values=0)
  msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
  msk2 = np.bitwise_and(cropped > zend, cropped != 0)
  cropped[msk1] = zstart
  cropped[msk2] = zend
  
  dsize = (img_size, img_size)
  wb = (xend - xstart)
  hb = (yend - ystart)
  if wb > hb:
    sz = (dsize[0], hb * dsize[0] / wb)
  else:
    sz = (wb * dsize[1] / hb, dsize[1])
  
  roi = cropped
  rz = cv2.resize(cropped, sz)

  ret = np.ones(dsize, np.float32) * zend
  xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
  xend = int(xstart + rz.shape[1])
  ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
  yend = int(ystart + rz.shape[0])
  ret[ystart:yend, xstart:xend] = rz
  
  return ret
    
dataset_path = paths['NYU_path']
J = 31
joint_id = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 5, 11, 17, 23, 32, 30, 31, 28, 27, 25, 24])
img_size = 128
    
fx = 588.03
fy = 587.07
fu = 320.
fv = 240.

data_names = ['train', 'test_1', 'test_2']
cube_sizes = [300, 300, 300 * 0.87]
id_starts = [0, 0, 2440]
id_ends = [72756, 2440, 8252]
num_packages = [3, 1, 1]

for D in range(0, len(data_names)):
  data_name = data_names[D]
  cube_size = cube_sizes[D]
  id_start = id_starts[D]
  id_end = id_ends[D]
  chunck_size = (id_end - id_start) / num_packages[D]
  
  data_type = 'train' if data_name == 'train' else 'test'
  data_path = '{}/{}'.format(dataset_path, data_type)
  label_path = '{}/joint_data.mat'.format(data_path)
  
  labels = sio.loadmat(label_path)
  joint_uvd = labels['joint_uvd'][0]
  joint_xyz = labels['joint_xyz'][0]

  cnt = 0
  chunck = 0
  depth_h5, joint_h5, com_h5, inds_h5 = [], [], [], []
  for id in range(id_start, id_end):
    img_path = '{}/depth_1_{:07d}.png'.format(data_path, id + 1)
    
    if not os.path.exists(img_path):
      print '{} Not Exists!'.format(img_path)
      continue
    print img_path
    img = cv2.imread(img_path)
    depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256)
    depth = CropImage(depth, joint_uvd[id, 34])

    com3D = joint_xyz[id, 34]
    joint = joint_xyz[id][joint_id] - com3D
    depth = ((depth - com3D[2]) / (cube_size / 2)).reshape(1, img_size, img_size)

    joint = np.clip(joint / (cube_size / 2), -1, 1)
    depth_h5.append(depth.copy())
    joint_h5.append(joint.copy().reshape(3 * J))
    com_h5.append(com3D.copy())
    inds_h5.append(id)
    cnt += 1
    if cnt % chunck_size == 0 or id == id_end - 1:
      rng = np.arange(cnt) if data_type == 'test' else np.random.choice(np.arange(cnt), cnt, replace = False)
      dset = h5py.File('h5data/{}_{}.h5'.format(data_name, chunck), 'w')
      dset['depth'] = np.asarray(depth_h5)[rng]
      dset['joint'] = np.asarray(joint_h5)[rng]
      dset['com'] = np.asarray(com_h5)[rng]
      dset['inds'] =np.asarray(inds_h5)[rng]
      dset.close()
      depth_h5, joint_h5, com_h5, inds_h5 = [], [], [], []
      chunck += 1
      cnt = 0
