import sys
paths = {}
with open('../path.config', 'r') as f:
  for line in f:
    name, path = line.split(': ')
    print name, path
    paths[name] = path
sys.path.insert(0, paths['pycaffe_root'])
import caffe
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import cv2

joints = np.arange(31)
Edges = [[0, 1], [1, 2], [2, 3], [3, 4], 
         [5, 6], [6, 7], [7, 8], [8, 9],
         [10, 11], [11, 12], [12, 13], [13, 14], 
         [15, 16], [16, 17], [17, 18], [18, 19],
         [4, 20], [9, 21], [14, 22], [19, 23], 
         [20, 24], [21, 24], [22, 24], [23, 24],
         [24, 25], [24, 26], [24, 27],
         [27, 28], [28, 29], [29, 30]]

J = len(joints)

if __name__ == '__main__':
  #caffe.set_mode_gpu()
  net = caffe.Net( 'DeepModel_deploy.prototxt',
                'weights/NYU.caffemodel',
                caffe.TEST)
  list_images = ['0.png', '772.png', '1150.png', '1350.png', '1739.png']
  for image_name in list_images:
      img = cv2.imread('test_images\\' + image_name)
      cv2.imshow('img_input', img)
      input = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255. * 2 - 1)
      blobs_in = {'data': input.reshape(1, 1, img.shape[0], img.shape[1])}
      out = net.forward(**blobs_in)
      joint = out['pred'][0]
      
      x = np.zeros(J)
      y = np.zeros(J)
      z = np.zeros(J)
      for j in range(J):
        x[j] = joint[joints[j] * 3]
        y[j] = joint[joints[j] * 3 + 1]   
        z[j] = joint[joints[j] * 3 + 2]
        cv2.circle(img, (int((x[j] + 1) / 2 * 128), int((- y[j] + 1) / 2 * 128)), 2, (255, 0, 0), 2)
      fig=plt.figure()
      ax=fig.add_subplot((111),projection='3d')
      ax.set_xlabel('z')
      ax.set_ylabel('x')
      ax.set_zlabel('y')
      ax.scatter(z, -x, y)
      for e in Edges:
        ax.plot(z[e], -x[e], y[e], c = 'b')
        
      #For axes equal
      max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
      Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
      Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
      Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
      for xb, yb, zb in zip(Xb, Yb, Zb):
         ax.plot([zb], [xb], [yb], 'w')

      cv2.imshow('img_pred', img)
      plt.show()
  