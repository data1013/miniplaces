import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)

def next_batch(self, batch_size):
    images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
    labels_batch = np.zeros(batch_size)
    for i in range(batch_size):
        image = scipy.misc.imread(self.list_im[self._idx])
        image = scipy.misc.imresize(image, (self.load_size, self.load_size)) #scale image to defined load size
        image = image.astype(np.float32)/255.
        image = image - self.data_mean #low contrast/tones it down?

        if self.randomize:
            flip = np.random.random_integers(0, 1)
            if flip>0:
                image = image[:,::-1,:] #randomly flips an image horizontally
            offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
            offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
        else:
            offset_h = (self.load_size-self.fine_size)/2
            offset_w = (self.load_size-self.fine_size)/2

        images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
        labels_batch[i, ...] = self.list_lab[self._idx]
            
        self._idx += 1
        if self._idx == self.num:
            self._idx = 0
        
    return images_batch, labels_batch
    