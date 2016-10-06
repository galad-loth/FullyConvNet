import numpy as npy
import matplotlib.pyplot as plt
import caffe
from PIL import Image
import cv2
import os

#caffe.set_device(0)
#caffe.set_mode_gpu()

def get_color_img(label,cmap):
    img_color=cmap[label.reshape(label.size),:]
    img_color=npy.reshape(img_color,(label.shape[0],label.shape[1],3))
    return img_color

def get_voc_colormap(N=256):
    cmap=npy.zeros((N,3),dtype=npy.uint8)
    for ii in npy.arange(N):
        r=0
        b=0
        g=0
        for jj in npy.arange(8):
            r=r+(1<<(7-jj))*((ii&(1<<(3*jj)))>>(3*jj))
            g=g+(1<<(7-jj))*((ii&(1<<(3*jj+1)))>>(3*jj+1))
            b=b+(1<<(7-jj))*((ii&(1<<(3*jj+2)))>>(3*jj+2))      
        cmap[ii,:]=npy.array([r,g,b])
    return cmap


def test_voc_seg(model_def,model_weights):
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    cmap=get_voc_colormap(256)
    data_root="E:/DevProj/Datasets/PascalVoc/2007/VOCdevkit/VOC2007/VOCtest/JPEGImages"
    img_mean=npy.array((104.00698793,116.66876762,122.67891434))
    
    num_test=0
    
    for img_name in os.listdir(data_root):
        if num_test>=100:
            break
        num_test+=1
        img_path=os.path.join(data_root,img_name)
        img=Image.open(img_path)
        img1=npy.array(img,dtype=npy.float32)
        img1=img1[:,:,::-1]-img_mean
        img1 = img1.transpose((2,0,1))
        net.blobs['data'].reshape(1, *img1.shape)
        net.blobs['data'].data[...] = img1
        # run net and take argmax for prediction
        net.forward()
        seg_label = net.blobs['score'].data[0].argmax(axis=0)
        seg_color=get_color_img(seg_label,cmap)
        img2=npy.array(img,dtype=npy.uint8)
        img_save=npy.zeros((img2.shape[0],2*img2.shape[1]+5,3),dtype=npy.uint8)
        img_save[:,0:img2.shape[1],:]=img2
        img_save[:,img2.shape[1]+5::,:]=seg_color
        cv2.imwrite("data/VOC2007/results/"+img_name+"seg.png",img_save)
    
    
    
if __name__=="__main__":
    model_def ="E:/DevProj/SemanticLabeling/FCN/voc-fcn8s/deploy.prototxt"
    model_weights ="E:/DevProj/SemanticLabeling/FCN/voc-fcn8s/fcn8s-heavy-pascal.caffemodel"
    test_voc_seg(model_def,model_weights)
    
