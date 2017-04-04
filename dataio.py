import sys
import os 
import numpy as npy
from PIL import Image
import mxnet as mx

class FCNDataBatch(object):
    def __init__(self, img, label, pad=0):
        self.data =[mx.nd.array(img)]
        self.label =[mx.nd.array(label)]
        self.pad = pad    
        
        
class VOCSegDataIter(mx.io.DataIter):
    def __init__(self, img_dir,label_dir,img_list,
                 img_mean = (127, 127, 127)):
        super(VOCSegDataIter, self).__init__()
        self._img_dir=img_dir
        self._img_list=img_list
        self._label_dir=label_dir
        self._img_mean=npy.reshape(npy.array(img_mean),(1,1,3))
        self._num_img = len(img_list)
        self._cur_batch=0
        self._data=npy.array([],dtype=npy.float32)
        self._label=npy.array([], dtype=npy.uint8)
        self._idx_rand=npy.random.permutation(self._num_img)
        if self._num_img>0:
            self._data,self._label=self._read_data()
        else:
            raise StopIteration
            
    def _read_data(self):
        img_name=self._img_list[self._idx_rand[self._cur_batch]].strip("\n")
        img_path=os.path.join(self._img_dir,img_name+".jpg")
        img = Image.open(img_path)
        img = npy.array(img, dtype=npy.float32) 
        img=img-self._img_mean        
        img=npy.pad(img,((100,100),(100,100),(0,0)),
                    'constant', constant_values=(0,))
        img = npy.swapaxes(img, 0, 2)
        img = npy.swapaxes(img, 1, 2)  # (c, h, w)
        img=img[npy.newaxis,:,:,:]
        label_path=os.path.join(self._label_dir,img_name+".png")
        label=Image.open(label_path)
        label=npy.array(label, npy.uint8)
        
        return (img,label)          
        
    def __iter__(self):
        return self 
        
    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()
        
    @property
    def provide_data(self):      
        return zip(["data"],[self._data.shape])

    @property
    def provide_label(self):       
        return zip(["label"],[self._label.shape])
        
    def next(self):
        self._cur_batch+=1
        if self._cur_batch<self._num_img:
            self._data,self._label=self._read_data()
            return FCNDataBatch(self._data, self._label)
        else:
            raise StopIteration
            
def get_voc_dataiter():
    root_dir= "E:\\DevProj\\Datasets\\PascalVoc\\2012\\VOCdevkit\\VOC2012"
    img_dir=os.path.join(root_dir,"JPEGImages")
    label_dir=os.path.join(root_dir,"SegmentationClass")
    train_list_file=os.path.join(root_dir,"ImageSets\\Segmentation\\train.txt")
    val_list_file=os.path.join(root_dir,"ImageSets\\Segmentation\\val.txt")
    with open(train_list_file,"r") as fr:
        train_img_list=fr.readlines()
    with open(val_list_file,"r") as fr:
        val_img_list=fr.readlines()
    train_iter=VOCSegDataIter(img_dir,label_dir,train_img_list,
                              img_mean= (123.68, 116.779, 103.939))
    val_iter=VOCSegDataIter(img_dir,label_dir,val_img_list,
                            img_mean= (123.68, 116.779, 103.939))
    return train_iter, val_iter
    
if __name__=="__main__":
    train_iter, val_iter=get_voc_dataiter()
    
