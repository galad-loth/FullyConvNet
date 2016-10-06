import os
import subprocess

def GetVOCList(train_path,test_path):
    num_label=0
    dict_class={}
    label=0
    img_ext=".jpg"
    for file_name in os.listdir(train_path):
        file_path=os.path.join(train_path,file_name)
        file_name_parts=file_name.split("_")        
        if len(file_name_parts)==1:
            continue
        if dict_class.has_key(file_name_parts[0]):
           label=dict_class[file_name_parts[0]]
        else:
           label=str(num_label)
           dict_class[file_name_parts[0]]=str(num_label)
           num_label=num_label+1
        with open(file_name_parts[1],"a") as fw,open(file_path,"r") as fr:
            for line in fr:
                content=line.split()
                if "1"==content[1]:
                    fw.write(content[0]+img_ext+" "+label+"\n")
     
    for file_name in os.listdir(test_path):
        file_path=os.path.join(test_path,file_name)
        file_name_parts=file_name.split("_")        
        if len(file_name_parts)==1:
            continue
        if dict_class.has_key(file_name_parts[0]):
           label=dict_class[file_name_parts[0]]
        else:
            continue
        with open(file_name_parts[1],"a") as fw,open(file_path,"r") as fr:
            for line in fr:
                content=line.split()
                if "1"==content[1]:
                    fw.write(content[0]+img_ext+" "+label+"\n")
           
    with open("class_map.txt","w") as fw:
        for key in dict_class.keys():
            fw.write(key+" "+dict_class[key]+"\n")
        
 

if __name__=="__main__":
    trainval_path="E:/DevProj/Datasets/PascalVoc/2007/VOCdevkit/VOC2007/VOCtrainval/ImageSets/Main"
    test_path="E:/DevProj/Datasets/PascalVoc/2007/VOCdevkit/VOC2007/VOCtest/ImageSets/Main"
    GetVOCList(trainval_path,test_path)  
     
    
    
