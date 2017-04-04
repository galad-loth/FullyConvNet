import mxnet as mx
import os
 

def get_fcn_symbol(model_dir, model_prefix, load_epoch, num_class):
    dict_layer_trunc={"resnet-50":"relu1",
              "resnet-101":"relu1",
              "vgg16":"pool5",
              "inception-BN":"ch_concat_5b_chconcat"}
    label = mx.symbol.Variable(name="label")
    load_net, load_arg_params, load_aux_params = \
        mx.model.load_checkpoint(os.path.join(model_dir,model_prefix), load_epoch)
    all_layers=load_net.get_internals()
    base_net = all_layers[dict_layer_trunc[model_prefix]+'_output']
    
    fcn_conv1 = mx.symbol.Convolution(data=base_net,kernel=(7, 7),
                                      num_filter=4096,name="fcn_layer_conv1")
    fcn_relu1 = mx.symbol.Activation(data=fcn_conv1, act_type="relu", name="fcn_layer_relu1")
    fcn_drop1 = mx.symbol.Dropout(data=fcn_relu1, p=0.5, name="fcn_layer_drop1")
    
    fcn_conv2 = mx.symbol.Convolution(data=fcn_drop1,kernel=(1, 1),
                                      num_filter=4096,name="fcn_layer_conv2")
    fcn_relu2 = mx.symbol.Activation(data=fcn_conv2, act_type="relu", name="fcn_layer_relu2")
    fcn_drop2 = mx.symbol.Dropout(data=fcn_relu2, p=0.5, name="fcn_layer_drop2")
    fcn_score = mx.symbol.Convolution(data=fcn_drop2, kernel=(1, 1), 
                                      num_filter=num_class, name="fcn_layer_score")
    fcn_upscore = mx.symbol.Deconvolution(
        data=fcn_score, kernel=(64,64), stride=(32,32), adj=(31, 31),
        num_filter=num_class, name="fcn_layer_upscore")   
    fcn_outscore = mx.symbol.Crop(*[fcn_upscore, label],
                                  offset=(19,19), name="fcn_layer_outscore")
     
    fcn_net=mx.symbol.SoftmaxOutput(
        data=fcn_outscore, label=label,multi_output=True, use_ignore=True, 
        ignore_label=255, name="fcn_layer_softmax")
    fcn_args = dict({k:load_arg_params[k] 
                     for k in load_arg_params if 'fcn_layer' not in k})
    return fcn_net,fcn_args,load_aux_params


if __name__=="__main__":
    model_dir="E:\\DevProj\\DeepLearningForCV\\pretrain_model\\mxnet\\"
    model_prefix="resnet-50"
    load_net, load_args_params, load_aux_params=get_fcn_symbol(model_dir,model_prefix,0,21)
    ex=load_net.simple_bind(ctx=mx.cpu(), data=(1,3,500,500),label=(1,1,300,300))
