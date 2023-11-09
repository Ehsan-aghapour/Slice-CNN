from ast import In
from posixpath import split
from pyexpat import model
from zlib import DEF_MEM_LEVEL
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format
from caffe import layers as L
import caffe
import os
import keras
from keras.utils.vis_utils import plot_model
import re


Main_Layers={}
Main_Layers['bvlc_alexnet.caffemodel']=['data','conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
Main_Layers['bvlc_googlenet.caffemodel']=['data','conv1/7x7_s2','conv2/3x3_reduce','conv2/3x3',
    'inception_3a/1x1','inception_3b/1x1','inception_4a/1x1','inception_4b/1x1','inception_4c/1x1',
    'inception_4d/1x1','inception_4e/1x1','inception_5a/1x1','inception_5b/1x1','loss3/classifier']
Main_Layers['squeezenet_v1.0.caffemodel']=['data','conv1','fire2/squeeze1x1','fire2/expand1x1',
    'fire3/squeeze1x1','fire3/expand1x1','fire4/squeeze1x1','fire4/expand1x1','fire5/squeeze1x1',
    'fire5/expand1x1','fire6/squeeze1x1','fire6/expand1x1','fire7/squeeze1x1','fire7/expand1x1',
    'fire8/squeeze1x1','fire8/expand1x1','fire9/squeeze1x1','fire9/expand1x1','conv10','prob']
Main_Layers['MobileNet.h5']=['input_2','conv1_pad','conv_dw_1','conv_pw_1','conv_pad_2','conv_pw_2',
    'conv_dw_3','conv_pw_3','conv_pad_4','conv_pw_4','conv_dw_5','conv_pw_5','conv_pad_6','conv_pw_6',
    'conv_dw_7','conv_pw_7','conv_dw_8','conv_pw_8','conv_dw_9','conv_pw_9','conv_dw_10','conv_pw_10',
    'conv_dw_11','conv_pw_11','conv_pad_12','conv_pw_12','conv_dw_13','conv_pw_13','conv_preds']

Main_Layers['ResNet50.h5']=['input_1','conv1_pad','res2a_branch2a','res2b_branch2a','res2c_branch2a',
    'res3a_branch2a','res3b_branch2a','res3c_branch2a','res3d_branch2a','res4a_branch2a','res4b_branch2a',
    'res4c_branch2a','res4d_branch2a','res4e_branch2a','res4f_branch2a','res5a_branch2a','res5b_branch2a',
    'res5c_branch2a','fc1000']


granularity="Conv"

if granularity=="Conv":
    formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*)"
    # If you do not want to skip input and output layers:
    # formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!.*batchnorm)(?!.*linear)(.*conv.*|.*fc.*|$)"

#
if granularity=="Operation":
    formatPattern = r"^(?!.*_g\d*)(?!.*relu)(?!$)"

index = 0

def check_name(name):
    global index
    if re.search(formatPattern, name, re.IGNORECASE):
        index += 1
        print(f"{index} layer: {name}")
        return True
    else:
        print(f"{index} skipping layer: {name}")
        return False



#_dir="/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Sub_Model/"
_dir="/home/ehsan/UvA/Sub_Model/"

def set_main_layers(Model_name):
    global Main_Layers
    layers=[l.name for l in net.layer]
    #sorted_layers=sort_strings_by_number(layers)
    #Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
    ##Main_Layers[Model_name]=[l for l in layers if 'relu' not in l and 'drop' not in l]
    Main_Layers[Model_name]=[l for l in layers if check_name(l)]
    print(f"main layers of the model {Model_name} are {Main_Layers[Model_name]}")
    print(len(Main_Layers[Model_name]))

def set_main_layers_keras(Model_name):
    global Main_Layers
    layers=[l.name for l in model.layers]
    #sorted_layers=sort_strings_by_number(layers)
    #Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
    #Main_Layers[Model_name]=[l for l in layers if 'relu' not in l and 'drop' not in l]
    Main_Layers[Model_name]=[l for l in layers if check_name(l)]
    print(f"main layers of the model {Model_name} are {Main_Layers[Model_name]}")
    print(len(Main_Layers[Model_name]))

def Load_Net(M='Alex/bvlc_alexnet.caffemodel',Structure='Alex/deploy.prototxt'):
    global net 
    net = caffe_pb2.NetParameter()
    global Model  
    Model = caffe.Net(Structure, 1, weights=M)
    with open(Structure, 'r') as f:
        pb.text_format.Merge(f.read(), net)
    set_main_layers(M.split('/')[-1])
    global main_layers
    main_layers=Main_Layers[M.split('/')[-1]]
    print(f'Model {M} loaded.')
    
    return net

def Load_Net_Keras(Model_name):
    global model
    model=keras.models.load_model(Model_name)
    set_main_layers_keras(Model_name.split('/')[-1])
    global main_layers
    main_layers=Main_Layers[Model_name.split('/')[-1]]
    print(f'Model {Model_name} loaded.')
    
    return model


def Save_Net(Name):
    #Name=Name+'.prototxt'
    with open(Name, 'w') as f:
        f.write(pb.text_format.MessageToString(net))

    print(f'Model saved as {Name}')



def Fill_Indexes():
    global dict
    dict={}
    layers=net.layer
    print(len(layers))
    layer=0
    started=0
    print(main_layers)
    for i in range(len(layers)):
        if layers[i].name in main_layers:
            if started:
                dict[layer].setdefault('end',i-1)
                print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])
                layer=layer+1
                dict.setdefault(layer,{})
                dict[layer].setdefault('name',main_layers[layer])
                dict[layer].setdefault('start',i)
            else:                               
                dict.setdefault(layer,{})
                dict[layer].setdefault('name',main_layers[layer])
                dict[layer].setdefault('start',i)
                started=1
        if i==(len(layers)-1):
            dict[layer].setdefault('end',i)
            print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])



def Fill_Indexes_keras(model_name):
    global dict
    dict={}
    #net=keras.models.load_model(model_name)
    layers=model.layers
    print(len(layers))
    layer=0
    started=0
    for i in range(len(layers)):
        #print(layers[i].name)
        
        if layers[i].name in main_layers:
            if started:
                dict[layer].setdefault('end',i-1)
                print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])
                layer=layer+1
                dict.setdefault(layer,{})
                dict[layer].setdefault('name',main_layers[layer])
                dict[layer].setdefault('start',i)
                print(f'\n\n\nlayer {main_layers[layer]}, input :{layers[i].input.shape}')
            else:                               
                dict.setdefault(layer,{})
                dict[layer].setdefault('name',main_layers[layer])
                dict[layer].setdefault('start',i)
                started=1
                print(f'\n\n\nlayer {main_layers[layer]}, input :{layers[i].name},{layers[i].input.shape}')
        if i==(len(layers)-1):
            dict[layer].setdefault('end',i)
            print(layer,dict[layer]['name'],dict[layer]['start'],dict[layer]['end'])

        print(f'sublayer {layers[i].name}, output shape :{layers[i].output.shape}')

        


def Slice(Start,End):
    # Extract Input shape of start layer
    
    Bottom_Name=Model.bottom_names[main_layers[Start]][0]
    print(f'Previous layer name: {Bottom_Name}')
    Input_Shape=Model.blobs[Bottom_Name].data.shape
    if len(Input_Shape) == 2:
    	Input_Shape=Input_Shape[:1]+(1,1,)+Input_Shape[1:]
    print(f'Input shape is: {Input_Shape}')
    #for b in Model.blobs:
    #	print(Model.blobs[b].data.shape)

    # Set input shape to Extracted Shape
    input=net.layer[0]
    shape=input.input_param.shape[0]
    shape.Clear()
    shape.dim.MergeFrom(Input_Shape)

    # Slice the model using indexed dict
    Start_index=dict[Start]['start']
    End_index=dict[End]['end']
    print(f'Start and end indexes are: {Start_index,End_index}')
    del net.layer[End_index+1:]
    previous_layers=[Bottom_Name]
    if Start_index>0:
        previous_layer_name=net.layer[Start_index-1].name
        previous_layers.append(previous_layer_name)
    if Start>0:     
        previous_layer_name2=main_layers[Start-1]
        previous_layers.append(previous_layer_name2)
    #print(f'start:{Start}, end:{End}, p:{previous_layer_name}, p2:{previous_layer_name2} ')
    del net.layer[1:Start_index]

    '''
    # Connect start layer to input layer 
    C1=net.layer[1]
    C1.ClearField('bottom')
    C1.bottom.append(input.name)
    '''

    # Connect start layers to input layer (Considering multiple parallel input layer)
    #print(f'Name of previousl layer {previous_layer_name} and {previous_layer_name2} and also {Bottom_Name}')
    print(f'Name of previous layers: {previous_layers}')
    for l in net.layer:
        print(f'bottom of {l.name}:{l.bottom}')
        #if l.bottom==[previous_layer_name] or l.bottom==[previous_layer_name2] or l.bottom==[Bottom_Name]:
        if l.bottom and l.bottom[0] in previous_layers:
            print(f'new first layer after data:{l}')
            l.ClearField('bottom')
            l.bottom.append(input.name)
    #print(net.layer)
    
    print(f'Model sliced')

def split_keras(model_name,Start,End):
    

    Start_index=dict[Start]['start']
    End_index=dict[End]['end']

    #model=keras.models.load_model(model_name)
    #DL_input = keras.layers.Input(model.layers[indx].input_shape[1:])
    print(f'Input shape is:{model.layers[Start_index].get_input_shape_at(0)[1:]}')
    print(f'Start and end indexes are: {Start_index,End_index}')
    
    p_layer=model.layers[Start_index-1]
    DL_input = keras.layers.Input(model.layers[Start_index].get_input_shape_at(0)[1:],name='my_input')
    DL_model = DL_input
    DL_model = model.layers[Start_index](DL_model)
    #ll=model.layers[:]
    for layer in model.layers[Start_index+1:End_index+1]:
        layer_in_shape=0
        if isinstance(layer.input, list):
            layer_in_shape=layer.input[0].shape
        else:
            layer_in_shape=layer.input.shape
        DL_model_name=0
        DL_model_shape=0
        if isinstance(DL_model,list):
            DL_model_shape=DL_model[0].shape
            DL_model_name=DL_model[0].name
        else:
            DL_model_shape=DL_model.shape
            DL_model_name=DL_model.name
        print(f'adding layer: {layer.name} with shape {layer_in_shape} to {DL_model_name} with shape {DL_model_shape}')
        if type(layer.input)==type([]): 
            if p_layer.output in layer.input:
                DL_model = layer([DL_input,DL_model])
            else:
                for l in model.layers:
                    if l.output in layer.input:
                        DL_model = layer([l.get_output_at(1),DL_model])
                        break
        else:
            DL_model = layer(DL_model)
        
    DL_model = keras.models.Model(inputs=DL_input, outputs=DL_model)
    
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print(f'Model sliced')
    _dir=os.path.dirname(model_name)
    _ext=os.path.splitext(model_name)[1]
    new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
    pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
    DL_model.save(new_name)
    #DL_model.summary()
    print(f'Model saved as {new_name}')
    #print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
    global pb_convert_args
    pb_convert_args={}
    pb_convert_args['h5name']=new_name
    pb_convert_args['pb_name']=pb_name
    pb_convert_args['input']=DL_model.layers[0].name
    pb_convert_args['output']=DL_model.layers[-1].get_output_at(0).op.name
    shape=''
    s=list(DL_model.layers[0].get_input_shape_at(0)[1:])
    for x in s:
        shape=shape+str(x)+','
    shape=shape[:-1]
    pb_convert_args['input_shape']=shape
    for l in DL_model.layers:
        print(f'name:{l.name} output:{l.get_output_at(0).name}')

    return DL_model


def split_keras_2(model_name,Start,End):
    Start_index=dict[Start]['start']
    End_index=dict[End]['end']

    #model=keras.models.load_model(model_name)
    
    Input_shape=model.layers[Start_index].get_input_shape_at(0)[1:]
    print(f'Input shape is:{Input_shape}')
    print(f'Start and end indexes are: {Start_index,End_index}')
    p_layer=model.layers[Start_index-1]
    nodes=p_layer._outbound_nodes
    Input_layer=keras.layers.InputLayer(input_shape=Input_shape,name="New_input")
    for node in nodes:
        if p_layer.output in node.input_tensors:
            node.input_tensors.remove(p_layer.output)
        node.input_tensors.append(Input_layer.output)

        if p_layer in node.inbound_layers:
            node.inbound_layers.remove(p_layer)
        node.inbound_layers.append(Input_layer)
    Input_layer._outbound_nodes=nodes
    if 'dropout' in model.layers[End_index].name:
        print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
        End_index=End_index-1
    new_model=keras.models.Model(inputs=Input_layer.input,outputs=model.layers[End_index].output)
    '''if 'dropout' in new_model.layers[-1].name:
        print(f'The last layer is dropout, it should be deleted because error in converting to rknn (dropout should be in training no inference!)')
        new_model.layers.pop()'''
        
    print(f'\n\nNew model First Layer:{new_model.layers[1].name}, Input shape:{new_model.layers[0].input.shape}\n\
            Input name:{new_model.layers[0].name} First layer input shape:{new_model.layers[1].input.shape},\n\
            Last layer name:{new_model.layers[-1].name} and {new_model.layers[-1].get_output_at(0).op.name}\n\
            output shape:{new_model.layers[-1].output.shape}')

    plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print(f'Model sliced')
    _dir=os.path.dirname(model_name)
    _ext=os.path.splitext(model_name)[1]
    new_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
    pb_name=_dir+'/'+model_name.split('/')[-2]+'_'+str(Start)+'_'+str(End)+'.pb'
    new_model.save(new_name)
    #DL_model.summary()
    print(f'Model saved as {new_name}')
    #print(f'Model input:{DL_model.input.name}, Output:{DL_model.output.name}, input_shape:{DL_model.input.shape}')
    global pb_convert_args
    pb_convert_args={}
    pb_convert_args['h5name']=new_name
    pb_convert_args['pb_name']=pb_name
    pb_convert_args['input']=new_model.layers[0].name
    pb_convert_args['output']=new_model.layers[-1].get_output_at(0).op.name
    shape=''
    s=list(new_model.layers[0].get_input_shape_at(0)[1:])
    for x in s:
        shape=shape+str(x)+','
    shape=shape[:-1]
    pb_convert_args['input_shape']=shape
    return new_model


def main(Start,End, M, Structure):
    Load_Net(M,Structure)
    print('inja')
    
    Fill_Indexes()
    print(dict)
    Slice(Start,End)

    
    _dir=os.path.dirname(Structure)
    _ext=os.path.splitext(Structure)[1]
    global Name
    Name=_dir+'/'+Structure.split('/')[-2]+'_'+str(Start)+'_'+str(End)+_ext
    Save_Net(Name)

def main_keras(M,Start,End):
    Load_Net_Keras(M)
    Fill_Indexes_keras(M)
    #print(dict)
    m=split_keras_2(M,Start,End)
    return m


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Slice a model')
    parser.add_argument('--Model', metavar='path', required=False,
                        help='Model')
    parser.add_argument('--Structure', metavar='path', required=False,
                        help='Structure of the model (prototxt)')
    parser.add_argument('--Start', metavar='number', required=True,
                        help='Starting layer')
    parser.add_argument('--End', metavar='number', required=True,
                        help='Ending layer')

    
    args = parser.parse_args()
    print(f'st is {args.Structure}')
    if args.Model.split('.')[-1]=='h5':
        main_keras(args.Model,int(args.Start),int(args.End))
        if pb:
            cmd1=f'python {_dir}keras_to_tensorflow.py --input_model={pb_convert_args["h5name"]} --output_model={pb_convert_args["pb_name"]}'
            print(f'command is: {cmd1}')
            ok=os.system(cmd1)
            print(f'Freezing graph return code is {ok} ')
            cmd2=f'python {_dir}convert.py {pb_convert_args["pb_name"]} {pb_convert_args["input"]} {pb_convert_args["output"]} {pb_convert_args["input_shape"]}'
            print(f'command is: {cmd2}')
            ok=os.system(cmd2)
            print(f'Convert to rknn return code is {ok} ')
    else:
        main(Start=int(args.Start), End=int(args.End), M=args.Model, Structure=args.Structure)
        cmd=f'python {_dir}convert.py {Name} {args.Model}'
        print(f'command is {cmd}')
        ok=os.system(cmd)
        print(f'Convert caffe to rknn return code is {ok}')


# +
def extract_number(string):
    # Extract the number from the string
    pattern = r"(\d+)"
    match = re.search(pattern, string)
    if match:
        return int(match.group(0))
    return 0

def sort_strings_by_number(strings):
    # Sort the strings based on the embedded numbers
    sorted_strings = sorted(strings, key=extract_number)
    return sorted_strings

def extract():
	ls=[l.name for l in model.layers]
	sls=sort_strings_by_number(ls)
	conv_index=0
	batch_norm_index=0
	layers_with_batch=-1
	#ls=model.layers[240]+model.layers[243]+model.layers[249]
	#ls=model.layers[241]+model.layers[244]+model.layers[250]
	#ls=model.layers[242]+model.layers[245]+model.layers[251]
	for kk,ll in enumerate(sls):
		#print(ll)
		l=model.get_layer(ll)
		for w in l.weights:
			name=w.name
			name2=name
			name=name.partition(':')[0]
			name=name.replace('/','_')
			name=name.replace('bnorm','batch_normalization')
			name=name.replace('moving_variance','var')
			name=name.replace('moving_mean','mean')
			pattern = r"(\d+)"
			#name = re.sub(pattern, lambda match: str(int(match.group(0)) + 1), name)
			#with each conv layer we increase the conve index by one (given that it is not conv_bias)
			if name.find('conv')==0:
				if name.find('bias') < 0:
					conv_index=conv_index+1
				name = re.sub(pattern, lambda match: str(conv_index), name)
				
			# the batch normalization indexing get different form conv indexing from layer conv59 which has not 
			# batch normalization (also layer 67 and 75 do not have bn)
			# so for batch-normaliztion layers for each 4 sub-layer (mean, var, gamma, beta) we increase its indexing
			if name.find('batch_normalization')==0:
				layers_with_batch+=1
				if layers_with_batch%4==0:
					batch_norm_index+=1
				name = re.sub(pattern, lambda match: str(batch_norm_index), name)
				
				
			name=name.replace('kernel','w')
			name=name.replace('conv','conv2d')
			name=name.replace('bias','b')
			name="yolov3_model/"+name+'.npy'
			# Extract the weights from the layer
			layer_weights = l.get_weights()

			# Get the list of variable names
			variable_names = [var.name for var in l.weights]
			for _name, _weight in zip(variable_names, layer_weights):
				if _name==name2:
					np.save(os.path.join(args.dumpPath, name),_weight)
			
			print(name,name2,kk)


# +
def set_main_layers_yolov3():
    global Main_Layers
    modelfile='Yolo/Yolov3.h5'
    model=keras.models.load_model(modelfile)
    model.summary()
    layers=[l.name for l in model.layers]
    sorted_layers=sort_strings_by_number(layers)
    #Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'padding' not in l and 'bias' not in l]
    Main_Layers['Yolov3.h5']=[l for l in sorted_layers if 'conv' in l and 'bias' not in l]
    print(len(Main_Layers['Yolov3.h5']))
    
'''set_main_layers_yolov3()
m=main_keras('Yolo/Yolov3.h5',5,15)
m.summary()
'''

# -

def Temp():
    input=net.layer[0];
    shape=kk=input.input_param.shape[0];
    shape.Clear();
    shape.dim.MergeFrom([10,256,13,13]);

    del net.layer[1:9]
    del net.layer[5:]
    C1=net.layer[1]
    C1.ClearField('bottom')
    C1.bottom.append(input.name)


