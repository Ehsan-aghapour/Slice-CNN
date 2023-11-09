'''
rk2
python3 convert.py model.onnx
python3 convert.py model.pb
python3 convert.py model.prototxt model.caffemodel

'''


from rknn.api import RKNN
import sys
from pathlib import Path

explicit_channel_reorder=True
explicit_mean_reduction=True
quantized=False
quantization_dataset=""

v=''
if quantized:
    v='_quantized'
name=sys.argv[1]
#rknn_name=name.split('/')[-1].split('.')[0]+'.rknn'
rknn_name=name.split('.')[0]+v+'.rknn'
rknn_name_precompiled=name.split('/')[-1].split('.')[0]+'_precompiled.rknn'
model_type=name.split('.')[-1]
precompile=False



'''
#MobileNet:
inputs='input_2'
outputs='act_softmax/Softmax'
INPUT_SIZE=[224,224,3]
'''


#ResNet:
inputs='input_1'
outputs='fc1000/Softmax'
INPUT_SIZE=[224,224,3]

inputs='input'
outputs='conv_dw_11_relu/Relu6'
INPUT_SIZE=[28,28,256]
'''
inputs='input_2'
outputs='act_softmax/Softmax'
INPUT_SIZE=[224,224,3]
'''

'''
#model.pb
inputs='test_in'
outputs='test_out/BiasAdd'
INPUT_SIZE = [475]
 '''
if __name__ == '__main__':
    rknn = RKNN()   # Create an RKNN execution object
 
    '''
    Configure model input for NPU preprocessing of input data
    channel_mean_value='0 0 0 255', when runing forward inference, the RGB data will be
    converted as follows (R - 0) / 255, (G - 0) / 255, (B - 0) / 255,
    The RKNN model automatically performs the mean and normalization.
    reorder_channel='0 1 2' , used to specify whether to adjust the image channel order, 
    set to 0 1 2, means no adjustment according to the input image channel order.
    reorder_channel='2 1 0' , indicates that 0 and 2 channels are exchanged.
    If the input is RGB, it will be adjusted to BGR. If it is BGR will be adjusted to RGB
    Image channel order is not adjusted
    '''
 
    #rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')
    #rknn.config(target_platform=["rk1806", "rk1808", "rk3399pro"])
    '''
    preprocess=0
	
    if preprocess: 
            #ARMCL:
        #rknn.config(channel_mean_value='122.68 116.67 104.01 1', reorder_channel='0 1 2')

            #mobilenetv1.tflite
        #rknn.config(channel_mean_value='103.94 116.78 123.68 58.82', reorder_channel='0 1 2')

            #mobilenet pb ARMCL:
        rknn.config(channel_mean_value='127.5 127.5 127.5 128', reorder_channel='0 1 2')
        
            #resnet: do not convert to bgr
        #rknn.config(channel_mean_value='122.68 116.67 104.01 1', reorder_channel='0 1 2')
    else:
        rknn.config()
    '''

    if explicit_channel_reorder:
        r_ch='0 1 2'
    else:
        r_ch='2 1 0'

    if explicit_mean_reduction:
        ch_m='0 0 0 1'
    else:
        ch_m='104.01 116.67 122.68 1'

    rknn.config(channel_mean_value=ch_m, reorder_channel=r_ch)
    #rknn.config()

    print('done')
 
    '''
    load TensorFlow model
    tf_pb='digital_gesture.pb' specify the TensorFlow model to be converted
    inputs specify the input node in the model
    outputs specify the output node in the model
    input_size_list specify the size of the model input
    '''
 
    print('--> Loading model')
    #rknn.load_tensorflow(tf_pb='model.pb',
    #                     inputs=['test_in'],
    #                     outputs=['test_out/BiasAdd'],
    #                     input_size_list=[[INPUT_SIZE]])
    #rknn.load_onnx(name)
    if model_type=='pb':
        print(f'args number {len(sys.argv)}')
        if len(sys.argv)>=5:   
            inputs=sys.argv[2]
            outputs=sys.argv[3]
            strOfNumbers = sys.argv[4]
            listOfNumbers= [int(x) for x in strOfNumbers.split(',')]
            INPUT_SIZE=listOfNumbers
            if len(sys.argv)==7:
                quantized=bool(int(sys.argv[5]))
                quantization_dataset=sys.argv[6]
        rknn.load_tensorflow(tf_pb=name,
            inputs=[inputs],
            outputs=[outputs],
            input_size_list=[INPUT_SIZE])
            

    if model_type=='onnx':
        rknn.load_onnx(name)

    if model_type=='prototxt':
        #p=Path(name)
        #proto_name=p.with_suffix('.prototxt')
        print(f'name:{name},blobs:{sys.argv[2]}')
        ret = rknn.load_caffe(model=name,
            proto='caffe',
            blobs=sys.argv[2])
    if model_type=='rknn':
        ret=rknn.load_rknn(name)
        precompile=True

    print('done')
 
    '''
    Create a parsing pb model
    do_quantization=False do not to be quantified
    Quantization will reduce the size of the model and increase the speed of the operation,
    but there will be loss of precision.
    '''
    
    if precompile:
        if model_type!='rknn':
            print('--> Building model')
            rknn.build(do_quantization=quantized)
            print('done')
        #rknn.export_rknn('./model.rknn')  # Export and save rknn model file
            print(rknn_name)
        print('--> Init runtime with precompile')
        ret = rknn.init_runtime(target='rk3399pro',rknn2precompile=True)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print(f'--> exporting precompiled model as:{rknn_name_precompiled}')
        ret = rknn.export_rknn_precompile_model(rknn_name_precompiled)
        if ret != 0:
            print('export failed')
            exit(ret)
    #ret = rknn.init_runtime(target='rk3399pro')
    else:
        if model_type!='rknn':
            print('--> Building model')
            if quantized:
                rknn_name=rknn_name.split('.')[0]+'_quantized'+rknn_name.split('.')[1]
                rknn.build(do_quantization=quantized,dataset=quantization_dataset)
            else:
                print("inja"+str(quantized))
                rknn.build(do_quantization=quantized)
            print('done')
        #rknn.export_rknn('./model.rknn')  # Export and save rknn model file
            print(rknn_name)
            rknn.export_rknn(rknn_name)  # Export and save rknn model file
    
    rknn.release()  # Release RKNN Context

