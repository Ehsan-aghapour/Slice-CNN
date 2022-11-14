import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
from caffe import layers as L

net = caffe_pb2.NetParameter()
with open('net.prototxt', 'r') as f:
    pb.text_format.Merge(f.read(), net)
with open('net2.prototxt', 'w') as f:
    f.write(pb.text_format.MessageToString(net))

'''
import caffe
net = caffe.Net('mnet.prototxt', 'mnet.caffemodel' , caffe.TEST)
del net.layer_dict['crop1']
del net.layer_dict['crop0']
net.save('new_model.caffemodel')
'''
#https://www.programcreek.com/python/example/104218/caffe.proto.caffe_pb2.NetParameter
def load_and_convert_caffe_model(prototxt_file_name, caffemodel_file_name):
    caffenet = caffe_pb2.NetParameter()
    caffenet_weights = caffe_pb2.NetParameter()
    text_format.Merge(open(prototxt_file_name).read(), caffenet)
    caffenet_weights.ParseFromString(open(caffemodel_file_name).read())
    # C2 conv layers current require biases, but they are optional in C1
    # Add zeros as biases is they are missing
    add_missing_biases(caffenet_weights)
    # We only care about getting parameters, so remove layers w/o parameters
    remove_layers_without_parameters(caffenet, caffenet_weights)
    # BatchNorm is not implemented in the translator *and* we need to fold Scale
    # layers into the new C2 SpatialBN op, hence we remove the batch norm layers
    # and apply custom translations code
    bn_weights = remove_spatial_bn_layers(caffenet, caffenet_weights)
    # Set num, channel, height and width for blobs that use shape.dim instead
    normalize_shape(caffenet_weights)
    # Translate the rest of the model
    net, pretrained_weights = caffe_translator.TranslateModel(
        caffenet, caffenet_weights
    )
    pretrained_weights.protos.extend(bn_weights)
    return net, pretrained_weights 


def DrpOut_OPT_Create_Prototxt(original_prototxt_path, original_model_path, optimized_prototxt_path):
    net_param = caffe_pb2.NetParameter()
    new_net_param = caffe_pb2.NetParameter()
    with open(original_prototxt_path, 'rt') as f:
        Parse(f.read(), net_param)
    for layer_idx in range(0, len(net_param.layer)):
        layer = net_param.layer[layer_idx]
        if layer.type == 'Dropout':
            if layer.top[0] == layer.bottom[0]:
                continue
            else:
                new_net_param.layer[-1].top[0] = layer.top[0]
        else:
            new_net_param.layer.extend([layer])
    new_net_param.name = net_param.name
    with open(optimized_prototxt_path, 'wt') as f:
        f.write(MessageToString(new_net_param))
    print "DROPOUT OPT : Create Optimized Prototxt Done."
    print bcolors.OKGREEN + "DROPOUT OPT : Model at " + original_model_path + "." + bcolors.ENDC
    print bcolors.OKGREEN + "DROPOUT OPT : Prototxt at " + optimized_prototxt_path + "." + bcolors.ENDC 


#https://nbviewer.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb


net = caffe_pb2.NetParameter()
with open('deploy.prototxt', 'r') as f:
    pb.text_format.Merge(f.read(), net)

input=net.layer[0];
shape=kk=input.input_param.shape[0];
shape.Clear();
shape.dim.MergeFrom([10,96,27,27]);

del net.layer[1:5]
C1=net.layer[1]
C1.ClearField('bottom')
C1.bottom.append(input.name)


with open('net2.prototxt', 'w') as f:
    f.write(pb.text_format.MessageToString(net))