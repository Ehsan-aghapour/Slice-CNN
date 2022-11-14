import sys
import caffe
from caffe import layers as cl

def create_neural_net(input_file, batch_size=50):
	net = caffe.NetSpec()  
	net.data, net.label = cl.HDF5Data(batch_size=batch_size, source=input_file, ntop=2)
	net.fc1 = cl.InnerProduct(net.data, num_output=100, weight_filler=dict(type='xavier'))
	net.relu1 = cl.ReLU(net.fc1, in_place=True)
	net.fc2 = cl.InnerProduct(net.relu1, num_output=50, weight_filler=dict(type='xavier'))
	net.relu2 = cl.ReLU(net.fc2, in_place=True)
	net.fc3 = cl.InnerProduct(net.relu1, num_output=20, weight_filler=dict(type='xavier'))
	net.relu3 = cl.ReLU(net.fc3, in_place=True)
	net.fc4 = cl.InnerProduct(net.relu3, num_output=1, weight_filler=dict(type='xavier'))
	net.loss = cl.SoftmaxWithLoss(net.fc4, net.label)
	return net.to_proto()

if __name__=='__main__':
	train_h5list_file = sys.argv[1]
	output_file = sys.argv[2]
	batch_size = 50
	with open(output_file, 'w') as f:
		f.write(str(create_neural_net(train_h5list_file, batch_size)))


