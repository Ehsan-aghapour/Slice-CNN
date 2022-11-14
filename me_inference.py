import numpy as np
import cv2
from rknn.api import RKNN
import sys

precompile=False
PC=True

name=sys.argv[1]
rknn_name=name.split('/')[-1].split('.')[0]+'.rknn'
rknn_name_precompiled=name.split('/')[-1].split('.')[0]+'_precompiled.rknn'
model_type=name.split('.')[-1]

def show_outputs(outputs):
	output = outputs[0][0]
	output_sorted = sorted(output, reverse=True)
	top5_str = 'mobilenet_v1\n-----TOP 5-----\n'
	for i in range(5):
		value = output_sorted[i]
		index = np.where(output == value)
		for j in range(len(index)):
			if (i + j) >= 5:
				break
			if value > 0:
				topi = '{}: {}\n'.format(index[j], value)
			else:
				topi = '-1: 0.0\n'
			top5_str += topi
	print(top5_str)

def show_perfs(perfs):
	perfs = 'perfs: {}\n'.format(outputs)
	print(perfs)

if __name__ == '__main__':

	# Create RKNN object
	rknn = RKNN()
	
	

	# Set inputs
	#'./dog_224x224.jpg'
	img = cv2.imread(sys.argv[2])
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# init runtime environment
	print('--> Init runtime environment')
	
	if model_type!='rknn':
		# pre-process config
		print('--> config model')
		#rknn.config(channel_mean_value='103.94 116.78 123.68 58.82', reorder_channel='0 1 2')
		rknn.config(channel_mean_value='122.68 116.67 104.01 1', reorder_channel='0 1 2')
		print('done')

		# Load tensorflow model
		print('--> Loading model')
		#ret = rknn.load_tflite(model='./mobilenet_v1.tflite')
		print('--> Loading model')
	#rknn.load_tensorflow(tf_pb='model.pb',
	#                     inputs=['test_in'],
	#                     outputs=['test_out/BiasAdd'],
	#                     input_size_list=[[INPUT_SIZE]])
	#rknn.load_onnx(name)
		if model_type=='pb':
			if len(sys.argv)==5:
				inputs=sys.argv[2]
				outputs=sys.argv[3]
				INPUT_SIZE=sys.argv[4]
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

		print('done')

		# Build model
		print('--> Building model')
		#ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
		ret = rknn.build(do_quantization=False)
		if ret != 0:
			print('Build mobilenet_v1 failed!')
			exit(ret)
		print('done')

		# Export rknn model
		print('--> Export RKNN model')
		ret = rknn.export_rknn(rknn_name)
		if ret != 0:
			print(f'Export {rknn_name} failed!')
			exit(ret)
		print('done')
	
		ret = rknn.init_runtime(target='RK3399Pro',rknn2precompile=True)
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		ret = rknn.export_rknn_precompile_model(rknn_name_precompiled)
		if ret != 0:
			print('export prcompile failed')
			exit(ret)
	else:
		#rknn.load_rknn('./mobilenet_v1_sample_test_precompiled.rknn')
		rknn.load_rknn(sys.argv[1])
		if 'precompiled' in sys.argv[1]:
			PC=0
		if PC:
			ret = rknn.init_runtime()
		else:
			ret = rknn.init_runtime(target='RK3399Pro')
		if ret != 0:
			print('Init runtime environment failed')
			exit(ret)
		print('done')

	# Inference
	print('--> Running model')
	outputs = rknn.inference(inputs=[img])
	show_outputs(outputs)
	print('done')
	print(f'output shape:{np.array(outputs).shape}')
	p=0
	if p:
		for i,output in enumerate(outputs[0][0]):
			if (i+1)%10:
				print(f'{i:<4}:{output:^10.4f}',end='\t')
			else:
				print(f'{i:<4}:{output:^10.4f}')

	perf=0
	if perf:
		# perf
		print('--> Begin evaluate model performance')
		perf_results = rknn.eval_perf(inputs=[img])
		print('done')

	rknn.release()

