# ai2
#https://github.com/onnx/keras-onnx
import keras
import numpy as np
import sys

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import keras2onnx
import onnxruntime
#pip install onnxruntime
import onnx

#name of keras model
name=sys.argv[1]
#name of onnx model
onx_name=name.split('/')[-1].split('.')[0]+'.onnx'

model=keras.models.load_model(name)
onnx_model = keras2onnx.convert_keras(model, model.name)

onnx.checker.check_model(onnx_model)

temp_model_file = onx_name
#keras2onnx.save_model(onnx_model, temp_model_file,keep_initializers_as_inputs=True)

print(onnx_model.graph.input[0].type.tensor_type.shape.dim)
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].Clear()
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1
print(onnx_model.graph.input[0].type.tensor_type.shape.dim)

onnx.save(onnx_model,temp_model_file)

#sess = onnxruntime.InferenceSession(temp_model_file)


if False:
	# image preprocessing
	img_path = 'street.jpg'   # make sure the image is in img_path
	img_size = 224
	img = image.load_img(img_path, target_size=(img_size, img_size))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	# load keras model
	from keras.applications.resnet50 import ResNet50
	model = ResNet50(include_top=True, weights='imagenet')

	# convert to onnx model
	onnx_model = keras2onnx.convert_keras(model, model.name)

	# runtime prediction
	content = onnx_model.SerializeToString()
	sess = onnxruntime.InferenceSession(content)
	x = x if isinstance(x, list) else [x]
	feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
	pred_onnx = sess.run(None, feed)

#for i,layer in enumerate(model.layers):
#	print(f'Layer {i}, Layer name:{layer.name}. Layer input:{layer.input}\n layer output: {layer.output}\n')
if False:
	mat=np.random.rand(475).astype('float32')
	for i in range(475):
		mat[i]=i
	#with open('test.npy', 'rb') as f:
	#	mat=np.load(f)

	out=model.predict(mat.reshape(1,-1))
	print(out)
