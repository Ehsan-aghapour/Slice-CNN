import numpy as np

def generate_dataset(shapes=[(224,224,3)]):
	print(f'generator is called with shapes: {shapes}')
	d=""
	for i,shape in enumerate(shapes):
		if type(shape)==list:
			shape=tuple(shape)
		#shape=(1,)+shape
		#original_shape = (None, None, 3)
		shape = tuple(10 if dim is None else dim for dim in shape)
		print(shape)

		print(f'first npy with shape: {shape}')
		test=f"/home/ehsan/UvA/Sub_Model/dataset/test{i}.npy"
		mat=np.random.rand(*shape).astype('float32')
		with open(test, 'wb') as f:
			np.save(f,mat)
		d=d+test+" "
	d=d[:-1]+'\n'
	with open('/home/ehsan/UvA/Sub_Model/dataset.txt', 'w') as f:
		f.write(d)
		

