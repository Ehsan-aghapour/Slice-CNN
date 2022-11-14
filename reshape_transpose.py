def transpose(array,shape=[96,27,27],perm=[1,2,0]):
	#shape=[96,27,27]
	stride=[shape[-2]*shape[-1],shape[-1],1]
	
	#perm=[1,2,0]

	shape_permuted=[shape[perm[0]],shape[perm[1]],shape[perm[2]]]
	stride_permuted=[stride[perm[0]],stride[perm[1]],stride[perm[2]]]

	g=[]
	print(f'shape is: {shape}\nstride is:{stride}\nnew shape is: {shape_permuted}\nnew stride is: {stride_permuted}')
	for s0 in range(shape_permuted[0]):
		for s1 in range(shape_permuted[1]):
			for s2 in range(shape_permuted[2]):
				i=s2+(s1*shape_permuted[2])+(s0*shape_permuted[1]*shape_permuted[2])
				indx=s2*stride_permuted[2]+s1*stride_permuted[1]+s0*stride_permuted[0]
				print(indx)
				g.append(array[indx])
	
	return g
