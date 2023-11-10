#include <set>
#include <map>
#include <string>
#include <unordered_set>



std::unordered_set<std::string> get_end_task_names(std::string graph_name){
	std::unordered_set<std::string> _end_task_names;
	if(graph_name=="alexnet" ){
		//_end_task_names={ "pool1", "pool2", "conv3", "conv4", "pool5", "fc6", "fc7", "prob" };
		_end_task_names={ "pool1", "pool2", "relu3", "relu4", "pool5", "relu6", "relu7", "prob" };
	}
	if(graph_name=="googlenet"){
		_end_task_names={ "pool1/norm1", "pool2/3x3_s2", "inception_3a/concat", "pool3/3x3_s2", "inception_4a/concat", "inception_4b/concat", "inception_4c/concat", "inception_4d/concat", "pool4/3x3_s2", "inception_5a/concat", "prob" };
	}
	if(graph_name=="mobilenetv1"){
		//_end_task_names={ "Conv2d_0+Conv2d_0/BatchNorm", "Conv2d_1_pointwise/Conv2D+Conv2d_1_pointwise/BatchNorm", "Conv2d_2_pointwise/Conv2D+Conv2d_2_pointwise/BatchNorm", "Conv2d_3_pointwise/Conv2D+Conv2d_3_pointwise/BatchNorm", "Conv2d_4_pointwise/Conv2D+Conv2d_4_pointwise/BatchNorm", "Conv2d_5_pointwise/Conv2D+Conv2d_5_pointwise/BatchNorm", "Conv2d_6_pointwise/Conv2D+Conv2d_6_pointwise/BatchNorm", "Conv2d_7_pointwise/Conv2D+Conv2d_7_pointwise/BatchNorm", "Conv2d_8_pointwise/Conv2D+Conv2d_8_pointwise/BatchNorm", "Conv2d_9_pointwise/Conv2D+Conv2d_9_pointwise/BatchNorm", "Conv2d_10_pointwise/Conv2D+Conv2d_10_pointwise/BatchNorm", "Conv2d_11_pointwise/Conv2D+Conv2d_11_pointwise/BatchNorm", "Conv2d_12_pointwise/Conv2D+Conv2d_12_pointwise/BatchNorm", "Softmax" };
		_end_task_names={
				"Conv2d_0/Relu6",
				"Conv2d_1_pointwise/Relu6",
				"Conv2d_2_pointwise/Relu6",
				"Conv2d_3_pointwise/Relu6",
				"Conv2d_4_pointwise/Relu6",
				"Conv2d_5_pointwise/Relu6",
				"Conv2d_6_pointwise/Relu6",
				"Conv2d_7_pointwise/Relu6",
				"Conv2d_8_pointwise/Relu6",
				"Conv2d_9_pointwise/Relu6",
				"Conv2d_10_pointwise/Relu6",
				"Conv2d_11_pointwise/Relu6",
				"Conv2d_12_pointwise/Relu6",
				//"Conv2d_13_pointwise/Relu6",
				"Logits/AvgPool_1a",
				"Softmax"
		};
	}
	if(graph_name=="resnetv1_50"){
		_end_task_names= { "pool1/MaxPool",
				"block1/unit1/bottleneck_v1/add",
				"block1/unit2/bottleneck_v1/add",
				"block1/unit3/bottleneck_v1/add",
				"block2/unit1/bottleneck_v1/add",
				"block2/unit2/bottleneck_v1/add",
				"block2/unit3/bottleneck_v1/add",
				"block2/unit4/bottleneck_v1/add",
				"block3/unit1/bottleneck_v1/add",
				"block3/unit2/bottleneck_v1/add",
				"block3/unit3/bottleneck_v1/add",
				"block3/unit4/bottleneck_v1/add",
				"block3/unit5/bottleneck_v1/add",
				"block3/unit6/bottleneck_v1/add",
				"block4/unit1/bottleneck_v1/add",
				"block4/unit2/bottleneck_v1/add",
				"pool5",
				"predictions/Softmax" };
	}
	if(graph_name=="squeezenetv1"){
		_end_task_names={ "pool1", "fire2/concat", "fire3/concat", "pool4", "fire5/concat", "fire6/concat", "fire7/concat", "pool8", "fire9/concat", "prob" };
	}
	if(graph_name=="test"){
		_end_task_names={ "pool1", "pool2"};
	}
	return _end_task_names;
}
	

inline bool check(std::string model, std::string layer){	
	return (ending_task_names[model].count(layer) > 0);
}
