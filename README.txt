keras summary: tensors, nodes and layers
https://stackoverflow.com/questions/53942291/what-does-the-00-of-the-layers-connected-to-in-keras-model-summary-mean

for each model there are layers:
model.layers

for each layer there is input and output tensor:
layer.input
layer.output

output tensor of a layer is equal to the input tensor of next layer:
l.ouput is exaclty same is next_l.input

when creating a model you can start by a tensor:
for example you can first create an input layer and start by its output:
input_layer=keras.layers.InoutLayer(input_shape=(227,227,3))
x=input_layer.output
x=desired_layer(x)
or:
x=keras.layers.Input(shape=(227,227,3))
x=model.layers[0](x)

** when we use x=model.layers[i](x) a new output tensor and new node is added to the model.layers[i] (with new name)
So the previous and new tenosors and nodes are added to the layer. Therefore when you try to split model with following method it give error:

# cereate an input layer with the shape of layer 19 input
in_layer=keras.layers.InputLayer(input_shape=model.layers[19].input.shap[1:])
x=in_layer.output
x=model.layers[19](x)

new_m=keras.models.Model(inputs=in_layer.input,outputs=model.output)
(this command give error even if you delete model.layers[0:18 with model.layers.pop(0) one by one] because 
the tensor x is new tensor and so model.layers[19] has two outputs now.) If you want to use this method you should continue x 
to the end of the model:
https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model/54517478#54517478

The othe method is to modified the graph instead of creating new tensors. I used this method by replacing input_tensor and inbound_layers
of model.layer[18]._outbound_nodes to the new inlayer instead of model.layer[18]

To generate a.ou:
g++ C_call.cpp 


