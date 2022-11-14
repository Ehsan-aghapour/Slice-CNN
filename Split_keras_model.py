import keras
def split(model,Start,End):
    #DL_input = keras.layers.Input(model.layers[indx].input_shape[1:])
    DL_input = keras.layers.Input(model.layers[Start].get_input_shape_at(0)[1:])
    DL_model = DL_input
    for layer in model.layers[Start:End]:
        DL_model = layer(DL_model)
    DL_model = keras.models.Model(inputs=DL_input, outputs=DL_model)
    return DL_model
    
def main():
    m=keras.models.load_model('MobileNet.h5')
    new_m=split(m,1,-1)
    new_m.summary()
    new_m.save('m_0_-1.h5')

'''
def split2(model,indx):
    new_model=keras.models.Sequential()
    for x in model.layers[indx:]:
        #new_model.add(model.get_layer(index=i))
        new_model.add(x)
    #new_model.build(input_shape=model.layers[indx].input_shape)
    new_model.build(input_shape=model.layers[indx].get_input_shape_at(0))
    return new_model
'''
