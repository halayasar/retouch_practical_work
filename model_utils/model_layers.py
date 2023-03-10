# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer
from keras import backend as K
from keras.layers import Layer


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[i] for i in range(len(input_shape)) ])


