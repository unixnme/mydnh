import numpy as np

class Layer(object):
    # takes an input and converts to an output
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self):
        pass

class Identity(Layer):
    # return the same array
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def backward(self):
        pass

class Sigmoid(Layer):
    # return sigmoid function
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def backward(self):
        pass

class Dense(Layer):
    # return product of matrix multiplication
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.randn(input_size, output_size).astype(np.float32)
        self.bias = np.zeros(output_size).astype(np.float32)
        super(Dense, self).__init__()

    def forward(self, input):
        input = input.reshape(1,-1)
        if len(input[0]) != self.input_size:
            raise Exception("input shape does not match")
        return np.dot(input, self.weight) + self.bias


if __name__ == '__main__':
    from numpy.random import randint
    layer1 = Identity()
    layer2 = Dense(5, 3)
    layer3 = Sigmoid()


    for _ in range(10):
        x = randint(0, 255, (5)).astype(np.float32)
        x = x / 255.0 - 0.5
        x = layer1.forward(x)
        x = layer2.forward(x)
        x = layer3.forward(x)
        print x