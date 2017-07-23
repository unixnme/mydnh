import numpy as np


class Layer(object):
    # takes an input and converts to an output
    def __init__(self):
        self.learning_rate = 1

    def forward(self, x):
        pass

    def backward(self, g):
        output = []
        for idx in range(len(g)):
            output.append(np.dot(g[idx], self.grad[idx]))
        return output


class Input(Layer):
    # takes an input and converts to an output
    def __init__(self):
        super(Input, self).__init__()

    def forward(self, x):
        # if not a list, create a list
        if not isinstance(x, list):
            return [x]
        for xi in x:
            if not isinstance(xi, np.ndarray):
                raise Exception("input must be np.array or a list of np.array")
        return x

    def backward(self, g):
        return g


class Identity(Layer):
    # return the same array
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def backward(self, g):
        return g


class Sigmoid(Layer):
    # return sigmoid function
    def __init__(self):
        super(Sigmoid, self).__init__()

    def eval(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        output = []
        grad = []
        for xi in x:
            temp = self.eval(xi)
            output.append(temp)
            grad.append(temp * (1 - temp))
        self.grad = grad
        return output

    def backward(self, g):
        return super(Sigmoid, self).backward(g)


class Dense(Layer):
    # return product of matrix multiplication
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.randn(input_size, output_size).astype(np.float32)
        self.bias = np.zeros(output_size).astype(np.float32)
        super(Dense, self).__init__()

    def forward(self, x):
        output = []
        grad = []
        grad_w = []
        grad_b = []
        for xi in x:
            xi = xi.reshape(1,-1)
            if len(xi[0]) != self.input_size:
                raise Exception("input shape does not match")
            output.append(np.dot(xi, self.weight) + self.bias)
            grad.append(self.weight)
            grad_w.append(xi)
            grad_b.append(np.ones(1, dtype=np.float32))
        self.grad = grad
        self.grad_w = grad_w
        self.grad_b = grad_b
        return output

    def backward(self, g):
        dw = np.zeros(self.weight.shape)
        db = np.zeros(self.bias.shape)
        for idx in range(len(g)):
            dw += np.dot(g[idx], self.grad_w[idx])
            db += np.dot(g[idx], self.grad_b[idx])
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        return super(Dense, self).backward(g)


class Square(Layer):
    # return squared function
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        output = []
        grad = []
        for xi in x:
            output.append(xi**2)
            grad.append(2*xi)
        self.grad = grad
        return output

    def backward(self, g):
        return super(Square, self).backward(g)


class Loss(Layer):
    # return 1 for backward
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x):
        self.length = len(x)
        return x

    def backward(self, g):
        output = []
        for _ in range(self.length):
            output.append(np.array(1, dtype=np.float32))
        return output

class Diff(Layer):
    # subtract from y
    def __init__(self):
        super(Diff, self).__init__()

    def forward(self, x, y):
        # if not a list, create a list
        if not isinstance(y, list):
            y = [y]
        for idx in range(len(y)):
            if not isinstance(y[idx], np.ndarray):
                y[idx] = np.array(y[idx])


        output = []
        grad = []
        if len(x) != len(y):
            raise Exception("x,y batch number mismatch")

        for idx in range(len(x)):
            output.append(y[idx] - x[idx])
            grad.append(np.array(-1, dtype=np.float32))

        self.grad = grad
        return output

    def backward(self, g):
        return super(Diff, self).backward(g)


if __name__ == '__main__':
    x = np.random.rand(1000,1)
    temp = -.1*x + .5
    y = 1.0 / (1.0 + np.exp(-temp))

    layers = []
    layers.append(Input())
    layers.append(Dense(1, 1))
    layers.append(Sigmoid())
    layers.append(Diff())
    layers.append(Square())
    layers.append(Loss())

    for xi,yi in zip(x, y):
        for layer in layers:
            if not isinstance(layer, Diff):
                xi = layer.forward(xi)
            else:
                xi = layer.forward(xi, yi)
        print (layers[1].weight, layers[1].bias)

        for layer in reversed(layers):
            xi = layer.backward(xi)

    # layer1 = Identity()
    # layer2 = Dense(5, 3)
    # layer3 = Sigmoid()
    #
    #
    # for _ in range(10):
    #     x = randint(0, 255, (5)).astype(np.float32)
    #     x = x / 255.0 - 0.5
    #     x = layer1.forward(x)
    #     x = layer2.forward(x)
    #     x = layer3.forward(x)
    #     print x