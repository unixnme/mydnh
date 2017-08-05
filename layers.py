import numpy as np


class Layer(object):
    # takes an input and converts to an output
    def __init__(self):
        self.learning_rate = .01

    # generic forward method where shape does not change
    def forward(self, x):
        self.grad = []
        y = []
        for xi in x:
            yi = self.eval(xi)
            dyi = self.derivative(xi, yi)
            y.append(yi)
            self.grad.append(dyi)
        return y

    def backward(self, g):
        output = []
        for idx in range(len(g)):
            output.append(g[idx] * self.grad[idx])
        return output


class Input(Layer):
    def forward(self, x):
        # if not a list, create a list
        # len(x) must be a batch size
        if not isinstance(x, list):
            return [x]
        for xi in x:
            if not isinstance(xi, (np.ndarray, np.float32)):
                raise Exception("input must be np.array or a list of np.array")
        return x

    def backward(self, g):
        return g


class Identity(Layer):
    def forward(self, x):
        return x

    def backward(self, g):
        return g


class Sigmoid(Layer):
    @classmethod
    def eval(cls, xi):
        return 1.0 / (1.0 + np.exp(-xi))

    @classmethod
    def derivative(cls, xi, yi):
        return yi * (1 - yi)

class Elu(Layer):
    @classmethod
    def eval(cls, xi):
        return (xi > 0) * xi + (xi <= 0) * (np.exp(xi) - 1)

    @classmethod
    def derivative(cls, xi, yi):
        return (xi > 0) * 1 + (xi <= 0) * (yi + 1)

class Relu(Layer):
    @classmethod
    def eval(cls, xi):
        return (xi > 0) * xi

    @classmethod
    def derivative(cls, xi, yi):
        return (xi > 0) * 1

class Dense(Layer):
    # return product of matrix multiplication
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.randn(input_size, output_size).astype(np.float32)
        self.bias = np.zeros((1, output_size)).astype(np.float32)
        super(Dense, self).__init__()

    def eval(self, xi):
        xi = xi.reshape(1, -1)
        return np.dot(xi, self.weight) + self.bias

    def derivative(self, xi, yi):
        return self.weight

    def forward(self, x):
        y = []
        self.grad = []
        self.grad_w = []
        for xi in x:
            xi = xi.reshape(1,-1)
            yi = self.eval(xi)
            y.append(yi)
            self.grad.append(self.derivative(xi, yi))
            self.grad_w.append(xi)
        return y

    def backward(self, g):
        dw = np.zeros(self.weight.shape)
        db = np.zeros(self.bias.shape)
        output = []
        for idx in range(len(g)):
            dw += np.dot(self.grad_w[idx].T, g[idx])
            db += g[idx]
            output.append(np.dot(g[idx], self.grad[idx].T))
        self.weight -= self.learning_rate * dw * len(g)
        self.bias -= self.learning_rate * db * len(g)
        return output

class Square(Layer):
    @classmethod
    def eval(cls, xi):
        return xi**2

    @classmethod
    def derivative(cls, xi, yi):
        return 2*xi

class Sum(Layer):
    def __init__(self, axis=-1):
        self.axis = axis
        super(Sum, self).__init__()

    def eval(self, xi):
        return np.sum(xi, axis=self.axis)

    def derivative(self, xi, yi):
        return np.ones(yi.shape, dtype=np.float32)

class BatchSum(Layer):
    # return 1 for backward
    def __init__(self):
        super(BatchSum, self).__init__()

    def forward(self, x):
        self.batch_size = len(x)
        output = 0
        for xi in x:
            output += xi
        return output

    def backward(self, g):
        return list(np.ones((self.batch_size)))

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
    #np.random.seed(1)
    batch_size = 10
    feature_size = 3
    x = np.random.rand(30000,feature_size)
    weight = np.array([[.1, .2], [.3, .4], [.5, .6]])
    bias = np.array([[-.1, -.2]])
    #weight = np.array([[.1]])
    #bias = np.array([[.2]])
    temp = np.dot(x, weight) + bias
    y = Sigmoid.eval(temp)

    layers = []
    layers.append(Input())
    layers.append(Dense(*weight.shape))
    layers.append(Sigmoid())
    layers.append(Diff())
    layers.append(Square())
    layers.append(Sum())
    layers.append(BatchSum())

    for idx in range(0, len(x), batch_size):
        xi = list(x[idx:idx+batch_size])
        yi = list(y[idx:idx+batch_size])
        for layer in layers:
            if not isinstance(layer, Diff):
                xi = layer.forward(xi)
            else:
                xi = layer.forward(xi, list(yi))
        print (layers[1].weight, layers[1].bias)

        for layer in reversed(layers):
            xi = layer.backward(xi)

