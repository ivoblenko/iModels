import numpy as np

class Perceptron:
    def __init__(self, data, classification_mode=False, activation_function_type="sigmoid", count_of_layers=3,
                 epochs=10000):
        self.data = data
        self.step = 0.001
        self.input_data = np.resize(self.data.iloc[:, 1:].to_numpy(), (self.data.shape[0], 28 * 28))
        self.output_data = np.resize(self.data.iloc[:, 0].to_numpy(), (self.data.shape[0], 1))
        self.output_layer_size = None
        self.recognized_dict = None
        self.layers = None
        self.weights = None
        self.activation_func = None
        self.count_of_layers = count_of_layers
        self.epochs = epochs
        self.activation_functions = {"sigmoid": self.sigmoid}
        if classification_mode:
            self.reorganized_y_to_classification()

        self.set_activation_function(activation_function_type)

        self.set_layers(count_of_layers)

    def set_activation_function(self, activation_function):
        self.activation_func = self.activation_functions.get(activation_function, self.sigmoid)

    def set_layers(self):
        self.layers = np.empty([self.count_of_layers])
        self.layers[0] = self.input_data

    def set_random_weights(self):
        self.weights = np.array()
        for i in range(self.count_of_layers - 1):
            cur_demension = self.layers[i].shape[1]
            if i < self.count_of_layers - 1:
                next_demesion = cur_demension // 2
            else:
                next_demesion = self.output_data.shape[1]
            np.append(self.weights, 2 * np.random.random((cur_demension, next_demesion)) - 1, axis=0)

    def reorganized_y_to_classification(self):
        unique_values = np.unique(self.y)
        count_of_bits = int(np.ceil(np.sqrt(len(unique_values))))
        change_dict = dict(map(lambda x: [x, str(bin(x))[2:].rjust(count_of_bits, "0")], unique_values))
        new_output_data = []
        for i in self.y:
            new_output_data.append(list(map(int, change_dict.get(i[0]))))
        self.output_data = np.array(new_output_data)
        self.output_layer_size = count_of_bits
        self.recognized_dict = dict(map(lambda x: [change_dict.get(x), x], change_dict.keys()))

    def learning(self):
        layer_delta = np.array()

        for i in range(self.epochs):
            for j in range(1, self.count_of_layers):
                self.layers[j] = self.activation_func(np.dot(self.layers[j - 1], self.weights[j - 1]))

            layer_delta[self.count_of_layers] = (self.layers[self.count_of_layers] - self.output_data) \
                                                * self.activation_func(self.layers[self.count_of_layers], True)
            for j in range(self.count_of_layers - 1, 1, -1):
                layer_delta[j] = layer_delta[j + 1].dot(self.weights[j + 1].T) * self.activation_func(self.layers[j],
                                                                                                      True)

            for j in range(self.count_of_layers - 1, 1, -1):
                self.weights[j] -= self.step * self.layers[j - 1].T.dot(layer_delta[j])

    def predict(self, x):
        self.layers = np.empty([self.count_of_layers])
        self.layers[0] = np.resize(x, (1, x.size))
        for i in range(1, self.count_of_layers):
            self.layers[i] = self.activation_func(np.dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[self.count_of_layers]

    # функции активации
    def sigmoid(self, x, derivative=False):
        if derivative:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        else:
            return 1 / (1 + np.exp(-x))
