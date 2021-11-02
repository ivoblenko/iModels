import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, data: pd.DataFrame, classification_mode: bool = False, activation_function_type: str = "sigmoid",
                 count_of_hidden_layers: int = 1,
                 epochs: int = 10000):

        """
        :param data: input data for fit model
        :param classification_mode: Flag to activate classification mode
        :param activation_function_type: Type of activation function to neuron
        :param count_of_hidden_layers: Count of hidden layers
        :param epochs: Count of fit's epochs
        """

        self.data = data
        self.epsilon = 0.001
        self.input_data = np.resize(self.data.iloc[:, 1:].to_numpy(), (self.data.shape[0], 28 * 28))
        self.output_data = np.resize(self.data.iloc[:, 0].to_numpy(), (self.data.shape[0], 1))
        self.output_layer_size = None
        self.recognized_dict = None
        self.layers = None
        self.weights = None
        self.activation_func = None
        self.count_of_hidden_layers = count_of_hidden_layers
        self.epochs = epochs
        self.activation_functions = {"sigmoid": self._sigmoid}
        self.classification_mode = classification_mode
        if self.classification_mode:
            self._reorganized_y_to_classification()

        self._set_activation_function(activation_function_type)

        self._set_layers()

        self._set_random_weights()

    def _set_activation_function(self, activation_function):
        self.activation_func = self.activation_functions.get(activation_function, self._sigmoid)

    def _set_layers(self):
        self.layers = [0 for i in range(self.count_of_hidden_layers + 2)]
        self.layers[0] = self.input_data

    def _set_random_weights(self):
        self.weights = [0 for i in range(self.count_of_hidden_layers + 1)]
        for i in range(self.count_of_hidden_layers + 1):
            if not i:
                cur_demension = self.layers[i].shape[1]
            else:
                cur_demension = next_demesion
            if i < self.count_of_hidden_layers:
                next_demesion = cur_demension // 2
            else:
                next_demesion = self.output_data.shape[1]
            self.weights[i] = 2 * np.random.random((cur_demension, next_demesion)) - 1

    def _reorganized_y_to_classification(self):
        unique_values = np.unique(self.output_data)
        count_of_bits = int(np.ceil(np.sqrt(len(unique_values))))
        change_dict = dict(map(lambda x: [x, str(bin(x))[2:].rjust(count_of_bits, "0")], unique_values))
        new_output_data = []
        for i in self.output_data:
            new_output_data.append(list(map(int, change_dict.get(i[0]))))
        self.output_data = np.array(new_output_data)
        self.output_layer_size = count_of_bits
        self.recognized_dict = dict(map(lambda x: [change_dict.get(x), x], change_dict.keys()))

    def fit(self):
        layer_delta = [0 for i in range(self.count_of_hidden_layers + 1)]

        for i in range(self.epochs):
            print(i)
            for j in range(1, self.count_of_hidden_layers + 2):
                self.layers[j] = self.activation_func(np.dot(self.layers[j - 1], self.weights[j - 1]))

            layer_delta[self.count_of_hidden_layers] = (self.layers[self.count_of_hidden_layers + 1] - self.output_data) \
                                                       * self.activation_func(
                self.layers[self.count_of_hidden_layers + 1],
                True)
            for j in range(self.count_of_hidden_layers - 1, -1, -1):
                layer_delta[j] = layer_delta[j + 1].dot(self.weights[j + 1].T) * self.activation_func(
                    self.layers[j + 1],
                    True)

            for j in range(self.count_of_hidden_layers, -1, -1):
                self.weights[j] -= self.epsilon * self.layers[j].T.dot(layer_delta[j])

    def predict(self, input: pd.DataFrame):
        input = input.to_numpy()
        self.layers[0] = np.resize(input, (1, input.size))
        for i in range(1, self.count_of_hidden_layers + 2):
            self.layers[i] = self.activation_func(np.dot(self.layers[i - 1], self.weights[i - 1]))
        if self.classification_mode:
            self.layers[self.count_of_hidden_layers + 1] = self.layers[self.count_of_hidden_layers + 1].round()
            predict = list(map(int, self.layers[self.count_of_hidden_layers + 1][0]))
            return self.recognized_dict.get(''.join(map(str, predict)), "_")
        else:
            return self.layers[self.count_of_hidden_layers + 1]

    # функции активации
    def _sigmoid(self, x, derivative=False):
        if derivative:
            return self._sigmoid(x) * (1 - self._sigmoid(x))
        else:
            return 1 / (1 + np.exp(-x))
