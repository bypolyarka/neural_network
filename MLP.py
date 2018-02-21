#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math


class Neuron:
    """Класс Neuron"""

    def __init__(self, bias):
        self.bias = bias  # смещение (тупой нейрон = 1)
        self.weights = []  # веса

    # передаточная функция (функция активации нейрона)
    def transfer_function(self, arg):
        self.arg = arg
        return ((1 - math.exp(-self.arg)) / (1 + math.exp(-self.arg)))

    # производная передаточной функции
    def derivative_of_transfer_function(self, arg):
        self.arg = arg
        return (0.5 * (1 - self.transfer_function(self.arg)) * (1 - self.transfer_function(self.arg)))

    # вычисляет взвешенную сумму для нейрона
    def calculate_weight_sum(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return (total + self.bias)

    # вычисление выхода нейрона (применяем передаточную функцию для взвешенной суммы)
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.transfer_function(self.calculate_weight_sum())
        return (self.output)

    # вычисление ошибки нейрона с указанием учителя (среднеквадратическая ошибка)
    def calculate_error(self, teacher_instruction):
        return (0.5 * (teacher_instruction - self.output) ** 2)

    # вычисление частной производной ошибки нейрона относительно выхода другого нейрона
    # pd - сокр. partial derivative (частная производная), wrt - сокр. with respect to (относительно)
    def calculate_pd_error_wrt_output(self, teacher_instruction):  # dE/dy
        return -(teacher_instruction - self.output)

    # вычисление частной производной выхода нейрона относительно выхода другого нейрона (производная передаточной функции по взвешенной сумме)
    def calculate_pd_output_wrt_weight_sum(self):  # dy/ds // y = f(s), f - transfer func, s - weight sum
        return 0.5 * (1 - (self.output) ** 2)

    # вычисление частной производной взвешенной суммы по весу
    def calculate_pd_weight_sum_wrt_weight(self, index):  # ds/dw
        return self.inputs[index]

    # вычисление частной производной ошибки относительно взвешенной суммы
    def calculate_pd_error_wrt_weight_sum(self, teacher_instruction):  # dE/ds
        return self.calculate_pd_error_wrt_output(teacher_instruction) * self.calculate_pd_output_wrt_weight_sum()


class NeuronLayer:
    """docstring for NeuronLayer"""

    # конструктор класса
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.uniform(-0.5, 0.5)  # инициализация смещения для каждого слоя (вес тупого нейрона)
        self.neurons = []  # список нейронов в слое
        for i in range(num_neurons):  # инициализируем список нейронов слоя объектами класса Neuron
            self.neurons.append(Neuron(self.bias))

    # метод выводит инфу про слой: число нейронов, веса для каждого нейрона, вес тупого нейрона слоя
    def print_layer(self):
        print('-----Number Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print('---Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print(' Weight:', self.neurons[n].weights[w])
            print(' Bias:', self.bias)

    # метод позволяет получить выходы каждого нейрона слоя (для каждой взвешенной суммы применяется функция активации)
    def get_outputs_for_layer(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


class Perceptron:
    ALPHA = 0.1

    def __init__(self, num_hidden_layers=None, num_inputs_neurons=1044, num_hidden_neurons=101, num_outputs_neurons=1):
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers else 1
        self.num_inputs_neurons = num_inputs_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_outputs_neurons = num_outputs_neurons

        self.hidden_layers = []
        [self.hidden_layers.append(NeuronLayer(self.num_hidden_neurons, None)) for i in range(self.num_hidden_layers)]

        self.output_layer = NeuronLayer(num_outputs_neurons, None)

        self.init_weights_from_inputs_to_first_hidden_layer()
        if self.num_hidden_layers > 1: self.init_weights_between_hidden_layers()
        self.init_weights_from_last_hidden_layer_to_output_layer()

    def init_weights_from_inputs_to_first_hidden_layer(self):
        for h in range(self.num_hidden_neurons):
            for i in range(self.num_inputs_neurons):
                self.hidden_layers[0].neurons[h].weights.append(random.uniform(-0.5, 0.5))

    def init_weights_between_hidden_layers(self):
        for i in range(1, self.num_hidden_layers):
            for h_next in range(self.num_hidden_neurons):
                for h_previous in range(self.num_hidden_neurons):
                    self.hidden_layers[i].neurons[h_next].weights.append(random.uniform(-0.5, 0.5))

    def init_weights_from_last_hidden_layer_to_output_layer(self):
        for o in range(self.num_outputs_neurons):
            for h in range(self.num_hidden_neurons):
                self.output_layer.neurons[o].weights.append(random.uniform(-0.5, 0.5))

    def get_outputs_for_network(self, inputs):
        hidden_layer_outputs = self.hidden_layers[0].get_outputs_for_layer(inputs)
        if self.num_hidden_layers > 1:
            for i in range(1, self.num_hidden_layers):
                hidden_layer_outputs = self.hidden_layers[i].get_outputs_for_layer(hidden_layer_outputs)
        return self.output_layer.get_outputs_for_layer(hidden_layer_outputs)

    def training_network(self, training_set):
        #global pd_errors_wrt_hidden_neuron1
        for z in training_set:
            training_inputs, training_outputs = z
            self.get_outputs_for_network(training_inputs)  # считаем выход сети для обуч входных данных

            # 1) считаем ошибки для нейронов выходного слоя (отклонения выходного слоя)
            pd_errors_wrt_output_neuron = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                pd_errors_wrt_output_neuron[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_weight_sum(training_outputs[o])

            # 2) вычисляем отклонения для последнего скрытого слоя
            pd_errors_wrt_hidden_neuron = [0] * len(self.hidden_layers[self.num_hidden_layers - 1].neurons)
            for h in range(len(self.hidden_layers[self.num_hidden_layers - 1].neurons)):
                d_error_wrt_hidden_neuron_output = 0
                for o in range(len(self.output_layer.neurons)):
                    d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].weights[h]
                pd_errors_wrt_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * self.hidden_layers[self.num_hidden_layers - 1].neurons[h].calculate_pd_output_wrt_weight_sum()

            # 3) если скрытых слоев несколько, то вычисляем отклонения для каждого из них
            if self.num_hidden_layers > 1:
                pd_errors_wrt_hidden_neuron1 = [[0] * self.num_hidden_neurons for z in range(self.num_hidden_layers - 1)]

                for x in range(len(self.hidden_layers) - 2, -1, -1):

                    for i in range(self.num_hidden_neurons):
                        d_error_wrt_hidden_neuron_output = 0
                        for j in range(self.num_hidden_neurons):
                            d_error_wrt_hidden_neuron_output += pd_errors_wrt_hidden_neuron[j] * self.hidden_layers[x].neurons[j].weights[i]
                        pd_errors_wrt_hidden_neuron1[x][i] = d_error_wrt_hidden_neuron_output * self.hidden_layers[x - 1].neurons[i].calculate_pd_output_wrt_weight_sum()
            #print (pd_errors_wrt_hidden_neuron1[0][0])

            # 4) обновляем веса между нейронами последнего скрытого слоя и выходными нейронами (w_ho => weights between hidden2 neurons and outputs)
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    pd_error_wrt_weight = pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].calculate_pd_weight_sum_wrt_weight(w_ho)
                    self.output_layer.neurons[o].weights[w_ho] -= self.ALPHA * pd_error_wrt_weight

            if self.num_hidden_layers > 1:
                # 5) обновляем веса между нейронами 1-го скрытого слоя и нейронами последнего скрытого слоя
                for x in range(len(self.hidden_layers) - 2, -1, -1):

                    for w_h2o in range(self.num_hidden_neurons):
                        for w_h1o in range(len(self.hidden_layers[x].neurons[w_h2o].weights)):
                            pd_error_wrt_weight = pd_errors_wrt_hidden_neuron1[x][w_h2o] * self.hidden_layers[x].neurons[w_h2o].calculate_pd_weight_sum_wrt_weight(w_h1o)
                            self.hidden_layers[x].neurons[w_h2o].weights[w_h1o] -= self.ALPHA * pd_error_wrt_weight

            # 6) обновляем веса между входными данными и скрытыми нейронами 1-го скрытого слоя(w_ih1 => weight input -> hidden1)
            for h1 in range(len(self.hidden_layers[0].neurons)):
                for w_ih1 in range(len(self.hidden_layers[0].neurons[h1].weights)):
                    if self.num_hidden_layers > 1:
                        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron1[0][h1] * self.hidden_layers[0].neurons[h1].calculate_pd_weight_sum_wrt_weight(w_ih1)
                    else:
                        pd_error_wrt_weight = pd_errors_wrt_hidden_neuron[h1] * self.hidden_layers[0].neurons[h1].calculate_pd_weight_sum_wrt_weight(w_ih1)
                    self.hidden_layers[0].neurons[h1].weights[w_ih1] -= self.ALPHA * pd_error_wrt_weight

    def calculate_total_network_error(self, set):
        total_error = 0
        for x in set:
            set_inputs, set_outputs = x
            self.get_outputs_for_network(set_inputs)
            for o in range(len(set_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(set_outputs[o])
        return (total_error / len(set))

    def testing_network(self, testing_set):
        return self.calculate_total_network_error(testing_set)

class MultiLayerPerceptron:
    """docstring for MultiLayerPerceptron"""
    ALPHA = 0.1  # шаг метода градиентного спуска (коэф., характеризующий скорость обучения сети)

    # конструктор класса
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None,
                 hidden_layer_bias=None, output_layer_weights=None, output_layer_bias=None):
        self.num_inputs = num_inputs  # количество входных данных сети
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)  # создаем скрытый слой перцептрона
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)  # создаем выходной слой

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)  # инициализация весов от входных данных к первому скрытому слою
        #self.init_weights_from_hidden_leyer1_neurons_to_hidden_layer2_neurons(hidden_layer2_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_neurons(output_layer_weights)

    # метод задает веса между входными данными и первым скрытым слоем нейронов
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.uniform(-0.5, 0.5))  # инициализируем веса случайными числами
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_leyer1_neurons_to_hidden_layer2_neurons(self, hidden_layer2_weights):
        weight_num = 0
        for h2 in range(len(self.hidden_layer2.neurons)):
            for h1 in range(len(self.hidden_layer1.neurons)):
                if not hidden_layer2_weights:
                    self.hidden_layer2.neurons[h2].weights.append(random.uniform(-0.5, 0.5))  # инициализируем веса случайными числами
                else:
                    self.hidden_layer2.neurons[h2].weights.append(hidden_layer2_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.uniform(-0.5, 0.5))
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    #
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.uniform(-0.5, 0.5))
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    # метод выводит структуру сети
    def print_MLP(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.print_layer()
        print('------')
        print('* Output Layer')
        self.output_layer.print_layer()
        print('------')

    # метод возвращает результат работы сети, принимая на вход входные данные
    def get_outputs_for_network(self, inputs):
        hidden_layer_outputs = self.hidden_layer.get_outputs_for_layer(inputs)
        return self.output_layer.get_outputs_for_layer(hidden_layer_outputs)

    def get_outputs_for_perceptron2(self, inputs):
        hidden_layer1_outputs = self.hidden_layer1.get_outputs_for_layer(inputs)
        hidden_layer2_outputs = self.hidden_layer2.get_outputs_for_layer(hidden_layer1_outputs)
        return self.output_layer.get_outputs_for_layer(hidden_layer2_outputs)

    # метод тестирует обучение сети и возвращает ошибку обобщения
    def network_testing(self, testing_set):
        return self.calculate_total_network_error(testing_set)

    # метод обучает нейронку на всех векторах обучающего множества
    def network_training(self, training_set):
        for x in training_set:
            training_inputs, training_outputs = x
            self.get_outputs_for_network(training_inputs)  # считаем выход сети для обуч входных данных

        # 1) считаем ошибки для нейронов выходного слоя (отклонения выходного слоя)
            pd_errors_wrt_output_neuron = [0] * len(self.output_layer.neurons)
            for o in range(len(self.output_layer.neurons)):
                pd_errors_wrt_output_neuron[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_weight_sum(training_outputs[o])

        # 2) вычисляем отклонения для скрытого слоя
            pd_errors_wrt_hidden_neuron = [0] * len(self.hidden_layer.neurons)
            for h in range(len(self.hidden_layer.neurons)):
                d_error_wrt_hidden_neuron_output = 0
                for o in range(len(self.output_layer.neurons)):
                    d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].weights[h]
                pd_errors_wrt_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_output_wrt_weight_sum()

        # 3) обновляем веса между скрытыми и выходными нейронами (w_ho => weights between hidden neurons and outputs)
            for o in range(len(self.output_layer.neurons)):
                for w_ho in range(len(self.output_layer.neurons[o].weights)):
                    pd_error_wrt_weight = pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].calculate_pd_weight_sum_wrt_weight(w_ho)
                    self.output_layer.neurons[o].weights[w_ho] -= self.ALPHA * pd_error_wrt_weight

        # обновляем веса между входными данными и скрытыми нейронами (w_ih => weight input -> hidden)
            for h in range(len(self.hidden_layer.neurons)):
                for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                    pd_error_wrt_weight = pd_errors_wrt_hidden_neuron[h] * self.hidden_layer.neurons[h].calculate_pd_weight_sum_wrt_weight(w_ih)
                    self.hidden_layer.neurons[h].weights[w_ih] -= self.ALPHA * pd_error_wrt_weight

    def perceptron2_training(self, training_inputs, training_outputs):
        self.get_outputs_for_perceptron2(training_inputs)  # считаем выход сети для обуч входных данных

        # 1) считаем ошибки для нейронов выходного слоя (отклонения выходного слоя)
        pd_errors_wrt_output_neuron = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_weight_sum(training_outputs[o])

        # 2) вычисляем отклонения для второго скрытого слоя
        pd_errors_wrt_hidden2_neuron = [0] * len(self.hidden_layer2.neurons)
        for h2 in range(len(self.hidden_layer2.neurons)):
            d_error_wrt_hidden2_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden2_neuron_output += pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].weights[h2]
            pd_errors_wrt_hidden2_neuron[h2] = d_error_wrt_hidden2_neuron_output * self.hidden_layer2.neurons[h2].calculate_pd_output_wrt_weight_sum()

        # 3) вычисляем отклонения для первого скрытого слоя
        pd_errors_wrt_hidden1_neuron = [0] * len(self.hidden_layer1.neurons)
        for h1 in range(len(self.hidden_layer1.neurons)):
            d_error_wrt_hidden1_neuron_output = 0
            for h2 in range(len(self.hidden_layer2.neurons)):
                d_error_wrt_hidden1_neuron_output += pd_errors_wrt_hidden2_neuron[h2] * self.hidden_layer2.neurons[h2].weights[h1]
            pd_errors_wrt_hidden1_neuron[h1] = d_error_wrt_hidden1_neuron_output * self.hidden_layer1.neurons[h1].calculate_pd_output_wrt_weight_sum()

        # 4) обновляем веса между нейронами gоследнего скрытого слоя и выходными нейронами (w_h2o => weights between hidden2 neurons and outputs)
        for o in range(len(self.output_layer.neurons)):
            for w_h2o in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].calculate_pd_weight_sum_wrt_weight(w_h2o)
                self.output_layer.neurons[o].weights[w_h2o] -= self.ALPHA * pd_error_wrt_weight

        # 5) обновляем веса между нейронами 1-го скрытого слоя и нейронами 2-го скрытого слоя
        for w_h2o in range(len(self.hidden_layer2.neurons)):
            for w_h1o in range(len(self.hidden_layer2.neurons[w_h2o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden2_neuron[w_h2o] * self.hidden_layer2.neurons[w_h2o].calculate_pd_weight_sum_wrt_weight(w_h1o)
                self.hidden_layer2.neurons[w_h2o].weights[w_h1o] -= self.ALPHA * pd_error_wrt_weight

        # обновляем веса между входными данными и скрытыми нейронами 1-го скрытого слоя(w_ih1 => weight input -> hidden1)
        for h1 in range(len(self.hidden_layer1.neurons)):
            for w_ih1 in range(len(self.hidden_layer1.neurons[h1].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden1_neuron[h1] * self.hidden_layer1.neurons[h1].calculate_pd_weight_sum_wrt_weight(w_ih1)
                self.hidden_layer1.neurons[h1].weights[w_ih1] -= self.ALPHA * pd_error_wrt_weight

    def network_training_for_avarage_error(self, training_sets):
        pd_errors_wrt_output_neuron = [0] * len(self.output_layer.neurons)  # скапливаем ошибки для выходов
        pd_errors_wrt_hidden_neuron = [0] * len(self.hidden_layer.neurons)  # скапливаем ошибки для скрытых нейронов

        for x in training_sets:
            training_inputs, training_outputs = x
            self.get_outputs_for_network(training_inputs)

            for o in range(len(self.output_layer.neurons)):
                pd_errors_wrt_output_neuron[o] += self.output_layer.neurons[o].calculate_pd_error_wrt_weight_sum(
                    training_outputs[o])

            for h in range(len(self.hidden_layer.neurons)):
                d_error_wrt_hidden_neuron_output = 0
                for o in range(len(self.output_layer.neurons)):
                    d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron[o] * \
                                                        self.output_layer.neurons[o].weights[h]
                pd_errors_wrt_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[
                    h].calculate_pd_output_wrt_weight_sum()

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron[o] * self.output_layer.neurons[o].calculate_pd_weight_sum_wrt_weight(w_ho)
                self.output_layer.neurons[o].weights[w_ho] -= self.ALPHA * pd_error_wrt_weight

        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron[h] * self.hidden_layer.neurons[h].calculate_pd_weight_sum_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.ALPHA * pd_error_wrt_weight

    # метод возвращает среднюю ошибку работы сети на множестве
    def calculate_total_network_error(self, set):
        total_error = 0
        for x in set:
            set_inputs, set_outputs = x
            self.get_outputs_for_network(set_inputs)
            for o in range(len(set_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(set_outputs[o])
        return (total_error / len(set))

    def calculate_total_perceptron2_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.get_outputs_for_perceptron2(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return (total_error)




# метод считывает из файла вектора обучающей выборки и возвращает список этих векторов
def get_set_from_file(name=None):
    with open(name) as f:
        training_sets = []
        train_set = [row.strip() for row in f]
        # print(train_set[-1][-1][-2:])
        for x in train_set:
            list1 = []
            # print(x)
            if len(x) == 1045:
                for i in range(len(x) - 1):
                    list1.append(float(x[i]))
                list2 = [float(x[-1])]
                training_sets.append([list1, list2])
            else:
                for i in range(len(x) - 2):
                    list1.append(float(x[i]))
                list2 = [float(x[-2:])]
                training_sets.append([list1, list2])
    return (training_sets)

