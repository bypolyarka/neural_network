#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import PIL.Image
import io
import pickle
import sys
from tkinter import *


from mlp import MLP, IMG

from absl import flags


class App(Frame):
    """docstring for App"""
    WIDTH = 232
    HEIGHT = 288

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.brush_size = 15
        self.color_brush = 'black'
        self.width = 306
        self.height = 380
        self.lab = Label(root, text='')
        self.lab.grid(row=5, column=1)
        self.setUI()

    def setUI(self):
        self.parent.title('Neural Network')

        self.canv = Canvas(root, width=self.width, height=self.height, bg='white')  #
        self.canv.grid(row=0, column=0, rowspan=5)  #
        self.canv.bind('<B1-Motion>', self.draw_canvas)  #

        self.clear_btn = Button(root, text='Clear all', width=15, command=lambda: self.canv.delete('all'))  #
        self.clear_btn.grid(row=0, column=1)  #

        self.learn_btn = Button(root, text='Поместить в\n обуч выборку', width=30)  #
        self.learn_btn.grid(row=1, column=1)  #
        self.learn_btn.bind('<Button-1>', self.save_to_training_set)  #

        self.test_btn = Button(root, text='Поместить букву в\nтренеровочное множество', width=25)
        self.test_btn.grid(row=1, column=4)
        self.test_btn.bind('<Button-1>', self.save_to_testing_set)

        self.var = IntVar()
        self.var.set(1)
        self.rbutton_yes = Radiobutton(root, text='Буква В', variable=self.var, value=1)
        self.rbutton_no = Radiobutton(root, text='Буква С', variable=self.var, value=-1)
        self.rbutton_yes.grid(row=1, column=2)  #
        self.rbutton_no.grid(row=1, column=3)  #

        self.train_btn = Button(root, text='Обучить сеть', width=15)  #
        self.train_btn.grid(row=2, column=1)  #
        self.train_btn.bind('<Button-1>', self.train)  #

        self.start_btn = Button(root, text='Start', width=15)  #
        self.start_btn.grid(row=3, column=1)  #
        self.start_btn.bind('<Button-1>', self.start)  #

    # метод рисования на холсте
    def draw_canvas(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color_brush, outline=self.color_brush)

    # метод переводит кортеж для каждого пикселя в 0 или 1
    def tuple_to_number(self, t):
        self.list1 = []
        for i in t:
            if i == (0, 0, 0):
                self.list1.append(1)  # the pixel's color is black
            else:
                self.list1.append(0)  # white
        return (self.list1)

    #
    def list_to_matrix(self, lst):
        self.list1 = []
        for i in range(0, len(self.lst), self.WIDTH):
            self.list1.append(self.lst[i:i + self.WIDTH])
        return (self.list1)

    #
    def grid_sum(self, i, j, X, input_matrix, grid_size=8):
        s = 0
        self.i = i
        self.j = j
        self.X = X
        self.input_matrix = input_matrix
        self.grid_size = grid_size
        for k in range(self.grid_size):
            for l in range(self.grid_size):
                s += self.input_matrix[self.i + k][self.j + l]
        if s < 32:
            self.X.append(0)
        else:
            self.X.append(1)

    #
    def grid_to_InputVector(self, input_matrix):
        self.X = []
        self.input_matrix = input_matrix
        self.grid_size = 8
        for i in range(0, self.HEIGHT, self.grid_size):
            for j in range(0, self.WIDTH, self.grid_size):
                self.grid_sum(i, j, self.X, self.input_matrix)
        return (self.X)

    #
    def save_to_training_set(self, event):
        self.ps = self.canv.postscript(colormode='mono')
        self.img = PIL.Image.open(io.BytesIO(self.ps.encode('utf-8')))
        IMG.save_img(self.img)
        self.img = PIL.Image.open('test.png')
        self.lst = list(self.img.getdata())
        # print(len(self.lst))
        self.lst = self.list_to_matrix(self.lst)
        self.lst = self.grid_to_InputVector(self.lst)
        # print(len(self.lst))
        origin_stdout = sys.stdout
        f = open('training_set.txt', 'a')
        sys.stdout = f
        for x in self.lst:
            print(x, sep='', end='')
        # for i in range(0, len(self.lst), self.WIDTH/8):
        #	print(self.lst[i: i + self.WIDTH/8])
        print(self.var.get())
        f.close()
        sys.stdout = origin_stdout

    def save_to_testing_set(self, event):
        self.ps = self.canv.postscript(colormode='mono')
        self.img = PIL.Image.open(io.BytesIO(self.ps.encode('utf-8')))
        IMG.save_img(self.img)
        self.img = PIL.Image.open('test.png')
        self.lst = list(self.img.getdata())
        self.lst = self.list_to_matrix(self.lst)
        self.lst = self.grid_to_InputVector(self.lst)
        origin_stdout = sys.stdout
        with open('testing_set.txt', 'a') as f:
            sys.stdout = f
            for x in self.lst: print(x, sep='', end='')
            print(self.var.get())
            sys.stdout = origin_stdout

    #
    def start(self, event):
        self.ps = self.canv.postscript(colormode='mono')
        self.img = PIL.Image.open(io.BytesIO(self.ps.encode('utf-8')))
        IMG.save_img(self.img)
        self.img = PIL.Image.open('test.png')
        self.lst = list(self.img.getdata())
        self.lst = self.list_to_matrix(self.lst)
        self.lst = self.grid_to_InputVector(self.lst)
        # print(self.lst)
        with open('perceptron.pickle', 'rb') as f:
            self.perceptron = pickle.load(f)
            # if ((perceptron2.get_outputs_for_perceptron2(self.lst)[0] > 0) and (0.5 * (1 - perceptron2.get_outputs_for_perceptron2(self.lst)[0])**2 < 0.2)):
            if self.perceptron.get_outputs_for_network(self.lst)[0] > 0:
                self.lab['text'] = 'ЭТО БУКВА "В"'
            # elif ((perceptron2.get_outputs_for_perceptron2(self.lst)[0] <= 0) and (0.5 * (- 1 - perceptron2.get_outputs_for_perceptron2(self.lst)[0])**2 < 0.2)):
            elif self.perceptron.get_outputs_for_network(self.lst)[0] <= 0:
                self.lab['text'] = 'ЭТО БУКВА "С"'
            # else: self.lab['text'] = 'Извините, не определило'
            print(self.perceptron.get_outputs_for_network(self.lst))

    #
    def train(self, event):
        self.perceptron = MLP.MultiLayerPerceptron(1044, 32, 1)
        self.training_set = MLP.get_set_from_file(name='training_set.txt')
        self.testing_set = MLP.get_set_from_file(name='testing_set.txt')

        learn_error_list = []
        testing_error_list = []
        for i in range(15):
            self.perceptron.network_training(self.training_set)
            print ('Эпоха: {0}\tОшибка обучения: {1}\tОшибка обобщения: {2}'.format(i, self.perceptron.calculate_total_network_error(self.training_set), self.perceptron.network_testing(self.testing_set)))

            with open('perceptron.pickle', 'wb') as f:
                pickle.dump(self.perceptron, f)

            learn_error = self.perceptron.calculate_total_network_error(self.training_set)
            testing_error = self.perceptron.network_testing(self.testing_set)

            learn_error_list.append(learn_error)
            testing_error_list.append(testing_error)

            if 0.5 * (learn_error - testing_error)**2 < 0.0001 and (learn_error + testing_error) / 2 < 0.01:
                break

            #self.training_set.reverse()
            #self.testing_set.reverse()

        '''for i in range(1, 11):
            perceptron = MLP.Perceptron(i)
            [perceptron.training_network(self.training_set) for j in range(10)]
            learn_error_list.append(perceptron.calculate_total_network_error(self.training_set))
            testing_error_list.append(perceptron.testing_network(self.testing_set))
            print('Для перцептрона с {0} скрытыми слоями:\nОшибка обучения: {1}\tОшибка обобщения{2}'.format(i, perceptron.calculate_total_network_error(self.training_set), perceptron.testing_network(self.testing_set)))

        plt.xlabel('Number of hidden layers')
        plt.ylabel('Error')
        plt.grid()
        plt.plot(learn_error_list, label='Learning error')
        plt.plot(testing_error_list, label='Generalization error')
        plt.legend()
        plt.show()'''

root = Tk()
root.geometry("870x430+100+100")
app = App(root)
root.mainloop()

