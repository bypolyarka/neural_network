#! /usr/bin/env python
# -*- coding: utf-8 -*-

import PIL.Image


def list_to_matrix(lst, img_size):
    list2 = []
    for i in range(0, len(lst), img_size[0]):
        list2.append(lst[i:i + img_size[0]])
    return (list2)


def save_img(img):
    img = img
    original_size = img.size
    img = img.convert('1')

    pixels = list(img.getdata())

    # список пикселей картинки переобразовываем в матрицу пикселей
    matr_pix = list_to_matrix(pixels, img.size)
    # print('Width img: {0}\nHeight img: {1}'.format(len(matr_pix[0]), len(matr_pix)))

    min_i = min_j = len(matr_pix[0])
    max_i = max_j = 0

    # ищем минимальный и максимальный по столбцам индексы пикселей черного цвета
    for i in range(img.size[1]):
        pix_row = []
        tmp = False
        for j in range(img.size[0]):
            if not matr_pix[i][j]:
                pix_row.append(j)
                tmp = True
        if (tmp and (min(pix_row) < min_j)): min_j = min(pix_row)
        if (tmp and (max(pix_row) > max_j)): max_j = max(pix_row)

    # ищем миним и максимальный по строкам индексы пикселей черного цвета
    for j in range(img.size[0]):
        pix_column = []
        tmp = False
        for i in range(img.size[1]):
            if not matr_pix[i][j]:
                pix_column.append(i)
                tmp = True
        if (tmp and (min(pix_column) < min_i)): min_i = min(pix_column)
        if (tmp and (max(pix_column) > max_i)): max_i = max(pix_column)
    # print('Min j: {0}\nMax j: {1}\nMin i: {2}\nMax i: {3}'.format(min_j, max_j, min_i, max_i))


    img_croped = img.crop((min_j, min_i, max_j, max_i))  # вырезаем нашу букву
    img1 = img_croped.resize(original_size, PIL.Image.ANTIALIAS)  # подгоняем к исходному размеру 
    img1.save('test.png')
