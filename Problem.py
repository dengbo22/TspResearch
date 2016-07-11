# -*- coding: utf-8 -*-
import re
import copy
import math
import matplotlib.pyplot as plt
from sys import float_info
import numpy as np

__author__ = 'tiny'

DEFAULT_FILE_PATH = "./dataset/selfdesign.tsp"
UNIFORMITY_WEIGHT = 1
PARALLEL_NUMBER = 5
AVERAGE_RATE = 0.5


def split_path(center_pos, path):
    size = len(path)
    start_pos = path.index(center_pos)
    cur_pos = (start_pos + 1) % size
    result_array = []
    cache_array = []
    while True:
        cur_data = path[cur_pos]
        cache_array.append(cur_data)
        if cur_data == center_pos:
            result_array.append(cache_array)
            cache_array = []
        if cur_pos == start_pos:
            break
        cur_pos = (cur_pos + 1) % size

    return result_array


def distance(x1, y1, x2, y2):
    sqr = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return sqr ** 0.5


class Problem(object):
    def __init__(self):
        pass


class TspProblem(Problem):
    def __init__(self, position=0):
        super().__init__()
        self.best_result = float_info.max
        self.best_result_path = []
        self.center_position = position
        self.parallel_number = PARALLEL_NUMBER
        self.points = []
        self.city_distance_array = []
        self.init_distance_matrix()
        self.city_size = len(self.city_distance_array)
        return

    def init_distance_matrix(self):
        file_path = input("请输入文件的绝对路径\n")
        if file_path.strip() == "":
            file_path = DEFAULT_FILE_PATH
        # 打开文件读取数据资料
        point_array = []
        file = open(file_path, 'r')
        for each_line in file:
            if re.match('\d+|0\s[1-9]\d*\.\d*|0\.\d*|[1-9]\d*\s[1-9]\d*\.\d*|0\.\d*|[1-9]\d*', each_line):
                each_line_array = each_line.strip().split()
                del each_line_array[0]
                for i in range(0, len(each_line_array)):
                    each_line_array[i] = float(each_line_array[i])
                point_array.append(each_line_array)
        file.close()

        self.points = point_array
        # 计算点阵之间的距离
        size = len(point_array)
        for i in range(0, size):
            single_distance = []
            for j in range(0, size):
                dis_cache = distance(point_array[i][0], point_array[i][1], point_array[j][0], point_array[j][1])
                single_distance.append(dis_cache)
            self.city_distance_array.append(single_distance)
        # 此时city_distance_array即为初始化完成后的城市两两距离
        return

    def get_city_distance(self, start, desti):
        return self.city_distance_array[start][desti]

    def split_path_length(self, path):
        split_array = split_path(self.center_position, path)
        cache_distance = []
        for each_split_array in split_array:
            cache = self.get_path_length(each_split_array)
            cache_distance.append(cache)

        return cache_distance

    def get_path_length(self, path):
        size = len(path)
        path_length = 0.0
        for i in range(size):
            m = path[i - 1]
            n = path[i]
            path_length += self.get_city_distance(m, n)
        return path_length

    def result_evaluation(self, path):
        split_distance = self.split_path_length(path)
        subtour_distance = np.array(split_distance)
        # 对split_distance做操作判断该解的优越性
        average = np.mean(subtour_distance)
        variance = np.var(subtour_distance)
        std = np.std(subtour_distance)

        result = average + UNIFORMITY_WEIGHT * variance
        return result

    def update_best_result(self, path):
        self.best_result_path = copy.copy(path)
        self.best_result = self.result_evaluation(path)

    def show_multi_result(self):
        fig = plt.figure()
        each_path = split_path(self.center_position, self.best_result_path)
        size = len(each_path)
        sqrt_size = math.sqrt(size)
        real_size = int(sqrt_size)
        if sqrt_size > real_size:
            sqrt_size += 1
            real_size += 1

        pos_x = []
        pos_y = []
        all_x = []
        all_y = []
        # Init All
        for item in self.points:
            all_x.append(item[0])
            all_y.append(item[1])

        # Draw Result
        i = 0
        for i in range(size):
            fig.add_subplot(sqrt_size, sqrt_size, i + 1)

            # Draw all the point
            plt.scatter(all_x, all_y)
            plt.scatter(all_x[self.center_position], all_y[self.center_position], color="red")
            # Draw Line

            for item in each_path[i]:
                pos_x.append(self.points[item][0])
                pos_y.append(self.points[item][1])
            plt.plot(pos_x, pos_y, linewidth=1.5)
            pos_x = []
            pos_y = []
        # Draw All
        if sqrt_size > real_size:
            fig.add_subplot(sqrt_size, sqrt_size, real_size ** 2)
            ary_x = []
            ary_y = []
            for item in self.best_result_path:
                ary_x.append(self.points[item][0])
                ary_y.append(self.points[item][1])
                plt.scatter(self.points[item][0], self.points[item][1])
            plt.plot(ary_x, ary_y, linewidth=1.5)

        plt.show()

    def show_result(self):
        ary_x = []
        ary_y = []
        for item in self.best_result_path:
            ary_x.append(self.points[item][0])
            ary_y.append(self.points[item][1])
            plt.scatter(self.points[item][0], self.points[item][1])

        plt.plot(ary_x, ary_y, linewidth=1.5)
        plt.show()
