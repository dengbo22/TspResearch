# -*- coding: utf-8 -*-
import random
import Problem
from sys import float_info

__author__ = 'tiny'

ITERATION = 100

ROU = 0.5
ALPHA = 0
BETA = 2
P_BEST = 0.5
ENLARGE_Q = 1000.0


class Ant(object):
    """
    tabu_table表示禁忌表，是蚂蚁下一步可以移动城市的表示数组
    current_city_index表示蚂蚁当前所在城市的下标，取值为0 ～ CITY_COUNT-1
    moved_city_count表示蚂蚁已经移动过的城市计数，也可以认为是需要移动的路径下标，取值为1 ～ CITY_COUNT
    path表示蚂蚁的移动路径
    """

    city_size = -1
    center_position = -1
    parallel_size = -1

    def init_ant_class(city_count, center, parallel):
        if city_count > 0 and center >= 0:
            Ant.city_size = city_count
            Ant.center_position = center
            Ant.parallel_size = parallel
        return

    def __init__(self):
        if Ant.city_size > 0 and Ant.center_position >= 0 and Ant.parallel_size > 0:
            self.tabu_table = [1 for x in range(Ant.city_size)]
            self.tabu_table[self.center_position] = Ant.parallel_size
            self.path = [-1 for i in range(Ant.city_size + Ant.parallel_size - 1)]
            self.current_city_index = -1
            self.moved_city_count = 0
            self.mark = 0  # 表示正常
        else:
            print('Ant类参数设置有误，请检查CitySize:%d, Center:%d, Parallel%d' % (
                Ant.city_size, Ant.center_position, Ant.parallel_size))

    def reset_ant(self):
        self.tabu_table = [1 for x in range(Ant.city_size)]
        self.tabu_table[self.center_position] = Ant.parallel_size
        self.path = [-1 for i in range(Ant.city_size + Ant.parallel_size - 1)]
        self.current_city_index = -1
        self.moved_city_count = 0
        self.mark = 0

    def move_to(self, position):

        if position == -1:
            print("选择目的城市错误，异常退出")
            return False

        if self.tabu_table[position] == 0 or self.current_city_index == position:
            # print("蚂蚁不慎移动，宣告死亡")
            self.mark = 1
            return False

        # 选择城市符合要求，进行移动
        self.path[self.moved_city_count] = position
        self.tabu_table[position] -= 1
        self.current_city_index = position
        self.moved_city_count += 1
        return True

    def random_choose(self):
        selected = random.randint(0, Ant.city_size - 1)
        self.move_to(selected)

    def is_finished(self):
        return self.moved_city_count == Ant.city_size + Ant.parallel_size - 1


class AntColonyAlgorithm(object):
    def __init__(self, center=0):
        self.problem = Problem.TspProblem(center)
        cities = self.problem.city_size
        # 初始化蚁群
        self.ant_size = int(cities * 2 / 3)
        Ant.init_ant_class(cities, self.problem.center_position, self.problem.parallel_number)
        self.search_ant_array = [Ant() for i in range(self.ant_size)]
        # 初始化信息素
        self.max_pheromone = 0.0
        self.min_pheromone = 0.0
        self.city_pheromone_array = [[1 for i in range(cities)] for j in range(cities)]
        return

    # 可调校：设置蚂蚁选择下一个城市的函数机制
    def heuristic_function(self, start, dest):
        if start > self.problem.city_size or start < 0 or dest > self.problem.city_size or start < 0:
            print("城市信息素计算异常：出现开始/目标城市下标超出范围")
            return

        if start == dest:
            return 0.0

        else:
            pheromone_percent = self.city_pheromone_array[start][dest] ** ALPHA
            distance_percent = (1.0 / self.problem.get_city_distance(start, dest)) ** BETA
            return pheromone_percent * distance_percent

    # 可调校：设置信息素的添加函数：
    def pheromone_add_function(self, ant):
        return ENLARGE_Q / self.ant_judge(ant)

    def do_search(self):
        self.first_loop_search()
        iterate_times = ITERATION

        while iterate_times:
            self.loop_search()
            iterate_times -= 1
            print("第%d次迭代，最优解%s" % (ITERATION - iterate_times, self.problem.best_result))
            print("最优解路径：", Problem.split_path(self.problem.center_position, self.problem.best_result_path))

        return

    def first_loop_search(self):
        self.loop_search()
        self.calc_max_min_pheromone()
        # 更新当前环境中的信息素为最大值
        current_pheromone = self.max_pheromone
        self.city_pheromone_array = [[current_pheromone for i in range(self.problem.city_size)]
                                     for j in range(self.problem.city_size)]

        # 设置EnLargeQ的值
        # print("max_pheromone:", current_pheromone)
        # print("best_result", self.problem.best_result)

    def ant_judge(self, ant):
        if ant.mark == 1 or ant.path[0] == ant.path[-1]:
            return float_info.max
        else:
            return self.problem.result_evaluation(ant.path)

    def loop_search(self):
        # 初始化每只蚂蚁
        for ant in self.search_ant_array:
            ant.reset_ant()
        # 进行搜索
        for ant_item in self.search_ant_array:
            # 每只蚂蚁的搜索
            ant_item.random_choose()
            while (not ant_item.is_finished()) and ant_item.mark == 0:
                selected_city = self.pheromone_select(ant_item)
                ant_item.move_to(selected_city)

            # 每只蚂蚁搜索获得自己的解，尝试改动当前最优解
            problem = self.problem
            if self.ant_judge(ant_item) < problem.best_result:
                problem.update_best_result(ant_item.path)

        # 计算允许的信息素最大最小值
        self.calc_max_min_pheromone()
        # 更新信息素
        self.update_pheromone()

    def calc_max_min_pheromone(self):
        question = self.problem
        self.max_pheromone = 1.0 / (1 - ROU) * question.result_evaluation(question.best_result_path)
        factor = P_BEST ** (1.0 / question.city_size)
        avg = int(question.city_size / 2) - 1
        self.min_pheromone = (self.max_pheromone * (1 - factor)) / ((avg - 1) * factor)
        if self.max_pheromone > self.min_pheromone:
            # print("Max:%d, Min:%d" %(self.max_pheromone, self.min_pheromone))
            return True
        else:
            print("最大最小信息素计算异常，最大值%s\t最小值%s" % (self.max_pheromone, self.min_pheromone))
            return False

    def update_pheromone(self):
        city_count = self.problem.city_size
        temp = [[0 for i in range(city_count)]
                for j in range(city_count)]
        if self.calc_max_min_pheromone():
            # 矩阵运算，可优化
            for ant_item in self.search_ant_array:
                for position in range(city_count):
                    m = ant_item.path[position]
                    n = ant_item.path[(position + 1) % city_count]
                    temp[m][n] += self.pheromone_add_function(ant_item)
                    temp[n][m] = temp[m][n]
            # 矩阵运算，可优化
            for pos_x in range(0, city_count):
                for pos_y in range(0, city_count):
                    self.city_pheromone_array[pos_x][pos_y] *= ROU
                    self.city_pheromone_array[pos_x][pos_y] += temp[pos_x][pos_y]
                    if self.city_pheromone_array[pos_x][pos_y] > self.max_pheromone:
                        self.city_pheromone_array[pos_x][pos_y] = self.max_pheromone
                    if self.city_pheromone_array[pos_x][pos_y] < self.min_pheromone:
                        self.city_pheromone_array[pos_x][pos_y] = self.min_pheromone

    def pheromone_select(self, ant):
        probability_array = []
        for i in range(len(ant.tabu_table)):
            if ant.tabu_table[i] >= 0:
                pro = ant.tabu_table[i] * self.heuristic_function(ant.current_city_index, i)
                probability_array.append(pro)
            else:
                raise ValueError("禁忌表中第%s项值小于零,其值为%s" % (i, ant.tabu_table[i]))

        probability_total = sum(probability_array)

        if probability_total > 0.0:
            temp = random.random()
            temp *= probability_total
            for j in range(self.problem.city_size):
                temp -= probability_array[j]
                if temp < 0.0:
                    return j

        # 如果之前的值未能够很好的选择出来，则进入扫描状态,经测试，扫描算法一般选择的是中心城市
        for k in range(self.problem.city_size):
            if ant.tabu_table[k] >= 1:
                return k


if __name__ == '__main__':
    ACO = AntColonyAlgorithm()
    ACO.do_search()
    ACO.problem.show_result()
