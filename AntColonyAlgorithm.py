# -*- coding: utf-8 -*-
import random
import Problem
import numpy as np
from abc import ABCMeta, abstractmethod
from sys import float_info

__author__ = 'tiny'

ITERATION = 500

ROU = 0.9
ALPHA = 1.0
BETA = 2.0
P_BEST = 0.5
ENLARGE_Q = 1.0


def average_level_rate(x):
    result = 0.1 + 0.9 / (1 + np.exp(10 * x - 5))
    return result


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

    def __init__(self, mark=0):
        if Ant.city_size > 0 and Ant.center_position >= 0 and Ant.parallel_size > 0:
            self.tabu_table = [1 for x in range(Ant.city_size)]
            self.tabu_table[self.center_position] = Ant.parallel_size
            self.path = [-1 for i in range(Ant.city_size + Ant.parallel_size - 1)]
            self.current_city_index = -1
            self.moved_city_count = 0
            self.result = -1
            self.mark = mark  # 表示正常
        else:
            print('Ant类参数设置有误，请检查CitySize:%d, Center:%d, Parallel%d' % (
                Ant.city_size, Ant.center_position, Ant.parallel_size))

    def reset_ant(self):
        self.tabu_table = [1 for x in range(Ant.city_size)]
        self.tabu_table[self.center_position] = Ant.parallel_size
        self.path = [-1 for i in range(Ant.city_size + Ant.parallel_size - 1)]
        self.current_city_index = -1
        self.moved_city_count = 0
        self.result = -1
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
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def heuristic_function(self, start, dest):
        pass

    @abstractmethod
    def update_pheromone(self):
        pass


class MultiAntColonyAlgorithm(AntColonyAlgorithm):
    def __init__(self, center=0):
        super().__init__()
        self.problem = Problem.TspProblem(center)
        cities = self.problem.city_size
        # 初始化蚁群
        self.ant_size = int(cities * 2 / 3)
        Ant.init_ant_class(cities, self.problem.center_position, self.problem.parallel_number)
        self.search_ant_array = [Ant() for i in range(self.ant_size)]
        # 初始化信息素
        self.max_pheromone = 0.0
        self.min_pheromone = 0.0
        self.city_pheromone_array = np.ones((cities, cities))
        return

    # 可调校：设置蚂蚁选择下一个城市的函数机制
    def heuristic_function(self, start, dest):
        # 优化代码
        # if start > self.problem.city_size or start < 0 or dest > self.problem.city_size or start < 0:
        #     print("城市信息素计算异常：出现开始/目标城市下标超出范围")
        #     return

        if start == dest:
            return 0.0

        else:
            pheromone_percent = (self.city_pheromone_array[start][dest]) ** ALPHA
            distance_percent = (1.0 / self.problem.get_city_distance(start, dest)) ** BETA
            return pheromone_percent * distance_percent

    # 可调校：设置信息素的添加函数：
    def pheromone_add_function(self, ant):
        if ant.result < 0:
            return ENLARGE_Q / self.ant_judge(ant)
        else:
            return ENLARGE_Q / ant.result

    def do_search(self):

        self.first_loop_search()
        iterate_times = ITERATION

        while iterate_times:
            result_changed = self.loop_search()
            iterate_times -= 1
            # Simplify the output
            if result_changed or iterate_times % 50 == 0:
                print("第%d次迭代，最优解%s" % (ITERATION - iterate_times, self.problem.best_result))
                # print("最优解路径：", Problem.split_path(self.problem.center_position, self.problem.best_result_path))
                print("每段长度：", self.problem.split_path_length(self.problem.best_result_path))

        return

    def first_loop_search(self):
        self.loop_search()
        self.calc_max_min_pheromone()
        # 更新当前环境中的信息素为最大值
        current_pheromone = self.max_pheromone
        self.city_pheromone_array = np.ones((self.problem.city_size, self.problem.city_size))
        self.city_pheromone_array *= current_pheromone

        # 设置EnLargeQ的值
        # print("max_pheromone:", current_pheromone)
        # print("best_result", self.problem.best_result)

    def ant_judge(self, ant):
        if ant.mark == 1 or ant.path[0] == ant.path[-1]:
            return float_info.max
        else:
            return self.problem.result_evaluation(ant.path)

    def loop_search(self):
        pre_add = id(self.problem.best_result)
        problem = self.problem

        # 初始化每只蚂蚁
        for ant in self.search_ant_array:
            ant.reset_ant()
        # 进行搜索
        iterate_best = Ant(1)
        for ant_item in self.search_ant_array:
            # 每只蚂蚁的搜索
            ant_item.random_choose()
            while (not ant_item.is_finished()) and ant_item.mark == 0:
                selected_city = self.pheromone_select(ant_item)
                ant_item.move_to(selected_city)
            # 蚂蚁构造解完成，测试蚂蚁是否迭代最优
            ant_item.result = self.ant_judge(ant_item)
            if self.ant_judge(iterate_best) > ant_item.result:
                iterate_best = ant_item

        if iterate_best.result < problem.best_result:
            problem.update_best_result(iterate_best.path)

        # 更新信息素
        self.update_pheromone()
        self.add_pheromone_by_average(iterate_best)
        post_addr = id(self.problem.best_result)
        return pre_add != post_addr

    def calc_max_min_pheromone(self):
        question = self.problem
        self.max_pheromone = 1.0 / ((1 - ROU) * question.result_evaluation(question.best_result_path))
        factor = P_BEST ** (1.0 / question.city_size)
        avg = int(question.city_size / 2) - 1
        self.min_pheromone = (self.max_pheromone * (1 - factor)) / ((avg - 1) * factor)
        if self.max_pheromone > self.min_pheromone:
            # print("Max:%d, Min:%d" %(self.max_pheromone, self.min_pheromone))
            return True
        else:
            print("最大最小信息素计算异常，最大值%s\t最小值%s" % (self.max_pheromone, self.min_pheromone))
            return False

    def update_pheromone(self, special_ant=None):
        # Pheromone Evaporate
        self.city_pheromone_array *= ROU
        # Each Ant Update Pheromone
        for ant_item in self.search_ant_array:
            # self.add_pheromone_by_ant(ant_item)
            self.add_pheromone_by_average(ant_item)

        # Use special ant to do the extra pheromone update.
        if special_ant is not None:
            self.add_pheromone_by_average(special_ant)

        # Adjust the Max-Min pheromone
        self.check_max_min_pheromone()

        return

    def check_max_min_pheromone(self):
        city_count = self.problem.city_size
        if self.calc_max_min_pheromone():
            max_array = np.ones((city_count, city_count)) * self.max_pheromone
            min_array = np.ones((city_count, city_count)) * self.min_pheromone
            if_larger = self.city_pheromone_array > self.max_pheromone
            if_less = self.city_pheromone_array < self.min_pheromone
            self.city_pheromone_array = np.where(if_larger, max_array, self.city_pheromone_array)
            self.city_pheromone_array = np.where(if_less, min_array, self.city_pheromone_array)
            return True

        return False

    def add_pheromone_by_ant(self, ant):
        # Add new Pheromone
        for position in range(self.problem.city_size):
            m = ant.path[position - 1]
            n = ant.path[position]
            self.city_pheromone_array[m][n] += self.pheromone_add_function(ant)
            if self.city_pheromone_array[m][n] > self.max_pheromone:
                self.city_pheromone_array[m][n] = self.max_pheromone
            if self.city_pheromone_array[m][n] < self.min_pheromone:
                self.city_pheromone_array[m][n] = self.min_pheromone
            self.city_pheromone_array[n][m] = self.city_pheromone_array[m][n]
        return

    def add_pheromone_by_average(self, ant):
        subpaths = Problem.split_path(self.problem.center_position, ant.path)
        subpaths_length = []
        for i in range(len(subpaths)):
            subpaths_length.append(self.problem.get_path_length(subpaths[i]))
        subpaths_rate = np.array(subpaths_length)
        average = np.mean(subpaths_rate)
        subpaths_rate -= average
        divider = max(subpaths_rate)
        subpaths_rate /= divider
        # 每段路径根据Rate来进行更新

        base_pheromone = self.pheromone_add_function(ant)
        # base_pheromone = 1.0 / (self.problem.get_path_length(ant.path))

        for i in range(len(subpaths)):
            subpath = subpaths[i]
            subpath_size = len(subpath)
            add_pheromone = base_pheromone * average_level_rate(subpaths_rate[i])
            for j in range(subpath_size):
                m = subpath[j - 1]
                n = subpath[j]
                self.city_pheromone_array[m][n] += add_pheromone

    def update_pheromone_by_average(self, ant):
        each_length = self.problem.split_path_length(ant.path)
        size = len(each_length)
        aver = sum(each_length) / size
        for i in range(size):
            each_length[i] = abs(each_length[i] - aver)
        index = list(range(size))
        index.sort(key=lambda k: each_length[k])
        # 下面进入信息素更新部分
        current_rate = MAX_RATE
        step = (current_rate - MIN_RATE) / size

        splited_path = Problem.split_path(self.problem.center_position, ant.path)
        for i in range(size):
            increase_path = splited_path[index[i]]
            for j in range(len(increase_path)):
                m = increase_path[j - 1]
                n = increase_path[j]
                self.city_pheromone_array[m][n] += self.problem.result_evaluation(ant.path) * current_rate
            current_rate -= step

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
    ACO = MultiAntColonyAlgorithm()
    ACO.do_search()
    ACO.problem.show_multi_result()
