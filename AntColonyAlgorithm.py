# -*- coding: utf-8 -*-
import random
import Problem
import numpy as np
from abc import abstractmethod
from sys import float_info
from copy import copy

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

    def choose_randomly(self):
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
        self.city_pheromone_array = np.ones((cities, cities))

    def loop_search(self):
        """
        :return: return whether the result has updated,
                 true for updated, False for no updated.
        """
        # Get current address of result.
        pre_addr = id(self.problem.best_result);
        # Search the iteration best ant.
        iterate_best_path = self.ant_search();
        # Compare the iteration best ant with global best ant.
        if self.path_evaluation(iterate_best_path) < self.problem.best_result:
            self.problem.update_best_result(iterate_best_path);

        # Update Pheromone.
        self.update_pheromone();

        post_addr = id(self.problem.best_result);
        return pre_addr != post_addr;

    def do_search(self):

        self.first_loop_search();
        iterate_times = ITERATION;

        while iterate_times:
            result_changed = self.loop_search();
            iterate_times -= 1;
            # Simplify the output
            if result_changed or iterate_times % int(ITERATION / 10) == 0:
                print("第%d次迭代，%s = %s" %
                      (ITERATION - iterate_times, self.problem.split_path_length(self.problem.best_result_path),
                       self.problem.best_result));
                # print("最优解路径：", Problem.split_path(self.problem.center_position, self.problem.best_result_path))

        return

    @abstractmethod
    def first_loop_search(self):
        pass

    @abstractmethod
    def path_evaluation(self, path):
        pass

    @abstractmethod
    def ant_search(self):
        """
        :return: return the iterator best ant's path.
        """
        pass

    @abstractmethod
    def update_pheromone(self):
        pass

    @abstractmethod
    def heuristic_function(self, start, end):
        pass

    # Tool function
    def choose_by_heuristic(self, current_city, tabu_table):
        probability_array = np.zeros(self.problem.city_size);

        for i in range(len(tabu_table)):
            if tabu_table[i] >= 0:
                pro = tabu_table[i] * self.heuristic_function(current_city, i);
                probability_array[i] = pro;
            else:
                raise ValueError("禁忌表中第%s项值小于零,其值为%s" % (i, tabu_table[i]))

        probability_total = sum(probability_array)
        # 此处跳过所有城市都不可选的情况
        if probability_total > 0.0:
            probability_array /= probability_total;
            temp = random.random()
            for j in range(self.problem.city_size):
                temp -= probability_array[j]
                if temp < 0.0:
                    return j

        # 如果之前的值未能够很好的选择出来，则进入扫描状态,经测试，扫描算法一般选择的是中心城市
        for k in range(self.problem.city_size):
            if tabu_table[k] >= 1:
                return k

    def check_pheromone(self, multiple):
        less = self.city_pheromone_array < (self.min_pheromone * multiple);
        true_count = len(self.city_pheromone_array[less])
        print("MultiPle:%s, Percent: %s" % (multiple, true_count / (self.problem.city_size * self.problem.city_size)));
        return true_count;

    def get_ant_pass_count(self):
        pass_array = np.ones(self.city_pheromone_array.shape)
        for ant in self.search_ant_array:
            for i in range(len(ant.path)):
                m = ant.path[i - 1];
                n = ant.path[i];
                pass_array[m][n] -= 1;
                pass_array[n][m] = pass_array[m][n];

        changed = pass_array < 1;
        return len(pass_array[changed]);


class MultiAntColonyAlgorithm(AntColonyAlgorithm):
    def __init__(self, center=0):
        super().__init__()
        return

    # Implement
    def first_loop_search(self):
        self.loop_search()
        self.calc_max_min_pheromone()
        # 更新当前环境中的信息素为最大值
        current_pheromone = self.max_pheromone
        self.city_pheromone_array = np.ones((self.problem.city_size, self.problem.city_size))
        self.city_pheromone_array *= current_pheromone

    def calc_max_min_pheromone(self):
        question = self.problem
        self.max_pheromone = 1.0 / ((1 - ROU) * question.result_evaluation(question.best_result_path))
        factor = P_BEST ** (1.0 / question.city_size)
        avg = int(question.city_size / 2) - 1
        self.min_pheromone = (self.max_pheromone * (1 - factor)) / ((avg - 1) * factor)
        if self.max_pheromone > self.min_pheromone:
            return True
        else:
            print("最大最小信息素计算异常，最大值%s\t最小值%s" % (self.max_pheromone, self.min_pheromone))
            return False

    # Implement
    def ant_search(self):
        # 初始化每只蚂蚁
        for ant in self.search_ant_array:
            ant.reset_ant();
        # 进行搜索
        best_path = [];
        for ant_item in self.search_ant_array:
            # 每只蚂蚁的搜索
            ant_item.choose_randomly();
            while (not ant_item.is_finished()) and ant_item.mark == 0:
                selected_city = self.choose_by_heuristic(ant_item.current_city_index, ant_item.tabu_table);
                ant_item.move_to(selected_city);

            # 蚂蚁构造解完成，测试蚂蚁是否迭代最优
            ant_item.result = self.path_evaluation(ant_item.path);
            if self.path_evaluation(best_path) > ant_item.result:
                best_path = copy(ant_item.path);
        return best_path;

    # Implement
    def update_pheromone(self, special_ant=None):
        # Pheromone Evaporate
        self.city_pheromone_array *= ROU
        # Each Ant Update Pheromone
        for ant_item in self.search_ant_array:
            self.add_pheromone_by_ant(ant_item)

        # Use special ant to do the extra pheromone update.
        if special_ant is not None:
            self.add_pheromone_by_ant(special_ant)

        # Adjust the Max-Min pheromone
        self.check_max_min_pheromone()
        return

    def add_pheromone_by_ant(self, ant):
        subpaths = Problem.split_path(self.problem.center_position, ant.path)
        subpaths_length = []
        for i in range(len(subpaths)):
            subpaths_length.append(self.problem.get_path_length(subpaths[i]))
        subpaths_rate = np.array(subpaths_length)
        average = np.mean(subpaths_rate)
        subpaths_rate -= average
        divider = max(subpaths_rate)
        if divider != 0:
            subpaths_rate /= divider
        else:
            print("死亡蚂蚁，不添加信息素...");
            return;
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
                self.city_pheromone_array[m][n] += add_pheromone;
                self.city_pheromone_array[n][m] = self.city_pheromone_array[m][n];
        return

    # 可调校：设置信息素的添加函数：
    def pheromone_add_function(self, ant):
        if ant.result < 0:
            return ENLARGE_Q / self.path_evaluation(ant.path)
        else:
            return ENLARGE_Q / ant.result

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

    # Implement
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

    # Implement
    def path_evaluation(self, path):
        if (not path) or (-1 in path):
            return float_info.max;
        else:
            return self.problem.result_evaluation(path)

    # Ready to deprecated
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
