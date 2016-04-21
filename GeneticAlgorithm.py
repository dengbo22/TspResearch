# -*- coding: utf-8 -*-

import Problem
import random

__author__ = 'tiny'

POPULATION_SIZE = 2000
ITERATION = 100


# 在结果无效的情况下，随机变异确保结果有效
def valuable_result(path):
    path_size = len(path)
    for i in range(path_size):
        if path[i] == path[i - 1]:
            while True:
                swap = random.randint(0, path_size - 1)
                if path[(swap - 1) % path_size] != path[i] and path[(swap + 1) % path_size] != path[i]:
                    if path[swap] != path[i]:
                        path[i], path[swap] = path[swap], path[i]
                        break

    return path


def roulette_choice(selection_array):
    array_sum = sum(selection_array)
    selection = []
    for item in selection_array:
        selection.append(item / array_sum)
    random_number = random.random()
    result = 0
    for item in selection:
        random_number -= item

        if random_number <= 0:
            break
        result += 1

    return result


class Individuals(object):
    def __init__(self, path, result_value):
        self.gene = path
        self.mutation_times = 0
        self.generation = 0
        self.result = result_value

    def mutation(self):
        size = len(self.gene)
        pos1 = random.randint(0, size - 1)
        pos2 = random.randint(0, size - 1)
        if pos1 != pos2:
            self.gene[pos1], self.gene[pos2] = self.gene[pos2], self.gene[pos1]
        # 结果合理化,确保结果合理
        valuable_result(self.gene)
        self.mutation_times += 1

    def split_path(self, center):
        return Problem.split_path(center, self.gene)

    def alive_rate(self):
        return 1.0 / self.result


class GeneticAlgorithm(object):
    def __init__(self, center=0):
        self.problem = Problem.TspProblem(center)
        # 创建种群
        self.population_size = POPULATION_SIZE
        self.population = []
        self.choice_rate = 0
        for i in range(0, self.population_size):
            gene = valuable_result(self.create_result())
            life = Individuals(gene, self.problem.result_evaluation(gene))
            self.population.append(life)
            self.choice_rate += life.alive_rate()

    def create_result(self):
        result_array = [i for i in range(0, self.problem.city_size)]

        center_pos = self.problem.center_position
        parallel = self.problem.parallel_number

        result_array.extend([center_pos for i in range(0, parallel - 1)])
        random.shuffle(result_array)

        # 生成结果后,为了确保结果不包含相邻的连续城市，故需要检查
        valuable_result(result_array)

        return result_array

    # 使用轮转选择算法淘汰个体
    def selection_roulette(self):
        current_size = len(self.population)
        remove_count = len(self.population) - self.population_size
        while remove_count:
            selected = random.uniform(0, self.choice_rate)
            for i in range(current_size):
                rate = self.population[i].alive_rate()
                # print("selected:%s, rate:%s" %(selected, rate))
                selected -= rate
                if selected <= 0:
                    self.choice_rate -= rate
                    del self.population[i]
                    remove_count -= 1
                    break

        return

    # 使用排序选择算法淘汰个体
    def selection_sort(self):
        rate = 0.0
        result = sorted(self.population, key=lambda gen: gen.result)
        gen = result[0:self.population_size]
        for item in gen:
            rate += item.alive_rate()

        self.population = gen
        self.choice_rate = rate

        return

    # 使用锦标赛的方法淘汰个体
    def selection_championship(self):
        pass

    def do_search(self):
        # first_time()
        for i in range(ITERATION):
            # 产生下一代个体，并且将个体加入population中
            self.add_next_generation()
            # 淘汰产生的个体到population_size的数量
            self.selection_sort()
            print("第%d代生成:" % i)
            print("最优解：%s, 路径：%s" % (self.population[0].result, Problem.split_path(self.problem.center_position,self.population[0].gene)))
        self.problem.update_best_result(self.population[0].gene)

    def crossover(self, main_individuals, sub_individuals):
        child_gene = []
        center_pos = self.problem.center_position
        split_main = main_individuals.split_path(center_pos)
        split_sub = sub_individuals.split_path(center_pos)
        index1 = random.randint(0, len(split_main) - 1)
        index2 = random.randint(0, len(split_sub) - 1)

        # 剔除基因片段中的重复元素
        swap_gene = split_sub[index2]
        swap_pos1 = len(swap_gene) - 1
        for item in split_main[index1]:
            if item not in swap_gene:
                swap_gene.append(item)
        swap_pos2 = len(swap_gene) - 1
        swap_gene[swap_pos1], swap_gene[swap_pos2] = swap_gene[swap_pos2], swap_gene[swap_pos1]
        split_main[index1] = swap_gene

        # 扫描去全局重复元素，修改目标Gene片段并构造子类Gene序列
        for each_path in split_main:
            if each_path == swap_gene:
                continue
            for each_item in each_path:
                child_gene.append(each_item)
                if each_item in swap_gene:
                    swap_gene.remove(each_item)
        child_gene.extend(swap_gene)
        child_gene.append(center_pos)

        valuable_result(child_gene)
        value = self.problem.result_evaluation(child_gene)
        result_child = Individuals(child_gene, value)
        result_child.generation = max(main_individuals.generation, sub_individuals.generation) + 1
        return result_child

    def create_next_generation(self):
        next_generation = []
        for i in range(self.population_size):
            main = random.randint(0, self.population_size - 1)
            sub = random.randint(0, self.population_size - 1)
            child_item = self.crossover(self.population[main], self.population[sub])
            next_generation.append(child_item)

        return next_generation

    def add_next_generation(self):
        next_gener = []
        next_alive = 0.0
        for i in range(self.population_size):
            main = random.randint(0, self.population_size - 1)
            sub = random.randint(0, self.population_size - 1)
            child_item = self.crossover(self.population[main], self.population[sub])
            next_gener.append(child_item)
            next_alive += child_item.alive_rate()
        # 添加个体，更新choice_rate值
        self.population.extend(next_gener)
        self.choice_rate += next_alive


if __name__ == '__main__':
    GA = GeneticAlgorithm()
    GA.do_search()
    GA.problem.show_multi_result()

