from multiprocessing import Pool
from functools import lru_cache, partial
from itertools import combinations
from sko.GA import GA
# from regex import subf
from fact import FactSet
from typing import Tuple, List
from sko.tools import set_run_mode
from numba import jit
from tqdm import tqdm
import numpy as np
import random
import abc
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4-0613")


@lru_cache(maxsize=1280)
def binary_combinations(n):
    # 二进制解空间
    reu = []
    for i in range(2**n):
        binary_string = bin(i)[2:].zfill(n)
        combination = [int(bit) for bit in binary_string]
        reu.append(combination)
    return reu


@jit(nopython=True)
def cac_entropy(prior_p, ans_p_post_o, ans_p):
    cur_h = 0
    o_p_post_ans = prior_p * ans_p_post_o / ans_p
        # 质量增益 论文中的 ΔQ(F|T)
    for i2 in o_p_post_ans:
        cur_h -= (i2 * np.log(i2)).item()  # H(o|AS T CE)
    return cur_h


def expectation_cond(facts: FactSet, selection, worker_accuracy) -> float:
    """
    可变变量 selection, worker_accuracy
    计算query of correspondences set 的 
    return h(V|T) 
    """
    length = len(selection)
    # space 生成过程
    combinations = binary_combinations(length)
    prior_p = facts.get_prior_p()  # 初始化prior_p
    # 排序 后去除 顺序
    selection.sort()

    set_expected_h = 0
    for ans in combinations:
        # 传进去sub_facts[i]，相当于一个 o --- (当CE答案为sub_facts[i]时，获得P(ATCE) 和 P(ATCE|o))
        ans_p, ans_p_post_o = facts.compute_ans_p(ans,
                                                list(selection),
                                                worker_accuracy)
        # h -= p_ans * np.log(p_ans)
        # 获得P(o|ATCE) = P(o) * P(ATCE|o) / P(ATCE)
        assert ans_p!=0, "ans_p = 0 cause exception"
        cur_h = cac_entropy(prior_p, ans_p_post_o, ans_p)
        set_expected_h += ans_p*cur_h
    return set_expected_h


def get_values(source_pd, target_pd, correspondence_count, c_name):
    """return the specific values of correspondence at this round"""
    num = correspondence_count[tuple(c_name)]
    start = num*3
    end = 3+start
    values_1 = source_pd[c_name[0]].values.tolist()[start:end]
    values_2 = target_pd[c_name[1]].values.tolist()[start:end]
    return values_1,values_2
    

def cost_function(c_name, values_1, values_2):
    "return the cost of correspondence "
    tokens_num = 0
    tokens_num+=len(encoding.encode(c_name[0]+c_name[1]))
    for value_s in values_1:
        tokens_num += len(encoding.encode(str(value_s)))
    for value_s in values_2:
        tokens_num += len(encoding.encode(str(value_s)))
    return tokens_num
    
class QuerySelector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select(self, facts: FactSet,
               budget: int,
               worker_accuracy: np.array) -> Tuple[np.ndarray, "FactSet", float]:
        """
        根据该类的策略，固定budget 下 选择 correspondence
        :param facts: 需要被选择作为问题的事实集
        :param budget: 每轮的成本
        :param worker_accuracy: 工人的回答准确率
        :return: 返回facts的一个子集, 包括其相对于原来的索引
        """
        raise NotImplemented("implement me")


class BaseQuerySelector(QuerySelector):
    """
    暴力法的问题选择器
    """
    def select(self, facts: FactSet, budget: int, worker_accuracy: np.ndarray) -> Tuple[np.ndarray, "FactSet", float]:
        """budget each round 成本 cost """
        cost_list = facts.len_list()
        cost_list.sort()
        sum_cost = 0
        max_num = 0
        least_num = 0
        low_num = 0
        print("start brute")
        for i in cost_list[::-1]:
            if least_num + i <= budget:
                least_num += i
                low_num += 1
        # limit of num for brute
        for i in cost_list:
            if sum_cost+i<=budget:
                sum_cost+=i
                max_num+=1
            else:
                break
        assert low_num <= max_num, f"{low_num}, low, {max_num} max"
        num_fact: int = facts.num_fact()
        
        max_selection = []
        max_h = float('-inf')
        print("brute low_num:", low_num, "max_num:", max_num)
        for num in range(low_num, max_num+1):
            print("num:", num, "num_fact:", num_fact)
            selections = [i for i in combinations(range(num_fact), num)]
            
            for selection in tqdm(selections):
                selection_cost = 0.
                # 求和
                for ix in selection:
                    selection_cost+=cost_list[ix]
                    
                # 若超过预算 跳过
                if selection_cost > budget:
                    continue
                set_expected_h = expectation_cond(facts, list(selection), worker_accuracy)
                # print("selection:", selection,"set_expeted_h:", set_expected_h)
                
                if -set_expected_h > max_h:
                    max_h = -set_expected_h
                    max_selection = selection
        return np.array(max_selection), -max_h


def select_unit(three_index, budget_round, facts, worker_accuracy, c_index_list):
        T2 = []
        T2_h = -float('inf')
        three_cost = 0
        num_fact = facts.num_fact()
        cost_list = facts.len_list() 
        for c_index in list(three_index):
            three_cost+= cost_list[c_index]
            
        if budget_round - three_cost < 0:
            return T2, T2_h
        else:
            budget_round -= three_cost # update new budget
            candidates_list = [i for i in c_index_list if i not in three_index] #
            max_selection = list(three_index)
            max_hsum = -expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
            
            # greedy strategy starts 
        while budget_round>0:  # 近似找到fact最优组合
            max_h_gain = 0.  # 质量增益最低也得为0
            max_idx = -1
                # 原始
            h = expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)

            for idx in candidates_list:
                if 2 == 1:
                    w = 1.0
                else:
                    w= cost_list[idx]
                    
                if budget_round-w<0:
                    continue
                max_selection.append(idx) # meet budget add it
                cur_h =  expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
                h_gain = (h-cur_h)/w  #
                if h_gain >= (max_h_gain-0.000001):
                    max_h_gain = h_gain
                    if idx>num_fact:
                        raise
                    max_idx = idx   # 每次找出最大的idx在循环外部append
                assert max_h_gain >=-0.00000001, f"wrong selection, idx {max_idx}, {h} and {cur_h}"
                    # 删除 增加的 元素
                max_selection.remove(idx)  # 每次删除的位置一定改用pop()会更快
                
            if max_idx == -1:
                    # 无法再加入任何correspondence
                break
            max_hsum += max_h_gain
            assert max_idx < num_fact, f"exceed num_fact {num_fact},{max_idx}"
            w = cost_list[max_idx]
            
            budget_round -= w
                
            if budget_round >= 0:
                max_selection.append(max_idx)
            candidates_list.remove(max_idx)
            if len(max_selection)==num_fact:
                budget_round =-1
        return_h = expectation_cond(facts=facts, selection=max_selection, worker_accuracy=worker_accuracy)
        if -return_h > T2_h:
            T2 = max_selection
            T2_h = -return_h
        return T2, T2_h

# 贪心法的问题选择器
class GreedyQuerySelector(QuerySelector):  # 改6
    """
    贪心法的问题选择器
    """
    def select(self, facts: FactSet,
               budget: int,
               worker_accuracy: np.ndarray,
               process_num=12,
               c_index_list=[]
               ) -> Tuple[np.ndarray, "FactSet", float]:
        T1 = []
        T2 = []
        max_selection: list = []
        max_hsum = -100000000.
        # 穷举 计算所有2个 correspondence 最大化 H
        ######################
        for two_index in combinations(c_index_list, 2):
            two_cur_h = expectation_cond(facts=facts, selection=list(two_index), worker_accuracy=worker_accuracy)
            if -two_cur_h > max_hsum:
                max_selection = list(two_index)
                max_hsum = -two_cur_h
        for one_index in c_index_list:
            one_cur_h = expectation_cond(facts=facts, selection=[one_index], worker_accuracy=worker_accuracy)
            if -one_cur_h > max_hsum:
                max_selection = [one_index]
                max_hsum = -one_cur_h
        T1 = max_selection 
        T1_h = max_hsum
        
        T2 = ""
        T2_h = -float('inf')
        
        max_selection = []
        three_groups = [three_index for three_index in combinations(c_index_list, 3)]
        select_new = partial(select_unit, budget_round=budget, facts=facts, worker_accuracy=worker_accuracy, c_index_list=c_index_list)
        pool = Pool(process_num)
        return_result = list(tqdm(pool.imap(select_new, three_groups), total=len(three_groups)))
        pool.close()
        pool.join()
        if return_result:
            max_element = max(return_result, key=lambda r:r[1])
            T2, T2_h = max_element
        
        if T1_h >= T2_h:
            print("two_c win:", T2, "num:", T2_h)
            return np.array(T1), T1_h
        else:
            return np.array(T2), T2_h


class RandomQuerySelector(QuerySelector):  #2.6
    """
    随机法的问题选择器
    """
    def select(self, facts,
               budget,
               worker_accuracy,
               c_index_list=[]
               ) -> Tuple[np.ndarray, "FactSet", float]:
        """
        每轮固定budget 的选择
        """
        cost_list = facts.len_list()
        selection = []
        candidate_list = c_index_list
        while budget>0 and candidate_list!=[]:
            tmp_ix = random.sample(candidate_list,1)
            candidate_list.remove(tmp_ix[0])
            cost_tokens =cost_list[tmp_ix[0]]
            if budget - cost_tokens:
                selection.extend(tmp_ix)
            budget -= cost_tokens
            
        # selection = np.random.choice(num_fact,num,replace=False)
        # sub_facts = facts.get_subset(list(selection))
        h = expectation_cond(facts, selection, worker_accuracy)
        return selection, -h
    
    
class HeuristicQuerySelector(QuerySelector):
    """
    启发式算法问题选择器
    """
    def select(self, facts:FactSet, budget: int, worker_accuracy: np.ndarray, max_iters:int, cost_func:int) -> Tuple[np.ndarray, float]:
        num = facts.num_fact()
        if cost_func==1:
            cost = np.array([1 for _ in range(num)])
        else:
            cost = np.array(facts.len_list())
        
        def func(x):
            k = np.array(x)
            selection = list(np.where(k==1)[0])
            return expectation_cond(facts, selection, worker_accuracy)

        constraint_ueq = [lambda x: np.sum(cost*x)-budget]
        lb = [0 for i in range(num)]
        ub = [1 for i in range(num)]
        precision = [1 for _ in range(num)]
        set_run_mode(func, 'cached')
        ga = GA(func=func, n_dim=num, constraint_ueq= constraint_ueq ,size_pop=50, max_iter=max_iters, prob_mut=0.001, lb=lb, ub=ub, precision=precision)
        best_x, best_y = ga.run()
        k = np.array(best_x)
        selection = np.where(k==1)[0]
        return  list(selection), -best_y