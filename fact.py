from typing import overload, List, Tuple, Union

import numpy as np
from collections.abc import Sequence


class FactSet(Sequence):

    def __getitem__(self, i: int):
        return self.facts[i]

    def __len__(self) -> int:
        return len(self.facts)

    def __init__(self, facts: np.ndarray,
                 prior_p: np.ndarray,
                 ground_true: int,
                 fact_space: List[List[int]] = None, 
                 len_list:List[int]=None):
        """
        创建输出集
        :param fact_space: 描述所有fact的取值
        :param facts: 对应的事实输出集
        :param prior_p: 每一个事实对应的先验概率
        :param ground_true: 真值对应的facts索引
        Ground Truth： 就是指正确打标签的训练数据 或 简单来说就是有效的正确的数据
        """
        assert facts.ndim == 2, "data shape should be 2 dim"
        assert facts.dtype == int
        assert facts.shape[0] == prior_p.shape[0]
        assert prior_p.dtype == float
         
        self.facts = facts  # 描述所有fact的可能状态(通常是所有的组合)
        self._prior_p = prior_p  # 先验概率
        self._post_p = np.zeros_like(prior_p)
        self._num_fact = facts.shape[1]
        self._ground_true = ground_true
        self._len_list = len_list
        if fact_space is not None:
            self._fact_space = fact_space
        else:
            self._fact_space: List[List[int]] = []
            for i in range(self._num_fact):
                fact_val: List[int] = np.unique(  # type:ignore
                    self.facts[:, i]).tolist()
                self._fact_space.append(fact_val)

    def num_fact(self) -> int:
        return self._num_fact

    def len_list(self)->List[int]:
        return self._len_list 
        
    def post_p(self) -> np.ndarray:
        """
        后验概率
        :return:
        """
        return self._post_p

    def get_prior_p(self) -> np.ndarray:
        """
        先验概率
        :return:
        """
        return self._prior_p

    def set_prior_p(self, p: np.ndarray):
        assert p.shape == self._prior_p.shape
        self._prior_p = p

    def get_subset(self, f_indexes: List[int]) -> "FactSet":
        """
        返回facts的子集对应的FactSet
        :param f_indexes: 选取的fact在facts下的索引序列
        :return: 返回一个F中的子集T
        """
        # 恢复从小到大
        f_indexes.sort()
        sub_facts: np.ndarray = np.unique(self.facts[:, f_indexes], axis=0)
        sub_p: np.array = np.zeros(len(sub_facts))
        
        for i in range(len(sub_facts)):
            #print("h", np.prod(self.facts[:, f_indexes] == sub_facts[i], axis=1))
            sub_p[i] = np.sum(
                self._prior_p[np.prod(self.facts[:, f_indexes] == sub_facts[i], axis=1) == 1]
            )
        
        # ground_true: 真值对应的facts索引
        sub_ground_true_val = self.facts[self._ground_true, f_indexes]
        sub_ground_true, = np.where(
            np.all(sub_facts == sub_ground_true_val, axis=1))
        # sub_ground_true = self._ground_true[f_indexes]
        return FactSet(sub_facts, sub_p, sub_ground_true.item())

    def get_fact_space(self, fact_idx: int) -> List[int]:
        """
        获取某一个fact的值域
        :param fact_idx:
        :return:
        """
        if fact_idx < 0 or fact_idx >= self._num_fact:
            raise ValueError("fact_idx out of range")
        return self._fact_space[fact_idx]

    def get_ground_true(self) -> np.ndarray:
        """
        获取所有fact的ground true
        :return:
        """
        return self.facts[self._ground_true]

    def compute_ans_p(self, ans: np.ndarray,
                      fact_idxes: Union[List[int], np.ndarray],
                      worker_accuracy: np.array) -> \
            Tuple[float, np.ndarray]:
        """
        对于给定的一个回答, 以及工人对各个fact的正确率，计算该回答的概率
        :param ans:
        :param fact_idxes: 回答的问题的索引, 比如回答的是第一第三个，对应(0, 2)
        :param worker_accuracy:
        :return: 该回答的概率 p(ans) 以及对应各个输出的边缘概率 p(ans|o)
        """
        # assert len(worker_accuracy) == self.facts.shape[1]
        mask: np.ndarray = (self.facts[:, fact_idxes] == ans)
        acc = worker_accuracy[0]
        Cumprod = np.prod(acc[fact_idxes]**mask *((1-acc[fact_idxes])**(1-mask)),  axis=1)
        ###
        # 注意multiple random variable joint distribution will lead to a new 
        # probability
        # 获取p(u)
        p_ans = np.sum(Cumprod*self.get_prior_p())
        return p_ans, Cumprod

    def compute_entropy(self)->float:
        """
        计算问题集合的墒
        """
        return -np.sum(self._prior_p * np.log(self._prior_p))

    def compute_ansset_entropy(self, worker_accuracy: np.array)->float:
        """
        计算回答集合的墒
        :param worker_accuracy: 工人对每一个fact的回答准确率
        monotone submodualr function property 
        """
        h = 0.
        for i in range(len(self.facts)):
            
            p_ans, _ = self.compute_ans_p(self.facts[i],
                                          list(range(self.num_fact())),
                                          worker_accuracy)
            h -= p_ans * np.log2(p_ans)
        return h