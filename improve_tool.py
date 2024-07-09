"""
 this a tool script 
 some functions are collected in this script
 
"""
import math 
import copy
import json

def neg_entropy(prob):
    ## metric of data quality 
    neg_entropy = 0
        
    for i in prob:
        neg_entropy += (math.log(i) * i) 
        
    return neg_entropy

def total_join(o, p):
    """
    join two list
    """
    result = []
    for match_o in o:
        for match_p in p:
            matching_o = copy.deepcopy(match_o)
            matching_o.extend(match_p)
            result.append(matching_o)
    return result

def generate_matchings(matches, counter_p, num_match):
    "pandas DataFrame 全连接"
    prob = [1 for i in range(num_match)]
    start = 0 
    # 遍历
    keys_list = list(matches.keys())
    keys_list.sort()
    
    split_keys = list(counter_p.keys())
    matchings_list = []
    for key in split_keys:
        tmp_matching=[]
        num=0
        for match in keys_list:
            if match[0] == key:   
                tmp_matching.append([match])
                   
        matchings_list.append(tmp_matching)
    r = matchings_list[0]
    for idx in range(1,len(matchings_list)):
        r = total_join(r,matchings_list[idx])
    
    # 统计每个possible matching的概率
    for idx,matching_i in enumerate(r):
        for match_i in matching_i:
            prob[idx]*=matches[match_i] 
    sum_prob = 0 
    for ix in range(len(prob)):
        sum_prob+=prob[ix]
    prob = [float(i/sum_prob) for i in prob]
    
    assert len(prob)==len(r), f"prob_len:{len(prob)}, and matchings_num:{len(r)}"
    
    return r, prob, set(matches.keys())

def p_ans_v(inconsistent,  consistent, matchings, p_w):
    """
    inconsistent: dict of len(inconsistent_set) 
    consistent: dict of len(consistent_set) 
    caculate the p(Answer_set|V) of answer set 
    """
    p_ans_v = []
    for idx, _ in enumerate(matchings):
        
        p_ans_v.append( (p_w**consistent[idx]) * ((1-p_w)**inconsistent[idx]))
    return p_ans_v
    
def p_ans(probs,  p_ans_v):
        """ P(Answer_set)
        p_ans_v 是由 c_l 产生的条件概率 
        prob is the distribution of matchings
        """
        p_ans = 0
        for idx, prob in enumerate(probs):
            p_ans += prob * p_ans_v[idx]
        return p_ans

def p_v_ans(probs,p_a_v, p_a):
    
    p_v_ans = []
    for idx, prob in enumerate(probs):
        p_v_ans.append(prob*p_a_v[idx] / p_a)
    return p_v_ans

def inconsistent_or_consistent(c_list, matchings):
    """
    For the checking correspondences set T(correspondences_list), 
    get the inconsistent set and consistent set of correspondences_list,
    and assign them to "self.inconsistent/consistent"
    return len(self.inconsistent), len(self.consistent)  
    """ 
    inconsistent_dict = {}
    consistent_dict = {}
        
    for idx, matching in enumerate(matchings):
        tmp_consistent = []
        tmp_inconsistent = []
        for correspondence in c_list:
                # 蕴含判断
            if  correspondence[0] in matching:
                if correspondence[1] == "yes":
                    tmp_consistent.append(correspondence)
                elif correspondence[1] == "no":
                    tmp_inconsistent.append(correspondence)
            else:
                if correspondence[1]=="no":
                    tmp_consistent.append(correspondence)
                elif correspondence[1]=="yes":
                    tmp_inconsistent.append(correspondence)
        assert len(tmp_consistent)+len(tmp_inconsistent)==len(c_list),"一致集和非一致集并集为全集"
        # 对于答案集  
        # matchings 中的各个matching 的 一致集和不一致集
        inconsistent_dict[idx] = len(tmp_inconsistent)
        consistent_dict[idx] = len(tmp_consistent)
    return inconsistent_dict, consistent_dict



def cost_func(c):
    # cost default 1 
    # chatgpt len ~ price
    return len(c[0][1]+c[1][1])


def answer_from_file(answer_file):
        with open(answer_file, "r") as f:
            ans = json.load(f)
        return ans