# CRS 读入
import json
import numpy as np
import pandas as pd
from query import QuerySelector, BaseQuerySelector, GreedyQuerySelector,RandomQuerySelector, HeuristicQuerySelector
from query import cost_function, get_values
from fact import FactSet
import tiktoken
import openai
import numpy as np
import time
from query import get_values
import yaml
openai.api_base = "https://openkey.cloud/v1"
openai.api_key = "sk-XXX"


def prompt_make(attribute_name1, attribute_name2, values1, values2, schema_aspect):
    k = """Determine the two attributes can be took as the same attribute in schema match. Remember some tips.
Tips:
(1) These two schemas are used to store {schema_aspect} information
(2) Some letters are extracted from the full names and merged into an abbreviation word.
(3) Schema information sometimes is also added as the prefix of abbreviation.
(4) values exchange verification: match would be likely correct, if the second value instances are also suitable for the first attribute name.
Input:
First Attribute Name: {attribute_name} 
its Value instances: {values} 

Second Attribute Name: {attribute_name2} 
Its Value instances: {values2}. \n
Please answer with [yes or no]""".format(attribute_name=attribute_name1, attribute_name2=attribute_name2, values=values1, values2=values2, schema_aspect=schema_aspect)
    return k

def query_chatgpt(message_param):
    sentence = openai.ChatCompletion.create(
                                    model="gpt-4-0613",
                                    messages= [{"role": "user", "content": message_param}],
                                    # 流式输出
                                    temperature=0.1,
                                    stream = False)
    
    return sentence["choices"][0]["message"]["content"]

def process_crs(path:str):
    correspondence_set = []
    with open(path, "r") as f:
        matchings = json.load(f)
    for m in matchings:
        for c in m:
            if c not in correspondence_set:
                correspondence_set.append(c)
    Views = []
    for match in matchings:
        view = []
        for c in correspondence_set:
            if c in match:
                view.append(1)
            else:
                view.append(0)
        Views.append(view)
    prob = np.array([float(1/len(matchings)) for i in range(len(matchings))])
    
    correspondence_count = {tuple(i):0 for i in correspondence_set}
    
    return np.array(Views, dtype=int), matchings, prob, correspondence_set, correspondence_count

def read_correspondence_pd(source_path, target_path):
    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)
    return source_df, target_df


def excute_experiment(crs_path,query_selector, p_w, target_pth, source_pth, turns, budget_per_round, select_name="greedy"):
    start_time = time.time()
    turns_num = turns
    View, _, prob, c_set, correspondence_count = process_crs(crs_path)
    print(crs_path, "c_set:", len(c_set))
    source_df, target_df = read_correspondence_pd(source_path=source_pth, target_path=target_pth)
    len_list = [ ]
    for c_name in c_set:
        v1, v2 = get_values(source_pd=source_df, target_pd=target_df, correspondence_count=correspondence_count, c_name=c_name)
        cost_n = cost_function(c_name, v1, v2)
        len_list.append(cost_n)
    print("most_three:", sorted(len_list, reverse=True)[:3])
    print("mean:", sum(len_list)/len(len_list))
    
    ex_fact = FactSet(facts=View, prior_p=prob, ground_true=2, len_list=len_list)
    c_len = ex_fact.num_fact()
    acc = np.array([[p_w for i in range(c_len)]])
    print("num fact:",ex_fact.num_fact())
    c_index_list = [i for i in range(ex_fact.num_fact())]
    h_list = [ex_fact.compute_entropy()]
    
    ans_record = []
    while turns>0:
        
        if select_name == "brute":
            selection_idxes, h = query_selector.select(ex_fact, budget_per_round, acc)
        else:
            selection_idxes, h = query_selector.select(ex_fact, budget_per_round, acc , c_index_list=c_index_list)
        c_index_list = [i for i in c_index_list if i not in selection_idxes]
        ans = []
        
        for c_id in selection_idxes:
            c_name = c_set[c_id]
            correspondence_count[tuple(c_name)]+=1
            information = 'Musician'
            v1, v2 = get_values(source_pd=source_df, target_pd=target_df, correspondence_count=correspondence_count, c_name=c_name)
            prompt = prompt_make(c_name[0], c_name[1], v1, v2, information)
            answer = query_chatgpt(prompt).lower()
            print(c_name, answer)
            ans_record.append((c_name, answer))
            if "yes" in answer:
                ans = [1]
            else:
                ans = [0]
            p_a,p_a_v = ex_fact.compute_ans_p(ans, [c_id], acc)
            p_post = ex_fact.get_prior_p()*p_a_v/p_a
            ex_fact.set_prior_p(p_post)
        h_list.append(ex_fact.compute_entropy())
        turns -=1
        end_time = time.time()
        info_dic = {"time":end_time - start_time, "uncertainty":h_list, "ans":ans_record}
        with open(f"{crs_path}_info_{select_name}_{turns_num-turns}_{budget_per_round}", "w") as wf:
            json.dump(info_dic, wf, ensure_ascii=False, indent=2)
        with open(f"{crs_path}_prob_{select_name}_{turns_num-turns}_{budget_per_round}", "w") as f:
            json.dump( list(ex_fact.get_prior_p()), f, ensure_ascii=False, indent=2)
        

def effect_experiment(categories ,crs_dir, target_pth, source_pth, turns, budgets):
    
    query_selector = GreedyQuerySelector()
    random_selector = RandomQuerySelector()
    params = []
    for i,budget,turn in zip(categories, budgets, turns):
        # greedy selection
        params.append((crs_dir.format(i), query_selector, 0.918, target_pth.format(i,i.lower()), source_pth.format(i,i.lower()), turn, budget))
        # random selection
        params.append((crs_dir.format(i), random_selector, 0.918, target_pth.format(i,i.lower()), source_pth.format(i,i.lower()), turn, budget, "random"))
    for param in params:
        excute_experiment(*param)
        
def time_cost(categories, crs_dir, target_pth, source_pth, turns, budget):
    query_selector = GreedyQuerySelector()
    brute_selector = BaseQuerySelector()
    params = []
    for i in categories[:]:
        params.append((crs_dir.format(i), brute_selector, 0.918, target_pth.format(i,i.lower()), source_pth.format(i,i.lower()), turns, budget, "brute"))
        params.append((crs_dir.format(i), query_selector, 0.918, target_pth.format(i,i.lower()), source_pth.format(i,i.lower()), turns, budget))
    for param in params:
        excute_experiment(*param)

def run_experiment(config, data_name="musician"):
    "parse various parameters"
    print("config", config)
    crs_dir = config[data_name]["path"]
    print(crs_dir)
    # (crs_path, query_selector, p_w, target_pth, source_pth, turns)
    categories = config[data_name]["names"]
    budgets = config[data_name]["budgets"]
    turns = config[data_name]["turns"]
    target_pth = config[data_name]["target_pth"]
    source_pth = config[data_name]["source_pth"]
    effect_experiment(categories=categories, crs_dir=crs_dir, target_pth=target_pth, source_pth=source_pth, turns=turns, budgets=budgets)
    # time_cost(crs_dir=crs_dir, target_pth=target_pth, source_pth=source_pth, turns=5, budget=100)


if __name__=="__main__":
    # template of params of experiments
    with open('/root/autodl-tmp/prompt-matcher-for-schema-matching/configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    for name in ["miller2"]:
        run_experiment(config, name)
