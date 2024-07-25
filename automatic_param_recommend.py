## 
from main import process_crs, read_correspondence_pd
from query import cost_function, get_values
import yaml

def cost_info(config, config_name):
    names = config[config_name]["names"]
    for name in names:
        print("the cost info of "+name+":")
        crs_path = config[config_name]["path"].format(name)
        source_pth = config[config_name]["source_pth"].format(name, name.lower())
        target_pth = config[config_name]["target_pth"].format(name, name.lower())
        
        View, _, prob, c_set, correspondence_count = process_crs(crs_path)
        print("CRS",View.shape)
        print(crs_path, "c_set:", len(c_set))
        source_df, target_df = read_correspondence_pd(source_path=source_pth, target_path=target_pth)
        len_list = [ ]
        for c_name in c_set:
            v1, v2 = get_values(source_pd=source_df, target_pd=target_df, correspondence_count=correspondence_count, c_name=c_name)
            cost_n = cost_function(c_name, v1, v2)
            len_list.append(cost_n)
        print("sum_cost: ", sum(len_list))
        print("most_three:", sorted(len_list, reverse=True)[:3])
        print("mean:", sum(len_list)/len(len_list))
        start_n = 3*sum(len_list)/len(len_list) if 3*sum(len_list)/len(len_list)>max(len_list) else max(len_list)
        print("budget for each round must be larger than ",start_n)
        print("all budget for K rounds must be less than ", sum(len_list))
        print("##############################################")
        
if __name__ == "__main__":
    with open("/root/autodl-tmp/prompt-matcher-for-schema-matching/configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    for name in ["musician", "assays", "miller2", "prospect"]:
        cost_info(config, name)
        