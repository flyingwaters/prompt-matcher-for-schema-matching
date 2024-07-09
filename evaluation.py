import json


def parse_label(ground_truth_path):
    with open(ground_truth_path, "r") as f:
        content = json.load(f)
        return [(i["source_column"], i["target_column"]) for i in content["matches"] ]

def caculate_metric(path, ground_truth_path):
    """
    path: test file
    ground_truth_path: ground label
    """
    label_list = parse_label(ground_truth_path)
    label_len = len(label_list)
    with open(path, "r") as f:
        c = json.load(f)
    save_l = {"label_list":label_list}
    
    for idx, matching in enumerate(c):
            inter_set = []
            for i in label_list:
                if list(i) in matching:
                    inter_set.append(i)
            acc_length = len(inter_set)
            predict_len = len(matching)
            precision = acc_length/float(predict_len)
            recall = acc_length/float(label_len)
            if acc_length==0:
                f1 = 0
            else:
                f1 = 2*(precision*recall)/(precision+recall)
            save_l[idx]= {"precision":precision, "recall":recall,"f1":f1, "matches":matching}
    with open(f"{path}_metrics", "w") as w:
           json.dump(save_l, w, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    path = "/root/autodl-tmp/prompt-matcher-reduce-uncertainty/CRS/{}"
    ground_truth_path = "/root/autodl-tmp/prompt-matcher-reduce-uncertainty/Valentine-datasets/Wikidata/Musicians/{}/{}_mapping.json"
    for i in ["Musicians_joinable", "Musicians_unionable", "Musicians_viewunion", "Musicians_semjoinable"]:
        caculate_metric(path.format(i), ground_truth_path.format(i, i.lower()))