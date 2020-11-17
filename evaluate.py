import model
import config
import torch
import tools
import json
import _pickle
from transformers import AutoTokenizer, AutoModel


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

if __name__ == "__main__":

    c = config.best_config
        
    coref_model = model.CorefModel(c).eval().to(c["device"])
    tokenizer = AutoTokenizer.from_pretrained(c["transformer_model_name"])

    transformer_model = AutoModel.from_pretrained(c["checkpoint_path"] + ".transformer.max").eval().to(c["device"])
    checkpoint = torch.load(c["checkpoint_path"] + ".max")
    coref_model.load_state_dict(checkpoint["model"])

    # ==============================================================================================
    # data
    # ==============================================================================================
    
    # -----------------------------
    # 使用自己的数据
    # val_data = [
    #     {
    #         "sentences": [["打", "雷", "了", "怎", "么", "发", "短", "信", "安", "慰", "女", "朋", "友", "？", "打", "雷", "时", "还", "给", "她", "发", "？"]],
    #         "clusters": [[[10, 12], [19, 19]]],
    #         "speaker_ids": [["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b"]],
    #         "sentence_map": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
    #         "subtoken_map": [[1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16]],
    #         "genre": 0,
    #         "doc_key": "dummy_data"
    #     }
    # ]

    # ------------------------------
    # 使用 val_file_path 文件数据
    val_data = list()
    with open(c["val_file_path"], "r", encoding="utf-8") as fd:
        for line in fd:
            item = json.loads(line.strip())
            val_data.append(item)

    # ================================================================================================

    tokenized_val_data = list()
    for data_i in val_data:
        if len(data_i["sentences"]) == 0:
            print("Warning: `sentences` in %s is empty." % data_i["doc_key"])
        else:
            tokenized_val_data.append((tools.tokenize_example(data_i, tokenizer, c)))

    
    p, r, f = coref_model.evaluate(tokenized_val_data, transformer_model)
    print("Average Precision:", p)
    print("Average Recall:", r)
    print("Average F1:", f)
