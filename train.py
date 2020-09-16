import model
import config
import torch
import time
import tools
import json
import _pickle
import os
import copy
import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from transformers import AutoTokenizer, AutoModel, AdamW

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


def find_clusters(span_loc, index_clusters):
    """ 找到某个位置在clusters中的哪个cluster

    return:
        cluster的下标，若未找到，返回-1
    """
    if span_loc in index_clusters:
        return index_clusters[span_loc]
    return -1


def train():

    c = config.best_config
    tokenizer = AutoTokenizer.from_pretrained(c["transformer_model_name"])
    transformer_model = AutoModel.from_pretrained(c["transformer_model_name"]).to(c["device"])
    transformer_optim = AdamW(transformer_model.parameters(), lr=c["transformer_lr"])
    transformer_model.train()

    print("preparing data...")

    # =========使用自己的数据=========
    train_data = [
        {
            "sentences": [["打", "雷", "了", "怎", "么", "发", "短", "信", "安", "慰", "女", "朋", "友", "？", "打", "雷", "时", "还", "给", "她", "发", "？"]],
            "clusters": [[[10, 12], [19, 19]]],
            "speaker_ids": [["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b"]],
            "sentence_map": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
            "subtoken_map": [[1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16]],
            "genre": "dummy_genre",
            "doc_key": "dummy_data"
        }
    ]

    val_data = copy.deepcopy(train_data)

    # =========end:使用自己的数据=========

    # =========使用已生成的数据文件=========
    train_data = list()
    with open(c["train_file_path"], "r", encoding="utf-8") as fd:
        for line in fd:
            item = json.loads(line.strip())
            train_data.append(item)

    val_data = list()
    with open(c["val_file_path"], "r", encoding="utf-8") as fd:
        for line in fd:
            item = json.loads(line.strip())
            val_data.append(item)

    # =========end:使用已生成的数据文件=========


    tokenized_train_data = list()
    for data_i in train_data:
        if len(data_i["sentences"]) == 0:
            print("Warning: `sentences` in %s is empty." % data_i["doc_key"])
        else:
            tokenized_train_data.append((tools.tokenize_example(data_i, tokenizer, c)))

    tokenized_val_data = list()
    for data_i in val_data:
        if len(data_i["sentences"]) == 0:
            print("Warning: `sentences` in %s is empty." % data_i["doc_key"])
        else:
            tokenized_val_data.append((tools.tokenize_example(data_i, tokenizer, c)))


    coref_model = model.CorefModel(c).to(c["device"])
    optimizer = torch.optim.Adam(coref_model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    accumulated_loss = 0.0
    max_f1 = 0.0

    print("start training...")

    init_time = time.time()
    steps = 0
    while True:
        for idx in RandomSampler(SequentialSampler(tokenized_train_data)):
            steps += 1
            sentences_ids, sentences_masks, sentences_valid_masks, clusters, speaker_ids, sentence_map, subtoken_map, genre = tokenized_train_data[idx]
            if len(sentences_ids) > c["max_training_sentences"]:
                sentences_ids, sentences_masks, sentences_valid_masks, clusters, speaker_ids, sentence_map, subtoken_map = tools.truncate_example(sentences_ids, sentences_masks, sentences_valid_masks, clusters, speaker_ids, sentence_map, subtoken_map, c["max_training_sentences"])

            top_antecedents_score, top_antecedents_index, top_m_spans_masks, top_m_spans_start, top_m_spans_end = coref_model(sentences_ids, sentences_masks, sentences_valid_masks, speaker_ids, sentence_map, subtoken_map, genre, transformer_model)

            num_spans = len(top_m_spans_start)
            index_clusters = dict()
            for i, cluster in enumerate(clusters):
                for loc in cluster:
                    index_clusters[tuple(loc)] = i
            top_m_spans_cluster_idx = list()        # size: m
            for i in range(num_spans):
                cluster_idx = find_clusters((top_m_spans_start[i].item(), top_m_spans_end[i].item()), index_clusters)
                top_m_spans_cluster_idx.append(cluster_idx)
            top_m_spans_cluster_idx = torch.LongTensor(top_m_spans_cluster_idx).to(device=torch.device(c["device"]))
            top_m_spans_in_gold_clusters = (top_m_spans_cluster_idx != -1)
            # gold labels, size: m * k
            top_antecedents_label = (top_m_spans_cluster_idx[top_antecedents_index].t() == top_m_spans_cluster_idx).t()
            top_antecedents_label[~top_m_spans_in_gold_clusters] = False
            top_antecedents_label[~torch.gather(top_m_spans_masks.to(device=torch.device(c["device"])), 1, top_antecedents_index)] = False 
            # gold labels with dummy label, size: m * (k+1)
            top_antecedents_label = torch.cat((top_antecedents_label, ~torch.sum(top_antecedents_label, dim=1).bool().view(-1,1)), dim=1)


            gold_scores = top_antecedents_score + torch.log(top_antecedents_label.float())     # size: m * (k+1)
            marginalized_gold_scores = torch.logsumexp(gold_scores, 1)                  # size: m
            log_norm = torch.logsumexp(top_antecedents_score, 1)                          # size: m

            loss = torch.sum(log_norm - marginalized_gold_scores)

            optimizer.zero_grad()
            transformer_optim.zero_grad()
            loss.backward()
            optimizer.step()
            transformer_optim.step()

            accumulated_loss += loss.item()

            if steps % c["report_frequency"] == 0:
                total_time = time.time() - init_time
                print("[%d] loss=%.2f, steps/s=%.4f" % (steps, accumulated_loss / c["report_frequency"], steps / total_time))
                accumulated_loss = 0.0

            if steps % c["eval_frequency"] == 0:
                # 每 c["eval_frequency"] 轮保存一次
                coref_model.eval()
                transformer_model.eval()
                torch.save({"model": coref_model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps}, c["checkpoint_path"] + "." + str(steps))
                transformer_model.save_pretrained(c["checkpoint_path"] + ".transformer." + str(steps))
                try:
                    p, r, f = coref_model.evaluate(tokenized_val_data, transformer_model)
                    if f >= max_f1:
                        max_f1 = f
                        torch.save({"model": coref_model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps}, c["checkpoint_path"] + ".max")
                        transformer_model.save_pretrained(c["checkpoint_path"] + ".transformer.max")
                    print("evaluation result:\np:%.4f,r:%.4f,f:%.4f(max f:%.4f)" % (p, r, f, max_f1))
                except Exception as e:
                    print("Error: evaluation error:", e)
                coref_model.train()
                transformer_model.train()
                torch.cuda.empty_cache()





if __name__ == "__main__":
    train()
