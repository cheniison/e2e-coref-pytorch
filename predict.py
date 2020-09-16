import model
import config
import torch
import tools
import json
from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":

    c = config.best_config
    
    print("load model...")
    coref_model = model.CorefModel(c).eval().to(c["device"])
    tokenizer = AutoTokenizer.from_pretrained(c["transformer_model_name"])

    transformer_model = AutoModel.from_pretrained(c["checkpoint_path"] + ".transformer.max").eval().to(c["device"])
    checkpoint = torch.load(c["checkpoint_path"] + ".max", map_location=c["device"])
    coref_model.load_state_dict(checkpoint["model"])
    print("success!")

    
    while True:
        
        sentence = input("input: ")
        res = tokenizer(sentence)
        sentence_ids, sentence_masks = res["input_ids"], res["attention_mask"]
        sentence_ids = torch.LongTensor([sentence_ids])
        sentence_masks = torch.LongTensor([sentence_masks])

        
        # 去掉[CLS][SEP]
        sentence_valid_masks = sentence_masks.clone()
        sentence_valid_masks[0][0] = 0
        sentence_valid_masks[0][-1] = 0
        
        valid_num = torch.sum(sentence_valid_masks).item()
        speaker_ids = [[0] * valid_num]
        sentence_map = [[0] * valid_num]
        subtoken_map = [list(range(valid_num))]

        top_antecedents_score, top_antecedents_index, top_m_units_masks, top_m_units_start, top_m_units_end = coref_model(sentence_ids, sentence_masks, sentence_valid_masks, speaker_ids, sentence_map, subtoken_map, len(c["genres"]), transformer_model)
        predicted_antecedents = coref_model.get_predicted_antecedents(top_antecedents_index, top_antecedents_score)
        top_m_units = list()
        for i in range(len(top_m_units_start)):
            top_m_units.append([top_m_units_start[i], top_m_units_end[i]])
        predicted_clusters, _ = coref_model.get_predicted_clusters(top_m_units, predicted_antecedents)

        print("============================")
        print("tokenized context:")
        tokens = tokenizer.convert_ids_to_tokens(sentence_ids[sentence_valid_masks.bool()])
        print(tokens)
        print("predicted clusters:", predicted_clusters)