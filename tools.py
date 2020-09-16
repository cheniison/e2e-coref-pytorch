import numpy as np
import random
import torch


def print_spans_loc(spans_start, spans_end):
    """ 输出spans的位置
    """
    assert len(spans_start) == len(spans_end)
    for i in range(len(spans_start)):
        try:
            print((spans_start[i].item(), spans_end[i].item()))
        except:
            print((spans_start[i], spans_end[i]))



def truncate_example(sentences_ids, sentences_masks, sentences_valid_masks, clusters, speaker_ids, sentence_map, subtoken_map, max_training_sentences):
    """ 分割文档
    """
    line_offset = random.randint(0, len(sentences_ids) - max_training_sentences)
    truncated_sentences_ids = sentences_ids[line_offset:line_offset + max_training_sentences]
    truncated_sentences_masks = sentences_masks[line_offset:line_offset + max_training_sentences]
    truncated_sentences_valid_masks = sentences_valid_masks[line_offset:line_offset + max_training_sentences]
    truncated_speaker_ids = speaker_ids[line_offset:line_offset + max_training_sentences]
    truncated_sentence_map = sentence_map[line_offset:line_offset + max_training_sentences]
    truncated_subtoken_map = subtoken_map[line_offset:line_offset + max_training_sentences]

    token_offset = torch.sum(sentences_valid_masks[:line_offset]).item()
    token_num = torch.sum(truncated_sentences_valid_masks).item()

    truncated_clusters = list()
    for cluster in clusters:
        truncated_cluster = list()
        for start_loc, end_loc in cluster:
            if start_loc - token_offset >= 0 and end_loc <= token_offset + token_num:
                truncated_cluster.append([start_loc-token_offset, end_loc-token_offset])
        if len(truncated_cluster) > 0:
            truncated_clusters.append(truncated_cluster)
    return truncated_sentences_ids, truncated_sentences_masks, truncated_sentences_valid_masks, truncated_clusters, truncated_speaker_ids, truncated_sentence_map, truncated_subtoken_map


def add_bias_to_clusters(clusters, bias, coref_filter):
    for cluster in clusters:
        for coref in cluster:
            if coref_filter(coref) == True:
                coref[0] += bias
                coref[1] += bias


def tokenize_example(example, tokenizer, c):
    """ tokenize example
    """

    sentences = example["sentences"]
    clusters = example["clusters"]
    speaker_ids = example["speaker_ids"]
    sentence_map = example["sentence_map"]
    subtoken_map = example["subtoken_map"]
    example_genre_name = example["genre"]

    # genre to idx
    genre = 0
    while genre < len(c["genres"]):
        if example_genre_name == c["genres"][genre]:
            break
        genre += 1
    
    # token to ids
    sentences_ids = list()
    sentences_masks = list()

    max_seq_len = max([len(s) for s in sentences]) + 2      # 样本中最长句子的长度，加上[CLS]和[SEP]

    for sentence in sentences:
        sentence_ids = tokenizer.convert_tokens_to_ids(sentence)
        sentence_ids = [101] + sentence_ids + [102]
        token_num = len(sentence_ids)
        sentence_masks = [1] * token_num
        if token_num > c["bert_max_seq_length"]:
            raise Exception("the length of sentence is out the range of bert_max_seq_length.")
        else:
            sentence_ids += [0] * (max_seq_len - token_num)
            sentence_masks += [0] * (max_seq_len - token_num)
        sentences_ids.append(sentence_ids)
        sentences_masks.append(sentence_masks)

    sentences_ids = torch.LongTensor(sentences_ids)
    sentences_masks = torch.LongTensor(sentences_masks)

    # convert speaker_ids to long type
    speaker_dict = dict()
    speaker_index = 0
    speaker_ids_long = list()
    for sentence_speaker_ids in speaker_ids:
        sentence_speaker_ids_long = list()
        for speaker in sentence_speaker_ids:
            if speaker not in speaker_dict:
                speaker_dict[speaker] = speaker_index
                speaker_index += 1
            speaker_id = speaker_dict[speaker]
            sentence_speaker_ids_long.append(speaker_id)
        speaker_ids_long.append(sentence_speaker_ids_long)
    speaker_ids = speaker_ids_long

    sentence_tokens_num = torch.sum(sentences_masks, dim=1)
    # 去除embed中无关的符号
    # [CLS]位置置为0
    sentences_valid_masks = sentences_masks.clone()
    sentences_valid_masks[:, 0] = 0
    # [SEP]位置置为0
    for i in range(len(sentences_valid_masks)):
        sentences_valid_masks[i][sentence_tokens_num[i] - 1] = 0

    # 验证数据的正确性
    for i in range(len(sentences)):
        if not (len(sentences[i]) == len(speaker_ids[i]) == len(sentence_map[i]) == len(subtoken_map[i])):
            raise Exception("The length of sentence/speaker_ids/sentence_map/subtoken_map is inconsistent.")


    return sentences_ids, sentences_masks, sentences_valid_masks, clusters, speaker_ids, sentence_map, subtoken_map, genre


