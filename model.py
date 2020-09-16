import torch
import numpy as np
from collections import OrderedDict
import metrics
import tools
import math


class CorefModel(torch.nn.Module):

    def __init__(self, config):
        """ 模型初始化函数

        parameters:
            config: 模型配置
                embedding_dim: 词向量的维度
                span_dim: span的维度
                gap_dim: 零指代维度
                ffnn_depth/size: 前馈神经网络深度和大小
                max_span_width: span的最多字符数
        """
        super(CorefModel, self).__init__()
        # 配置初始化
        self.config = config
        self.span_dim = 2 * self.config["embedding_dim"]
        mm_input_dim = 0

        if self.config["use_features"]:
            self.span_dim += self.config["feature_dim"]
            self.span_width_embeddings = torch.nn.Embedding(self.config["max_span_width"], self.config["feature_dim"])
            self.bucket_distance_embeddings = torch.nn.Embedding(10, self.config["feature_dim"])
            mm_input_dim += self.config["feature_dim"]

        if self.config["model_heads"]:
            self.span_dim += self.config["embedding_dim"]
            self.Sh = torch.nn.Linear(self.config["embedding_dim"], 1)      # token head score

        if self.config["use_metadata"]:
            self.genre_embeddings = torch.nn.Embedding(len(self.config["genres"]) + 1, self.config["feature_dim"])
            self.same_speaker_emb = torch.nn.Embedding(2, self.config["feature_dim"])
            mm_input_dim += 2 * self.config["feature_dim"]

        mm_input_dim += 3 * self.span_dim

        # 模型初始化
        self.Sm = self._create_score_ffnn(self.span_dim)     # mention score        
        self.Smm = self._create_score_ffnn(mm_input_dim)      # pairwise score between spans
        self.c2fP = torch.nn.Linear(self.span_dim, self.span_dim)       # coarse to fine pruning span projection
        self.hoP = torch.nn.Linear(2 * self.span_dim, self.span_dim)    # high order projection



    def _create_ffnn(self, input_size, output_size, ffnn_size, ffnn_depth, dropout=0):
        """ 创建前馈神经网络
        """
        current_size = input_size
        model_seq = OrderedDict()
        for i in range(ffnn_depth):
            model_seq['fc' + str(i)] = torch.nn.Linear(current_size, ffnn_size)
            model_seq['relu' + str(i)] = torch.nn.ReLU()
            model_seq['dropout' + str(i)] = torch.nn.Dropout(dropout)
            current_size = ffnn_size
        model_seq['output'] = torch.nn.Linear(current_size, output_size)

        return torch.nn.Sequential(model_seq)


    def _create_score_ffnn(self, input_size):
        """ 创建评分前馈神经网络
        """
        return self._create_ffnn(input_size, 1, self.config["ffnn_size"], self.config["ffnn_depth"], self.config["dropout"])

    
    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        float_distances = distances.float()
        combined_idx = torch.floor(torch.log(float_distances) / math.log(2)) + 3
        use_identity = distances <= 4
        combined_idx[use_identity] = float_distances[use_identity]
        combined_idx = combined_idx.long()

        return torch.clamp(combined_idx, 0, 9)


    def get_span_embed(self, tokens_embed, spans_start, spans_end):
        """
        得到token序列中的所有span表示
        """
        span_embed_list = list()
        start_embed = tokens_embed[spans_start]           # 第一个token表示
        end_embed = tokens_embed[spans_end]            # 最后一个token表示

        span_embed_list.append(start_embed)
        span_embed_list.append(end_embed)

        if self.config["use_features"]:
            spans_width = (spans_end - spans_start).to(device=torch.device(self.config["device"]))
            span_width_embed = self.span_width_embeddings(spans_width)
            span_width_embed = torch.nn.functional.dropout(span_width_embed, p=self.config["dropout"], training=self.training)
            span_embed_list.append(span_width_embed)

        if self.config["model_heads"]:
            tokens_score = self.Sh(tokens_embed).view(-1)           # size: num_tokens
            tokens_locs = torch.arange(start=0, end=len(tokens_embed), dtype=torch.long).repeat(len(spans_start), 1)        # size: num_spans * num_tokens
            tokens_masks = (tokens_locs >= spans_start.view(-1, 1)) & (tokens_locs <= spans_end.view(-1, 1))             # size: num_spans * num_tokens
            tokens_weights = torch.nn.functional.softmax(
                (tokens_score + torch.log(tokens_masks.float()).to(device=torch.device(self.config["device"]))), 
                dim=1
            )
            
            span_head_emb = torch.matmul(tokens_weights, tokens_embed)
            span_embed_list.append(span_head_emb)

        return torch.cat(span_embed_list, dim=1)


    def extract_spans(self, spans_score, spans_start, spans_end, m):
        """ 得到top m个不互斥的spans
        """
        sorted_spans_score, indices = torch.sort(spans_score, 0, True)
        top_m_spans_index = list()
        top_m_spans_start = torch.zeros(m, dtype=torch.long)
        top_m_spans_end = torch.zeros(m, dtype=torch.long)
        top_m_len = 0
        i = 0
        while top_m_len < m and i < len(sorted_spans_score):
            span_index = indices[i]
            span_start = spans_start[span_index]
            span_end = spans_end[span_index]

            res = (((span_start < top_m_spans_start) & (span_end < top_m_spans_end) & (span_end >= top_m_spans_start)) 
                | ((span_start > top_m_spans_start) & (span_start <= top_m_spans_end) & (span_end > top_m_spans_end)))

            if torch.sum(res) == 0:
                top_m_spans_index.append(span_index)
                top_m_spans_start[top_m_len] = span_start
                top_m_spans_end[top_m_len] = span_end
                top_m_len += 1
            i += 1

        return torch.stack(top_m_spans_index)


    def coarse_to_fine_pruning(self, k, spans_masks, spans_embed, spans_score):
        """ 由粗到精得到每个span的候选先行语

        parameters:
            k: int, 第二阶段候选先行词个数，若配置中coarse_to_fine设置为false，此参数将被忽略
            spans_masks: m * m, 候选spans间的可见性
            spans_embed: m * span_dim, 候选spans embed
            spans_score: m * 1, 候选spans的当前得分

        return
            score: FloatTensor m * k
            index: Long m * k
        """

        m = len(spans_embed)
        all_score = torch.zeros(m, m).to(device=torch.device(self.config["device"]))
        all_score[~spans_masks] = float("-inf")

        # add span score
        all_score += spans_score

        antecedents_offset = (torch.arange(0, m).view(-1, 1) - torch.arange(0, m).view(1, -1))
        antecedents_offset = antecedents_offset.to(device=torch.device(self.config["device"]))

        if self.config["coarse_to_fine"] == True:
            source_top_span_emb = torch.nn.functional.dropout(self.c2fP(spans_embed), p=self.config["dropout"], training=self.training)
            target_top_span_emb = torch.nn.functional.dropout(spans_embed, p=self.config["dropout"], training=self.training)
            all_score += source_top_span_emb.matmul(target_top_span_emb.t())     # m * m
        else:
            # 使用所有候选span
            k = m

        top_antecedents_fast_score, top_antecedents_index = torch.topk(all_score, k)
        top_antecedents_offset = torch.gather(antecedents_offset, dim=1, index=top_antecedents_index)

        return top_antecedents_fast_score, top_antecedents_index, top_antecedents_offset


    def get_spans_similarity_score(self, top_antecedents_index, spans_embed, top_antecedents_offset, speaker_ids, genre_emb):
        """ 得到span间相似度得分

        parameters:
            top_antecedents_index: Long m * k, 每个span的topk候选先行词下标
            spans_embed: m * span_dim, 候选spans embed
            top_antecedents_offset: m * k, 候选spans间的相对偏移
            speaker_ids: m, 候选spans的speaker id
            genre_emb: feature_dim, 文档类型embed
        return:
            score: FloatTensor m * k
        """
        m = len(spans_embed)
        k = top_antecedents_index.shape[1]

        span_index = torch.arange(0, m, dtype=torch.long).repeat(k, 1).t()

        mm_ffnn_input_list = list()
        mm_ffnn_input_list.append(spans_embed[span_index])
        mm_ffnn_input_list.append(spans_embed[top_antecedents_index])
        mm_ffnn_input_list.append(mm_ffnn_input_list[0] * mm_ffnn_input_list[1])

        if self.config["use_features"]:
            top_antecedents_distance_bucket = self.bucket_distance(top_antecedents_offset)
            top_antecedents_distance_emb = self.bucket_distance_embeddings(top_antecedents_distance_bucket)
            mm_ffnn_input_list.append(top_antecedents_distance_emb)

        if self.config["use_metadata"]:
            same_speaker_ids = (speaker_ids.view(-1, 1) == speaker_ids[top_antecedents_index]).long().to(device=torch.device(self.config["device"]))
            speaker_emb = self.same_speaker_emb(same_speaker_ids)
            mm_ffnn_input_list.append(speaker_emb)
            mm_ffnn_input_list.append(genre_emb.repeat(m, k, 1))

        mm_ffnn_input = torch.cat(mm_ffnn_input_list, dim=2)
        mm_slow_score = self.Smm(mm_ffnn_input)

        return mm_slow_score.squeeze()


    def forward(self, sentences_ids, sentences_masks, sentences_valid_masks, speaker_ids, sentence_map, subtoken_map, genre, transformer_model):
        """ 
        parameters:
            sentences_ids: num_sentence * max_sentence_len
            sentences_masks: num_sentence * max_sentence_len
            sentences_valid_masks: num_sentence * max_sentence_len
            speaker_ids: list[list]
            sentence_map: list[list]
            subtoken_map: list[list]
            genre: genre_id
            transformer_model: AutoModel
        """

        # 得到所有候选spans
        sentences_embed, _ = transformer_model(sentences_ids.to(device=torch.device(self.config["device"])), sentences_masks.to(device=torch.device(self.config["device"])))      # num_sentence * max_sentence_len * embed_dim

        tokens_embed = sentences_embed[sentences_valid_masks.bool()]          # num_tokens * embed_dim

        flattened_sentence_indices = list()
        for sm in sentence_map:
            flattened_sentence_indices += sm
        flattened_sentence_indices = torch.LongTensor(flattened_sentence_indices)

        candidate_spans_start = torch.arange(0, len(tokens_embed)).repeat(self.config["max_span_width"], 1).t()
        candidate_spans_end = candidate_spans_start + torch.arange(0, self.config["max_span_width"])
        candidate_spans_start_sentence_indices = flattened_sentence_indices[candidate_spans_start]
        candidate_spans_end_sentence_indices = flattened_sentence_indices[torch.min(candidate_spans_end, torch.tensor(len(tokens_embed) - 1))]
        candidate_spans_mask = (candidate_spans_end < len(tokens_embed)) & (candidate_spans_start_sentence_indices == candidate_spans_end_sentence_indices)

        spans_start = candidate_spans_start[candidate_spans_mask]
        spans_end = candidate_spans_end[candidate_spans_mask]
        
        spans_embed = self.get_span_embed(tokens_embed, spans_start, spans_end)       # size: num_spans * span_dim
        spans_score = self.Sm(spans_embed)          # size: num_spans * 1

        spans_len = len(spans_score)
        spans_score_mask = torch.zeros(spans_len, dtype=torch.bool)
        m = min(int(self.config["top_span_ratio"] * len(tokens_embed)), spans_len)    # 文本中最多的span数量

        # 根据得分得到topk的spans
        if self.config["extract_spans"]:
            span_indices = self.extract_spans(spans_score, spans_start, spans_end, m)
        else:
            _, span_indices = torch.topk(spans_score, m, dim=0, largest=True)

        m = len(span_indices)
        spans_score_mask[span_indices] = True

        top_m_spans_embed = spans_embed[spans_score_mask]        # size: m * span_dim
        top_m_spans_score = spans_score[spans_score_mask]        # size: m * 1
        top_m_spans_start = spans_start[spans_score_mask]        # size: m
        top_m_spans_end = spans_end[spans_score_mask]            # size: m

        # BoolTensor, size: m * m
        top_m_spans_masks = \
            ((top_m_spans_start.repeat(m, 1).t() > top_m_spans_start) \
            | ((top_m_spans_start.repeat(m, 1).t() == top_m_spans_start) \
                & (top_m_spans_end.repeat(m, 1).t() > top_m_spans_end)))

        if self.config['use_metadata']:
            flattened_speaker_ids = list()
            for si in speaker_ids:
                flattened_speaker_ids += si
            flattened_speaker_ids = torch.LongTensor(flattened_speaker_ids)
            top_m_spans_speaker_ids = flattened_speaker_ids[top_m_spans_start]
            genre_emb = self.genre_embeddings(torch.LongTensor([genre]).to(device=torch.device(self.config["device"]))).squeeze()
        else:
            top_m_spans_speaker_ids = None
            genre_emb = None

        # 对每个span得到其topk候选先行词（antecedent）
        k = min(self.config["max_top_antecedents"], m)

        top_antecedents_fast_score, top_antecedents_index, top_antecedents_offset = self.coarse_to_fine_pruning(
            k, top_m_spans_masks, top_m_spans_embed, top_m_spans_score
        )

        for i in range(self.config["coref_depth"]):
            top_antecedents_slow_score = self.get_spans_similarity_score(top_antecedents_index, top_m_spans_embed, top_antecedents_offset, top_m_spans_speaker_ids, genre_emb)
            top_antecedents_score = top_antecedents_fast_score + top_antecedents_slow_score         # size: m * k
            dummy_score = torch.zeros(m, 1).to(device=torch.device(self.config["device"]))          # add dummy
            top_antecedents_score = torch.cat((top_antecedents_score, dummy_score), dim=1)          # size: m * (k+1)
            top_antecedents_weight = torch.nn.functional.softmax(top_antecedents_score, dim=1)      # size: m * (k+1)
            top_antecedents_emb = torch.cat((top_m_spans_embed[top_antecedents_index], top_m_spans_embed.unsqueeze(1)), dim=1)     # size: m * (k+1) * embed
            attended_spans_emb = torch.sum(top_antecedents_weight.unsqueeze(2) * top_antecedents_emb, dim=1)            # size: m * embed
            f = torch.sigmoid(self.hoP(torch.cat([top_m_spans_embed, attended_spans_emb], dim=1)))  # size: m * embed
            top_m_spans_embed = f * attended_spans_emb + (1 - f) * top_m_spans_embed                # size: m * embed

        return top_antecedents_score, top_antecedents_index, top_m_spans_masks, top_m_spans_start, top_m_spans_end
    

    def evaluate(self, data, transformer_model):
        """ 评估函数
        """
        coref_evaluator = metrics.CorefEvaluator()

        with torch.no_grad():
            for idx, data_i in enumerate(data):
                sentences_ids, sentences_masks, sentences_valid_masks, gold_clusters, speaker_ids, sentence_map, subtoken_map, genre = data_i
                top_antecedents_score, top_antecedents_index, top_m_spans_masks, top_m_spans_start, top_m_spans_end = self.forward(sentences_ids, sentences_masks, sentences_valid_masks, speaker_ids, sentence_map, subtoken_map, genre, transformer_model)
                predicted_antecedents = self.get_predicted_antecedents(top_antecedents_index, top_antecedents_score)
                top_m_spans = list()
                for i in range(len(top_m_spans_start)):
                    top_m_spans.append(tuple([top_m_spans_start[i].item(), top_m_spans_end[i].item()]))

                # all spans
                gold_clusters = [tuple(tuple([m[0], m[1]]) for m in gc) for gc in gold_clusters]
                mention_to_gold = {}
                for gc in gold_clusters:
                    for mention in gc:
                        mention_to_gold[tuple(mention)] = gc
                predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_m_spans, predicted_antecedents)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        return coref_evaluator.get_prf()


    def get_predicted_antecedents(self, top_antecedents_index, top_antecedents_score):
        """ 得到每个span的得分最高的先行词
        """
        predicted_antecedents = []
        for i, index in enumerate(torch.argmax(top_antecedents_score, axis=1)):
            if index == len(top_antecedents_score[i]) - 1:
                # 空指代
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(top_antecedents_index[i][index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_m_spans, predicted_antecedents, remove_zp=False):
        """ 根据预测的先行词得到指代簇
        """
        idx_to_clusters = {}
        predicted_clusters = []
        for i in range(len(top_m_spans)):
            idx_to_clusters[i] = set([i])

        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            else:
                union_cluster = idx_to_clusters[predicted_index.item()] | idx_to_clusters[i]
                for j in union_cluster:
                    idx_to_clusters[j] = union_cluster
        
        tagged_index = set()
        for i in idx_to_clusters:
            if (len(idx_to_clusters[i]) == 1) or (i in tagged_index):
                continue
            cluster = idx_to_clusters[i]
            predicted_cluster = list()
            for j in cluster:
                tagged_index.add(j)
                predicted_cluster.append(tuple(top_m_spans[j]))

            predicted_clusters.append(tuple(predicted_cluster))

        if remove_zp == True:
            predicted_clusters = self.remove_zp_cluster(predicted_clusters)

        mention_to_predicted = {}
        for pc in predicted_clusters:
            for mention in pc:
                mention_to_predicted[tuple(mention)] = pc

        return predicted_clusters, mention_to_predicted

    def remove_zp_cluster(self, clusters):
        # 移除聚类中的零指代，和tools中的略有不同
        clusters_wo_zp = list()
        for cluster in clusters:
            cluster_wo_zp = list()
            for sloc, eloc in cluster:
                if eloc - sloc > 0:
                    cluster_wo_zp.append(tuple([sloc, eloc]))
            if len(cluster_wo_zp) > 1:
                clusters_wo_zp.append(tuple(cluster_wo_zp))

        return clusters_wo_zp