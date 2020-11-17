import re
from pathlib import Path
import config
from transformers import AutoTokenizer
import json



class Onf(object):

    SECTION_SEP = "========================================================================================================================"
    SENTENCE_SEP = "------------------------------------------------------------------------------------------------------------------------"

    def __init__(self, onf_file_path, config):
        self.onf_file_path = onf_file_path
        self.config = config


    def read_block(self, lines, start_line):
        """ 读入一个块
        返回块内容，块结束行号
        """
        if lines[start_line] != self.SENTENCE_SEP and lines[start_line] != self.SECTION_SEP:
            raise Exception("块格式有误")
        
        block_content = list()
        while True:
            block_content.append(lines[start_line])
            start_line += 1
            if start_line >= len(lines) or lines[start_line] == self.SENTENCE_SEP or lines[start_line] == self.SECTION_SEP:
                break
        
        return block_content, start_line


    def onf_to_sections(self):
        """ 将onf文件分割成多个sections
        """

        # 读入所有内容
        lines = list()
        with open(self.onf_file_path, "r", encoding="utf-8") as fd:
            for line in fd:
                lines.append(line.strip())

        # 按行处理
        line_i = 0
        sections = list()
        section = Section(start_line=0)

        while True:

            line = lines[line_i]

            if line == self.SECTION_SEP:
                # section 基本结束，读入指代汇总信息
                block_content, line_i = self.read_block(lines, line_i)
                section.add_coref_block(block_content)
                if section.check_section() == False:
                    raise Exception("Section 解析出错")
                sections.append(section)
                start_line = section.get_lines_num() + section.start_line
                section = Section(start_line=start_line)

            elif line == self.SENTENCE_SEP:
                # 句子开始
                block_content, line_i = self.read_block(lines, line_i)
                section.add_sentence_block(block_content)
            else:
                raise Exception("Section 格式有误")

            if line_i >= len(lines):
                break
            
        return sections


    def onf_to_examples(self, tokenizer):
        """ 将onf文件转化为模型易处理的输入-按section分成多个样本
        """
        sections = self.onf_to_sections()
        examples = list()
        for i, section in enumerate(sections):
            example = section.section_to_example(tokenizer, self.config["max_seq_length"])
            example["genre"] = self.get_onf_genre()
            example["doc_key"] = str(self.onf_file_path) + "_" + str(i)
            examples.append(example)
        return examples


    def onf_to_example(self, tokenizer):
        """ 将onf文件转化为模型易处理的输入-将多个section合并成一个样本
        """

        def add_bias(src, bias):
            if type(src) is list:
                res = list()
                for element in src:
                    new_element = add_bias(element, bias)
                    res.append(new_element)
                return res
            else:
                return src + bias


        sections = self.onf_to_sections()
        final_example = {"sentences": list(), "clusters": list(), "speaker_ids": list(), "sentence_map": list(), "subtoken_map": list(), "genre": self.get_onf_genre(), "doc_key": str(self.onf_file_path)}
        clusters_bias = 0
        for i, section in enumerate(sections):
            example = section.section_to_example(tokenizer, self.config["max_seq_length"])
            final_example["sentences"].extend(example["sentences"])
            final_example["clusters"].extend(add_bias(example["clusters"], clusters_bias))
            final_example["speaker_ids"].extend(example["speaker_ids"])
            final_example["sentence_map"].extend(example["sentence_map"])
            final_example["subtoken_map"].extend(example["subtoken_map"])
            clusters_bias += sum([len(s) for s in example["sentences"]])

        return [final_example]


    def get_onf_genre(self):
        """ 得到onf文件类型
        """
        words = self.onf_file_path.parts
        for word in words:
            for i, genre in enumerate(self.config["genres"]):
                if word == genre:
                    # 匹配了对应的类型
                    return genre

        # 未匹配上类型
        return "Unknown"



class Section(object):

    PLAIN_SENTENCE_MARK = "Plain sentence:"
    TREEBANKED_SENTENCE_MARK = "Treebanked sentence:"
    SPEAKER_MARK = "Speaker information:"
    TREE_MARK = "Tree:"
    LEAVES_MARK = "Leaves:"
    COREF_MARK = "Coreference chains for section"
    COREF_LOC_PATTERN = r"\d+\.\d+-\d+"


    def __init__(self, start_line=0):
        self.plain_sentence_list = list()
        self.treebanked_sentence_list = list()
        self.tree_list = list()
        self.speaker_list = list()
        self.leaves_list = list()
        self.coref_chains = list()
        self.start_line = start_line


    def get_lines_num(self):
        """ 得到section中句子数量
        """
        return len(self.plain_sentence_list)

    def sentence_is_break(self, sentence):
        """ 判断句子是否是分界句
        """
        for t in sentence:
            if t != "-":
                return False
        return True

    def sentence_is_valid(self, sentence, speaker_ids):
        """ 判断句子是否是有效的
        """
        return len(speaker_ids) > 0 and speaker_ids[0] != ""


    def check_section(self):
        """ 检测该section的有效性
        """
        return len(self.plain_sentence_list) == len(self.treebanked_sentence_list) == len(self.tree_list) == len(self.speaker_list) == len(self.leaves_list)

    def read_piece(self, sentence_block, line_i):
        """ 读入块中的片
        """
        piece = list()
        line_i += 1
        if not self.sentence_is_break(sentence_block[line_i]):
            raise Exception("片格式有误")
        
        while True:
            line_i += 1
            if line_i + 2 >= len(sentence_block) or (sentence_block[line_i + 2] != "" and self.sentence_is_break(sentence_block[line_i + 2])):
                break
            piece.append(sentence_block[line_i])
        
        line_i += 1
        return piece, line_i

    def read_coref_chain(self, coref_block, line_i):
        """ 读入一条指代链
        """
        if coref_block[line_i][:5] != "Chain":
            raise Exception("指代链格式有误")
        
        _, cid, _ = coref_block[line_i].split()
        corefs = list()
        while True:
            line_i += 1
            if line_i >= len(coref_block) or coref_block[line_i] == "":
                break
            corefs.append(coref_block[line_i].split())

        line_i += 1
        return cid, corefs, line_i


    def section_to_example(self, tokenizer, max_seq_length):
        """ 将section转化为模型输入
        """

        # 将treebanked_sentence分割为词
        sentence_words = list()
        for treebanked_sentence in self.treebanked_sentence_list:
            words = treebanked_sentence.split(" ")
            sentence_words.append(words)
        
        # 将指代链解析成需要格式
        formatted_coref_chains = list()
        line_corefs = dict()      # 按行组织指代
        for coref_chain in self.coref_chains:
            formatted_coref_chain = list()
            for coref in coref_chain:
                if re.match(self.COREF_LOC_PATTERN, coref[0]) is not None:
                    loc = coref[0]
                elif len(coref) > 1 and re.match(self.COREF_LOC_PATTERN, coref[1]) is not None:
                    loc = coref[1]      # 有部分指代的格式开头不是行号
                else:
                    continue            # 有部分换行的指代
                line_num, line_loc = loc.split(".")
                line_start, line_end = line_loc.split("-")
                line_num = int(line_num) - self.start_line
                formatted_coref = [line_num, int(line_start), int(line_end) + 1]
                formatted_coref_chain.append(formatted_coref)
                if line_num not in line_corefs:
                    line_corefs[line_num] = list()
                line_corefs[line_num].append(formatted_coref)

            formatted_coref_chains.append(formatted_coref_chain)
        
        sentences = list()
        subtoken_map = list()
        sentence_map = list()
        speaker_ids = list()
        subtoken_index = 0
        # tokenize每个词，更新指代链
        for line_num, words in enumerate(sentence_words):
            tokens = list()
            sentence_subtoken_map = list()

            for word in words:
                if word[0] != "*":
                    tokenized_word = tokenizer.tokenize(word)
                    subtoken_index += 1
                else:
                    tokenized_word = list()

                for _ in range(len(tokenized_word)):
                    sentence_subtoken_map.append(subtoken_index)

                # 更新指代链
                if line_num in line_corefs:
                    for coref in line_corefs[line_num]:
                        if coref[1] > len(tokens):
                            coref[1] += len(tokenized_word) - 1
                            coref[2] += len(tokenized_word) - 1

                        elif coref[2] > len(tokens):
                            coref[2] += len(tokenized_word) - 1

                tokens.extend(tokenized_word)

            sentences.append(tokens)
            subtoken_map.append(sentence_subtoken_map)
            sentence_map.append([line_num] * len(tokens))
            speaker_ids.append([self.speaker_list[line_num]] * len(tokens))     

        output_sentences = list()
        output_subtoken_map = list()
        output_sentence_map = list()
        output_speaker_ids = list()
        tmp_sentences = list()
        tmp_subtoken_map = list()
        tmp_sentence_map = list()
        tmp_speaker_ids = list()
        bias = 0            # 位置偏置
        line_num = 0
        # 按配置划分聚合句子
        while line_num < len(sentences):
            sentence = sentences[line_num]
            sentence_token_num = len(sentence)
            if len(tmp_sentences) > 0 and sentence_token_num + len(tmp_sentences) > max_seq_length:
                # 句子超出了范围，作为下一个长句的开始
                output_sentences.append(tmp_sentences)
                output_subtoken_map.append(tmp_subtoken_map)
                output_sentence_map.append(tmp_sentence_map)
                output_speaker_ids.append(tmp_speaker_ids)
                tmp_sentences = list()
                tmp_subtoken_map = list()
                tmp_sentence_map = list()
                tmp_speaker_ids = list()
            else:
                # 句子没超出范围
                tmp_sentences.extend(sentence)
                tmp_subtoken_map.extend(subtoken_map[line_num])
                tmp_sentence_map.extend(sentence_map[line_num])
                tmp_speaker_ids.extend(speaker_ids[line_num])
                # 更新指代链
                if line_num in line_corefs:
                    for coref in line_corefs[line_num]:
                        coref[1] += bias
                        coref[2] += bias
                bias += sentence_token_num
                line_num += 1

        if len(tmp_sentences) > 0:
            output_sentences.append(tmp_sentences)
            output_subtoken_map.append(tmp_subtoken_map)
            output_sentence_map.append(tmp_sentence_map)
            output_speaker_ids.append(tmp_speaker_ids)

        output_clusters = list()
        for formatted_coref_chain in formatted_coref_chains:
            if len(formatted_coref_chain) > 1:
                cluster = list()
                for coref in formatted_coref_chain:
                    if coref[1] != coref[2]:
                        cluster.append([coref[1], coref[2] - 1])
                if len(cluster) > 1:
                    output_clusters.append(cluster)

        return {"sentences": output_sentences, "clusters": output_clusters, "speaker_ids": output_speaker_ids, "sentence_map": output_sentence_map, "subtoken_map": output_subtoken_map}


    def add_sentence_block(self, sentence_block):
        """ 将句子块的信息加入到Section中
        """
        plain_sentence, treebanked_sentence, tree, speaker, leaves = self.parse_sentence_block(sentence_block)
        self.plain_sentence_list.append(plain_sentence)
        self.treebanked_sentence_list.append(treebanked_sentence)
        self.tree_list.append(tree)
        self.speaker_list.append(speaker)
        self.leaves_list.append(leaves)


    def parse_sentence_block(self, sentence_block):
        """ 解析句子块
        """
        plain_sentence = ""
        treebanked_sentence = ""
        tree = list()
        speaker = ""
        leaves = list()
        
        line_i = 2
        
        if sentence_block[line_i] == self.PLAIN_SENTENCE_MARK:
            plain_sentence, line_i = self.read_piece(sentence_block, line_i)
            if len(plain_sentence) == 0:
                raise Exception("Plain sentence 格式有误")
            plain_sentence = " ".join(plain_sentence)
        else:
            raise Exception("块格式有误")

        if sentence_block[line_i] == self.TREEBANKED_SENTENCE_MARK:
            treebanked_sentence, line_i = self.read_piece(sentence_block, line_i)
            if len(treebanked_sentence) == 0:
                raise Exception("Treebanked sentence 格式有误")
            treebanked_sentence = " ".join(treebanked_sentence)
        else:
            raise Exception("块格式有误")

        if sentence_block[line_i] == self.SPEAKER_MARK:
            speaker, line_i = self.read_piece(sentence_block, line_i)
            if len(speaker) != 3:
                raise Exception("Speaker information 格式有误")
            speaker = speaker[0].split()[-1]
        else:
            pass
        
        if sentence_block[line_i] == self.TREE_MARK:
            tree, line_i = self.read_piece(sentence_block, line_i)
        else:
            raise Exception("块格式有误")

        if sentence_block[line_i] == self.LEAVES_MARK:
            leaves, line_i = self.read_piece(sentence_block, line_i)
        else:
            raise Exception("块格式有误")

        return plain_sentence, treebanked_sentence, tree, speaker, leaves


    def add_coref_block(self, coref_block):
        """ 将指代块的信息加入到Section中
        """ 
        self.coref_chains = self.parse_coref_block(coref_block)


    def parse_coref_block(self, coref_block):
        """ 解析指代块
        """
        line_i = 4
        coref_chains = list()
        while line_i < len(coref_block) and coref_block[line_i] != "":
            cid, corefs, line_i = self.read_coref_chain(coref_block, line_i)
            coref_chains.append(corefs)
        return coref_chains



def find_onf_files(root_path):
    # 找到目录下所有 .onf 文件
    onf_files_path = list()
    for fp in root_path.iterdir():
        if fp.is_dir():
            onf_files_path.extend(find_onf_files(fp))
        if fp.suffix == ".onf":
            onf_files_path.append(fp)
    return onf_files_path



if __name__ == "__main__":

    import tqdm

    c = config.best_config
    tokenizer = AutoTokenizer.from_pretrained(c["transformer_model_name"], do_lower_case=False)
    ontonotes_root_path = Path(c["ontonotes_root_dir"])
    onf_files_path = find_onf_files(ontonotes_root_path)

    train_list_file = "./data/train_list"
    val_list_file = "./data/val_list"
    test_list_file = "./data/test_list"

    train_list = list()
    with open(train_list_file, "r", encoding="utf-8") as fd:
        for line in fd:
            train_list.append(line.strip())
    
    val_list = list()
    with open(val_list_file, "r", encoding="utf-8") as fd:
        for line in fd:
            val_list.append(line.strip())

    test_list = list()
    with open(test_list_file, "r", encoding="utf-8") as fd:
        for line in fd:
            test_list.append(line.strip())

    train_file_fd = open(c["train_file_path"], "w", encoding="utf-8")
    val_file_fd = open(c["val_file_path"], "w", encoding="utf-8")
    test_file_fd = open(c["test_file_path"], "w", encoding="utf-8")

    for onf_file_path in tqdm.tqdm(onf_files_path):
        onf = Onf(onf_file_path, c)
        examples = onf.onf_to_examples(tokenizer)
        # examples = onf.onf_to_example(tokenizer)
        examples_json = ""
        for example in examples:
            examples_json += json.dumps(example, ensure_ascii=False) + "\n"
        if onf_file_path.name in train_list:
            train_file_fd.write(examples_json)
        elif onf_file_path.name in val_list:
            val_file_fd.write(examples_json)
        elif onf_file_path.name in test_list:
            test_file_fd.write(examples_json)
        else:
            print("Warning: %s not in train/test/val list." % onf_file_path.name)

    train_file_fd.close()
    val_file_fd.close()
    test_file_fd.close()

