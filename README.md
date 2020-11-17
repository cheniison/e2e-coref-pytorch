# e2e-coref-pytorch

[README_ENG](./README_ENG.md)

基于[transformers](https://github.com/huggingface/transformers)的端到端指代消解pytorch实现，模型架构及tensorflow实现参考[bert-coref](https://github.com/mandarjoshi90/coref)

## 目录结构

+ data/: 数据目录，用于存放 train/test/val、checkpoint
+ config.py: 配置文件
+ evaluate.py: 测试文件
+ metrics.py: 评价指标文件
+ model.py: 模型文件
+ onf_to_data.py: ontonotes数据集处理文件，将onf格式转化为所需格式
+ predict.py: 预测文件
+ tools.py: 工具文件
+ train.py: 训练文件


## 要求配置

### 硬件需求

程序需要超过16G的内存/显存，训练时可通过修改 config.py 中的 max_training_sentences 等配置降低显存使用

### python依赖包

+ python=3.6.6
+ torch=1.5.0
+ transformers=3.0.2
+ numpy=1.18.1


## 数据集收集和处理

### 使用 ontonotes 数据集（仅在中文上做了测试）

1. 从 LDC 网站下载 ontonotes 数据集，修改 config.py 中的 "ontonotes_root_dir" 配置。若做中文指代，可将配置修改为 "/path/to/ontonotes/data/files/data/chinese/annotations"
2. 检查 data/ 目录下是否存在 train_list/test_list/val_list。若不存在则需建立软链接到对应文件上（也可直接复制，更推荐软链接），相关命令为 `ln -si ./data/list/train_chinese_list ./data/train_list`
3. 运行命令 `python onf_to_data.py`
4. 若没出现问题，则 data/ 下会生成 train.json/test.json/val.json 三个文件
5. 注意：数据文件（英文、中文、阿拉伯文）也可以简单的由 [bert-coref](https://github.com/mandarjoshi90/coref) 中生成的 jsonlines 转换而来


### 使用自己的数据

将数据根据需要分割成 train.json/test.json/val.json

三个json文件的格式为：每行代表一个文档，每行都是一个json对象，json对象具体内容为：（注：此处为了美观将json对象格式化为多行，在真实文件中需为1行）

```
{
    "sentences": [["token1", "token2", ...], ...],
    "clusters": [[[sloc1, eloc1], [sloc2, eloc2], ...], ...],
    "speaker_ids" [["speaker#1", ...], ...]
    "sentence_map": [[0, 0, 0, ..., 3, 3, 3], ...],
    "subtoken_map": [[0, 0, 1, 2, 3, ...], ...],
    "genre": "文档类型",
    "doc_key": "文档的名称"
}
```

说明：sentences中一个元素代表一个长句子，可由若干个短句子组成，使用onf_to_data.py生成的长句子长度在 max_seq_length 附近，max_seq_length 可在 config.py 文件中设置；sentences中组成长句子的若干个短句子由 **tokenize 后的 token** 组成；clusters 是文档中所有指代链的集合，由若干指代链组成，一个指代链由若干个mention组成，mention使用位置表示：`[在文档中的token起始位置, 在文档中的token结束位置]`。

例子（注：此处为了美观将json对象格式化为多行，在真实文件中每个文档需为1行）：

```
{
    "sentences": [["打", "雷", "了", "怎", "么", "发", "短", "信", "安", "慰", "女", "朋", "友", "？", "打", "雷", "时", "还", "给", "她", "发", "？"]],
    "clusters": [[[10, 12], [19, 19]]],         # （女朋友， 她）
    "speaker_ids": [["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b"]],
    "sentence_map": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
    "subtoken_map": [[1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16]],
    "genre": "dummy_genre",
    "doc_key": "dummy_data"
}
```

## 模型训练

1. 按照“数据集收集和处理”得到符合输入格式的训练数据
2. 配置 config.py 中相关参数，将 train.py 中的数据部分修改为希望训练的数据；将 transformer_model_name 修改为需要的 transformer 模型
3. 运行 `python train.py`


## 模型测试

1. 配置 config.py 相关参数，将 evaluate.py 中的数据部分修改为希望测试的数据
2. 运行 `python evaluate.py`


## 模型预测

1. 配置 config.py 相关参数
2. 运行 `python predict.py`


## 模型效果

使用预设的配置训练 70000 steps，OntoNotes中文测试集 F1 值约为 0.67，英文测试F1 值为 0.73。由于水平有限，英文数据集上与论文效果差距 0.01。


## 反馈
如有错误和建议，欢迎指正和说明。
欢迎 star 和 fork 本项目。

