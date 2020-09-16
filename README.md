# e2e-coref-pytorch

基于bert的端到端指代消解pytorch实现，仅在Ontonotes中文数据集上进行了测试（ONLY TESTED ON ONTONOTES 5.0 CHINESE DATASET），模型架构及tensorflow实现参考[bert-coref](https://github.com/mandarjoshi90/coref)

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

程序需要超过16G的内存/显存，训练时可通过修改 config.py 中的 max_training_sentences 配置降低显存使用

### python依赖包

+ python=3.6.6
+ torch=1.15.0
+ transformers=3.0.2
+ numpy=1.18.1


## 数据集收集和处理

### 使用 ontonotes 数据集（仅中文上测试过，ONLY TESTED ON CHINESE DATASET）

1. 从 LDC 网站下载 ontonotes 数据集，修改 config.py 中的 "ontonotes_root_dir" 配置。若做中文指代，可将配置修改为 "/path/to/ontonotes/data/files/data/chinese/annotations"
2. 检查 data/ 目录下是否存在 train_list/test_list/val_list
3. 运行命令 ```python onf_to_data.py```
4. 若没出现问题，则 data/ 下会生成 train.json/test.json/val.json 三个文件



### 使用自己的数据

将数据根据需要分割成 train.json/test.json/val.json

三个json文件的格式为：每行代表一个文档，每行都是一个json对象，json对象具体内容为：（注：此处为了美观将json对象格式化为多行，在真实文件中需为1行）

```
{
    "doc_key": "文档的名称", 
    "sentences": [["token1", "token2", ...], ...],
    "clusters": [[[sloc1, eloc1], [sloc2, eloc2], ...], ...],
    "speaker_ids" [["speaker#1", ...], ...]
    "sentence_map": [[0, 0, 0, ..., 3, 3, 3], ...],
    "subtoken_map": [[-1, 0, 0, 1, 2, 3, ..., -1], ...]
}
```

说明：sentences中一个元素代表一个长句子，可由若干个短句子组成，使用onf_to_data.py生成的长句子长度在 max_seq_length 附近，max_seq_length 可在 config.py 文件中设置；sentences中组成长句子的若干个短句子由 **tokenize 后的 token** 组成；clusters 是文档中所有指代链的集合，由若干指代链组成，一个指代链由若干个指代词或先行语组成，指代词/先行语使用位置和token个数表示：```[在文档中的token起始位置, 在文档中的token结束位置]```。

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
2. 配置 config.py 中相关参数，将 train.py 中的数据部分修改为希望训练的数据
3. 运行 ```python train.py```


## 模型测试

1. 配置 config.py 相关参数，将 evaluate.py 中的数据部分修改为希望测试的数据
2. 运行 ```pyhton evaluate.py```


## 模型预测

1. 配置 config.py 相关参数
2. 运行 ```pyhton predict.py```
