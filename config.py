
# best config
best_config = {
    "embedding_dim": 768,
    "max_span_width": 10,
    # max training sentences depends on size of memery 
    "max_training_sentences": 11,
    # max seq length
    "max_seq_length": 128,
    "bert_max_seq_length": 512,

    "device": "cuda",
    "checkpoint_path": "./data/checkpoint",
    "lr": 0.0002,
    "weight_decay": 0.0005,
    "dropout": 0.3,

    "report_frequency": 5,
    "eval_frequency": 50,

    # ontonotes dir
    "ontonotes_root_dir": "./data/ontonotes",
    "train_file_path": "./data/train.json",
    "test_file_path": "./data/test.json",
    "val_file_path": "./data/val.json",

    # max candidate mentions size in first/second stage
    "top_span_ratio": 40,
    "max_top_antecedents": 500,
    # use coarse to fine pruning
    "coarse_to_fine": True,
    # high order coref depth
    "coref_depth": 2,

    # FFNN config
    "ffnn_depth": 2,
    "ffnn_size": 150,

    # use span features, such as distance
    "use_features": True,
    "feature_dim": 20,
    "model_heads": True,
    # use metadata, such as genre and speaker info
    "use_metadata": True,
    "genres": ["bc", "bn", "mz", "nw", "tc", "wb"],

    # 选择topk时是否考虑单元互斥
    "extract_spans": False,

    # transformer model
    "transformer_model_name": 'bert-base-chinese',
    "transformer_lr": 0.00001,
}
