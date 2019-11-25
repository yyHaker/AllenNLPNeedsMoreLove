{
    "dataset_reader": {
        "type": "squad_bert",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "bert-base-uncased",
            "do_lowercase": true,
            "start_tokens": [],
            "end_tokens": []
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": "bert-base-uncased",
                "do_lowercase": true
            }
        }
    },
    "train_data_path": "data/squad/train-v1.1.json",
    "validation_data_path": "data/squad/dev-v1.1.json",
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 64
  },
    "model": {
        "type": "BERT_QA",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "tokens": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "tokens": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased",
                    "requires_grad":true
                }
            }
        },
        "initializer": [],
    },
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 2,
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.00003,
            "warmup":0.1,
            "t_total": 29199
        },
        "patience": 10,
        "validation_metric": "+f1"
    }
}