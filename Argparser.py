import argparse
import os

File = os.getcwd()

coat_params = {
    "name": "coat",
    "threshold": 4.0,
    "init": 0.01,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "item_emb_side": "item",
    "train_path": File + "/DataSet/coat/train.csv",
    "random_path": File + "/DataSet/coat/random.csv",
    'dense_layer_dim': 512,
    'dense_layer_num': 3,
    "user_feature_label": File + "/DataSet/coat/user_feat_onehot.csv",
    "user_feature_label_no": File + "/DataSet/coat/user_feat_label.csv",
    "item_feature_label": File + "/DataSet/coat/item_feat_onehot.csv",
    "item_popular": File + "/DataSet/coat/item_counts.csv",
    "user_feature_dim": [2, 6, 3, 3,],
    "Weight_path": File + "/Weight/coat_weight/",
    "vae_path": File + "/Weight/vae/coat_weight/",
    "ivae_path": File + "/Weight/ivae/coat_weight/",
    "pop_item_number":23
}

yahoo_params = {
    "name": "yahoo",
    "threshold": 4.0,
    "init": 0.001,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "item_emb_side": "user",
    "train_path": File + "/DataSet/yahoo_R3/train.csv",
    "random_path": File + "/DataSet/yahoo_R3/random.csv",
    "item_popular": File + "/DataSet/yahoo_R3/item_counts.csv",
    "user_feature_label": File + "/DataSet/yahoo_R3/user_feat_onehot.csv",
    "item_feature_label": File + "/DataSet/yahoo_R3/item_features_onehot.csv",
    "user_feature_dim": [5, 5, 5, 5, 5, 5, 5],
    'dense_layer_dim': 1024,
    'dense_layer_num': 3,
    "Weight_path": File + "/Weight/yahoo_weight/",
}

kuai_rand_params = {
    "name": "kuai_rand",
    "threshold": 0.9,
    "init": 0.001,
    "test_ratio": 0.7,
    "train_ratio": 0.8,
    "item_emb_side": "user",
    "train_path": File + "/DataSet/kuai_rand/train.csv",
    "random_path": File + "/DataSet/kuai_rand/random.csv",
    'dense_layer_dim': 512,
    'dense_layer_num': 3,
    "user_feature_label": File + "/DataSet/kuai_rand/user_feat_onehot.csv",
    "item_feature_label": File + "/DataSet/kuai_rand/item_feat_onehot_base.csv",
    "user_feature_label_no": File + "/DataSet/kuai_rand/user_feat_label.csv",
    "vae_path": File + "/Weight/vae/coat_weight/",
    "ivae_path": File + "/Weight/ivae/coat_weight/",
    "item_popular": File + "/DataSet/kuai_rand/item_counts.csv",
    "user_feature_dim": [9, 2, 2, 8, 9, 7, 8, 2, 7, 50, 3, 7, 5, 4],
    "Weight_path": File + "/Weight/kuai_rand_weight/",
    "pop_item_number":241
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=str, default="coat")

    args = parser.parse_args()
    data_params = None
    if args.dataset == "yahoo":
        data_params = yahoo_params
    elif args.dataset == "coat":
        data_params = coat_params
    elif args.dataset == "kuai_rand":
        data_params = kuai_rand_params
    elif args.dataset == "sim_data":
        data_params = sim_data
    else:
        raise Exception("invalid dataset")

    setattr(args, "data_params", data_params)
    return args
