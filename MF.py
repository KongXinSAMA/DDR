import Argparser
import itertools
import torch.nn as nn
from Utils.Utils import *
from Utils.BaseModel import MF
from Utils.Dataloader import PopMFRatingDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def Cal(uids, iids, predicts, labels, k=10):
    predict_matrix = -np.inf * np.ones((max(uids) + 1, max(iids) + 1))
    predict_matrix[uids, iids] = predicts
    label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids)))).toarray()
    ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=k).mean().item()
    recall = Recall_at_k_batch(predict_matrix, label_matrix, k=k).mean()
    return ndcg, recall

def Evaluate(data_loader, test_model, device="cpu", k=10):
    test_model.eval()
    with torch.no_grad():
        uids, iids, predicts, labels = list(), list(), list(), list()
        for index, (uid, iid, rating) in enumerate(data_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            predict = test_model(uid, iid).view(-1)
            uids.extend(uid.cpu())
            iids.extend(iid.cpu())
            predicts.extend(predict.cpu())
            labels.extend(rating.cpu())
        predict_matrix = -np.inf * np.ones((max(uids) + 1, max(iids) + 1))
        predict_matrix[uids, iids] = predicts
        label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids)))).toarray()
        ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=k).mean().item()
        recall = Recall_at_k_batch(predict_matrix, label_matrix, k=k).mean()
        return ndcg, recall

def Pop_eva(config, model):
    data_params = config["data_params"]
    _, _, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])

    threshold = data_params["threshold"]

    test_mat[:, 2] = test_mat[:, 2] >= threshold

    print("test size:", test_mat.shape[0])


    item_popular_raw = pd.read_csv(data_params["item_popular"])
    item_popular = torch.tensor(item_popular_raw['count'].values, dtype=torch.int16)

    test_dataset = PopMFRatingDataset(test_mat[:, 0].astype(int),
                                   test_mat[:, 1].astype(int),
                                   test_mat[:, 2],
                                   item_popular)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=4,
                             pin_memory=True)

    model.eval()
    with torch.no_grad():
        pop_uids, pop_iids, pop_predicts, pop_labels, pops = list(), list(), list(), list(), list()
        uids, iids, predicts, labels = list(), list(), list(), list()

        for index, (uid, iid, rating, pop) in enumerate(test_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            predict = model(uid, iid).view(-1)
            # predict = model(uid, iid).view(-1)
            mask = (pop >= data_params["pop_item_number"])

            uids.extend(uid[~mask].cpu())
            iids.extend(iid[~mask].cpu())
            predicts.extend(predict[~mask].cpu())
            labels.extend(rating[~mask].cpu())

            pop_uids.extend(uid[mask].cpu())
            pop_iids.extend(iid[mask].cpu())
            pop_predicts.extend(predict[mask].cpu())
            pop_labels.extend(rating[mask].cpu())

        print("pop items:", len(pop_predicts), "unpop items:", len(predicts))

    pop_ndcg, pop_recall = Cal(pop_uids, pop_iids, pop_predicts, pop_labels)
    ndcg, recall = Cal(uids, iids, predicts, labels)


    print("pop_ndcg:{:.4f}, pop_recall:{:.4f}, ndcg:{:.4f}, recall:{:.4f}".format(pop_ndcg, pop_recall, ndcg, recall))

    return [pop_ndcg, pop_recall, ndcg, recall]

def train_eval(config):
    seed_everything(config['seed'])
    data_params = config["data_params"]
    train_loader, val_loader, test_loader, n_users, n_items = construct_mf_dataloader(config)

    item_popular_raw = pd.read_csv(data_params["item_popular"])
    item_popular = torch.tensor(item_popular_raw['count'].values, dtype=torch.int16) >= 23

    model = MF(n_users, n_items, config["embedding_dim"]).to(device)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    mf_loss_function = nn.MSELoss()

    best_val = (0, 0)
    patience_count = 0
    patience = config["patience"]
    test_performance = (0, 0)

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            preds = model(uid, iid).view(-1)
            loss = mf_loss_function(preds, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)
            total_len += len(rating)

        train_loss = total_loss / total_len
        model.eval()
        validation_performance = Evaluate(val_loader, model, device=device)
        test = Evaluate(test_loader, model, device=device)
        if config['show_log']:
            print(train_loss, validation_performance, test)
        patience_count += 1
        if validation_performance[0] > best_val[0]:
            patience_count = 0
            best_val = validation_performance
            test_performance = test
            torch.save(model.state_dict(), "Best_MF_Rec.pt")

        if patience_count > patience:
            if config['show_log']:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    print("best val performance = {0}, test performance is {1}".format(best_val, test_performance))
    model.load_state_dict(torch.load("Best_MF_Rec.pt"))
    # model.get_item_em(item_popular)
    return list(best_val), list(test_performance), model


if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        if args.data_params["name"] == "kuai_rand":
            search_grid = {'lr': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6], 'weight_decay': [1e-5, 1e-6],
                           'embedding_dim': [512, 1024]}
        else:
            search_grid = {'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 'weight_decay': [1e-5, 1e-6],
                'embedding_dim': [256, 512]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
        print("tune seed:", random_seed)
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            Train_eval_config = {
                "tune": True,
                "seed": random_seed,
                "epochs": 120,
                "show_log": False,
                "patience": 5,
                "lr": supparams['lr'],
                "weight_decay": supparams['weight_decay'],
                "embedding_dim": supparams['embedding_dim'],
                "batch_size": 512,
                "data_params": args.data_params,
            }
            print("index:", index, "index:", supparams)
            _, result, _ = train_eval(Train_eval_config)
            if result[0] > best_val[0]:
                best_val = result
                best_param = supparams
        print(best_val, best_param)
    elif args.test:
        test_param = None
        if args.data_params["name"] == 'coat':
            test_param = {'lr': 0.0001, 'weight_decay': 1e-05, 'embedding_dim': 512}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 5e-04, 'weight_decay': 1e-05, 'embedding_dim': 512}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-05, 'weight_decay': 1e-06, 'embedding_dim': 512}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 0.0001, 'weight_decay': 1e-05, 'embedding_dim': 256}
        lisT = []
        pop_list = []
        for i in range(10):
            random_seed = random.randint(1000, 2000)
            Train_eval_config = {
                "tune": False,
                "seed": random_seed,
                "epochs": 120,
                "show_log": True,
                "patience": 5,
                "lr": test_param['lr'],
                "weight_decay": test_param['weight_decay'],
                "embedding_dim": test_param['embedding_dim'],
                "batch_size": 512,
                "data_params": args.data_params,
            }
            _, result, model = train_eval(Train_eval_config)
            lisT.append(result)
            pop_Res = Pop_eva(Train_eval_config, model)
            pop_list.append(pop_Res)

        mean = np.mean(lisT, 0)
        std = np.std(lisT, 0)
        print("{} ± {}  {} ± {}".format(mean[0], std[0], mean[1], std[1]))

        pop_mean = np.mean(pop_list, 0)
        pop_std = np.std(pop_list, 0)
        print("{} ± {}  {} ± {} {} ± {} {} ± {}".format(pop_mean[0], pop_std[0], pop_mean[1], pop_std[1], pop_mean[2], pop_std[2], pop_mean[3], pop_std[3]))
    else:
        Train_eval_config = {
            "tune": False,
            "seed": 1208,
            "epochs": 120,
            "show_log": True,
            "patience": 5,
            "lr": 0.0001,
            "weight_decay": 1e-5,
            "embedding_dim": 256,
            "batch_size": 512,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)