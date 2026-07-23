import Argparser
import itertools
import torch.nn as nn
from Utils.Utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from Utils.Dataloader import PopMFRatingDataset
from Utils.Utils import *
from Utils.BaseModel import *
from Utils.BaseModel import DDR, AffineTransform, RealNVP
device = "cuda" if torch.cuda.is_available() else "cpu"

def Cal(uids, iids, predicts, labels, k=5):
    predict_matrix = -np.inf * np.ones((max(uids) + 1, max(iids) + 1))
    predict_matrix[uids, iids] = predicts
    label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids)))).toarray()
    ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=k).mean().item()
    recall = Recall_at_k_batch(predict_matrix, label_matrix, k=k).mean()
    return ndcg, recall

def Evaluate(data_loader, test_model, user_ivae_mean, item_ivae_mean, real_nvp_s0, real_nvp_s1, P_Oeq1, P_Oeq0, OR_tilde, device="cpu", k=5, val=False):
    test_model.eval()
    with torch.no_grad():
        uids, iids, predicts, labels = list(), list(), list(), list()
        for index, (uid, iid, rating) in enumerate(data_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            User_shadow = user_ivae_mean[uid.type(torch.long)]
            Item_Rep = item_ivae_mean[iid.type(torch.long)]
            predict = test_model(uid, iid, User_shadow, Item_Rep).view(-1)
            if val == True:
                Ratio = torch.exp(real_nvp_s1.log_prob(User_shadow)) / torch.exp(real_nvp_s0.log_prob(User_shadow))
                predict = predict / (P_Oeq1 + OR_tilde[1] * Ratio * P_Oeq0)
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

    user_ivae_mean = torch.load(data_params["Weight_path"] + "user_mean.pt", map_location='cpu', weights_only=False).to(device)
    item_ivae_mean = torch.load(data_params["Weight_path"] + "item_mean.pt", map_location='cpu', weights_only=False).to(device)

    item_popular_raw = pd.read_csv(data_params["item_popular"])
    counts = item_popular_raw['count'].values
    max_count = np.max(counts)
    normalized_counts = 1 - counts.astype(np.float32) / max_count
    popularity_weight = torch.tensor(normalized_counts, dtype=torch.float32).to(device)

    # n_users = test_mat[:, 0].astype(int).max() + 1
    # n_items = test_mat[:, 1].astype(int).max() + 1

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
            User_shadow = user_ivae_mean[uid.type(torch.long)]
            Item_Rep = item_ivae_mean[iid.type(torch.long)]
            predict = model(uid, iid, User_shadow, Item_Rep).view(-1)
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
    train_loader, val_loader, test_loader, n_users, n_items, P_Oeq1, P_Oeq0, y_unique, OR_tilde = construct_or_dataloader(config, device)

    P_Oeq1, P_Oeq0, y_unique, OR_tilde = torch.Tensor(P_Oeq1).to(device), torch.Tensor(P_Oeq0).to(device), torch.Tensor(y_unique).to(device), torch.Tensor(OR_tilde).to(device)

    user_ivae_mean = torch.load(data_params["Weight_path"] + "user_mean.pt", map_location='cpu', weights_only=False).to(device)
    item_ivae_mean = torch.load(data_params["Weight_path"] + "item_mean.pt", map_location='cpu', weights_only=False).to(device)
    # item_ivae_mean = torch.load(data_params["Shadow_path"] + "item_mean_show_1.pt", map_location='cpu', weights_only=False).to(device)

    # item_show = torch.load(data_params["Shadow_path"] + "item_mean_show_1.pt", map_location='cpu', weights_only=False)

    item_popular_raw = pd.read_csv(data_params["item_popular"])
    counts = item_popular_raw['count'].values
    max_count = np.max(counts)
    normalized_counts = np.power(1 - counts.astype(np.float32) / max_count, 9)
    popularity_weight = torch.tensor(normalized_counts, dtype=torch.float32).to(device)

    # item_popular_raw['label'] = 0
    # item_popular_raw.loc[pop_index, 'label'] = 1

    # plt.figure(figsize=(12, 12))
    # plt.scatter(item_show[:, 0], item_show[:, 1], c=item_popular_raw['label'], cmap=ListedColormap(['tab:orange','tab:green']))
    # plt.show()

    inp_size = user_ivae_mean.shape[1]
    layer_dim = data_params['dense_layer_dim']
    layer_num = data_params['dense_layer_num']
    real_nvp_s1 = RealNVP([AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)]).to(device)

    real_nvp_s0 = RealNVP([AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device),
                           AffineTransform("right", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim*2, device=device),
                           AffineTransform("left", input_size=inp_size, layer_num=layer_num, layer_dim=layer_dim, device=device)]).to(device)

    real_nvp_s1.load_state_dict(torch.load(data_params["Weight_path"] + "Best_S1_UDensety_Estimator.pt", weights_only=False))
    real_nvp_s0.load_state_dict(torch.load(data_params["Weight_path"] + "Best_S0_UDensety_Estimator.pt", weights_only=False))

    model = DDR(n_users, n_items, config["embedding_dim"], U_Shadow_size=user_ivae_mean.shape[1], I_Rep_size = item_ivae_mean.shape[1], item_emb_side = data_params["item_emb_side"], popularity_weight = popularity_weight, init=data_params["init"], device=device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    mf_loss_function = nn.MSELoss()

    patience = config["patience"]
    best_val = (0, 0)
    test_performance = (0, 0)
    patience_count = 0
    real_nvp_s1.eval()
    real_nvp_s0.eval()
    for epoch in range(config["epochs"]):
        total_loss = 0
        total_len = 0
        model.train()
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(device), iid.to(device), rating.float().to(device)
            User_shadow = user_ivae_mean[uid.type(torch.long)]
            Item_Rep = item_ivae_mean[iid.type(torch.long)]
            preds = model(uid, iid, User_shadow, Item_Rep)
            Ratio = torch.exp(real_nvp_s1.log_prob(User_shadow) - real_nvp_s0.log_prob(User_shadow))
            Adjustment = (P_Oeq1 + OR_tilde[1] * Ratio * P_Oeq0)
            unbias_preds = preds / Adjustment
            # unbias_preds = preds
            loss = mf_loss_function(rating, unbias_preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rating)
            total_len += len(rating)

        train_loss = total_loss / total_len

        model.eval()
        validation_performance = Evaluate(val_loader, model, user_ivae_mean, item_ivae_mean, real_nvp_s0, real_nvp_s1, P_Oeq1, P_Oeq0, OR_tilde, val=True, device=device)
        test = Evaluate(test_loader, model, user_ivae_mean, item_ivae_mean, real_nvp_s0, P_Oeq1, P_Oeq0, OR_tilde, real_nvp_s1, device=device)
        if config['show_log']:
            print(train_loss, validation_performance, test)
        patience_count += 1
        if validation_performance[0] > best_val[0]:
            patience_count = 0
            best_val = validation_performance
            test_performance = test
            torch.save(model.state_dict(), "Best_DDR_Rec.pt")

        if patience_count > patience:
            if config['show_log']:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    print("best val performance = {0}, test performance is {1}".format(best_val, test_performance))
    model.load_state_dict(torch.load("Best_DDR_Rec.pt"))
    return list(best_val), list(test_performance), model

# lisT = []
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
                'embedding_dim': [128, 256, 512]}
        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [0, 0]
        random_seed = random.randint(1000, 2000)
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
            _, result, model = train_eval(Train_eval_config)
            if result[0] > best_val[0]:
                best_val = result
                best_param = supparams


    elif args.test:
        test_param = None
        if args.data_params["name"] == 'coat':
            test_param = {'lr': 0.0005, 'weight_decay': 1e-06, 'embedding_dim': 256}
        elif args.data_params["name"] == 'yahoo':
            # test_param = {'lr': 5e-05, 'weight_decay': 1e-05, 'embedding_dim': 512}
            test_param = {'lr': 1e-05, 'weight_decay': 1e-05, 'embedding_dim': 256}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-06, 'weight_decay': 1e-05, 'embedding_dim': 1024}
        elif args.data_params["name"] == 'sim_data':
            test_param = {'lr': 0.00005, 'weight_decay': 1e-05, 'embedding_dim': 512}
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

        pop_mean = np.mean(pop_list, 0)
        pop_std = np.std(pop_list, 0)

        print("{} ± {}  {} ± {}".format(mean[0], std[0], mean[1], std[1]))
        print("{} ± {}  {} ± {} {} ± {} {} ± {}".format(pop_mean[0], pop_std[0], pop_mean[1], pop_std[1], pop_mean[2], pop_std[2], pop_mean[3], pop_std[3]))
    else:
        Train_eval_config = {
            "tune": False,
            "seed": 1208,
            "epochs": 120,
            "show_log": True,
            "patience": 5,
            "lr": 0.001,
            "weight_decay": 1e-5,
            "embedding_dim": 512,
            "batch_size": 512,
            "data_params": args.data_params,
        }
        _, result= train_eval(Train_eval_config)
        print(result)
