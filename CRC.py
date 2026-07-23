import Argparser
import itertools
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from Utils.Utils import *
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from Utils.BaseModel import Item_Effect_Catcher
from Utils.Dataloader import Pop_TrainingDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
from matplotlib.colors import ListedColormap
CUDA_LAUNCH_BLOCKING=1

def iVae_loss_function(x, x_hat, mean, log_var, prior_mean, log_prior_var, anneal=1.):
    reproduction_loss = torch.mean(
        torch.sum(nn.functional.binary_cross_entropy(x_hat, x, reduction="none"), dim=1))

    kld_loss = -0.5 * torch.mean(
        torch.sum(
            1 + log_var - log_prior_var - ((mean - prior_mean).pow(2) + log_var.exp()) / log_prior_var.exp(),
            dim=1)
    )

    return reproduction_loss + kld_loss * anneal

def InfoNCE_i(view1, view2, view3, temperature, gama):
    view1, view2, view3 = torch.nn.functional.normalize(
        view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score_1 = torch.exp(ttl_score_1 / temperature)

    # 创建对角线掩码（batch_size x batch_size）
    mask = torch.eye(view1.size(0), dtype=torch.bool, device=view1.device)
    # 将对角线元素（自身相似度）置零
    ttl_score_1 = ttl_score_1.masked_fill(mask, 0)
    # 对每行求和（排除自身）
    ttl_score_1 = ttl_score_1.sum(dim=1)

    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
    ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / (gama * ttl_score_2+ttl_score_1+pos_score))

    # cl_loss = -torch.log(pos_score / (gama * ttl_score_2 + pos_score))
    return torch.mean(cl_loss)

def similar_test(config):

    data_params = config["data_params"]

    crc_rep = torch.load(data_params["Weight_path"] + "item_mean.pt", map_location='cpu',
                                    weights_only=False).to(device)

    ivae_rep = torch.load("ivae_item_mean.pt", map_location='cpu',
                                       weights_only=False).to(device)

    item_popular_raw = pd.read_csv(data_params["item_popular"])

    item_popular = torch.tensor(item_popular_raw['count'].values, dtype=torch.int16)

    pop_mask = (item_popular >= 241)

    # 反事实表征
    pop_crc_rep = crc_rep[pop_mask]

    unpop_crc_rep = crc_rep[~pop_mask]
    # 基线表征
    pop_ivae_rep = ivae_rep[pop_mask]

    unpop_ivae_rep = ivae_rep[~pop_mask]

    # ===================== 全局分布对比函数 =====================
    def global_dist_analysis(pop_feat, unpop_feat):
        """
        计算冷热物品全局分布相似性指标
        :param pop_feat: 热门物品表征 [N_pop, D]
        :param unpop_feat: 冷门物品表征 [N_unpop, D]
        :return: 各类分布指标字典
        """
        with torch.no_grad():
            # 1. 全局均值（每一维特征的均值）
            pop_mean = torch.mean(pop_feat, dim=0)  # [D]
            unpop_mean = torch.mean(unpop_feat, dim=0)  # [D]

            # 2. 全局方差（衡量分布离散程度）
            pop_var = torch.var(pop_feat, dim=0)
            unpop_var = torch.var(unpop_feat, dim=0)

            # 3. 均值欧式距离（核心指标：两组整体向量的距离，越小越相似）
            mean_eu_dist = torch.norm(pop_mean - unpop_mean, p=2).item()

            # 4. 均值余弦相似度（衡量两组整体语义方向，越大越相似）
            mean_cos_sim = F.cosine_similarity(pop_mean, unpop_mean, dim=0).item()

            # 5. 展平特征，计算Wasserstein距离（衡量整体分布差异，越小分布越接近）
            # 对每一维分别计算Wasserstein再取平均
            from scipy.stats import wasserstein_distance
            D = pop_feat.shape[-1]
            wd_list = []
            for dim in range(D):
                pop_dim = pop_feat[:, dim].cpu().numpy()
                unpop_dim = unpop_feat[:, dim].cpu().numpy()
                wd = wasserstein_distance(pop_dim, unpop_dim)
                wd_list.append(wd)
            avg_wasserstein = np.mean(wd_list)

            # 6. 全局特征均值差、方差差（辅助参考）
            mean_diff = torch.mean(torch.abs(pop_mean - unpop_mean)).item()
            var_diff = torch.mean(torch.abs(pop_var - unpop_var)).item()

        return {
            "mean_eu_distance": round(mean_eu_dist, 4),  # 均值欧式距离
            "mean_cosine_similarity": round(mean_cos_sim, 4),  # 均值余弦相似度
            "avg_wasserstein": round(avg_wasserstein, 4),  # 平均Wasserstein距离
            "mean_abs_diff": round(mean_diff, 4),  # 特征均值绝对差
            "var_abs_diff": round(var_diff, 4)  # 特征方差绝对差
        }

    # ===================== 计算两组模型的全局分布指标 =====================
    print("=" * 70)
    print("【1. 基线表征 (crc_rep) 热门 & 冷门 全局分布指标】")
    ivae_dist = global_dist_analysis(pop_ivae_rep, unpop_ivae_rep)
    for k, v in ivae_dist.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 70)
    print("【2. 反事实表征 (ivae_rep) 热门 & 冷门 全局分布指标】")
    crc_dist = global_dist_analysis(pop_crc_rep, unpop_crc_rep)
    for k, v in crc_dist.items():
        print(f"{k}: {v}")

    # ===================== 指标对比总结 =====================
    print("\n" + "=" * 70)
    print("【全局分布对比总结】")
    print(f"均值欧式距离 | 基线: {ivae_dist['mean_eu_distance']} | 反事实: {crc_dist['mean_eu_distance']} (越小越好)")
    print(f"均值余弦相似度 | 基线: {ivae_dist['mean_cosine_similarity']} | 反事实: {crc_dist['mean_cosine_similarity']} (越大越好)")
    print(f"平均Wasserstein距离 | 基线: {ivae_dist['avg_wasserstein']} | 反事实: {crc_dist['avg_wasserstein']} (越小越好)")
    print("=" * 70)

    # ===================== 可选：可视化 特征均值对比图 =====================
    def plot_feature_mean(pop_mean, unpop_mean, title_name):
        plt.figure(figsize=(10, 4))
        dim = np.arange(len(pop_mean.cpu().numpy()))
        plt.plot(dim, pop_mean.cpu().numpy(), label='Popular Items', linewidth=1.2)
        plt.plot(dim, unpop_mean.cpu().numpy(), label='Unpopular Items', linewidth=1.2)
        plt.xlabel("Feature Dimension")
        plt.ylabel("Feature Mean Value")
        plt.title(title_name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{title_name}.png", dpi=300, bbox_inches="tight")
        plt.show()

    # 绘制基线、反事实的特征均值分布图
    plot_feature_mean(torch.mean(pop_crc_rep, dim=0), torch.mean(unpop_crc_rep, dim=0),
                      "Baseline(crc_rep) Feature Mean")
    plot_feature_mean(torch.mean(pop_ivae_rep, dim=0), torch.mean(unpop_ivae_rep, dim=0),
                      "Counterfactual(ivae_rep) Feature Mean")

    # 返回所有结果，便于保存日志/后续分析
    return {
        "crc_global_dist": crc_dist,
        "ivae_global_dist": ivae_dist
    }


def item_rep_train_eval(config):
    seed_everything(config["seed"])
    # Read Data
    data_params = config["data_params"]
    fullmatrix, train_matrix, val_matrix, train_item_index, val_item_index = construct_item_dataset(
                                                                                data_params["train_path"],
                                                                                train_ratio=data_params["train_ratio"])

    item_feat = pd.read_csv(data_params["item_feature_label"]).to_numpy().astype(float)
    item_feat = torch.tensor(item_feat, dtype=torch.float).to(device)
    train_Y_data = torch.tensor(train_matrix >= data_params['threshold'] + 1, dtype=torch.float).to(device)
    train_data = torch.tensor(train_matrix > 0, dtype=torch.float).to(device)
    val_Y_data = torch.tensor(val_matrix >= data_params['threshold'] + 1, dtype=torch.float).to(device)
    val_data = torch.tensor(val_matrix > 0, dtype=torch.float).to(device)

    item_popular_raw = pd.read_csv(data_params["item_popular"])

    val_item_feat = item_feat[val_item_index]

    item_popular = torch.tensor(item_popular_raw['count'].values, dtype=torch.int16)

    pop_train_dataset = Pop_TrainingDataset(Y_data=train_Y_data, O_data=train_data,
                                             Feature=item_feat[train_item_index], PopValue=item_popular[train_item_index])

    pop_train_dataloader = DataLoader(pop_train_dataset, batch_size=config["batch_size"], shuffle=True)

    gl_pop_train_dataset = Pop_TrainingDataset(Y_data=train_Y_data, O_data=train_data,
                                             Feature=item_feat[train_item_index], PopValue=item_popular[train_item_index])

    gl_pop_train_dataloader = DataLoader(gl_pop_train_dataset, batch_size=config["batch_size"], shuffle=False)

    global_p = []
    for y, o, x, p in gl_pop_train_dataloader:
        global_p.append(p)
    # 全局表征
    global_p = torch.cat(global_p, dim=0)
    # 排序划分
    sorted_p, _ = torch.sort(global_p, descending=True)
    threshold = sorted_p[int(len(sorted_p) * config["popular_rate"])]
    gl_pop_mask = global_p >= threshold

    val_p = item_popular[val_item_index]
    val_pop_mask = val_p >= threshold


    # Creat model
    layer_dim = config["i_layer_dim"]
    rep_dim = config["i_rep_dim"]
    item_feat_dim = item_feat.shape[1]
    input_size = train_matrix.shape[1]
    prior_mean = True if data_params['name'] == 'sim_data' else False
    model = Item_Effect_Catcher(input_dim=input_size,
                 feature_dim=item_feat_dim,
                 effect_dim=rep_dim,
                 layer_dim=layer_dim,
                 n_layers=config["n_layers"],
                 device=device,
                 prior_mean=prior_mean)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.to(device)

    # Train Preparation
    best_val_loss = [np.inf, np.inf, np.inf, np.inf]

    Mse_loss_list = []
    val_loss_list = []
    training_loss_list = []

    epochs = config["epochs"]
    patience_count = 0
    patience = config["patience"]
    anneal_count = 0
    use_anneal = config["anneal"]
    anneal_max = config["beta_max"]
    total_batches = int(epochs * train_data.shape[0] / config["batch_size"])
    anneal_max_count = int(0.2 * total_batches / anneal_max)
    for epoch in range(epochs):
        model.train()
        total_len = 0
        total_loss = 0
        total_Main_loss = 0

        # 信息捕获
        for y, o, x, p in pop_train_dataloader:

            o_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, pos_mean = model(o, x)


            if use_anneal:
                anneal = min(anneal_max, 1. * anneal_count / anneal_max_count)
            else:
                anneal = anneal_max

            l2_reg = torch.tensor([0]).to(device).float()
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param)

            Vae_loss, Y_z_loss = iVae_loss_function(o, o_hat, mean, log_var, prior_mean, prior_log_var, anneal),\
                                         torch.mean(torch.sum(F.binary_cross_entropy(Y_hat, y, reduction="none"), dim=1))

            # 超参数控制正样本数量
            # 正负样本对比
            # 正负样本区分训练
            # 通过逆概率加权调整正样本的权重，不让正负样本过于区分

            # sorted_p, _ = torch.sort(p, descending=True)
            # threshold = sorted_p[int(len(o) * config["popular_rate"])]
            pop_mask = p >= threshold
            pop_items = mean[pop_mask]
            pop_pos_mean = pos_mean[pop_mask]
            unpop_items = mean[~pop_mask]
            unpop_pos_mean = pos_mean[~pop_mask]

            item_cl_pop = InfoNCE_i(pop_items, pop_pos_mean,
                                                         unpop_pos_mean, 0.2, config["cl_pop_item_weight"])

            item_cl_unpop = InfoNCE_i(unpop_items, unpop_pos_mean,
                                                                 pop_pos_mean, 0.2, config["cl_pop_item_weight"])



            item_cl_loss = (item_cl_pop + item_cl_unpop) * config["unpop_item_weight"]
            item_cl_loss = 0
            loss = Vae_loss + Y_z_loss + item_cl_loss + l2_reg * config["l2_penalty"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            anneal_count += 1
            total_len += len(x)
            total_loss += loss.item() * len(x)
            total_Main_loss += item_cl_pop.item() * len(x)

        # 全局一致性监督
        global_mean = []
        for y, o, x, _ in gl_pop_train_dataloader:
            _, mean, _, _, _, _, _ = model(o, x)
            global_mean.append(mean)
        global_mean = torch.cat(global_mean, dim=0)
        pop_items = global_mean[gl_pop_mask]
        unpop_items = global_mean[~gl_pop_mask]

        # 模型l2或者表征l2
        l1_loss = torch.norm(global_mean, p=2)


        align_loss = (pop_items[torch.mm(F.normalize(unpop_items, dim=-1), F.normalize(pop_items, dim=-1).T).argmax(
            dim=1)] - unpop_items).norm(p=2, dim=1).pow(2).mean()

        loss = align_loss * config["align_rate"] + l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val, Y_hat, val_pos_mean = model(val_data, val_item_feat)

        val_pop_pos_mean = val_pos_mean[val_pop_mask]
        val_unpop_items = mean_val[~val_pop_mask]
        val_unpop_pos_mean = val_pos_mean[~val_pop_mask]

        val_item_cl_unpop = InfoNCE_i(val_unpop_items, val_unpop_pos_mean,
                                  val_pop_pos_mean, 0.2, config["cl_pop_item_weight"])

        val_Y_loss = torch.mean(torch.sum(nn.functional.binary_cross_entropy(Y_hat, val_Y_data, reduction="none"), dim=1)).detach().item()

        val_ivae_loss = iVae_loss_function(val_data, test_o_hat, mean_val, log_var_val, prior_mean_val, prior_log_var_val, anneal=anneal_max).detach().item()

        val_loss = val_item_cl_unpop

        patience_count += 1

        if val_loss < best_val_loss[0]:
            patience_count = 0
            best_val_loss = [val_loss, val_Y_loss, val_ivae_loss, align_loss]
            # print(best_val_loss)
            torch.save(model.state_dict(), data_params["Weight_path"] + "{}_{}_Best_I_Effect_Model.pt".format(data_params["name"], "Val"))

        if patience_count > patience:
            # print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

        if config["show_log"]:
            training_loss = total_loss / total_len
            val_loss_list.append(val_loss.item())
            training_loss_list.append(training_loss)
            Mse_loss_list.append(total_Main_loss / total_len)
            print("Epoch {}, Training Loss = {}, Val uncl = {}, Val Loss = {}, Align Loss = {}".format(epoch, training_loss, val_item_cl_unpop, val_Y_loss, align_loss))

    if config["show_log"]:
        plt.plot(training_loss_list, label="Training Loss")
        plt.title("iVAE")
        plt.legend()
        plt.show()
        plt.plot(val_loss_list, label="MSE loss")
        plt.title("Unpop_cl loss")
        plt.legend()
        plt.show()


    print("Best Unpop_loss = {}, val_Y_loss = {}, Ivae_Val loss = {}, Align Loss = {}".format(best_val_loss[0], best_val_loss[1], best_val_loss[2], best_val_loss[3]))
    if config['save_mean']:
        # save to local
        model.load_state_dict(torch.load(data_params["Weight_path"] + "{}_{}_Best_I_Effect_Model.pt".format(data_params["name"], "Val"), weights_only=False))
        fullmatrix = torch.tensor(fullmatrix > 0, dtype=torch.float).to(device)
        O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, pos_mean = model(fullmatrix, item_feat)
        torch.save(mean.detach(), data_params["Weight_path"] + "item_mean.pt")
        # torch.save(log_var.detach().exp().sqrt(), data_params["Weight_path"] + "item_std_show_1.pt")
        print('Already Save Prama')

        if config["i_rep_dim"] == 2:
            data_params = config["data_params"]

            mean = mean.cpu().detach().numpy()

            item_popular_raw = pd.read_csv(data_params["item_popular"])

            pop_index = item_popular_raw['count'].nlargest(int(len(item_popular_raw) * config["popular_rate"])).index
            item_popular_raw['label'] = 0
            item_popular_raw.loc[pop_index, 'label'] = 1

            plt.figure(figsize=(18, 12))
            plt.scatter(mean[:, 0], mean[:, 1], c=item_popular_raw['label'],
                        cmap=ListedColormap(['tab:blue', 'tab:red']), s=128)
            plt.xlim(-2.4, 2.4)
            plt.ylim(-2.8, 2.4)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(r"C:\Users\Administrator\Desktop\resu\CRC.png", bbox_inches='tight')
            plt.show()

    return best_val_loss


if __name__ == '__main__':
    args = Argparser.parse_args()
    print("Dataset:", args.data_params["name"])
    if args.tune:
        print("--tune model--")
        i_layer_dim_search = []
        i_shadow_dim_search = []
        popular_rate = []
        if args.data_params["name"] == 'coat':
            i_layer_dim_search = [512, 1024]
            i_shadow_dim_search = [8, 16]
            popular_rate = [0.39]
        elif args.data_params["name"] == 'yahoo':
            i_layer_dim_search = [512, 1024]
            i_shadow_dim_search = [16, 32]
            popular_rate = [0.35]
        elif args.data_params["name"] == 'kuai_rand':
            i_layer_dim_search = [512, 1024]
            i_shadow_dim_search = [8]
            popular_rate = [0.22]
            # 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
            #  500, 1000
        search_grid = {'lr': [5e-4, 1e-4], 'weight_decay': [1e-4, 1e-5], 'l2_penalty': [1e-4, 1e-5], 'align_rate': [10, 100, 200],
                "popular_rate": popular_rate,  "cl_pop_item_weight": [0.5, 0.4, 0.3, 0.2], "unpop_item_weight": [10, 100, 200],
                'i_layer_dim': i_layer_dim_search, 'i_rep_dim': i_shadow_dim_search}

        combinations = list(itertools.product(*search_grid.values()))
        best_param, best_val = {}, [np.inf, np.inf]
        for index, combination in enumerate(combinations):
            supparams = dict(zip(search_grid.keys(), combination))
            print("index:", index, "params:", supparams)
            grid_config = {
                "seed": 1208,
                "save_mean": False,
                "show_log": False,
                "anneal": True,
                "beta_max": 1.0,
                "patience": 10,
                "epochs": 1000,
                "popular_rate": supparams['popular_rate'],
                "align_rate": supparams['align_rate'],
                "unpop_item_weight":supparams['unpop_item_weight'],
                "cl_pop_item_weight": supparams['cl_pop_item_weight'],
                "lr": supparams['lr'],
                "l2_penalty": supparams['l2_penalty'],
                "n_layers": 3,
                "i_layer_dim": supparams['i_layer_dim'],
                "i_rep_dim": supparams['i_rep_dim'],
                "batch_size": 512,
                "weight_decay": supparams['weight_decay'],
                "data_params": args.data_params,
            }
            result = item_rep_train_eval(grid_config)
            if result[0] < best_val[0]:
                best_val = result
                best_param = supparams
        print("best val is:", best_val, "params is:", best_param)
    elif args.test:
        test_param = None
        if args.data_params["name"] == 'coat':
            # test_param = {'lr': 0.001, 'weight_decay': 0.0001, 'l2_penalty': 0.001, 'align_rate': 10, 'popular_rate': 0.39,
            #  'cl_pop_item_weight': 0.8, 'unpop_item_weight': 10, 'i_layer_dim': 512, 'i_rep_dim': 16}
            # test_param = {'lr': 0.001, 'weight_decay': 1e-06, 'l2_penalty': 0.0001, 'align_rate': 100, 'popular_rate': 0.39, 'cl_pop_item_weight': 0.2, 'unpop_item_weight': 200, 'i_layer_dim': 512, 'i_rep_dim': 16}
            test_param = {'lr': 0.001, 'weight_decay': 1e-06, 'l2_penalty': 0.0001, 'align_rate': 100,
                          'popular_rate': 0.39, 'cl_pop_item_weight': 0.6, 'unpop_item_weight': 200, 'i_layer_dim': 512,
                          'i_rep_dim': 2}
            # test_param =  {'lr': 0.0005, 'weight_decay': 0.0001, 'l2_penalty': 0.0001, 'align_rate': 200, 'popular_rate': 0.39, 'cl_pop_item_weight': 0.7, 'unpop_item_weight': 10, 'i_layer_dim': 512, 'i_rep_dim': 8}
        elif args.data_params["name"] == 'yahoo':
            test_param = {'lr': 0.0005, 'weight_decay': 1e-05, 'l2_penalty': 0.0001, 'align_rate': 200, 'popular_rate': 0.33, 'cl_pop_item_weight': 0.3, 'unpop_item_weight': 200, 'i_layer_dim': 1024, 'i_rep_dim': 32}
        elif args.data_params["name"] == 'kuai_rand':
            test_param = {'lr': 5e-05, 'weight_decay': 1e-05, 'l2_penalty': 0.001, 'align_rate': 10, 'popular_rate': 0.22, 'cl_pop_item_weight': 0.3, 'unpop_item_weight': 100, 'i_layer_dim': 1024, 'i_rep_dim': 8}
        lisT = []
        Train_eval_config = {
            "seed": 1208,
            "save_mean": True,
            "show_log": False,
            "anneal": True,
            "tune": False,
            "beta_max": 1.0,
            "epochs": 1000,
            "patience": 10,
            "n_layers": 3,
            "popular_rate": test_param['popular_rate'],
            "align_rate": test_param['align_rate'],
            "unpop_item_weight":test_param['unpop_item_weight'],
            "cl_pop_item_weight": test_param['cl_pop_item_weight'],
            "lr": test_param['lr'],
            "weight_decay": test_param['weight_decay'],
            "i_rep_dim": test_param['i_rep_dim'],
            "i_layer_dim": test_param['i_layer_dim'],
            "l2_penalty": test_param['l2_penalty'],
            "batch_size": 512,
            "data_params": args.data_params,
        }
        item_rep_train_eval(Train_eval_config)
    else:
        Train_eval_config = {
            "seed": 1208,
            "save_mean": True,
            "show_log": False,
            "anneal": True,
            "beta_max": 1.0,
            "patience": 10,
            "epochs": 1000,
            "popular_rate": 0.3,
            "align_rate": 50,
            "unpop_item_weight":5,
            "cl_pop_item_weight": 0.4,
            "beta": 0.5,
            "lr": 0.001,
            "l2_penalty": 1e-03,
            "n_layers": 3,
            "i_layer_dim": 64,
            "i_rep_dim": 8,
            "batch_size": 512,
            "weight_decay": 1e-03,
            "data_params": args.data_params,
        }
        # similar_test(Train_eval_config)
        item_rep_train_eval(Train_eval_config)

