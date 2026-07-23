import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activations = activations

        self.linear_nets = nn.Sequential()
        prev_dim = input_dim
        for i, (hidden_dim, activation) in enumerate(zip(hidden_dims, activations)):
            self.linear_nets.add_module("fc_{}".format(i), nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            if activation == "relu":
                self.linear_nets.add_module("act_{}".format(i), nn.ReLU())
            elif activation == "lrelu":
                self.linear_nets.add_module("act_{}".format(i), nn.LeakyReLU(0.2))
            elif activation == "sigmoid":
                self.linear_nets.add_module("act_{}".format(i), nn.Sigmoid())
            elif activation == "softmax":
                self.linear_nets.add_module("act_{}".format(i), nn.Softmax(dim=1))
            elif activation == "tanh":
                self.linear_nets.add_module("act_{}".format(i), nn.Tanh())
        # 调用初始化函数
        self.initialize_weights()

    def initialize_weights(self):
        # 遍历所有模块并应用初始化
        for m in self.linear_nets:
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.xavier_normal_(m.weight)
                # init.normal_(m.weight, mean=0.0, std=0.001)

    def forward(self, x):
        return self.linear_nets(x)

class ShadowConstructor(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(ShadowConstructor, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

        # Q_Test
        self.Y_By_Z = MLP(input_dim=shadow_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                          activations=[activation] * (n_layers - 1) + ['sigmoid'])

        self.P_Y = MLP(input_dim=input_dim,
                       hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                       activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        # tx = torch.cat((t, x), 1)
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        Y_hat = self.Y_By_Z(z)
        Q = self.P_Y(Y_hat)

        O_hat = self.decode(z, x)
        return O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, Q

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class Effect_Catcher(nn.Module):
    def __init__(self, input_dim, effect_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(Effect_Catcher, self).__init__()
        self.latent_dim = effect_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=self.latent_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

        self.Y_By_Z = MLP(input_dim=self.latent_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                          activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        # tx = torch.cat((t, x), 1)
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        Y_hat = self.Y_By_Z(z)
        O_hat = self.decode(z, x)
        return O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class Item_Effect_Catcher(nn.Module):
    def __init__(self, input_dim, effect_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(Item_Effect_Catcher, self).__init__()
        self.latent_dim = effect_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=self.latent_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

        self.Y_By_Z = MLP(input_dim=self.latent_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                          activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        # tx = torch.cat((t, x), 1)
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # pos_z = []
        # for _ in range(nums):
        #     pos_z.append(self.reparameterization(mean, torch.exp(0.5 * log_var)))
        # pos_z = torch.cat(pos_z, dim=1)

        pos_z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        Y_hat = self.Y_By_Z(z)
        O_hat = self.decode(z, x)

        return O_hat, mean, log_var, prior_mean, prior_log_var, Y_hat, pos_z

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class Item_Effect_Catcher_VAE(nn.Module):
    def __init__(self, input_dim, effect_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu"):
        super(Item_Effect_Catcher_VAE, self).__init__()
        self.latent_dim = effect_dim
        self.device = device


        # encoder
        self.mean_z = MLP(input_dim=input_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [self.latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=self.latent_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

        self.Y_By_Z = MLP(input_dim=self.latent_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                          activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t):
        mean = self.mean_z(t)
        log_var = self.log_var_z(t)
        return mean, log_var

    def decode(self, t):
        return self.decoder(t)

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o):

        mean, log_var = self.encode(o)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        pos_z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        Y_hat = self.Y_By_Z(z)
        O_hat = self.decode(z)

        return O_hat, mean, log_var, Y_hat, pos_z


class Shadow_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, I_Shadow_size, U_Shadow_size, device="cpu"):
        super(Shadow_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 给对撞建模
        self.shadow_u_emb = nn.Embedding(num_users, I_Shadow_size)
        self.shadow_u_emb.weight.data.uniform_(-0.01, 0.01)
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow, ItemShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        UShadows = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        IShadows = (ItemShadow * self.shadow_u_emb(u_id)).sum(1)
        Shadows_Feature = (UShadows + IShadows)
        # Shadows_Feature = UShadows
        # Shadows_Feature = IShadows
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

    def model_debias_loss(self, u_id, i_id, UserShadow, ItemShadow, PCAShadow, real_nvp_s0, real_nvp_s1, y_train, loss_f):
        preds = self.forward(u_id, i_id, UserShadow, ItemShadow).view(-1)

        Ratio = torch.exp(real_nvp_s1.log_prob(PCAShadow)) / torch.exp(real_nvp_s0.log_prob(PCAShadow))
        preds = preds * Ratio
        loss = loss_f(preds, y_train)
        return loss

class MF_Direct(nn.Module):
    def __init__(self, nums, Shadow_size, device="cpu"):
        super(MF_Direct, self).__init__()
        # 给对撞建模
        self.shadow_emb = nn.Embedding(nums, Shadow_size)
        self.shadow_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, id, Shadow):
        Shadows_Feature = (Shadow * self.shadow_emb(id)).sum(1)
        return Shadows_Feature

class U_Shadow_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, device="cpu"):
        super(U_Shadow_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 给对撞建模
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        # self.shadow_i_emb = nn.Embedding(num_users, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        Shadows_Feature = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class SBR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, init, device="cpu"):
        super(SBR, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-init, init)
        self.user_bias.weight.data.uniform_(-init, init)
        self.item_emb.weight.data.uniform_(-init, init)
        self.item_bias.weight.data.uniform_(-init, init)
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-init*0.1, init*0.1)
        self.shadow_u_emb = nn.Embedding(num_users, U_Shadow_size)
        self.shadow_u_emb.weight.data.uniform_(-init*0.1, init*0.1)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        Shadows_Feature = (UserShadow * self.shadow_i_emb(i_id)).sum(1)

        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class DDR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, I_Rep_size, item_emb_side, popularity_weight, init, device="cpu"):
        super(DDR, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-init, init)
        self.user_bias.weight.data.uniform_(-init, init)
        self.item_emb.weight.data.uniform_(-init, init)
        self.popularity_weight = popularity_weight
        self.item_bias.weight.data.uniform_(-init, init)

        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-init*0.1, init*0.1)

        self.item_emb_side = item_emb_side

        self.di_emb = nn.Embedding(num_users, I_Rep_size) if self.item_emb_side == 'user' else nn.Embedding(num_items, I_Rep_size)
        self.di_emb.weight.data.uniform_(-init*0.1, init*0.1)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow, Item_Rep, Train = True):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()

        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        Item_Effect = (Item_Rep * self.di_emb(u_id)).sum(1) if self.item_emb_side == 'user' else (Item_Rep * self.di_emb(i_id)).sum(1)

        Shadows_Effect = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        # Shadows_Effect = 0
        return (U * I).sum(1) + Shadows_Effect + self.popularity_weight[i_id] * Item_Effect + b_u + b_i + self.mean

class Medi_Effect(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, U_Shadow_size, I_Shadow_size, init, device="cpu"):
        super(Medi_Effect, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-init, init)
        self.user_bias.weight.data.uniform_(-init, init)
        self.item_emb.weight.data.uniform_(-init, init)
        self.item_bias.weight.data.uniform_(-init, init)
        self.shadow_i_emb = nn.Embedding(num_items, U_Shadow_size)
        self.shadow_i_emb.weight.data.uniform_(-init*0.1, init*0.1)
        self.shadow_u_emb = nn.Embedding(num_users, I_Shadow_size)
        self.shadow_u_emb.weight.data.uniform_(-init*0.1, init*0.1)
        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserEffect, ItemEffect):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        # Shadows_Feature = (UserEffect * self.shadow_i_emb(i_id)).sum(1) + (ItemEffect * self.shadow_u_emb(u_id)).sum(1)
        Shadows_Feature = 0
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

# class Medi_MF(nn.Module):
#     def __init__(self, num_users, num_items, embedding_size, user_feature_size, item_feature_size, device="cpu"):
#         super(Medi_MF, self).__init__()
#         self.user_emb = nn.Embedding(num_users, embedding_size)
#         self.user_bias = nn.Embedding(num_users, 1)
#         self.item_emb = nn.Embedding(num_items, embedding_size)
#         self.item_bias = nn.Embedding(num_items, 1)
#         self.user_emb.weight.data.uniform_(-0.01, 0.01)
#         self.user_bias.weight.data.uniform_(-0.01, 0.01)
#         self.item_emb.weight.data.uniform_(-0.01, 0.01)
#         self.item_bias.weight.data.uniform_(-0.01, 0.01)
#
#         self.feature_embs_u = []
#         self.feature_embs_ui = []
#         for feature_dim in user_feature_size:
#             emb_u = nn.Embedding(feature_dim, 32)
#             emb_u.weight.data.uniform_(-0.01, 0.01)
#             emb_i = nn.Embedding(num_items, 32)
#             emb_i.weight.data.uniform_(-0.01, 0.01)
#             self.feature_embs_ui.append(emb_i.to(device))
#             self.feature_embs_u.append(emb_u.to(device))
#
#         self.feature_embs_i = []
#         self.feature_embs_iu = []
#         for feature_dim in item_feature_size:
#             emb_u = nn.Embedding(feature_dim, 32)
#             emb_u.weight.data.uniform_(-0.01, 0.01)
#             emb_i = nn.Embedding(num_users, 32)
#             emb_i.weight.data.uniform_(-0.01, 0.01)
#             self.feature_embs_iu.append(emb_i.to(device))
#             self.feature_embs_i.append(emb_u.to(device))
#
#         self.mean = nn.Parameter(torch.FloatTensor([0]), False)
#
#     def forward(self, u_id, i_id, u_feat, i_feat):
#         U = self.user_emb(u_id)
#         b_u = self.user_bias(u_id).squeeze()
#         I = self.item_emb(i_id)
#         b_i = self.item_bias(i_id).squeeze()
#         Medi = U * I
#         F = torch.zeros_like(u_id).to(self.device)
#         for i in range(u_feat.shape[1]):
#             F = F + (self.feature_embs_u[i](u_feat[:, i]) * self.feature_embs_ui[i](i_id)).sum(1)
#
#         for i in range(i_feat.shape[1]):
#             F = F + (self.feature_embs_i[i](i_feat[:, i]) * self.feature_embs_iu[i](u_id)).sum(1)
#         return Medi, (Medi).sum(1) + b_u + b_i + self.mean + F

class Medi_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(Medi_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        Medi = U * I
        return Medi, (Medi).sum(1) + b_u + b_i + self.mean

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean

    def get_item_em(self, pop_mask):
        import numpy as np
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        original_cmap = plt.get_cmap('brg', 10)  # 创建有10个颜色的brg颜色映射
        selected_colors = [(1, 0.2, 0.1, 1), (0, 0.4, 0.5, 0.6)]  # 选择第一个和第六个颜色

        # 创建新的颜色映射
        custom_cmap = ListedColormap(selected_colors)

        # cmap = plt.get_cmap('brg', 2)
        c = custom_cmap(pop_mask)

        for i in range(2):
            color = custom_cmap(i)
            print(f"Value {i}: {color}")

        # 假设你有一个形状为(300,256)的张量
        # 这里我们随机生成一个示例张量，你可以替换为你的实际数据
        # original_tensor = your_tensor  # 你的(300,256)张量
        all_iid = torch.arange(0, 300, dtype=torch.int32).cuda()
        c = c
        all_iem = self.item_emb(all_iid).detach().cpu()
        # 将PyTorch张量转换为NumPy数组以供scikit-learn使用
        data_numpy = all_iem.numpy()

        # 创建PCA实例并降维
        # pca = PCA(n_components=2)
        # data_numpy = pca.fit_transform(data_numpy)
        edgecolors = [(1, 0.2, 0.1, 0.8) if cat == 0 else (0, 0.4, 0.5, 0.2) for cat in pop_mask]
        linewidths = [100 if cat == 0 else 60 for cat in pop_mask]

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        # 移除刻度线和刻度值
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.legend(fontsize=18)

        ax.scatter(data_numpy[:, 0], data_numpy[:, 1], c=pop_mask,cmap=ListedColormap(['tab:blue', 'tab:red']))
        x_min, x_max = -0.05, 0.05  # 评分范围假设为0-5分
        # plt.xlim(x_min, x_max)
        # plt.ylim(x_min + 0.01, x_max + 0.01)
        plt.grid(True)
        plt.show()

        # 输出解释方差比例
        # print(f"解释方差比例: {pca.explained_variance_ratio_}")
        # print(f"累计解释方差比例: {sum(pca.explained_variance_ratio_)}")


        return

class IPS_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, corY, InverP, device="cpu"):
        super(IPS_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.corY = corY
        self.invP = nn.Embedding(num_items, 2)
        self.invP.weight = torch.nn.Parameter(InverP)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean

    def model_ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = torch.mean(weight * cost)
        return loss

class RD_IPS_MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, corY, upBound, lowBound, InverP, device="cpu"):
        super(RD_IPS_MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.corY = corY
        self.upBound = upBound
        self.lowBound = lowBound
        self.invP = nn.Embedding(num_items, 2)
        self.invP.weight = torch.nn.Parameter(InverP)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        return (U * I).sum(1) + b_u + b_i + self.mean

    def model_ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = torch.mean(weight * cost)
        return loss

    def ips_loss(self, u_id, i_id, y_train, loss_f):
        preds = self.forward(u_id, i_id).view(-1)

        weight = torch.ones(y_train.shape).to(self.device)
        weight[y_train == self.corY[0]] = self.invP(i_id)[y_train == self.corY[0], 0]
        weight[y_train == self.corY[1]] = self.invP(i_id)[y_train == self.corY[1], 1]

        cost = loss_f(preds, y_train)
        loss = - torch.mean(weight * cost)
        return loss

    def update_ips(self):
        with torch.no_grad():
            self.invP.weight.data[self.invP.weight.data > self.upBound] = self.upBound[
                self.invP.weight.data > self.upBound]
            self.invP.weight.data[self.invP.weight.data < self.lowBound] = self.lowBound[
                self.invP.weight.data < self.lowBound]

class MFwithFeature(nn.Module):
    def __init__(self, num_users, num_items, feature_size, embedding_size, device="cpu"):
        super(MFwithFeature, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.feature_embs_u = []
        self.feature_embs_i = []
        for feature_dim in feature_size:
            emb_u = nn.Embedding(feature_dim, 32)
            emb_u.weight.data.uniform_(-0.01, 0.01)
            emb_i = nn.Embedding(num_items, 32)
            emb_i.weight.data.uniform_(-0.01, 0.01)
            self.feature_embs_i.append(emb_i.to(device))
            self.feature_embs_u.append(emb_u.to(device))

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, features):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        Y = torch.zeros_like(u_id).to(self.device)
        for i in range(features.shape[1]):
            Y = Y + (self.feature_embs_u[i](features[:, i]) * self.feature_embs_i[i](i_id)).sum(1)
        return (U * I).sum(1) + b_u + b_i + self.mean + Y

class RealNVP_MLP(nn.Module):
    def __init__(self, input_size, layer_num, layer_dim, output_size):
        super().__init__()
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(input_size, layer_dim))
            layers.append(nn.GELU())
            input_size = layer_dim
        layers.append(nn.Linear(layer_dim, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AffineTransform(nn.Module):
    def __init__(self, type, input_size=2, layer_num=2, layer_dim=64, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.mask = self.build_mask(type=type, input_size=input_size).to(device)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True).to(device)

        self.mlp = RealNVP_MLP(input_size=input_size, layer_num=layer_num, layer_dim=layer_dim, output_size=2).to(device)

    def build_mask(self, type, input_size):
        if input_size // 2 == 0:
            size = input_size // 2
        else :
            size = input_size // 2 + 1
        assert type in {"left", "right"}
        if type == "left":
            mask = torch.cat(
                (torch.FloatTensor([1]), torch.FloatTensor([0]))).repeat(size)
        elif type == "right":
            mask = torch.cat(
                (torch.FloatTensor([0]), torch.FloatTensor([1]))).repeat(size)
        else:
            raise NotImplementedError
        return mask[:input_size]

    def forward(self, x):

        batch_size = x.shape[0]
        mask = self.mask.repeat(batch_size, 1)

        x_ = x * mask
        log_s, t = self.mlp(x_).split(1, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift
        t = t * (1.0 - mask)
        log_s = log_s * (1.0 - mask)
        x = x * torch.exp(log_s) + t
        return x, log_s

class RealNVP(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))

    def flow(self, x):
        z, log_det = x, torch.zeros_like(x)
        for op in self.transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.flow(x)
        return torch.sum(log_det, dim=1) + torch.sum(self.prior.log_prob(z), dim=1)

    def nll(self, x):
        return - self.log_prob(x).mean()

class DeepCausal(nn.Module):
    def __init__(self,num_users,num_items,feature_dim,embedding_size,vae_mean, vae_std,device="cpu"):
        super(DeepCausal, self).__init__()
        self.mf_layer = MFwithFeature(num_users, num_items, feature_dim, embedding_size, device=device)
        self.vae_mean = vae_mean
        self.vae_std = vae_std

        self.item_emb = nn.Embedding(num_items, self.vae_mean.shape[1])
        self.item_emb.weight.data.uniform_(-0.05, 0.05)

        self.device = device
        self.to(device)

    def forward(self, uid, iid, u_feat, sample=False):
        mf_output = self.mf_layer(uid, iid, u_feat)
        mean = self.vae_mean[uid.type(torch.long)]

        if sample:
            std = self.vae_std[uid]
            eps = torch.randn_like(std).to(self.device)
            z = mean + std * eps
        else:
            z = mean

        i_emb = self.item_emb(iid)

        latent_regression = (i_emb * z).sum(1)
        return mf_output + latent_regression

    def predict(self, uid, iid, u_feat):
        return self.forward(uid, iid, u_feat, sample=False)

class IDCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, ivae_dim, device="cpu"):
        super(IDCF, self).__init__()

        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.shadow_i_emb = nn.Embedding(num_items, ivae_dim)
        self.shadow_i_emb.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)
        self.device = device

    def forward(self, u_id, i_id, UserShadow):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()

        UShadows = (UserShadow * self.shadow_i_emb(i_id)).sum(1)
        Shadows_Feature = UShadows
        return (U * I).sum(1) + Shadows_Feature + b_u + b_i + self.mean

class Ivae(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(Ivae, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t):
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        O_hat = self.decode(z)
        return O_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat

class DisIvae(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(Ivae, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t):
        return self.decoder(t)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        O_hat = self.decode(z)
        return O_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat

class CIvae(nn.Module):
    def __init__(self, input_dim, shadow_dim, feature_dim, layer_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(CIvae, self).__init__()
        self.latent_dim = shadow_dim
        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=feature_dim,
                                    hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation])
        self.prior_log_var_z = MLP(input_dim=feature_dim,
                                   hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation])

        # encoder
        self.mean_z = MLP(input_dim=input_dim + feature_dim,
                          hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation])
        self.log_var_z = MLP(input_dim=input_dim + feature_dim,
                             hidden_dims=[layer_dim] * (n_layers - 1) + [shadow_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation])

        # decoder
        self.decoder = MLP(input_dim=shadow_dim + feature_dim,
                           hidden_dims=[layer_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'])

    def encode(self, t, x):
        tx = torch.cat((t, x), 1)
        mean = self.mean_z(tx)
        log_var = self.log_var_z(tx)
        return mean, log_var

    def decode(self, t, x):
        tx = torch.cat((t, x), 1)
        return self.decoder(tx)

    def prior(self, x):
        log_var_z = self.prior_log_var_z(x)
        if self.prior_mean:
            mean_z = self.prior_mean_z(x)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, o, x):
        prior_log_var = self.prior_log_var_z(x)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(x)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(o, x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        O_hat = self.decode(z, x)
        return O_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z, x)
        return x_hat

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers=3, activation="lrelu", out_activation=None,
                 device="cpu"):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        # encoder
        self.mean_z = MLP(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],)
        self.log_var_z = MLP(input_dim=input_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],)

        # decoder
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + ['sigmoid'],)


    def encode(self, x):
        mean = self.mean_z(x)
        log_var = self.log_var_z(x)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def reconstruct(self, x, sample=False):
        mean, log_var = self.encode(x)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat

class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class InvPref(nn.Module):
    def __init__(self, num_users, num_items, num_envs, embedding_size):
        super(InvPref, self).__init__()
        self.user_emb_inv = nn.Embedding(num_users, embedding_size)
        self.user_bias_inv = nn.Embedding(num_users, 1)
        self.item_emb_inv = nn.Embedding(num_items, embedding_size)
        self.item_bias_inv = nn.Embedding(num_items, 1)

        self.user_emb_env = nn.Embedding(num_users, embedding_size)
        self.user_bias_env = nn.Embedding(num_users, 1)
        self.item_emb_env = nn.Embedding(num_items, embedding_size)
        self.item_bias_env = nn.Embedding(num_items, 1)

        self.env_emb = nn.Embedding(num_envs, embedding_size)
        self.env_bias = nn.Embedding(num_envs, 1)

        self.user_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.env_emb.weight.data.uniform_(-0.01, 0.01)
        self.env_bias.weight.data.uniform_(-0.01, 0.01)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(embedding_size, num_envs)

    def forward(self, users_id, items_id, envs_id, alpha=0):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)

        users_embed_env_aware = self.user_emb_env(users_id)
        items_embed_env_aware = self.item_emb_env(items_id)

        envs_embed = self.env_emb(envs_id)

        invariant_preferences = users_embed_invariant * items_embed_invariant
        env_aware_preferences = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        env_aware_mid_score = torch.sum(env_aware_preferences, dim=1) \
                              + self.user_bias_env(users_id).view(-1) \
                              + self.item_bias_env(items_id).view(-1) \
                              + self.env_bias(envs_id).view(-1)

        env_aware_score = invariant_score + env_aware_mid_score

        reverse_invariant_preferences = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.view(-1), env_aware_score.view(-1), env_outputs.view(-1, self.env_emb.num_embeddings)

    def predict(self, users_id, items_id):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)
        invariant_preferences = users_embed_invariant * items_embed_invariant

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        return invariant_score

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

class MF_Stable_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_size
        self.prediction_model = MF(
            num_users=self.num_users, num_items=self.num_items, embedding_size=self.embedding_k)
        self.imputation = MF(
            num_users=self.num_users, num_items=self.num_items, embedding_size=self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips, mu=0, eta=1, stop=5,
            num_epoch=1000, batch_size=128, lr=0.05, lr1=10, lamb=0,
            tol=1e-4, G=1, verbose=False):

        mu = torch.Tensor([mu])
        mu.requires_grad_(True)

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        optimizer_propensity = torch.optim.Adam(
            [mu], lr=lr1, weight_decay=lamb)

        last_loss = 1e9

        observation = torch.zeros([self.num_users, self.num_items])
        for i in range(len(x)):
            observation[x[i][0], x[i][1]] = 1
        observation = observation.reshape(self.num_users * self.num_items)

        y1 = []
        for i in range(len(x)):
            if y[i] == 1:
                y1.append(self.num_items * x[i][0] + x[i][1])

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)  # 6960
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y, y1, mu)
        else:
            one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)

        one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)].detach()

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)  # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])  # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                # propensity score
                inv_prop = one_over_zl_obs[selected_idx]

                sub_y = torch.Tensor(sub_y)

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.forward(sub_x)
                pred = self.sigmoid(pred)
                imputation_y = self.sigmoid(imputation_y)

                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop.detach()).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()

                x_all_idx = ul_idxs[G * idx * batch_size: G * (idx + 1) * batch_size]
                x_sampled = x_all[x_all_idx]

                imputation_y1 = self.imputation.predict(x_sampled)
                imputation_y1 = self.sigmoid(imputation_y1)

                prop_loss = F.binary_cross_entropy(1 / one_over_zl[x_all_idx], observation[x_all_idx], reduction="sum")
                pred_y1 = self.prediction_model.predict(x_sampled)
                pred_y1 = self.sigmoid(pred_y1)

                imputation_loss = F.binary_cross_entropy(imputation_y1, pred_y1, reduction="none")

                loss = prop_loss + eta * ((1 - observation[x_all_idx] * one_over_zl[x_all_idx]) * (
                            imputation_loss - imputation_loss.mean())).sum() ** 2

                optimizer_propensity.zero_grad()
                loss.backward()
                optimizer_propensity.step()

                # print("mu = {}".format(mu))

                one_over_zl = self._compute_IPS(x, y, y1, mu, y_ips)
                one_over_zl_obs = one_over_zl[np.where(observation.cpu() == 1)]
                inv_prop = one_over_zl_obs[selected_idx].detach()

                pred = self.prediction_model.forward(sub_x)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction="sum")
                xent_loss = (xent_loss) / (inv_prop.detach().sum())

                optimizer_prediction.zero_grad()
                xent_loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-Stable-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-Stable-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self, x, y, y1, mu, y_ips=None):
        if y_ips is None:
            y_ips = 1
            print("y_ips is none")
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = (len(x) + mu) / (x[:, 0].max() * x[:, 1].max() + 2 * mu)
            py1o1 = (y.sum() + mu) / (len(y) + 2 * mu)
            py0o1 = 1 - py1o1

            propensity = torch.zeros(self.num_users * self.num_items)

            propensity += (py0o1 * po1) / py0

            propensity[np.array(y1)] = (py1o1 * po1) / py1

            one_over_zl = (1 / propensity)

        # one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl