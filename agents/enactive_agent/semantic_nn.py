# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config


class Semantic_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(horizontal_cmd_dim=0, vertical_cmd_dim=0, shoot_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(512, 512),
                 native_hidden_size=(512, 256),
                 policy_hidden_size=(256, 256, 128),
                 value_hidden_size=(256, 128),
                 atten_depth=2,
                 msl_token_embed_dim=32, msl_token_num_heads=4,
                 activation='tanh', init_method='xavier', aircraft_num=1):
        super().__init__()

        self.log_protect = Config.log_protect
        self.multinomial_protect = Config.multinomial_protect
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.aircraft_num = aircraft_num

        # *** layers init *** #
        # ** 1.global hidden layers ** #
        self.global_hidden_layers = nn.ModuleList()
        last_global_dim = ground_truth_dim
        for dim in ground_truth_size_before_cat:
            self.global_hidden_layers.append(nn.Linear(last_global_dim, dim))
            last_global_dim = dim
        # ** 2.attention layers ** #
        self.attn_depth = atten_depth

        self.msl_token_embed_dim = msl_token_embed_dim
        self.msl_token_num_heads = msl_token_num_heads

        self.self_msl_token_embed_layer = nn.Linear(self_msl_token_dim, msl_token_embed_dim)
        self.bandit_msl_token_embed_layer = nn.Linear(bandit_msl_token_dim, msl_token_embed_dim)

        self.self_msl_attn_layers = nn.ModuleList()
        self.bandit_msl_attn_layers = nn.ModuleList()

        self.w_k_self_msl_token = nn.ModuleList()
        self.w_v_self_msl_token = nn.ModuleList()
        self.w_q_self_msl_token = nn.ModuleList()

        self.w_k_bandit_msl_token = nn.ModuleList()
        self.w_v_bandit_msl_token = nn.ModuleList()
        self.w_q_bandit_msl_token = nn.ModuleList()

        for _ in range(self.attn_depth):
            self.self_msl_attn_layers.append(
                nn.MultiheadAttention(embed_dim=self.msl_token_embed_dim, num_heads=self.msl_token_num_heads))
            self.bandit_msl_attn_layers.append(
                nn.MultiheadAttention(embed_dim=self.msl_token_embed_dim, num_heads=self.msl_token_num_heads))

            self.w_k_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_v_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_q_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))

            self.w_k_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_v_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_q_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))

        # for normalization #
        self.self_msl_token_norm_layer = nn.LayerNorm([self.msl_token_embed_dim])
        self.bandit_msl_token_norm_layer = nn.LayerNorm([self.msl_token_embed_dim])

        # ** 3.native hidden layers ** #
        self.native_hidden_layers = nn.ModuleList()
        last_native_dim = native_dim
        for dim in native_hidden_size:
            self.native_hidden_layers.append(nn.Linear(last_native_dim, dim))
            last_native_dim = dim

        # ** 4.cat global and tokens ** #
        #  ---------------------------------actor -----------------------------------------------------#
        flatten_dim_without_global = self.msl_token_embed_dim * (self_msl_token_num + bandit_msl_token_num) + \
                                     last_native_dim
        flatten_dim_with_global = flatten_dim_without_global + last_global_dim

        self.policy_affine_layers = nn.ModuleList()
        last_policy_dim = flatten_dim_without_global
        for dim in policy_hidden_size:
            self.policy_affine_layers.append(nn.Linear(last_policy_dim, dim))
            last_policy_dim = dim

        action_head_hidden_size = int(last_policy_dim / 4)
        self.horizontal_cmd_hidden = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.vertical_cmd_hidden = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.shoot_hidden = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.v_c_hidden = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"], action_head_hidden_size)
        self.nn_c_hidden = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"], action_head_hidden_size)

        self.horizontal_cmd_head = nn.Linear(action_head_hidden_size, action_dims["horizontal_cmd_dim"])
        self.vertical_cmd_head = nn.Linear(action_head_hidden_size, action_dims["vertical_cmd_dim"])
        self.shoot_head = nn.Linear(action_head_hidden_size, action_dims["shoot_dim"])
        self.v_c_head = nn.Linear(action_head_hidden_size, action_dims["v_c_dim"])
        self.nn_c_head = nn.Linear(action_head_hidden_size, action_dims["nn_c_dim"])

        # ---------------------------------------------critic --------------------------------------------- #
        self.value_affine_layers = nn.ModuleList()
        last_value_dim = flatten_dim_with_global
        for dim in value_hidden_size:
            self.value_affine_layers.append(nn.Linear(last_value_dim, dim))
            last_value_dim = dim

        value_head_hidden_size = int(last_value_dim / 4)
        self.value_head_hidden = nn.Linear(last_value_dim, value_head_hidden_size)
        self.value_head = nn.Linear(value_head_hidden_size, 1)

        # ------------------init layers-------------------------------------------------------------------- #
        set_init(self.global_hidden_layers, method=init_method)  # global part
        set_init([self.self_msl_token_embed_layer, self.bandit_msl_token_embed_layer], method=init_method)

        self.self_msl_token_norm_layer.weight.requires_grad_(False)
        self.self_msl_token_norm_layer.bias.requires_grad_(False)
        self.bandit_msl_token_norm_layer.weight.requires_grad_(False)
        self.bandit_msl_token_norm_layer.bias.requires_grad_(False)

        set_init(self.w_q_self_msl_token, method=init_method)
        set_init(self.w_k_self_msl_token, method=init_method)
        set_init(self.w_v_self_msl_token, method=init_method)
        set_init(self.w_q_bandit_msl_token, method=init_method)
        set_init(self.w_k_bandit_msl_token, method=init_method)
        set_init(self.w_v_bandit_msl_token, method=init_method)

        set_init(self.native_hidden_layers, method=init_method)
        set_init(self.policy_affine_layers, method=init_method)

        set_init([self.shoot_hidden], method=init_method)
        set_init([self.horizontal_cmd_hidden], method=init_method)
        set_init([self.vertical_cmd_hidden], method=init_method)
        set_init([self.v_c_hidden], method=init_method)
        set_init([self.nn_c_hidden], method=init_method)
        set_init([self.shoot_head], method=init_method)
        set_init([self.horizontal_cmd_head], method=init_method)
        set_init([self.vertical_cmd_head], method=init_method)
        set_init([self.v_c_head], method=init_method)
        set_init([self.nn_c_head], method=init_method)

        set_init(self.value_affine_layers, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init([self.value_head], method=init_method)

    def macro_forward(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                      hor_mask, ver_mask, shoot_mask):
        # *** attention forward *** #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

        self_msl_token_embedding = self.self_msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.bandit_msl_token_embed_layer(bandit_msl_token_state)

        # print(self.state_token_norm_layer.weight)
        for i in range(self.attn_depth):
            q_self_msl = self.w_q_self_msl_token[i](self_msl_token_embedding)
            k_self_msl = self.w_k_self_msl_token[i](self_msl_token_embedding)
            v_self_msl = self.w_v_self_msl_token[i](self_msl_token_embedding)
            q_bandit_msl = self.w_q_bandit_msl_token[i](bandit_msl_token_embedding)
            k_bandit_msl = self.w_k_bandit_msl_token[i](bandit_msl_token_embedding)
            v_bandit_msl = self.w_v_bandit_msl_token[i](bandit_msl_token_embedding)
            # print("new forward")

            self_msl_tokens_out, _ = self.self_msl_attn_layers[i](q_self_msl, k_self_msl, v_self_msl)
            bandit_msl_tokens_out, _ = self.bandit_msl_attn_layers[i](q_bandit_msl, k_bandit_msl, v_bandit_msl)  #
            self_msl_token_sum = self_msl_tokens_out + self_msl_token_embedding
            bandit_msl_token_sum = bandit_msl_tokens_out + bandit_msl_token_embedding

            # print("token_embedding", token_embedding.mean(), token_embedding.std())
            self_msl_token_embedding = self.self_msl_token_norm_layer(self_msl_token_sum)
            bandit_msl_token_embedding = self.bandit_msl_token_norm_layer(bandit_msl_token_sum)

        self_msl_tokens_out = self_msl_token_embedding.transpose(0, 1)
        bandit_msl_tokens_out = bandit_msl_token_embedding.transpose(0, 1)

        # *** flat and cat *** #
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** actor *** #
        policy_state = torch.cat([native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        hor_hidden = self.activation(self.horizontal_cmd_hidden(policy_state))
        ver_hidden = self.activation(self.vertical_cmd_hidden(policy_state))
        s_hidden = self.activation(self.shoot_hidden(policy_state))

        hor_head = self.horizontal_cmd_head(hor_hidden)
        ver_head = self.vertical_cmd_head(ver_hidden)

        # *** add mask operation *** #
        hor_exp = torch.exp(self.horizontal_cmd_head(hor_hidden))
        ver_exp = torch.exp(self.vertical_cmd_head(ver_hidden))
        shoot_exp = torch.exp(self.shoot_head(s_hidden))

        hor_prob = (hor_exp * hor_mask) / torch.sum(hor_exp * hor_mask, dim=-1, keepdim=True)
        ver_prob = (ver_exp * ver_mask) / torch.sum(ver_exp * ver_mask, dim=-1, keepdim=True)
        shoot_prob = (shoot_exp * shoot_mask) / torch.sum(shoot_exp * shoot_mask, dim=-1, keepdim=True)

        # *** critic *** #
        value_state = torch.cat([global_state, native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return hor_prob, ver_prob, shoot_prob, v_head, policy_state, hor_head, ver_head

    def hybrid_forward(self, policy_state, hor_c_one_hot, ver_c_one_hot):
        steer_state = torch.cat((policy_state, hor_c_one_hot, ver_c_one_hot), dim=-1)

        v_c_hidden = self.activation(self.v_c_hidden(steer_state))
        nn_c_hidden = self.activation(self.nn_c_hidden(steer_state))

        v_c_prob = torch.softmax(self.v_c_head(v_c_hidden), dim=-1)
        nn_c_prob = torch.softmax(self.nn_c_head(nn_c_hidden), dim=-1)

        return v_c_prob, nn_c_prob

    def select_action(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                      hor_mask: list, ver_mask: list, shoot_mask: list):
        global_state = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)
        native_state = torch.tensor(native_state, dtype=torch.float32).unsqueeze(0)
        self_msl_token_state = torch.tensor(self_msl_token_state, dtype=torch.float32).unsqueeze(0)
        bandit_msl_token_state = torch.tensor(bandit_msl_token_state, dtype=torch.float32).unsqueeze(0)

        hor_mask = torch.tensor(hor_mask, dtype=torch.float32).squeeze()  # todo need squeeze 0 if single target
        ver_mask = torch.tensor(ver_mask, dtype=torch.float32).squeeze()
        shoot_mask = torch.tensor(shoot_mask, dtype=torch.float32).squeeze()

        hor_prob, ver_prob, shoot_prob, v_head, policy_state, hor_head, ver_head = self.macro_forward(global_state,
                                                                             native_state,
                                                                             self_msl_token_state,
                                                                             bandit_msl_token_state,
                                                                             hor_mask,
                                                                             ver_mask,
                                                                             shoot_mask)

        hor = (hor_prob * hor_mask + hor_mask * self.multinomial_protect).multinomial(1)
        ver = (ver_prob * ver_mask + ver_mask * self.multinomial_protect).multinomial(1)
        shoot = (shoot_prob * shoot_mask + shoot_mask * self.multinomial_protect).multinomial(1)

        # change to one hot #
        hor_one_hot = index_to_one_hot(hor.tolist(), hor_prob.size(0))
        hor_one_hot = torch.tensor(hor_one_hot).detach()
        ver_one_hot = index_to_one_hot(ver.tolist(), ver_prob.size(0))
        ver_one_hot = torch.tensor(ver_one_hot).detach()

        v_c_prob, nn_c_prob = self.hybrid_forward(policy_state, hor_one_hot, ver_one_hot)
        v_c = (v_c_prob + self.multinomial_protect).multinomial(1)
        nn_c = (nn_c_prob + self.multinomial_protect).multinomial(1)

        return hor, ver, shoot, v_c, nn_c, v_head, hor_head, ver_head

    def batch_forward(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                      hor_masks, ver_masks, shoot_masks, hor_one_hots, ver_one_hots):
        # function used for training #
        hor_probs, ver_probs, shoot_probs, v_head, policy_state = self.macro_forward(global_state,
                                                                                     native_state,
                                                                                     self_msl_token_state,
                                                                                     bandit_msl_token_state,
                                                                                     hor_masks,
                                                                                     ver_masks,
                                                                                     shoot_masks)
        v_c_probs, nn_c_probs = self.hybrid_forward(policy_state, hor_one_hots, ver_one_hots)

        return hor_probs, ver_probs, shoot_probs, v_c_probs, nn_c_probs, v_head

    def get_log_prob_and_values(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                                hors, vers, shoots, v_cs, nn_cs, hor_masks, ver_masks, shoot_masks,
                                hor_one_hots, ver_one_hots):

        hor_probs, ver_probs, shoot_probs, v_c_probs, nn_c_probs, value = self.batch_forward(
            global_state, native_state, self_msl_token_state, bandit_msl_token_state, hor_masks, ver_masks, shoot_masks,
            hor_one_hots, ver_one_hots)

        hor = hors.clone()
        hor[torch.isnan(hor)] = 0
        ver = vers.clone()
        ver[torch.isnan(ver)] = 0
        s = shoots.clone()
        s[torch.isnan(s)] = 0
        v = v_cs.clone()
        v[torch.isnan(v)] = 0
        nn_c = nn_cs.clone()
        nn_c[torch.isnan(nn_c)] = 0

        hor_probs = hor_probs.gather(-1, hor.long())
        hor_probs[torch.isnan(hors)] = 1
        ver_probs = ver_probs.gather(-1, ver.long())
        ver_probs[torch.isnan(vers)] = 1
        shoot_probs = shoot_probs.gather(-1, s.long())
        shoot_probs[torch.isnan(shoots)] = 1
        v_c_probs = v_c_probs.gather(-1, v.long())
        v_c_probs[torch.isnan(v_cs)] = 1
        nn_c_probs = nn_c_probs.gather(-1, nn_c.long())
        nn_c_probs[torch.isnan(nn_cs)] = 1

        ans = torch.log(hor_probs + self.log_protect) + \
              torch.log(ver_probs + self.log_protect) + \
              torch.log(shoot_probs + self.log_protect) + \
              torch.log(v_c_probs + self.log_protect) + \
              torch.log(nn_c_probs + self.log_protect)

        return ans, value








