# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config
from utils.math import index_to_one_hot
from models.independ_nn.Commnet import CommNetWork


class Ltt_NN(nn.Module):
    def __init__(self, id_dim, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(horizontal_cmd_dim=0, vertical_cmd_dim=0, shoot_dim=0, target_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(512, 512),
                 native_hidden_size=(512, 256),
                 policy_hidden_size=(256, 256, 128),
                 value_hidden_size=(256, 128, 1),
                 state_token_embed_dim=100, state_token_num_heads=4, atten_depth=2,
                 msl_token_embed_dim=32, msl_token_num_heads=4,
                 activation='tanh', init_method='xavier', aircraft_num=2):
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
        self.action_dims = action_dims

        # *** layers init *** #
        # ** 1.global hidden layers ** #
        self.global_hidden_layers = nn.ModuleList()
        last_global_dim = ground_truth_dim
        for dim in ground_truth_size_before_cat:
            self.global_hidden_layers.append(nn.Linear(last_global_dim, dim))
            last_global_dim = dim
        # ** 2.attention layers ** #
        self.attn_depth = atten_depth
        self.state_token_embed_dim = state_token_embed_dim
        self.state_token_num_heads = state_token_num_heads

        self.msl_token_embed_dim = msl_token_embed_dim
        self.msl_token_num_heads = msl_token_num_heads

        self.state_token_embed_layer = nn.Linear(state_token_dim, state_token_embed_dim)
        # self.msl_token_embed_layer = nn.Linear(self_msl_token_dim, msl_token_embed_dim)  #todo consider self and bandit msl same dim
        self.self_msl_token_embed_layer = nn.Linear(self_msl_token_dim, msl_token_embed_dim)
        self.bandit_msl_token_embed_layer = nn.Linear(bandit_msl_token_dim, msl_token_embed_dim)

        self.state_attn_layers = nn.ModuleList()
        self.self_msl_attn_layers = nn.ModuleList()
        self.bandit_msl_attn_layers = nn.ModuleList()

        self.w_k_state_token = nn.ModuleList()
        self.w_v_state_token = nn.ModuleList()
        self.w_q_state_token = nn.ModuleList()

        self.w_k_self_msl_token = nn.ModuleList()
        self.w_v_self_msl_token = nn.ModuleList()
        self.w_q_self_msl_token = nn.ModuleList()

        self.w_k_bandit_msl_token = nn.ModuleList()
        self.w_v_bandit_msl_token = nn.ModuleList()
        self.w_q_bandit_msl_token = nn.ModuleList()

        for _ in range(self.attn_depth):
            self.state_attn_layers.append(
                nn.MultiheadAttention(embed_dim=self.state_token_embed_dim, num_heads=self.state_token_num_heads))
            self.self_msl_attn_layers.append(
                nn.MultiheadAttention(embed_dim=self.msl_token_embed_dim, num_heads=self.msl_token_num_heads))
            self.bandit_msl_attn_layers.append(
                nn.MultiheadAttention(embed_dim=self.msl_token_embed_dim, num_heads=self.msl_token_num_heads))

            self.w_k_state_token.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))
            self.w_v_state_token.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))
            self.w_q_state_token.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))

            self.w_k_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_v_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_q_self_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))

            self.w_k_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_v_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))
            self.w_q_bandit_msl_token.append(nn.Linear(self.msl_token_embed_dim, self.msl_token_embed_dim))

        # for normalization #
        self.state_token_norm_layer = nn.LayerNorm([self.state_token_embed_dim])
        self.self_msl_token_norm_layer = nn.LayerNorm([self.msl_token_embed_dim])
        self.bandit_msl_token_norm_layer = nn.LayerNorm([self.msl_token_embed_dim])

        # ** 3.native hidden layers ** #
        self.native_hidden_layers = nn.ModuleList()
        last_native_dim = native_dim
        for dim in native_hidden_size:
            self.native_hidden_layers.append(nn.Linear(last_native_dim, dim))
            last_native_dim = dim

        # ** 4.cat native, tokens and global ** #
        #  ---------------------------------actor -----------------------------------------------------#
        flatten_dim_without_global = self.state_token_embed_dim * state_token_num + \
                                     self.msl_token_embed_dim * (self_msl_token_num + bandit_msl_token_num) + \
                                     last_native_dim + id_dim

        self.policy_commnet = CommNetWork(flatten_dim_without_global, policy_hidden_size, n_agents=self.aircraft_num)
        last_policy_dim = policy_hidden_size[-1]

        action_head_hidden_size = int(last_policy_dim / 4)

        # self.ltt_hiddens = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"], action_head_hidden_size)
        self.ltt_hiddens = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] * action_dims["vertical_cmd_dim"], action_head_hidden_size)

        # 13sigmoid
        self.ltt_heads = nn.Linear(action_head_hidden_size, action_dims["horizontal_cmd_dim"] * action_dims["vertical_cmd_dim"])
        # softmax
        # self.ltt_heads = nn.Linear(action_head_hidden_size, 2)

        # ---------------------------------------------critic --------------------------------------------- #
        flatten_dim_with_global = flatten_dim_without_global + last_global_dim
        self.value_affine_layers = nn.ModuleList()
        last_value_dim = flatten_dim_with_global
        for dim in value_hidden_size[:2]:
            self.value_affine_layers.append(nn.Linear(last_value_dim, dim))
            last_value_dim = dim

        value_head_hidden_size = int(last_value_dim / 4)
        self.value_head_hidden = nn.Linear(last_value_dim, value_head_hidden_size)
        self.value_head = nn.Linear(value_head_hidden_size, 1)

        # ------------------init layers-------------------------------------------------------------------- #
        set_init(self.global_hidden_layers, method=init_method)  # global part
        set_init(self.native_hidden_layers, method=init_method)  # native part
        set_init([self.state_token_embed_layer], method=init_method)  # atten part
        set_init([self.self_msl_token_embed_layer, self.bandit_msl_token_embed_layer], method=init_method)

        self.state_token_norm_layer.weight.requires_grad_(False)
        self.state_token_norm_layer.bias.requires_grad_(False)
        self.self_msl_token_norm_layer.weight.requires_grad_(False)
        self.self_msl_token_norm_layer.bias.requires_grad_(False)
        self.bandit_msl_token_norm_layer.weight.requires_grad_(False)
        self.bandit_msl_token_norm_layer.bias.requires_grad_(False)

        set_init(self.w_q_state_token, method=init_method)
        set_init(self.w_k_state_token, method=init_method)
        set_init(self.w_v_state_token, method=init_method)
        set_init(self.w_q_self_msl_token, method=init_method)
        set_init(self.w_k_self_msl_token, method=init_method)
        set_init(self.w_v_self_msl_token, method=init_method)
        set_init(self.w_q_bandit_msl_token, method=init_method)
        set_init(self.w_k_bandit_msl_token, method=init_method)
        set_init(self.w_v_bandit_msl_token, method=init_method)

        set_init([self.ltt_hiddens], method=init_method)
        set_init([self.ltt_heads], method=init_method)

        set_init(self.value_affine_layers, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init([self.value_head], method=init_method)

    def macro_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state,
                      ids_one_hot_state, hor_one_hots, ver_one_hots, maneuver_one_hots, sample):

        token_state_split = token_state.split(1, dim=1)
        self_msl_token_state_split = self_msl_token_state.split(1, dim=1)
        bandit_msl_token_state_split = bandit_msl_token_state.split(1, dim=1)

        state_tokens_out_flat_list = []
        self_msl_tokens_out_flat_list = []
        bandit_msl_tokens_out_flat_list = []

        for air_id in range(self.aircraft_num):
            # *** attention forward *** #
            token_state = token_state_split[air_id].squeeze(1).transpose(0, 1)
            self_msl_token_state = self_msl_token_state_split[air_id].squeeze(1).transpose(0, 1)
            bandit_msl_token_state = bandit_msl_token_state_split[air_id].squeeze(1).transpose(0, 1)

            token_embedding = self.state_token_embed_layer(token_state)
            self_msl_token_embedding = self.self_msl_token_embed_layer(self_msl_token_state)
            bandit_msl_token_embedding = self.bandit_msl_token_embed_layer(bandit_msl_token_state)

            # print(self.state_token_norm_layer.weight)
            for i in range(self.attn_depth):
                q_state = self.w_q_state_token[i](token_embedding)
                k_state = self.w_k_state_token[i](token_embedding)
                v_state = self.w_v_state_token[i](token_embedding)
                q_self_msl = self.w_q_self_msl_token[i](self_msl_token_embedding)
                k_self_msl = self.w_k_self_msl_token[i](self_msl_token_embedding)
                v_self_msl = self.w_v_self_msl_token[i](self_msl_token_embedding)
                q_bandit_msl = self.w_q_bandit_msl_token[i](bandit_msl_token_embedding)
                k_bandit_msl = self.w_k_bandit_msl_token[i](bandit_msl_token_embedding)
                v_bandit_msl = self.w_v_bandit_msl_token[i](bandit_msl_token_embedding)
                # print("new forward")

                state_tokens_out, _ = self.state_attn_layers[i](q_state, k_state, v_state)
                self_msl_tokens_out, _ = self.self_msl_attn_layers[i](q_self_msl, k_self_msl, v_self_msl)
                bandit_msl_tokens_out, _ = self.bandit_msl_attn_layers[i](q_bandit_msl, k_bandit_msl, v_bandit_msl)  #
                # print(tokens_out.size())  # todo problems here of dimention operation
                state_token_sum = state_tokens_out + token_embedding
                self_msl_token_sum = self_msl_tokens_out + self_msl_token_embedding
                bandit_msl_token_sum = bandit_msl_tokens_out + bandit_msl_token_embedding

                token_embedding = self.state_token_norm_layer(state_token_sum)
                # print("token_embedding", token_embedding.mean(), token_embedding.std())
                self_msl_token_embedding = self.self_msl_token_norm_layer(self_msl_token_sum)
                bandit_msl_token_embedding = self.bandit_msl_token_norm_layer(bandit_msl_token_sum)

                state_tokens_out = token_embedding.transpose(0, 1)
                self_msl_tokens_out = self_msl_token_embedding.transpose(0, 1)
                bandit_msl_tokens_out = bandit_msl_token_embedding.transpose(0, 1)

                # *** flat and cat *** #
                state_tokens_out_flat = torch.flatten(state_tokens_out, start_dim=-2, end_dim=-1)
                self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
                bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

            state_tokens_out_flat_list.append(state_tokens_out_flat)
            self_msl_tokens_out_flat_list.append(self_msl_tokens_out_flat)
            bandit_msl_tokens_out_flat_list.append(bandit_msl_tokens_out_flat)

        state_tokens_out_flat = torch.stack(state_tokens_out_flat_list, dim=1)
        self_msl_tokens_out_flat = torch.stack(self_msl_tokens_out_flat_list, dim=1)
        bandit_msl_tokens_out_flat = torch.stack(bandit_msl_tokens_out_flat_list, dim=1)

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** actor *** #
        policy_state = torch.cat(
            [ids_one_hot_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)

        policy_state = self.policy_commnet(policy_state)
        policy_state = policy_state.squeeze(0)

        steer_state = torch.cat((policy_state, maneuver_one_hots), dim=-1)

        ltt_hiddens = self.activation(self.ltt_hiddens(steer_state))
        ltt_heads = self.ltt_heads(ltt_hiddens)

        # 13sigmoid
        ltt_probs = torch.sigmoid(ltt_heads)
        if sample:
            noise = torch.normal(mean=0.1, std=torch.tensor(0.03))
            ltt_probs = ltt_probs + noise
            ltt_probs = torch.clamp(ltt_probs, 0, 1)
        # softmax
        # ltt_probs = torch.softmax(ltt_heads, dim=-1).unsqueeze(0)

        # *** critic *** #
        value_state = torch.cat(
            [ids_one_hot_state, global_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat],
            dim=-1)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return ltt_probs, v_head

    def select_action(self, x0, x1, x2, x3, x4, x5, hor_one_hots, ver_one_hots, maneuver_one_hots, flag):
        x0 = torch.FloatTensor(x0).unsqueeze(0)
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)
        x3 = torch.FloatTensor(x3).unsqueeze(0)
        x4 = torch.FloatTensor(x4).unsqueeze(0)
        x5 = torch.FloatTensor(x5).unsqueeze(0)

        ltt_probs, _ = self.macro_forward(x0, x1, x2, x3, x4, x5, hor_one_hots, ver_one_hots, maneuver_one_hots, flag)

        ltt_probs.squeeze(0)

        # 13sigmoid
        ltt_probs = torch.masked_select(ltt_probs, maneuver_one_hots.to(torch.bool)).unsqueeze(-1)
        ltt = (ltt_probs + self.multinomial_protect).bernoulli().to(torch.int16)

        # softmax
        # ltt = (ltt_probs + self.multinomial_protect).multinomial(1)

        #print(hor_ltt_probs.tolist())

        return ltt

    def get_log_prob_and_values(self, x0, x1, x2, x3, x4, x5, ltts, hor_one_hots, ver_one_hots, maneuver_one_hots, flag):

        ltt_probs, value = self.macro_forward(x0, x1, x2, x3, x4, x5, hor_one_hots, ver_one_hots, maneuver_one_hots, flag)

        # 13sigmoid
        ltt_probs = torch.masked_select(ltt_probs, maneuver_one_hots.to(torch.bool))
        ltt_probs = ltt_probs.view(-1, maneuver_one_hots.shape[1], 1)
        # ltt_probs[torch.isnan(ltts)] = 1
        # ans = torch.log(ltt_probs + self.log_protect)
        ltt_probs[torch.isnan(ltts)] = 0

        value[torch.isnan(ltts)] = 0  # if agent die, v = 0, cut backward

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        # ans = torch.log(ans + self.log_protect)
        return ltt_probs, value








