# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config
from models.independ_nn.Commnet import CommNetWork
# from tensorboardX import SummaryWriter


class Release_NN(nn.Module):
    def __init__(self, id_dim, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 seq_len=4,
                 is_lstm=True,
                 action_dims=dict(horizontal_cmd_dim=0, vertical_cmd_dim=0, shoot_dim=0, target_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(64, 32),
                 native_hidden_size=(128, 128),
                 policy_hidden_size=(128, 128, 128),
                 value_hidden_size=(64, 32),
                 state_token_embed_dim=64, state_token_num_heads=4, atten_depth=1,
                 msl_token_embed_dim=32, msl_token_num_heads=4,
                 activation='tanh', init_method='xavier', aircraft_num=2, last_decision_embedding_size=8):
        super().__init__()

        self.seq_len = seq_len if is_lstm else 1
        self.log_protect = Config.log_protect
        self.multinomial_protect = Config.multinomial_protect
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.aircraft_num = aircraft_num
        self.is_lstm = is_lstm

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

        self.policy_commnet = CommNetWork(flatten_dim_without_global, policy_hidden_size, n_agents=self.aircraft_num,
                                          seq_len=self.seq_len)
        last_policy_dim = policy_hidden_size[-1]
        # for lstm
        self.lstm_hidden_size = last_policy_dim
        self.lstm_layer = nn.LSTM(input_size=last_policy_dim, hidden_size=last_policy_dim,
                                  num_layers=1, batch_first=True)

        action_head_hidden_size = int(last_policy_dim / 4)

        last_decision_dim = action_dims["target_dim"] + \
                            action_dims["horizontal_cmd_dim"] + \
                            action_dims["vertical_cmd_dim"] + \
                            action_dims["v_c_dim"] + \
                            action_dims["nn_c_dim"]

        self.lts_embedding = nn.Linear(last_decision_dim, last_decision_embedding_size)
        lts_input_dim = last_decision_embedding_size + last_policy_dim
        self.lts_hiddens = nn.Linear(lts_input_dim, action_head_hidden_size)
        self.lts_heads = nn.Linear(action_head_hidden_size, 2)  # stop and not stop, 2 dim

        self.horizontal_cmd_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.vertical_cmd_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.shoot_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.target_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.v_c_hiddens = nn.Linear(
            last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"],
            action_head_hidden_size)
        self.nn_c_hiddens = nn.Linear(
            last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"],
            action_head_hidden_size)

        self.horizontal_cmd_heads = nn.Linear(action_head_hidden_size, action_dims["horizontal_cmd_dim"])
        self.vertical_cmd_heads = nn.Linear(action_head_hidden_size, action_dims["vertical_cmd_dim"])
        self.shoot_heads = nn.Linear(action_head_hidden_size, action_dims["shoot_dim"])
        self.target_heads = nn.Linear(action_head_hidden_size, action_dims["target_dim"])
        self.v_c_heads = nn.Linear(action_head_hidden_size, action_dims["v_c_dim"])
        self.nn_c_heads = nn.Linear(action_head_hidden_size, action_dims["nn_c_dim"])

        # ---------------------------------------------critic --------------------------------------------- #
        if self.is_lstm:
            flatten_dim_with_global = last_policy_dim + last_global_dim + id_dim
        else:
            flatten_dim_with_global = flatten_dim_without_global + last_global_dim
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

        set_init([self.shoot_hiddens], method=init_method)
        set_init([self.target_hiddens], method=init_method)
        set_init([self.shoot_heads], method=init_method)
        set_init([self.target_heads], method=init_method)

        set_init([self.lts_embedding], method=init_method)
        set_init([self.lts_hiddens], method=init_method)
        set_init([self.lts_heads], method=init_method)

        set_init([self.horizontal_cmd_hiddens], method=init_method)
        set_init([self.vertical_cmd_hiddens], method=init_method)
        set_init([self.horizontal_cmd_heads], method=init_method)
        set_init([self.vertical_cmd_heads], method=init_method)

        set_init([self.v_c_hiddens], method=init_method)
        set_init([self.nn_c_hiddens], method=init_method)
        set_init([self.v_c_heads], method=init_method)
        set_init([self.nn_c_heads], method=init_method)

        set_init(self.value_affine_layers, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        # print(type(self.lstm_layer.all_weights))
        set_init([self.lstm_layer], method="lstm_orthogonal_init")
        set_init([self.value_head], method=init_method)

    def macro_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      ids_one_hot_state, hor_mask, ver_mask, shoot_mask, target_mask, last_decision, lts_mask,
                      deliberation_cost, hidden_h, hidden_c):

        next_hiddens = [hidden_h, hidden_c]
        batch_size = token_state.shape[0]
        seq_len = token_state.shape[2]

        token_state_split = token_state.split(1, dim=1)
        self_msl_token_state_split = self_msl_token_state.split(1, dim=1)
        bandit_msk_token_state_split = bandit_msk_token_state.split(1, dim=1)

        state_tokens_out_flat_list = []
        self_msl_tokens_out_flat_list = []
        bandit_msl_tokens_out_flat_list = []

        for air_id in range(self.aircraft_num):
            # *** attention forward *** #
            token_state = token_state_split[air_id].squeeze(1).transpose(2, 1).transpose(1, 0)
            self_msl_token_state = self_msl_token_state_split[air_id].squeeze(1).transpose(2, 1).transpose(1, 0)
            bandit_msk_token_state = bandit_msk_token_state_split[air_id].squeeze(1).transpose(2, 1).transpose(1, 0)

            token_state = token_state.contiguous().view(token_state.shape[0], -1, token_state.shape[-1])
            self_msl_token_state = self_msl_token_state.contiguous().view(self_msl_token_state.shape[0], -1,
                                                                          self_msl_token_state.shape[-1])
            bandit_msk_token_state = bandit_msk_token_state.contiguous().view(bandit_msk_token_state.shape[0], -1,
                                                                              bandit_msk_token_state.shape[-1])

            token_embedding = self.activation(self.state_token_embed_layer(token_state))
            self_msl_token_embedding = self.activation(self.self_msl_token_embed_layer(self_msl_token_state))
            bandit_msl_token_embedding = self.activation(self.bandit_msl_token_embed_layer(bandit_msk_token_state))

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

            state_tokens_out_flat = state_tokens_out_flat.view(batch_size, seq_len, state_tokens_out_flat.shape[-1])
            self_msl_tokens_out_flat = self_msl_tokens_out_flat.view(batch_size, seq_len,
                                                                     self_msl_tokens_out_flat.shape[-1])
            bandit_msl_tokens_out_flat = bandit_msl_tokens_out_flat.view(batch_size, seq_len,
                                                                         bandit_msl_tokens_out_flat.shape[-1])

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
            [ids_one_hot_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat,
             bandit_msl_tokens_out_flat], dim=-1)    # batch_size * num_aircraft * seq_len * feature_dim #

        comm_state = self.policy_commnet(policy_state)   # batch_size * num_aircraft * (seq_len == 1  *) feature_dim #
        if self.is_lstm:
            policy_state = comm_state.view(-1, seq_len, comm_state.shape[-1])
            policy_state, next_hiddens = self.lstm_layer(policy_state, (hidden_h, hidden_c))  # 2 4 128   1 2 128
            policy_state_v = policy_state[:, -1, :].view(batch_size, self.aircraft_num, -1)
            policy_state = policy_state_v.squeeze(0)
        else:
            policy_state = comm_state[:, :, :].squeeze(0)

        # print("last decision size", last_decision.size())
        last_decision_embedding = self.activation(self.lts_embedding(last_decision))
        lts_state = torch.cat([policy_state, last_decision_embedding], dim=-1)
        lts_hiddens = self.activation(self.lts_hiddens(lts_state))
        hor_hiddens = self.activation(self.horizontal_cmd_hiddens(policy_state))
        ver_hiddens = self.activation(self.vertical_cmd_hiddens(policy_state))
        s_hiddens = self.activation(self.shoot_hiddens(policy_state))
        t_hiddens = self.activation(self.target_hiddens(policy_state))

        # *** add mask operation *** #
        hor_exp = torch.exp(self.horizontal_cmd_heads(hor_hiddens))
        ver_exp = torch.exp(self.vertical_cmd_heads(ver_hiddens))
        shoot_exp = torch.exp(self.shoot_heads(s_hiddens))
        target_exp = torch.exp(self.target_heads(t_hiddens))
        lts_exp = torch.exp(self.lts_heads(lts_hiddens))

        hor_probs = (hor_exp * hor_mask) / torch.sum(hor_exp * hor_mask, dim=-1, keepdim=True)
        ver_probs = (ver_exp * ver_mask) / torch.sum(ver_exp * ver_mask, dim=-1, keepdim=True)
        shoot_probs = (shoot_exp * shoot_mask) / torch.sum(shoot_exp * shoot_mask, dim=-1, keepdim=True)
        target_probs = (target_exp * target_mask) / torch.sum(target_exp * target_mask, dim=-1, keepdim=True)

        lts_probs = lts_exp / torch.sum(lts_exp, dim=-1, keepdim=True)

        lts_probs += deliberation_cost
        lts_probs = (lts_probs * lts_mask) / torch.sum(lts_probs * lts_mask, dim=-1, keepdim=True)

        # *** critic *** #
        if self.is_lstm:
            value_state = torch.cat(
                [ids_one_hot_state[:, :, -1, :], global_state, policy_state_v],
                dim=-1)
        else:
            value_state = torch.cat(
                [ids_one_hot_state[:, :, -1, :], global_state, native_state[:, :, -1, :], state_tokens_out_flat[:, :, -1, :], self_msl_tokens_out_flat[:, :, -1, :],
                 bandit_msl_tokens_out_flat[:, :, -1, :]],
                dim=-1)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return hor_probs, ver_probs, shoot_probs, target_probs, v_head, policy_state, lts_probs, next_hiddens

    def hybrid_forward(self, policy_state, hor_c_one_hots, ver_c_one_hots):
        steer_state = torch.cat((policy_state, hor_c_one_hots, ver_c_one_hots), dim=-1)

        v_c_hiddens = self.activation(self.v_c_hiddens(steer_state))
        nn_c_hiddens = self.activation(self.nn_c_hiddens(steer_state))

        v_c_probs = torch.softmax(self.v_c_heads(v_c_hiddens), dim=-1)
        nn_c_probs = torch.softmax(self.nn_c_heads(nn_c_hiddens), dim=-1)

        return v_c_probs, nn_c_probs

    def select_action(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      ids_one_hot_state, hor_masks: list, ver_masks: list, shoot_masks: list, target_masks: list,
                      last_decision, lts_masks, deliberation_cost, hidden_h,
                      hidden_c):
        global_state = torch.FloatTensor(global_state).unsqueeze(0)
        native_state = torch.FloatTensor(native_state).unsqueeze(0)
        token_state = torch.FloatTensor(token_state).unsqueeze(0)  # add one dim #
        self_msl_token_state = torch.FloatTensor(self_msl_token_state).unsqueeze(0)
        bandit_msk_token_state = torch.FloatTensor(bandit_msk_token_state).unsqueeze(0)
        ids_one_hot_state = torch.FloatTensor(ids_one_hot_state).unsqueeze(0)
        last_decision = torch.FloatTensor(last_decision)
        deliberation_cost = torch.FloatTensor(deliberation_cost)
        hidden_h = torch.FloatTensor(hidden_h).unsqueeze(0)[:, :, 0, :]
        hidden_c = torch.FloatTensor(hidden_c).unsqueeze(0)[:, :, 0, :]

        # last_hor = torch.FloatTensor(last_hor)
        # last_ver = torch.FloatTensor(last_ver)

        hor_masks = torch.tensor(hor_masks, dtype=torch.float32)  # todo need squeeze 0 if single target
        ver_masks = torch.tensor(ver_masks, dtype=torch.float32)
        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)
        lts_masks = torch.tensor(lts_masks, dtype=torch.float32)

        hor_probs, ver_probs, shoot_probs, target_probs, _, policy_state, lts_probs, next_hiddens = self.macro_forward(
            global_state,
            native_state,
            token_state,
            self_msl_token_state,
            bandit_msk_token_state,
            ids_one_hot_state,
            hor_masks,
            ver_masks,
            shoot_masks,
            target_masks,
            last_decision, lts_masks, deliberation_cost,
            hidden_h,
            hidden_c)

        hor_probs = hor_probs.squeeze(0)
        ver_probs = ver_probs.squeeze(0)
        shoot_probs = shoot_probs.squeeze(0)
        target_probs = target_probs.squeeze(0)

        hors = (hor_probs * hor_masks + hor_masks * self.multinomial_protect).multinomial(1)
        vers = (ver_probs * ver_masks + ver_masks * self.multinomial_protect).multinomial(1)
        shoots = (shoot_probs * shoot_masks + shoot_masks * self.multinomial_protect).multinomial(1)
        targets = (target_probs * target_masks + target_masks * self.multinomial_protect).multinomial(1)

        # change to one hot #
        hor_one_hots = [index_to_one_hot(hors.tolist()[i][0], hor_probs.size(1))
                        for i in range(self.aircraft_num)]
        hor_one_hots = torch.tensor(hor_one_hots).detach()
        ver_one_hots = [index_to_one_hot(vers.tolist()[i][0], ver_probs.size(1))
                        for i in range(self.aircraft_num)]
        ver_one_hots = torch.tensor(ver_one_hots).detach()

        v_c_probs, nn_c_probs = self.hybrid_forward(policy_state, hor_one_hots, ver_one_hots)
        v_c = (v_c_probs + self.multinomial_protect).multinomial(1)
        nn_c = (nn_c_probs + self.multinomial_protect).multinomial(1)

        lts = (lts_probs + self.multinomial_protect).multinomial(1)

        return hors, vers, shoots, targets, v_c, nn_c, lts, next_hiddens

    def batch_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      ids_one_hot_state, hor_masks, ver_masks, shoot_masks, target_masks, hor_one_hots, ver_one_hots,
                      last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c):
        # function used for training #
        hor_probs, ver_probs, shoot_probs, target_probs, v_head, policy_state, lts_prob, next_hiddens = self.macro_forward(
            global_state,
            native_state,
            token_state,
            self_msl_token_state,
            bandit_msk_token_state,
            ids_one_hot_state,
            hor_masks,
            ver_masks,
            shoot_masks,
            target_masks, last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c)
        v_c_probs, nn_c_probs = self.hybrid_forward(policy_state, hor_one_hots, ver_one_hots)
        return hor_probs, ver_probs, shoot_probs, target_probs, v_c_probs, nn_c_probs, v_head, lts_prob

    def forward(self, x):
        pass

    def get_log_prob_and_values(self, x0, x1, x2, x3, x4, x5, hors, vers, shoots, targets, hor_masks, ver_masks,
                                shoot_masks, target_masks, hor_one_hots, ver_one_hots, v_cs, nn_cs, last_decision,
                                lts_masks, lts_choice, deliberation_cost, hidden_h, hidden_c):

        hor_probs, ver_probs, shoot_probs, target_probs, v_c_probs, nn_c_probs, value, lts_probs = self.batch_forward(
            x0, x1, x2, x3, x4, x5, hor_masks, ver_masks, shoot_masks, target_masks, hor_one_hots, ver_one_hots,
            last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c)

        lts_choice_c = lts_choice.clone()
        lts_choice_c[torch.isnan(lts_choice_c)] = 0
        lts_choice_c = lts_choice_c.unsqueeze(-1)

        hor = hors.clone()
        hor[torch.isnan(hor)] = 0
        value[torch.isnan(hor)] = 0  # if agent die, v = 0, cut backward
        hor = hor.unsqueeze(-1)
        ver = vers.clone()
        ver[torch.isnan(ver)] = 0
        ver = ver.unsqueeze(-1)
        s = shoots.clone()
        s[torch.isnan(s)] = 0
        s = s.unsqueeze(-1)
        t = targets.clone()
        t[torch.isnan(t)] = 0
        t = t.unsqueeze(-1)
        v = v_cs.clone()
        v[torch.isnan(v)] = 0
        v = v.unsqueeze(-1)
        nn_c = nn_cs.clone()
        nn_c[torch.isnan(nn_c)] = 0
        nn_c = nn_c.unsqueeze(-1)

        hor_probs = hor_probs.gather(-1, hor.long())
        hor_probs[torch.isnan(hors)] = 1
        hor_probs.squeeze(-1)
        ver_probs = ver_probs.gather(-1, ver.long())
        ver_probs[torch.isnan(vers)] = 1
        ver_probs.squeeze(-1)
        shoot_probs = shoot_probs.gather(-1, s.long())
        shoot_probs[torch.isnan(shoots)] = 1
        shoot_probs.squeeze(-1)
        target_probs = target_probs.gather(-1, t.long())
        target_probs[torch.isnan(targets)] = 1
        target_probs.squeeze(-1)
        v_c_probs = v_c_probs.gather(-1, v.long())
        v_c_probs[torch.isnan(v_cs)] = 1
        v_c_probs.squeeze(-1)
        nn_c_probs = nn_c_probs.gather(-1, nn_c.long())
        nn_c_probs[torch.isnan(nn_cs)] = 1
        nn_c_probs.squeeze(-1)
        lts_probs = lts_probs.gather(-1, lts_choice_c.long())
        lts_probs[torch.isnan(lts_choice)] = 1
        lts_probs.squeeze(-1)

        # todo origin re-normalize method
        # maneuver_probs = (maneuver_probs + Config.devide_protect) / (maneuver_probs_sum + Config.devide_protect)
        # target_probs = (target_probs + Config.devide_protect) / (target_probs_sum + Config.devide_protect)
        # shoot_probs = (shoot_probs + Config.devide_protect) / (shoot_probs_sum + Config.devide_protect)

        # ans = torch.log(torch.prod(maneuver_probs, 0) + self.log_protect) + \
        #       torch.log(torch.prod(shoot_probs, 0) + self.log_protect) + \
        #       torch.log(torch.prod(target_probs, 0) + self.log_protect)

        ans = torch.log(hor_probs + self.log_protect) + \
              torch.log(ver_probs + self.log_protect) + \
              torch.log(shoot_probs + self.log_protect) + \
              torch.log(target_probs + self.log_protect) + \
              torch.log(v_c_probs + self.log_protect) + \
              torch.log(nn_c_probs + self.log_protect) + \
              torch.log(lts_probs + self.log_protect)

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        # ans = torch.log(ans + self.log_protect)
        return ans, value


if __name__ == "__main__":
    from state_method import state_method_independ_refactor

    state = state_method_independ_refactor.get_kteam_aircraft_state(Config.env, 0)
    red_state_global = state[0]
    red_state_native = state[1]
    red_state_atten = state[2]
    msl_token_self = state[3]
    msl_token_bandit = state[4]
    id_one_hot_state = state[5]

    model_action_dims = dict(
        horizontal_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["horizontal_cmd"]["mask_len"],
        vertical_cmd_dim=Config.env.action_interface["AMS"][0]["SemanticManeuver"]["vertical_cmd"]["mask_len"],
        shoot_dim=0,
        target_dim=0,
        v_c_dim=len(Config.hybrid_v_c),
        nn_c_dim=len(Config.hybrid_nn_c)
    )

    model_action_dims["shoot_dim"] = Config.env.blue + 1
    model_action_dims["target_dim"] = Config.env.red + Config.env.blue - 1

    model = Release_NN(len(id_one_hot_state[0]),
                       len(red_state_global[0]),
                       len(red_state_native[0]),
                       len(red_state_atten[0][0]), len(red_state_atten[0]),
                       len(msl_token_self[0][0]), len(msl_token_self[0]),
                       len(msl_token_bandit[0][0]), len(msl_token_bandit[0]),
                       seq_len=1,
                       action_dims=model_action_dims,
                       is_lstm=True)

    hor_masks = [[1] * 8] * 2
    ver_masks = [[1] * 7] * 2
    shoot_masks = [[1] * 3] * 2
    target_masks = [[1] * 3] * 2
    hor_one_hots = [[1] * 8] * 2
    ver_one_hots = [[1] * 7] * 2
    last_decision = [[1.0] * 24] * 2
    lts_masks = [[1] * 2] * 2
    deliberation_cost = [0.0] * 2
    hidden_h = [[0.0] * 128] * 2
    hidden_c = [[0.0] * 128] * 2
    hor_masks = torch.tensor(hor_masks).unsqueeze(0)
    ver_masks = torch.tensor(ver_masks).unsqueeze(0)
    shoot_masks = torch.tensor(shoot_masks).unsqueeze(0)
    target_masks = torch.tensor(target_masks).unsqueeze(0)
    hor_one_hots = torch.tensor(hor_one_hots)
    ver_one_hots = torch.tensor(ver_one_hots)
    last_decision = torch.tensor(last_decision)
    lts_masks = torch.tensor(lts_masks).unsqueeze(0)
    deliberation_cost = torch.tensor(deliberation_cost).unsqueeze(0)
    hidden_h = torch.tensor(hidden_h).unsqueeze(0)
    hidden_c = torch.tensor(hidden_c).unsqueeze(0)

    red_state_global = torch.tensor(red_state_global).unsqueeze(0)
    red_state_native = torch.tensor(red_state_native).unsqueeze(0).unsqueeze(0).transpose(1, 2)
    red_state_atten = torch.tensor(red_state_atten).unsqueeze(0).unsqueeze(0).transpose(1, 2)
    msl_token_self = torch.tensor(msl_token_self).unsqueeze(0).unsqueeze(0).transpose(1, 2)
    msl_token_bandit = torch.tensor(msl_token_bandit).unsqueeze(0).unsqueeze(0).transpose(1, 2)
    id_one_hot_state = torch.tensor(id_one_hot_state).unsqueeze(0).unsqueeze(0).transpose(1, 2)

    # model.forward(red_state_global, red_state_native, red_state_atten, msl_token_self, msl_token_bandit,
    #               id_one_hot_state, hor_masks, ver_masks, shoot_masks, target_masks, hor_one_hots, ver_one_hots,
    #               last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c)

    input_t = (red_state_global, red_state_native, red_state_atten, msl_token_self, msl_token_bandit,
               id_one_hot_state, hor_masks, ver_masks, shoot_masks, target_masks, hor_one_hots, ver_one_hots,
               last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c)

    writer = SummaryWriter('runs/exp')
    # writer.add_graph(model, input_t)

    hor_probs, ver_probs, shoot_probs, target_probs, v_c_probs, nn_c_probs, v_head, lts_prob = \
        model.forward(red_state_global, red_state_native, red_state_atten, msl_token_self, msl_token_bandit,
                      id_one_hot_state, hor_masks, ver_masks, shoot_masks, target_masks, hor_one_hots, ver_one_hots,
                      last_decision, lts_masks, deliberation_cost, hidden_h, hidden_c)

    loss = hor_probs.mean() + ver_probs.mean() + shoot_probs.mean() + target_probs.mean() + v_c_probs.mean() + nn_c_probs.mean() + v_head.mean() + lts_prob.mean()
    loss.backward()

    for i, (name, param) in enumerate(model.named_parameters()):
        # if 'bn' not in name:
        # writer.add_histogram(name, param, 0)
        # writer.add_scalar('loss', loss, i)
        # loss = loss * 0.5
        print(i)
        writer.add_histogram(name, param, global_step=i)
        if param.grad is None:
            print(name)
        else:
            writer.add_histogram(name + "grad", param.grad, global_step=i)
