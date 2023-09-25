# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config

from state_method.state_method_refactor import get_kteam_aircraft_state


class LSTM_Discrete_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
                 ground_truth_size_before_cat=(128, 256),
                 native_hidden_size=(128, 128),
                 # policy_hidden_size=(256, 256, 256),
                 policy_hidden_layer_num=2,  # use residue network, no need to add dim
                 value_hidden_size=(128, 64),
                 state_token_embed_dim=64, state_token_num_heads=4, atten_depth=2,
                 msl_token_embed_dim=32, msl_token_num_heads=2,
                 activation='tanh', init_method='xavier', aircraft_num=2, msl_per_aircraft=4):
        super().__init__()

        self.log_protect = Config.log_protect
        self.multinomial_protect = Config.multinomial_protect
        self.init_method = init_method
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
        self.state_token_embed_dim = state_token_embed_dim
        self.state_token_num_heads = state_token_num_heads

        self.msl_token_embed_dim = msl_token_embed_dim
        self.msl_token_num_heads = msl_token_num_heads

        self.state_token_embed_layer = nn.Linear(state_token_dim, state_token_embed_dim)
        self.self_msl_token_embed_layer = nn.Linear(self_msl_token_dim, msl_token_embed_dim)  #todo consider self and bandit msl same dim
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

        # ** 4.cat native and tokens ** #
        # flatten_dim_without_global = last_native_dim + self.state_token_embed_dim * aircraft_num + self.msl_token_embed_dim + self.msl_token_embed_dim
        flatten_dim_without_global = last_native_dim + self.state_token_embed_dim * aircraft_num + \
                                     self.msl_token_embed_dim * aircraft_num * msl_per_aircraft + \
                                     self.msl_token_embed_dim * aircraft_num * msl_per_aircraft

        # ** 5.global output to lstm layer ** #
        self.lstm_layer = nn.LSTM(input_size=flatten_dim_without_global, hidden_size=flatten_dim_without_global,
                                  num_layers=1, batch_first=True)
        self.lstm_hidden_size = flatten_dim_without_global  # agent read this to get hidden size

        self.policy_affine_residue_norm_layer = nn.LayerNorm([flatten_dim_without_global])

        # ** 6.output value and policy ** #
        flatten_dim_with_global = flatten_dim_without_global + last_global_dim

        self.policy_affine_layers = nn.ModuleList()
        self.value_affine_layers = nn.ModuleList()
        last_policy_dim = flatten_dim_without_global
        for _ in range(policy_hidden_layer_num):
            self.policy_affine_layers.append(nn.Linear(last_policy_dim, last_policy_dim))

        last_value_dim = flatten_dim_with_global
        for dim in value_hidden_size:
            self.value_affine_layers.append(nn.Linear(last_value_dim, dim))
            last_value_dim = dim

        self.value_head = nn.Linear(last_value_dim, 1)

        # # ** head output ** #
        # value_head_hidden_size = int(last_value_dim / 4)
        # self.value_head_hidden = nn.Linear(last_value_dim, value_head_hidden_size)
        # self.value_head = nn.Linear(value_head_hidden_size, 1)  # todo slim agent

        action_head_hidden_size = int(last_policy_dim / 4)
        self.maneuver_hiddens = nn.ModuleList()
        self.shoot_hiddens = nn.ModuleList()
        self.target_hiddens = nn.ModuleList()

        self.maneuver_heads = nn.ModuleList()
        self.shoot_heads = nn.ModuleList()
        self.target_heads = nn.ModuleList()

        for _ in range(self.aircraft_num):
            self.maneuver_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))
            self.shoot_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))
            self.target_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))

            self.maneuver_heads.append(nn.Linear(action_head_hidden_size, action_dims["maneuver_dim"]))
            self.shoot_heads.append(nn.Linear(action_head_hidden_size, action_dims["shoot_dim"]))
            self.target_heads.append(nn.Linear(action_head_hidden_size, action_dims["target_dim"]))

        self._init_layers()

    def _init_layers(self):

        set_init(self.global_hidden_layers, method=self.init_method)  # global part #
        set_init(self.native_hidden_layers, method=self.init_method)  # native part #
        set_init([self.state_token_embed_layer,
                  self.self_msl_token_embed_layer,
                  self.bandit_msl_token_embed_layer], method=self.init_method)  # token embed layers #

        set_init(self.w_q_state_token, method=self.init_method)
        set_init(self.w_k_state_token, method=self.init_method)
        set_init(self.w_v_state_token, method=self.init_method)
        set_init(self.w_q_self_msl_token, method=self.init_method)
        set_init(self.w_k_self_msl_token, method=self.init_method)
        set_init(self.w_v_self_msl_token, method=self.init_method)
        set_init(self.w_q_bandit_msl_token, method=self.init_method)
        set_init(self.w_k_bandit_msl_token, method=self.init_method)
        set_init(self.w_v_bandit_msl_token, method=self.init_method)

        self.state_token_norm_layer.weight.requires_grad_(False)   # todo norm layer without init_method
        self.state_token_norm_layer.bias.requires_grad_(False)
        self.self_msl_token_norm_layer.weight.requires_grad_(False)
        self.self_msl_token_norm_layer.bias.requires_grad_(False)
        self.bandit_msl_token_norm_layer.weight.requires_grad_(False)
        self.bandit_msl_token_norm_layer.bias.requires_grad_(False)
        self.policy_affine_residue_norm_layer.weight.requires_grad_(False)
        self.policy_affine_residue_norm_layer.bias.requires_grad_(False)
        # set_init([self.msl_token_embed_layer], method=init_method)

        set_init(self.policy_affine_layers, method=self.init_method)  # after cat to output part
        set_init(self.value_affine_layers, method=self.init_method)

        set_init(self.shoot_hiddens, method=self.init_method)  # output head part
        set_init(self.target_hiddens, method=self.init_method)
        set_init(self.maneuver_hiddens, method=self.init_method)
        set_init(self.shoot_heads, method=self.init_method)
        set_init(self.target_heads, method=self.init_method)
        set_init(self.maneuver_heads, method=self.init_method)
        set_init([self.value_head], method=self.init_method)

    def forward_with_mask(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state,
                          hidden_h, hidden_c,
                          target_mask, maneuver_mask, shoot_mask):
        # *** attention forward *** #
        # print(token_state.size())
        batch_num = token_state.size(0)
        seq_len = token_state.size(1)
        token_state = token_state.reshape(-1, token_state.size(-2), token_state.size(-1)).transpose(0, 1)
        # print(token_state.size())
        # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        self_msl_token_state = self_msl_token_state.reshape(-1, self_msl_token_state.size(-2), self_msl_token_state.size(-1)).transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.reshape(-1, bandit_msl_token_state.size(-2), bandit_msl_token_state.size(-1)).transpose(0, 1)

        # token_state = token_state.transpose(0, 1)
        # # print(token_state.size())
        # self_msl_token_state = self_msl_token_state.transpose(0, 1)
        # bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

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

            # print("q_self_msl size", q_self_msl.size())
            state_tokens_out, _ = self.state_attn_layers[i](q_state, k_state, v_state)
            self_msl_tokens_out, _ = self.self_msl_attn_layers[i](q_self_msl, k_self_msl, v_self_msl)
            bandit_msl_tokens_out, _ = self.bandit_msl_attn_layers[i](q_bandit_msl, k_bandit_msl, v_bandit_msl)  #
            # print(tokens_out.size())  # todo problems here of dimention operation
            # state_token_sum = state_tokens_out + token_embedding
            # self_msl_token_sum = self_msl_tokens_out + self_msl_token_embedding
            # bandit_msl_token_sum = bandit_msl_tokens_out + bandit_msl_token_embedding

            token_embedding = self.state_token_norm_layer(state_tokens_out + token_embedding)
            self_msl_token_embedding = self.self_msl_token_norm_layer(self_msl_tokens_out + self_msl_token_embedding)
            bandit_msl_token_embedding = self.bandit_msl_token_norm_layer(bandit_msl_tokens_out + bandit_msl_token_embedding)

        # print("token_embedding size", token_embedding.size())
        token_embedding = token_embedding.transpose(0, 1)
        # print("token_embedding size", token_embedding.size())
        self_msl_token_embedding = self_msl_token_embedding.transpose(0, 1)
        bandit_msl_token_embedding = bandit_msl_token_embedding.transpose(0, 1)

        token_embedding = token_embedding.reshape(batch_num, seq_len, token_embedding.size(-2), token_embedding.size(-1))
        # print("token_embedding size", token_embedding.size())
        self_msl_token_embedding = self_msl_token_embedding.reshape(batch_num, seq_len, self_msl_token_embedding.size(-2), self_msl_token_embedding.size(-1))
        bandit_msl_token_embedding = bandit_msl_token_embedding.reshape(batch_num, seq_len, bandit_msl_token_embedding.size(-2), bandit_msl_token_embedding.size(-1))

        state_tokens_out = token_embedding
        # print("state_tokens_out", state_tokens_out.size())
        self_msl_tokens_out = self_msl_token_embedding
        bandit_msl_tokens_out = bandit_msl_token_embedding

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** flat and cat *** #
        # print(state_tokens_out.size(), "token_out_size")
        # state_tokens_out_flat = torch.flatten(state_tokens_out, start_dim=-2, end_dim=-1)
        # self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        # bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        # state_tokens_out_mean = torch.mean(state_tokens_out, dim=2, dtype=torch.float32)  # use mean pooling
        # print(state_tokens_out.size())

        state_tokens_out_mean_0 = torch.mean(state_tokens_out[:, :, 0:2, :], dim=2, dtype=torch.float32)  # todo use segment mean pooling ****** #
        state_tokens_out_mean_1 = torch.mean(state_tokens_out[:, :, 3:5, :], dim=2, dtype=torch.float32)

        # self_msl_tokens_mean = torch.mean(self_msl_tokens_out, dim=2, dtype=torch.float32)
        # bandit_msl_tokens_mean = torch.mean(bandit_msl_tokens_out, dim=2, dtype=torch.float32)
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)  # todo consider mean pooling losing info #
        # todo todo
        # print(state_tokens_out_mean.size(), "token_mean_size")
        # print(self_msl_tokens_mean.size(), "self_msl_tokens_out_size")
        # print(bandit_msl_tokens_mean.size(), "bandit_msl_tokens_out_size")

        # policy_state = torch.cat(
        #     [native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        policy_state = torch.cat(
            [native_state, state_tokens_out_mean_0, state_tokens_out_mean_1, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        # value_state = torch.cat(
        #     [global_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat],
        #     dim=-1)
        # print(policy_state.size())
        # policy_state = policy_state.unsqueeze(0)  # dim[batch_size, seq_len, logic]
        # print(policy_state.size())
        policy_state_out, _ = self.lstm_layer(policy_state, (hidden_h, hidden_c))  # next_hidden # not used #
        # print(policy_state.size())
        # print((policy_state[:, 0, :].unsqueeze(1)).size())
        # get next hiddens # fix in 2020/09/14 #
        _, next_hiddens = self.lstm_layer(policy_state[:, 0, :].unsqueeze(1), (hidden_h, hidden_c))  # get next hiddens #
        policy_state_out = policy_state_out[:, -1, :]

        # print("policy state size", policy_state_out.shape)
        policy_state_out = policy_state_out.squeeze(0)
        # print("policy state size", policy_state_out.shape)
        # add residue in policy affine layers #
        for affine in self.policy_affine_layers:
            # print("policy_state", policy_state)
            policy_state_out_n = affine(policy_state_out)
            policy_state_out_n = self.activation(policy_state_out_n)
            policy_state_out = policy_state_out + policy_state_out_n  # need same dim policy affine layer
            policy_state_out = self.policy_affine_residue_norm_layer(policy_state_out)

        m_hiddens = [self.activation(self.maneuver_hiddens[i](policy_state_out)) for i in range(self.aircraft_num)]
        s_hiddens = [self.activation(self.shoot_hiddens[i](policy_state_out)) for i in range(self.aircraft_num)]
        t_hiddens = [self.activation(self.target_hiddens[i](policy_state_out)) for i in range(self.aircraft_num)]

        # *** add mask operation *** #

        maneuver_exp = torch.cat([torch.exp(self.maneuver_heads[i](m_hiddens[i])).unsqueeze(0)
                                 for i in range(self.aircraft_num)], 0)  # todo why for times
        shoot_exp = torch.cat([torch.exp(self.shoot_heads[i](s_hiddens[i])).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)
        target_exp = torch.cat([torch.exp(self.target_heads[i](t_hiddens[i])).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)

        # maneuver_exp = maneuver_exp.squeeze(-2)
        # shoot_exp = shoot_exp.squeeze(-2)384
        # target_exp = target_exp.squeeze(-2)
        # print("maneuver shape", maneuver_exp.shape)
        # print("mask shape", maneuver_mask.shape)
        # print("multi", maneuver_exp * maneuver_mask)
        maneuver_probs = (maneuver_exp * maneuver_mask + Config.devide_protect) / torch.sum(maneuver_exp * maneuver_mask + Config.devide_protect, dim=-1, keepdim=True)
        shoot_probs = (shoot_exp * shoot_mask + Config.devide_protect) / torch.sum(shoot_exp * shoot_mask + Config.devide_protect, dim=-1, keepdim=True)
        target_probs = (target_exp * target_mask + Config.devide_protect) / torch.sum(target_exp * target_mask + Config.devide_protect, dim=-1, keepdim=True)

        # value out, cat global state and lstm out state #
        # print(global_state.size())
        # print(policy_state_out.size())
        value_state = torch.cat([global_state, policy_state_out], dim=-1)
        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_head = self.value_head(value_state)

        # print("target_prob", target_probs)
        # print("target_hidden_prob", t_hiddens)
        # print(maneuver_mask, maneuver_probs)
        # print(shoot_mask, shoot_probs)
        # print(target_mask, target_probs)

        return maneuver_probs, shoot_probs, target_probs, v_head, next_hiddens

    def select_action(self, x0, x1, x2, x3, x4, h, c,
                      maneuver_masks: list, shoot_masks: list, target_masks: list, mask_forward=True):
        x0 = torch.FloatTensor(x0)
        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        x3 = torch.FloatTensor(x3)
        x4 = torch.FloatTensor(x4)
        h = torch.FloatTensor(h).unsqueeze(0).unsqueeze(0)
        c = torch.FloatTensor(c).unsqueeze(0).unsqueeze(0)

        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)  # todo need squeeze 0 if single target
        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)

        if mask_forward:
            # print(x0.size())
            # print(x1.size())
            # print(x2.size())
            # print(x3.size())
            # print(x4.size())
            maneuver_probs, shoot_probs, target_probs, _, next_hiddens = self.forward_with_mask(x0, x1, x2, x3, x4, h, c,
                                                                                               target_masks, maneuver_masks, shoot_masks)
            # print("probs")
            # print(maneuver_probs)
            # print(shoot_probs)
            # print(target_probs)

        else:
            # not implement
            pass

        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        # print(target_masks, target_probs)

        # maneuvers = []
        # shoots = []
        # targets = []
        # print("masks")
        # print(maneuver_masks)
        # print(shoot_masks)
        # print(target_masks)
        maneuvers = (maneuver_probs * maneuver_masks).multinomial(1)  # delete multinomial protect #
        shoots = (shoot_probs * shoot_masks).multinomial(1)
        targets = (target_probs * target_masks).multinomial(1)

        return maneuvers, shoots, targets, next_hiddens

    def select_action_after_target(self, x0, x1, x2, x3, x4, maneuver_masks: list, shoot_masks: list, target_masks: list):
        # todo mask forward method is not available for only enemy method 2020/07/17, only valid for target part #
        # shoot mask based on target, only use for choosing enemy as target and decide if shoot this target #
        x0 = torch.FloatTensor(x0).unsqueeze(0)
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #
        x3 = torch.FloatTensor(x3).unsqueeze(0)
        x4 = torch.FloatTensor(x4).unsqueeze(0)

        maneuver_probs, shoot_probs, target_probs, _ = self.forward(x0, x1, x2, x3, x4)
        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        # print(target_masks, target_probs)
        # print("")

        # maneuvers = []
        # shoots = []
        # targets = []
        target_masks = torch.tensor(target_masks, dtype=torch.float32)
        targets = (target_probs * target_masks + target_masks * self.multinomial_protect).multinomial(1)

        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)
        maneuvers = (maneuver_probs * maneuver_masks + maneuver_masks * self.multinomial_protect).multinomial(1)

        chosen_target = targets.tolist()

        shoot_target_masks = []
        for i in range(len(targets.tolist())):
            shoot_target_masks.append([1, shoot_masks[i][chosen_target[i][0] + 1]])

        shoot_target_masks = torch.tensor(shoot_target_masks, dtype=torch.float32)
        shoots = (shoot_probs * shoot_target_masks + shoot_target_masks * self.multinomial_protect).multinomial(1)

        return maneuvers, shoots, targets

    def get_log_prob_and_values(self, x0, x1, x2, x3, x4, h, c, maneuvers, shoots, targets, maneuver_masks, shoot_masks, target_masks, mask_forward=True):

        if mask_forward:
            maneuver_masks = maneuver_masks.transpose(0, 1)
            shoot_masks = shoot_masks.transpose(0, 1)
            target_masks = target_masks.transpose(0, 1)
            maneuver_probs, shoot_probs, target_probs, value, _ = self.forward_with_mask(x0, x1, x2, x3, x4, h, c,
                                                                                         target_masks, maneuver_masks,
                                                                                         shoot_masks)
            # maneuver_probs, shoot_probs, target_probs, value = self.segment_forward_with_mask(x0, x1, x2, x3, x4, h, c,
            #                                                                                      target_masks, maneuver_masks,
            #                                                                                      shoot_masks)
        else:
            pass
            # maneuver_probs, shoot_probs, target_probs, value = self.forward(x0, x1, x2, x3, x4)

        # todo origin re-normalize method
        # maneuver_masks = maneuver_masks.transpose(0, 1)
        # shoot_masks = shoot_masks.transpose(0, 1)
        # target_masks = target_masks.transpose(0, 1)
        # maneuver_probs_sum = torch.sum(maneuver_probs * maneuver_masks, -1, keepdim=True).detach()
        # shoot_probs_sum = torch.sum(shoot_probs * shoot_masks, -1, keepdim=True).detach()
        # target_probs_sum = torch.sum(target_probs * target_masks, -1, keepdim=True).detach()

        maneuvers = torch.t(maneuvers)
        shoots = torch.t(shoots)
        targets = torch.t(targets)

        m = maneuvers.clone()
        m[torch.isnan(m)] = 0
        m = m.unsqueeze(-1)
        s = shoots.clone()
        s[torch.isnan(s)] = 0
        s = s.unsqueeze(-1)
        t = targets.clone()
        t[torch.isnan(t)] = 0
        t = t.unsqueeze(-1)

        maneuver_probs = maneuver_probs.gather(-1, m.long())
        maneuver_probs[torch.isnan(maneuvers)] = 1
        maneuver_probs.squeeze(-1)
        shoot_probs = shoot_probs.gather(-1, s.long())
        shoot_probs[torch.isnan(shoots)] = 1
        shoot_probs.squeeze(-1)
        target_probs = target_probs.gather(-1, t.long())
        target_probs[torch.isnan(targets)] = 1
        target_probs.squeeze(-1)

        # todo origin re-normalize method
        # maneuver_probs = (maneuver_probs + Config.devide_protect) / (maneuver_probs_sum + Config.devide_protect)
        # target_probs = (target_probs + Config.devide_protect) / (target_probs_sum + Config.devide_protect)
        # shoot_probs = (shoot_probs + Config.devide_protect) / (shoot_probs_sum + Config.devide_protect)

        ans = torch.log(torch.prod(maneuver_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(shoot_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(target_probs, 0) + self.log_protect)

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        # ans = torch.log(ans + self.log_protect)
        return ans, value


if __name__ == "__main__":
    # testing code below #

    env = Config.env
    env.reset()

    state_refactor = get_kteam_aircraft_state(env, 0)
    red_global_state = state_refactor[0]
    red_native_state = state_refactor[1]
    state_relative_token = state_refactor[2]
    self_msl_token = state_refactor[3]
    bandit_msl_token = state_refactor[4]

    # red_global_state = get_kteam_global_ground_truth_state(env, 0)
    # red_atten_state = get_kteam_aircraft_state_for_attention(env, 0)
    # msl_token_self = get_kteam_msl_tokens(env, 0)
    # msl_token_bandit = get_kteam_msl_tokens(env, 1)

    action_dims = dict(maneuver_dim=12, shoot_dim=2, target_dim=4)

    net_nn = LSTM_Discrete_NN(len(red_global_state), len(red_native_state),
                              len(state_relative_token[0]), len(state_relative_token),
                              len(self_msl_token[0]), len(self_msl_token),
                              len(bandit_msl_token[0]), len(bandit_msl_token), action_dims=action_dims,
                              activation="relu", init_method="kaiming")

    # for param in net_nn.parameters():
    #     print(param)
    # print(net_nn)

    # a,b,c,d, hidden = net_nn.forward_with_mask(torch.tensor(red_global_state), torch.tensor(red_atten_state[0]).unsqueeze(0),
    #                                    torch.tensor(red_atten_state[1]).unsqueeze(0),
    #                                    torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0),
    #                                    torch.tensor([0] * 1368, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    #                                    torch.tensor([0] * 1368, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    #                                    torch.tensor([[1] * 4, [1] * 4], dtype=torch.float32),
    #                                    torch.tensor([[1] * 12, [1] * 12], dtype=torch.float32),
    #                                    torch.tensor([[1] * 2, [1] * 2], dtype=torch.float32))
    #
    # print(a, a.size())
    # print(b, b.size())
    # print(c, c.size())
    # print(d, d.size())

    # sequence length size
    input_0 = red_global_state
    input_1 = [[red_native_state, red_native_state]]
    input_2 = [[state_relative_token, state_relative_token]]
    input_3 = [[self_msl_token, self_msl_token]]
    input_4 = [[bandit_msl_token, bandit_msl_token]]

    e, f, g, h, hidden_1 = net_nn.forward_with_mask(torch.tensor(input_0),
                                          torch.tensor(input_1),
                                          torch.tensor(input_2),
                                          torch.tensor(input_3),
                                          torch.tensor(input_4),
                                          torch.tensor([0] * 448, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                          torch.tensor([0] * 448, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                          torch.tensor([[1] * 4, [1] * 4], dtype=torch.float32),
                                          torch.tensor([[1] * 12, [1] * 12], dtype=torch.float32),
                                          torch.tensor([[1] * 2, [1] * 2], dtype=torch.float32))
    print(e, e.size())
    print(f, f.size())
    print(g, g.size())
    print(h, h.size())

    # i, j, k, h = net_nn.select_action(input_0, input_1, input_2, input_3, input_4, [0] * 1368, [0] * 1368,
    #                                [[1] * 12, [1] * 12], [[1] * 2, [1] * 2], [[1] * 4, [1] * 4])

    # print(i)
    # print(j)
    # print(k)
    #
    # e,f,g,h = net_nn.forward_with_mask(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0),
    #                                    torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0),
    #                                    torch.tensor([[1] * 4, [1] * 4]),
    #                                    torch.tensor([[1] * 12, [1] * 12]),
    #                                    torch.tensor([[1] * 2, [1] * 2]))
    # # ground_truth_dim, native_dim, state_token_dim, state_token_num,
    # # self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
    # # action_dims = dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
    # #
    # # a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0))
    # # print(e,f,g,h)
    # #
    # i,j,k = net_nn.select_action(red_global_state, red_atten_state[0], red_atten_state[1], msl_token_self, msl_token_bandit, [[1] * 4, [1] * 4], [[1] * 12, [1] * 12], [[1] * 2, [1] * 2])
    # # # print(net_nn)
    # print(i,j,k)
    #
    # # e,f,g = net_nn.select_action(red_global_state, red_atten_state[0], red_atten_state[1], msl_token_self, msl_token_bandit, [1]*12, [1]*2, [1]*4)
    # # print(e,f,g)






