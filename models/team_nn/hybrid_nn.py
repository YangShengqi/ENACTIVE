# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config

from state_method.state_method import get_kteam_aircraft_state_for_attention, get_kteam_global_ground_truth_state, \
    get_kteam_msl_tokens


class Hybrid_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(maneuver_dim=0, shoot_dim=0, target_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(512, 512),
                 native_hidden_size=(512, 256),
                 policy_hidden_size=(256, 256, 128),
                 value_hidden_size=(256, 128),
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
        self.msl_token_embed_layer = nn.Linear(self_msl_token_dim,
                                               msl_token_embed_dim)  # todo consider self and bandit msl same dim

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
        flatten_dim_without_global = self.state_token_embed_dim * state_token_num + \
                                     self.msl_token_embed_dim * (self_msl_token_num + bandit_msl_token_num) + \
                                     last_native_dim
        flatten_dim_with_global = flatten_dim_without_global + last_global_dim

        self.policy_affine_layers = nn.ModuleList()
        self.value_affine_layers = nn.ModuleList()
        last_policy_dim = flatten_dim_without_global
        for dim in policy_hidden_size:
            self.policy_affine_layers.append(nn.Linear(last_policy_dim, dim))
            last_policy_dim = dim
        last_value_dim = flatten_dim_with_global
        for dim in value_hidden_size:
            self.value_affine_layers.append(nn.Linear(last_value_dim, dim))
            last_value_dim = dim

        # ** head output ** #
        value_head_hidden_size = int(last_value_dim / 4)
        self.value_head_hidden = nn.Linear(last_value_dim, value_head_hidden_size)
        self.value_head = nn.Linear(value_head_hidden_size, 1)

        action_head_hidden_size = int(last_policy_dim / 4)
        self.maneuver_hiddens = nn.ModuleList()
        self.shoot_hiddens = nn.ModuleList()
        self.target_hiddens = nn.ModuleList()
        self.v_c_hiddens = nn.ModuleList()
        self.nn_c_hiddens = nn.ModuleList()

        self.maneuver_heads = nn.ModuleList()
        self.shoot_heads = nn.ModuleList()
        self.target_heads = nn.ModuleList()
        self.v_c_heads = nn.ModuleList()
        self.nn_c_heads = nn.ModuleList()

        for _ in range(self.aircraft_num):
            self.maneuver_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))
            self.shoot_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))
            self.target_hiddens.append(nn.Linear(last_policy_dim, action_head_hidden_size))
            self.v_c_hiddens.append(nn.Linear(last_policy_dim + action_dims["maneuver_dim"], action_head_hidden_size))
            self.nn_c_hiddens.append(nn.Linear(last_policy_dim + action_dims["maneuver_dim"], action_head_hidden_size))

            self.maneuver_heads.append(nn.Linear(action_head_hidden_size, action_dims["maneuver_dim"]))
            self.shoot_heads.append(nn.Linear(action_head_hidden_size, action_dims["shoot_dim"]))
            self.target_heads.append(nn.Linear(action_head_hidden_size, action_dims["target_dim"]))
            self.v_c_heads.append(nn.Linear(action_head_hidden_size, action_dims["v_c_dim"]))
            self.nn_c_heads.append(nn.Linear(action_head_hidden_size, action_dims["nn_c_dim"]))

        # init layers #
        set_init(self.global_hidden_layers, method=init_method)  # global part
        set_init([self.state_token_embed_layer], method=init_method)  # atten part
        set_init([self.msl_token_embed_layer], method=init_method)

        # set_init([self.state_token_norm_layer], method=init_method)  # todo norm layer without init_method
        self.state_token_norm_layer.weight.requires_grad_(False)
        self.state_token_norm_layer.bias.requires_grad_(False)
        self.self_msl_token_norm_layer.weight.requires_grad_(False)
        self.self_msl_token_norm_layer.bias.requires_grad_(False)
        self.bandit_msl_token_norm_layer.weight.requires_grad_(False)
        self.bandit_msl_token_norm_layer.bias.requires_grad_(False)
        # set_init([self.msl_token_embed_layer], method=init_method)

        # set_init(self.state_attn_layers, method=init_method)
        # set_init(self.self_msl_token_norm_layer, method=init_method)
        # set_init(self.bandit_msl_attn_layers, method=init_method)

        set_init(self.w_q_state_token, method=init_method)
        set_init(self.w_k_state_token, method=init_method)
        set_init(self.w_v_state_token, method=init_method)
        set_init(self.w_q_self_msl_token, method=init_method)
        set_init(self.w_k_self_msl_token, method=init_method)
        set_init(self.w_v_self_msl_token, method=init_method)
        set_init(self.w_q_bandit_msl_token, method=init_method)
        set_init(self.w_k_bandit_msl_token, method=init_method)
        set_init(self.w_v_bandit_msl_token, method=init_method)

        set_init(self.native_hidden_layers, method=init_method)  # native part
        set_init(self.policy_affine_layers, method=init_method)  # cat part
        set_init(self.value_affine_layers, method=init_method)

        set_init(self.shoot_hiddens, method=init_method)  # output part
        set_init(self.target_hiddens, method=init_method)
        set_init(self.maneuver_hiddens, method=init_method)
        set_init(self.v_c_hiddens, method=init_method)
        set_init(self.nn_c_hiddens, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init(self.shoot_heads, method=init_method)
        set_init(self.target_heads, method=init_method)
        set_init(self.maneuver_heads, method=init_method)
        set_init(self.v_c_heads, method=init_method)
        set_init(self.nn_c_heads, method=init_method)
        set_init([self.value_head], method=init_method)

    def forward_with_mask(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state,
                          target_mask, maneuver_mask, shoot_mask):
        # *** attention forward *** #
        token_state = token_state.transpose(0, 1)
        # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

        token_embedding = self.state_token_embed_layer(token_state)
        self_msl_token_embedding = self.msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.msl_token_embed_layer(bandit_msl_token_state)

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

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** flat and cat *** #
        state_tokens_out_flat = torch.flatten(state_tokens_out, start_dim=-2, end_dim=-1)
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        policy_state = torch.cat(
            [native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        value_state = torch.cat(
            [global_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat],
            dim=-1)

        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            # print("policy_state", policy_state)
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        m_hiddens = [self.activation(self.maneuver_hiddens[i](policy_state)) for i in range(self.aircraft_num)]
        s_hiddens = [self.activation(self.shoot_hiddens[i](policy_state)) for i in range(self.aircraft_num)]
        t_hiddens = [self.activation(self.target_hiddens[i](policy_state)) for i in range(self.aircraft_num)]

        # *** add mask operation *** #

        maneuver_exp = torch.cat([torch.exp(self.maneuver_heads[i](m_hiddens[i])).unsqueeze(0)
                                 for i in range(self.aircraft_num)], 0)  # todo why for times
        shoot_exp = torch.cat([torch.exp(self.shoot_heads[i](s_hiddens[i])).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)
        target_exp = torch.cat([torch.exp(self.target_heads[i](t_hiddens[i])).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)

        # print(maneuver_exp.shape)
        # print(maneuver_mask.shape)
        # print("multi", maneuver_exp * maneuver_mask)
        maneuver_probs = (maneuver_exp * maneuver_mask) / torch.sum(maneuver_exp * maneuver_mask, dim=-1, keepdim=True)
        shoot_probs = (shoot_exp * shoot_mask) / torch.sum(shoot_exp * shoot_mask, dim=-1, keepdim=True)
        target_probs = (target_exp * target_mask) / torch.sum(target_exp * target_mask, dim=-1, keepdim=True)

        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        # print("target_prob", target_probs)
        # print("target_hidden_prob", t_hiddens)
        # print(maneuver_mask, maneuver_probs)
        # print(shoot_mask, shoot_probs)
        # print(target_mask, target_probs)

        return maneuver_probs, shoot_probs, target_probs, v_head

    def forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state):

        # *** attention forward *** #
        token_state = token_state.transpose(0, 1)  # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

        token_embedding = self.state_token_embed_layer(token_state)
        self_msl_token_embedding = self.msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.msl_token_embed_layer(bandit_msl_token_state)

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

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** flat and cat *** #
        state_tokens_out_flat = torch.flatten(state_tokens_out, start_dim=-2, end_dim=-1)
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        policy_state = torch.cat([native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        value_state = torch.cat([global_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)

        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            # print("policy_state", policy_state)
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        m_hiddens = [self.activation(self.maneuver_hiddens[i](policy_state)) for i in range(self.aircraft_num)]
        s_hiddens = [self.activation(self.shoot_hiddens[i](policy_state)) for i in range(self.aircraft_num)]
        t_hiddens = [self.activation(self.target_hiddens[i](policy_state)) for i in range(self.aircraft_num)]
        maneuver_probs = torch.cat([torch.softmax(self.maneuver_heads[i](m_hiddens[i]), dim=-1).unsqueeze(0)
                                    for i in range(self.aircraft_num)], 0)
        shoot_probs = torch.cat([torch.softmax(self.shoot_heads[i](s_hiddens[i]), dim=-1).unsqueeze(0)
                                 for i in range(self.aircraft_num)], 0)
        target_probs = torch.cat([torch.softmax(self.target_heads[i](t_hiddens[i]), dim=-1).unsqueeze(0)
                                  for i in range(self.aircraft_num)], 0)

        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        #print("m_prob", maneuver_probs.tolist()[0])
        # print("target_hidden_prob", t_hiddens)

        return maneuver_probs, shoot_probs, target_probs, v_head

    def hybrid_forward(self, native_state, token_state, self_msl_token_state, bandit_msl_token_state, maneuver_one_hots, sample):
        # *** attention forward *** #
        token_state = token_state.transpose(0, 1)  # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

        token_embedding = self.state_token_embed_layer(token_state)
        self_msl_token_embedding = self.msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.msl_token_embed_layer(bandit_msl_token_state)

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

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** flat and cat *** #
        state_tokens_out_flat = torch.flatten(state_tokens_out, start_dim=-2, end_dim=-1)
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        policy_state = torch.cat(
            [native_state, state_tokens_out_flat, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)

        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            # print("policy_state", policy_state)
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        if sample:
            steer_state = [torch.cat((policy_state, maneuver_one_hots[i]), dim=-1) for i in range(maneuver_one_hots.shape[0])]
        else:
            maneuver_one_hots = maneuver_one_hots.transpose(0, 1)
            steer_state = [torch.cat((policy_state, maneuver_one_hots[i]), dim=-1) for i in range(maneuver_one_hots.shape[0])]

        v_c_hiddens = [self.activation(self.v_c_hiddens[i](steer_state[i])) for i in range(self.aircraft_num)]
        nn_c_hiddens = [self.activation(self.nn_c_hiddens[i](steer_state[i])) for i in range(self.aircraft_num)]

        v_c_probs = torch.cat([torch.softmax(self.v_c_heads[i](v_c_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)], 0)
        nn_c_probs = torch.cat([torch.softmax(self.nn_c_heads[i](nn_c_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)], 0)

        return v_c_probs, nn_c_probs

    def select_action(self, x0, x1, x2, x3, x4, maneuver_masks: list, shoot_masks: list, target_masks: list, mask_forward=True):
        x0 = torch.FloatTensor(x0).unsqueeze(0)
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #
        x3 = torch.FloatTensor(x3).unsqueeze(0)
        x4 = torch.FloatTensor(x4).unsqueeze(0)
        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)  # todo need squeeze 0 if single target
        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)

        if mask_forward:
            maneuver_probs, shoot_probs, target_probs, _ = self.forward_with_mask(x0, x1, x2, x3, x4,
                                                                                  target_masks, maneuver_masks,
                                                                                  shoot_masks)
        else:
            maneuver_probs, shoot_probs, target_probs, _ = self.forward(x0, x1, x2, x3, x4)

        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        # print(target_masks, target_probs)

        # maneuvers = []
        # shoots = []
        # targets = []
        maneuvers = (maneuver_probs * maneuver_masks + maneuver_masks * self.multinomial_protect).multinomial(1)
        shoots = (shoot_probs * shoot_masks + shoot_masks * self.multinomial_protect).multinomial(1)
        targets = (target_probs * target_masks + target_masks * self.multinomial_protect).multinomial(1)

        return maneuvers, shoots, targets

    def hybrid_select_action(self, x1, x2, x3, x4, maneuver_one_hots, sample=True):
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #
        x3 = torch.FloatTensor(x3).unsqueeze(0)
        x4 = torch.FloatTensor(x4).unsqueeze(0)

        v_c_probs, nn_c_probs = self.hybrid_forward(x1, x2, x3, x4, maneuver_one_hots, sample)
        v_c_probs.squeeze(0)
        nn_c_probs.squeeze(0)

        v_c = (v_c_probs + self.multinomial_protect).multinomial(1)

        nn_c = (nn_c_probs + self.multinomial_protect).multinomial(1)

        return v_c, nn_c

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

    def get_log_prob_and_values(self, x0, x1, x2, x3, x4, maneuvers, shoots, targets, maneuver_masks, shoot_masks, target_masks, mask_forward=True):

        if mask_forward:
            maneuver_masks = maneuver_masks.transpose(0, 1)
            shoot_masks = shoot_masks.transpose(0, 1)
            target_masks = target_masks.transpose(0, 1)
            maneuver_probs, shoot_probs, target_probs, value = self.forward_with_mask(x0, x1, x2, x3, x4,
                                                                                      target_masks, maneuver_masks,
                                                                                      shoot_masks)
        else:
            maneuver_probs, shoot_probs, target_probs, value = self.forward(x0, x1, x2, x3, x4)

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

    def get_hybrid_log_prob(self, x1, x2, x3, x4, v_cs, nn_cs, maneuver_one_hots, sample):

        v_c_probs, nn_c_probs = self.hybrid_forward(x1, x2, x3, x4, maneuver_one_hots, sample)

        v_cs = torch.t(v_cs)
        nn_cs = torch.t(nn_cs)

        v = v_cs.clone()
        v[torch.isnan(v)] = 0
        v = v.unsqueeze(-1)
        nn = nn_cs.clone()
        nn[torch.isnan(nn)] = 0
        nn = nn.unsqueeze(-1)

        v_c_probs = v_c_probs.gather(-1, v.long())
        v_c_probs[torch.isnan(v_cs)] = 1
        v_c_probs.squeeze(-1)
        nn_c_probs = nn_c_probs.gather(-1, nn.long())
        nn_c_probs[torch.isnan(nn_cs)] = 1
        nn_c_probs.squeeze(-1)

        ans = torch.log(torch.prod(v_c_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(nn_c_probs, 0) + self.log_protect)

        return ans


if __name__ == "__main__":
    # testing code below #

    env = Config.env
    env.reset()

    red_global_state = get_kteam_global_ground_truth_state(env, 0)
    red_atten_state = get_kteam_aircraft_state_for_attention(env, 0)
    msl_token_self = get_kteam_msl_tokens(env, 0)
    msl_token_bandit = get_kteam_msl_tokens(env, 1)

    action_dims = dict(maneuver_dim=12, shoot_dim=2, target_dim=4)

    net_nn = Hybrid_NN(len(red_global_state), len(red_atten_state[0]),
                                    len(red_atten_state[1][0]), len(red_atten_state[1]),
                                    len(msl_token_self[0]), len(msl_token_self),
                                    len(msl_token_bandit[0]), len(msl_token_bandit), action_dims=action_dims)
    print(net_nn)

    a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0),
                             torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0))
    # ground_truth_dim, native_dim, state_token_dim, state_token_num,
    # self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
    # action_dims = dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
    #
    # a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0))
    print(a, b, c, d)
    #
    # # print(net_nn)

    e,f,g = net_nn.select_action(red_global_state, red_atten_state[0], red_atten_state[1], msl_token_self, msl_token_bandit, [1]*12, [1]*2, [1]*4)
    print(e,f,g)
