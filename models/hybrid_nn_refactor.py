# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config

from state_method.state_method_refactor import get_kteam_aircraft_state
from utils.math import index_to_one_hot
from algorithm.layer_parameter_disturb import tune_layer_parameters


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
            self.v_c_hiddens.append(nn.Linear(last_policy_dim + action_dims["maneuver_dim"] + action_dims["shoot_dim"] + action_dims["target_dim"], action_head_hidden_size))
            self.nn_c_hiddens.append(nn.Linear(last_policy_dim + action_dims["maneuver_dim"] + action_dims["shoot_dim"] + action_dims["target_dim"], action_head_hidden_size))

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

    def macro_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      target_mask, maneuver_mask, shoot_mask):
        # *** attention forward *** #
        token_state = token_state.transpose(0, 1)
        # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msk_token_state = bandit_msk_token_state.transpose(0, 1)

        token_embedding = self.state_token_embed_layer(token_state)
        self_msl_token_embedding = self.msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.msl_token_embed_layer(bandit_msk_token_state)

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

        maneuver_probs = (maneuver_exp * maneuver_mask) / torch.sum(maneuver_exp * maneuver_mask, dim=-1, keepdim=True)
        shoot_probs = (shoot_exp * shoot_mask) / torch.sum(shoot_exp * shoot_mask, dim=-1, keepdim=True)
        target_probs = (target_exp * target_mask) / torch.sum(target_exp * target_mask, dim=-1, keepdim=True)

        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return maneuver_probs, shoot_probs, target_probs, v_head, policy_state

    def hybrid_forward(self, policy_state, maneuver_one_hots, target_one_hots, shoot_one_hots):
        steer_state = [torch.cat((policy_state, maneuver_one_hots[i], target_one_hots[i], shoot_one_hots[i]), dim=-1)
                       for i in range(self.aircraft_num)]

        v_c_hiddens = [self.activation(self.v_c_hiddens[i](steer_state[i])) for i in range(self.aircraft_num)]
        nn_c_hiddens = [self.activation(self.nn_c_hiddens[i](steer_state[i])) for i in range(self.aircraft_num)]

        v_c_probs = torch.cat([torch.softmax(self.v_c_heads[i](v_c_hiddens[i]), dim=-1).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)
        nn_c_probs = torch.cat([torch.softmax(self.nn_c_heads[i](nn_c_hiddens[i]), dim=-1).unsqueeze(0)
                               for i in range(self.aircraft_num)], 0)

        return v_c_probs, nn_c_probs

    def select_action(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      target_masks, maneuver_masks, shoot_masks):

        global_state = torch.FloatTensor(global_state).unsqueeze(0)
        native_state = torch.FloatTensor(native_state).unsqueeze(0)
        token_state = torch.FloatTensor(token_state).unsqueeze(0)  # add one dim #
        self_msl_token_state = torch.FloatTensor(self_msl_token_state).unsqueeze(0)
        bandit_msk_token_state = torch.FloatTensor(bandit_msk_token_state).unsqueeze(0)
        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)  # todo need squeeze 0 if single target
        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        target_masks = torch.tensor(target_masks, dtype=torch.float32)

        maneuver_probs, shoot_probs, target_probs, _, policy_state = \
            self.macro_forward(global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                               target_masks, maneuver_masks, shoot_masks)
        print(maneuver_probs, shoot_probs, target_probs)

        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        maneuvers = (maneuver_probs * maneuver_masks + maneuver_masks * self.multinomial_protect).multinomial(1)
        shoots = (shoot_probs * shoot_masks + shoot_masks * self.multinomial_protect).multinomial(1)
        targets = (target_probs * target_masks + target_masks * self.multinomial_protect).multinomial(1)

        # change to one hot #
        maneuver_one_hots = [index_to_one_hot(maneuvers.tolist()[i][0], maneuver_probs.size(1))
                             for i in range(self.aircraft_num)]
        target_one_hots = [index_to_one_hot(targets.tolist()[i][0], target_probs.size(1))
                             for i in range(self.aircraft_num)]
        shoot_one_hots = [index_to_one_hot(shoots.tolist()[i][0], shoot_probs.size(1))
                             for i in range(self.aircraft_num)]
        maneuver_one_hots = torch.tensor(maneuver_one_hots)
        target_one_hots = torch.tensor(target_one_hots)
        shoot_one_hots = torch.tensor(shoot_one_hots)

        v_c_probs, nn_c_probs = self.hybrid_forward(policy_state, maneuver_one_hots, target_one_hots, shoot_one_hots)
        v_c = (v_c_probs + self.multinomial_protect).multinomial(1)
        nn_c = (nn_c_probs + self.multinomial_protect).multinomial(1)

        return maneuvers, shoots, targets, v_c, nn_c

    def batch_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                      target_masks, maneuver_masks, shoot_masks,
                      target_one_hots, maneuver_one_hots, shoot_one_hots):
        # function used for training #
        maneuver_probs, shoot_probs, target_probs, v_head, policy_state = self.macro_forward(global_state, native_state, token_state, self_msl_token_state, bandit_msk_token_state,
                                                                                             target_masks, maneuver_masks, shoot_masks)
        v_c_probs, nn_c_probs = self.hybrid_forward(policy_state, maneuver_one_hots, target_one_hots, shoot_one_hots)
        return maneuver_probs, shoot_probs, target_probs, v_c_probs, nn_c_probs, v_head

    def get_log_prob_and_values(self, x0, x1, x2, x3, x4, maneuvers, shoots, targets,
                                maneuver_masks, shoot_masks, target_masks,
                                target_one_hots, maneuver_one_hots, shoot_one_hots,
                                v_cs, nn_cs):

        maneuver_masks = maneuver_masks.transpose(0, 1)
        shoot_masks = shoot_masks.transpose(0, 1)
        target_masks = target_masks.transpose(0, 1)
        maneuver_one_hots = maneuver_one_hots.transpose(0, 1)
        shoot_one_hots = shoot_one_hots.transpose(0, 1)
        target_one_hots = target_one_hots.transpose(0, 1)

        maneuver_probs, shoot_probs, target_probs, v_c_probs, nn_c_probs, value = self.batch_forward(x0, x1, x2, x3, x4,
                                                                                                     target_masks, maneuver_masks, shoot_masks,
                                                                                                     target_one_hots, maneuver_one_hots, shoot_one_hots)

        maneuvers = torch.t(maneuvers)
        shoots = torch.t(shoots)
        targets = torch.t(targets)
        v_cs = torch.t(v_cs)
        nn_cs = torch.t(nn_cs)

        m = maneuvers.clone()
        m[torch.isnan(m)] = 0
        m = m.unsqueeze(-1)
        s = shoots.clone()
        s[torch.isnan(s)] = 0
        s = s.unsqueeze(-1)
        t = targets.clone()
        t[torch.isnan(t)] = 0
        t = t.unsqueeze(-1)
        v = v_cs.clone()
        v[torch.isnan(v)] = 0
        v = v.unsqueeze(-1)
        nn = nn_cs.clone()
        nn[torch.isnan(nn)] = 0
        nn = nn.unsqueeze(-1)

        maneuver_probs = maneuver_probs.gather(-1, m.long())
        maneuver_probs[torch.isnan(maneuvers)] = 1
        maneuver_probs.squeeze(-1)
        shoot_probs = shoot_probs.gather(-1, s.long())
        shoot_probs[torch.isnan(shoots)] = 1
        shoot_probs.squeeze(-1)
        target_probs = target_probs.gather(-1, t.long())
        target_probs[torch.isnan(targets)] = 1
        target_probs.squeeze(-1)

        v_c_probs = v_c_probs.gather(-1, v.long())
        v_c_probs[torch.isnan(v_cs)] = 1
        v_c_probs.squeeze(-1)
        nn_c_probs = nn_c_probs.gather(-1, nn.long())
        nn_c_probs[torch.isnan(nn_cs)] = 1
        nn_c_probs.squeeze(-1)

        ans = torch.log(torch.prod(maneuver_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(shoot_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(target_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(v_c_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(nn_c_probs, 0) + self.log_protect)

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        # ans = torch.log(ans + self.log_protect)
        return ans, value

    def param_tune(self):
        layer_array = [self.global_hidden_layers, self.state_token_embed_layer, self.msl_token_embed_layer,
                       self.w_q_state_token, self.w_k_state_token, self.w_v_state_token,
                       self.w_q_self_msl_token, self.w_k_self_msl_token, self.w_v_self_msl_token,
                       self.w_q_bandit_msl_token, self.w_k_bandit_msl_token, self.w_v_bandit_msl_token,
                       self.native_hidden_layers, self.policy_affine_layers, self.value_affine_layers,
                       self.shoot_hiddens, self.target_hiddens, self.maneuver_hiddens,
                       self.v_c_hiddens, self.nn_c_hiddens, self.value_head_hidden,
                       self.shoot_heads, self.target_heads, self.maneuver_heads,
                       self.v_c_heads, self.nn_c_heads, self.value_head,
                       self.state_attn_layers, self.self_msl_attn_layers, self.bandit_msl_attn_layers]

        for layer in layer_array:
            tune_layer_parameters(layer)


if __name__ == "__main__":
    # testing code below #

    env = Config.env
    env.reset()

    state = get_kteam_aircraft_state(env, 0)
    # state_input = [ground_truth_state, native_state, aircraft_token_state, self_msl_token_state, bandit_msl_token_state]

    red_global_state = state[0]
    red_native_state = state[1]
    red_token_state = state[2]
    msl_token_self = state[3]
    msl_token_bandit = state[4]

    action_dims = dict(maneuver_dim=12, shoot_dim=2, target_dim=4, v_c_dim=len(Config.hybrid_v_c), nn_c_dim=len(Config.hybrid_nn_c))

    net_nn = Hybrid_NN(len(red_global_state), len(red_native_state),
                       len(red_token_state[0]), len(red_token_state),
                       len(msl_token_self[0]), len(msl_token_self),
                       len(msl_token_bandit[0]), len(msl_token_bandit), action_dims=action_dims)

    e, f, g, h, i = net_nn.select_action(red_global_state, red_native_state, red_token_state,
                                         msl_token_self, msl_token_bandit,
                                         [[1.0] * 4, [1.0] * 4], [[1.0] * 12, [1.0] * 12], [[1.0] * 2, [1.0] * 2])

    net_nn.param_tune()
    # print(net_nn)

    # a,b,c,d = net_nn.macro_forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_native_state).unsqueeze(0), torch.tensor(red_token_state).unsqueeze(0),
    #                                torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0),
    #                                torch.tensor([[1.0] * 4, [1.0] * 4], dtype=torch.float32),
    #                                torch.tensor([[1.0] * 12, [1.0] * 12], dtype=torch.float32),
    #                                torch.tensor([[1.0] * 2, [1.0] * 2], dtype=torch.float32))

    #
    # print(a, b, c, d)
    #
    e, f, g, h, i = net_nn.select_action(red_global_state, red_native_state, red_token_state,
                                         msl_token_self, msl_token_bandit,
                                         [[1.0] * 4, [1.0] * 4], [[1.0] * 12, [1.0] * 12], [[1.0] * 2, [1.0] * 2])

    print(e,f,g,h,i)
