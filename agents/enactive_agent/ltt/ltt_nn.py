# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config


class Ltt_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(horizontal_cmd_dim=0, vertical_cmd_dim=0, shoot_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(256, 256),
                 native_hidden_size=(256, 128),
                 policy_hidden_size=(128, 128, 64),
                 value_hidden_size=(128, 64),
                 atten_depth=2,
                 msl_token_embed_dim=32, msl_token_num_heads=4,
                 activation='tanh', init_method='xavier', last_maneuver_embedding_size=8):
        super().__init__()

        self.log_protect = Config.log_protect
        self.multinomial_protect = Config.multinomial_protect
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
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
        last_policy_dim = flatten_dim_with_global
        for dim in policy_hidden_size:
            self.policy_affine_layers.append(nn.Linear(last_policy_dim, dim))
            last_policy_dim = dim

        action_head_hidden_size = int(last_policy_dim / 2)

        # self.ltt_hidden = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] + action_dims["vertical_cmd_dim"], action_head_hidden_size)
        # self.ltt_hidden = nn.Linear(last_policy_dim + action_dims["horizontal_cmd_dim"] * action_dims["vertical_cmd_dim"], action_head_hidden_size)
        last_maneuver_dim = action_dims["horizontal_cmd_dim"] + \
                            action_dims["vertical_cmd_dim"] + \
                            action_dims["v_c_dim"] + \
                            action_dims["nn_c_dim"]
        self.ltt_embedding = nn.Linear(last_maneuver_dim, last_maneuver_embedding_size)
        self.ltt_hidden = nn.Linear(last_policy_dim + last_maneuver_embedding_size, action_head_hidden_size)
        self.ltt_head = nn.Linear(action_head_hidden_size, 1)

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

        set_init([self.ltt_embedding], method=init_method)
        set_init([self.ltt_hidden], method=init_method)
        set_init([self.ltt_head], method=init_method)

        set_init(self.value_affine_layers, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init([self.value_head], method=init_method)

    def macro_forward(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                      last_action_one_hot, deliberation_cost):
        # *** attention forward *** #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msl_token_state = bandit_msl_token_state.transpose(0, 1)

        self_msl_token_embedding = self.self_msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.bandit_msl_token_embed_layer(bandit_msl_token_state)

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
        policy_state = torch.cat([global_state, native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        last_maneuver_embedding = self.activation(self.ltt_embedding(last_action_one_hot))
        steer_state = torch.cat((policy_state, last_maneuver_embedding), dim=-1)
        ltt_hidden = self.activation(self.ltt_hidden(steer_state))
        ltt_head = self.ltt_head(ltt_hidden)

        # 13sigmoid
        ltt_prob = torch.sigmoid(ltt_head)
        ltt_prob_ = ltt_prob + deliberation_cost.unsqueeze(-1)
        ltt_prob_final = torch.clamp(ltt_prob_, 0, 1)

        # *** critic *** #
        value_state = torch.cat([global_state, native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return ltt_prob_final, v_head

    def select_action(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state, maneuver_one_hot,
                      deliberation_cost):
        global_state = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)
        native_state = torch.tensor(native_state, dtype=torch.float32).unsqueeze(0)
        self_msl_token_state = torch.tensor(self_msl_token_state, dtype=torch.float32).unsqueeze(0)
        bandit_msl_token_state = torch.tensor(bandit_msl_token_state, dtype=torch.float32).unsqueeze(0)
        deliberation_cost = torch.tensor(deliberation_cost, dtype=torch.float32)

        ltt_prob, _ = self.macro_forward(global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                                         maneuver_one_hot, deliberation_cost)

        # 13sigmoid
        ltt = (ltt_prob + self.multinomial_protect).bernoulli().to(torch.int16)

        # softmax
        # ltt = (ltt_prob + self.multinomial_protect).multinomial(1)

        #print(hor_ltt_prob.tolist())

        return ltt

    def get_log_prob_and_values(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                                ltts, last_action_one_hots, deliberation_cost):

        ltt_probs, value = self.macro_forward(global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                                              last_action_one_hots, deliberation_cost)

        # 13sigmoid
        ltt_probs_final = ltt_probs.unsqueeze(-1)
        # ltt_probs[torch.isnan(ltts)] = 1
        # ans = torch.log(ltt_probs + self.log_protect)
        ltt_probs_final[torch.isnan(ltts)] = 0

        value[torch.isnan(ltts)] = 0  # if agent die, v = 0, cut backward

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        # ans = torch.log(ans + self.log_protect)
        return ltt_probs_final, value








