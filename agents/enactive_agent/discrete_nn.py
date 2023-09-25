# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config


class Discrete_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(maneuver_dim=0, shoot_dim=0),
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

        # ** 4.cat native, tokens and global ** #
        flatten_dim_without_global = self.msl_token_embed_dim * (self_msl_token_num + bandit_msl_token_num) + \
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
        self.maneuver_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)
        self.shoot_hiddens = nn.Linear(last_policy_dim, action_head_hidden_size)

        self.maneuver_heads = nn.Linear(action_head_hidden_size, action_dims["maneuver_dim"])
        self.shoot_heads = nn.Linear(action_head_hidden_size, action_dims["shoot_dim"])

        # init layers #
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

        set_init(self.native_hidden_layers, method=init_method)  # native part
        set_init(self.policy_affine_layers, method=init_method)  # cat part
        set_init(self.value_affine_layers, method=init_method)

        set_init([self.shoot_hiddens], method=init_method)  # output part
        set_init([self.maneuver_hiddens], method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init([self.shoot_heads], method=init_method)
        set_init([self.maneuver_heads], method=init_method)
        set_init([self.value_head], method=init_method)

    def forward_with_mask(self, global_state, native_state, self_msl_token_state, bandit_msk_token_state,
                          maneuver_mask, shoot_mask):
        # *** attention forward *** #
        self_msl_token_state = self_msl_token_state.transpose(0, 1)
        bandit_msk_token_state = bandit_msk_token_state.transpose(0, 1)

        self_msl_token_embedding = self.self_msl_token_embed_layer(self_msl_token_state)
        bandit_msl_token_embedding = self.bandit_msl_token_embed_layer(bandit_msk_token_state)

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
            bandit_msl_tokens_out, _ = self.bandit_msl_attn_layers[i](q_bandit_msl, k_bandit_msl, v_bandit_msl)
            self_msl_token_sum = self_msl_tokens_out + self_msl_token_embedding
            bandit_msl_token_sum = bandit_msl_tokens_out + bandit_msl_token_embedding

            self_msl_token_embedding = self.self_msl_token_norm_layer(self_msl_token_sum)
            bandit_msl_token_embedding = self.bandit_msl_token_norm_layer(bandit_msl_token_sum)

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
        self_msl_tokens_out_flat = torch.flatten(self_msl_tokens_out, start_dim=-2, end_dim=-1)
        bandit_msl_tokens_out_flat = torch.flatten(bandit_msl_tokens_out, start_dim=-2, end_dim=-1)

        policy_state = torch.cat(
            [native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat], dim=-1)
        value_state = torch.cat(
            [global_state, native_state, self_msl_tokens_out_flat, bandit_msl_tokens_out_flat],
            dim=-1)

        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
            # print("policy_state", policy_state)
            policy_state = affine(policy_state)
            policy_state = self.activation(policy_state)

        m_hiddens = self.activation(self.maneuver_hiddens(policy_state))
        s_hiddens = self.activation(self.shoot_hiddens(policy_state))

        # *** add mask operation *** #
        maneuver_exp = torch.exp(self.maneuver_heads(m_hiddens))
        shoot_exp = torch.exp(self.shoot_heads(s_hiddens))

        maneuver_probs = (maneuver_exp * maneuver_mask) / torch.sum(maneuver_exp * maneuver_mask, dim=-1, keepdim=True)
        shoot_probs = (shoot_exp * shoot_mask) / torch.sum(shoot_exp * shoot_mask, dim=-1, keepdim=True)

        value_state = value_state.squeeze(0)
        for affine in self.value_affine_layers:
            value_state = affine(value_state)
            value_state = self.activation(value_state)

        v_hidden = self.activation(self.value_head_hidden(value_state))
        v_head = self.value_head(v_hidden)

        return maneuver_probs, shoot_probs, v_head

    def select_action(self, global_state, native_state, self_msl_token_state, bandit_msl_token_state,
                      maneuver_masks: list, shoot_masks: list):
        global_state = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)
        native_state = torch.tensor(native_state, dtype=torch.float32).unsqueeze(0)
        self_msl_token_state = torch.tensor(self_msl_token_state, dtype=torch.float32).unsqueeze(0)
        bandit_msl_token_state = torch.tensor(bandit_msl_token_state, dtype=torch.float32).unsqueeze(0)

        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32).squeeze()
        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32).squeeze()

        maneuver_probs, shoot_probs, _ = self.forward_with_mask(global_state, native_state,
                                                                self_msl_token_state, bandit_msl_token_state,
                                                                maneuver_masks, shoot_masks)

        maneuvers = (maneuver_probs * maneuver_masks + maneuver_masks * self.multinomial_protect).multinomial(1)
        shoots = (shoot_probs * shoot_masks + shoot_masks * self.multinomial_protect).multinomial(1)

        return maneuvers, shoots

    def get_log_prob_and_values(self, global_state, native_state, self_msl_token_state,
                                bandit_msl_token_state,  maneuvers, shoots, maneuver_masks, shoot_masks):

        maneuver_probs, shoot_probs, value = self.forward_with_mask(global_state, native_state,
                                                                    self_msl_token_state, bandit_msl_token_state,
                                                                    maneuver_masks, shoot_masks)

        m = maneuvers.clone()
        m[torch.isnan(m)] = 0
        s = shoots.clone()
        s[torch.isnan(s)] = 0

        maneuver_probs = maneuver_probs.gather(-1, m.long())
        maneuver_probs[torch.isnan(maneuvers)] = 1
        shoot_probs = shoot_probs.gather(-1, s.long())
        shoot_probs[torch.isnan(shoots)] = 1

        ans = torch.log(maneuver_probs + self.log_protect) + \
              torch.log(shoot_probs + self.log_protect)

        return ans, value


if __name__ == "__main__":
    # testing code below #

    env = Config.env
    env.reset()

    red_global_state = get_kteam_global_ground_truth_state(env, 0)
    red_atten_state = get_kteam_aircraft_state_for_attention(env, 0)
    msl_token_self = get_kteam_msl_tokens(env, 0)
    msl_token_bandit = get_kteam_msl_tokens(env, 1)

    action_dims = dict(maneuver_dim=12, shoot_dim=2, target_dim=4)

    net_nn = MultiHead_Attention_NN(len(red_global_state), len(red_atten_state[0]),
                                    len(red_atten_state[1][0]), len(red_atten_state[1]),
                                    len(msl_token_self[0]), len(msl_token_self),
                                    len(msl_token_bandit[0]), len(msl_token_bandit), action_dims=action_dims)
    print(net_nn)

    a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0),
                             torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0))

    e,f,g,h = net_nn.forward_with_mask(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0),
                                       torch.tensor(msl_token_self).unsqueeze(0), torch.tensor(msl_token_bandit).unsqueeze(0),
                                       torch.tensor([[1] * 4, [1] * 4]),
                                       torch.tensor([[1] * 12, [1] * 12]),
                                       torch.tensor([[1] * 2, [1] * 2]))
    # ground_truth_dim, native_dim, state_token_dim, state_token_num,
    # self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
    # action_dims = dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
    #
    # a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0))
    # print(e,f,g,h)
    #
    i,j,k = net_nn.select_action(red_global_state, red_atten_state[0], red_atten_state[1], msl_token_self, msl_token_bandit, [[1] * 4, [1] * 4], [[1] * 12, [1] * 12], [[1] * 2, [1] * 2])
    # # print(net_nn)
    print(i,j,k)

    # e,f,g = net_nn.select_action(red_global_state, red_atten_state[0], red_atten_state[1], msl_token_self, msl_token_bandit, [1]*12, [1]*2, [1]*4)
    # print(e,f,g)






