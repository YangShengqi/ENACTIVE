# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config

from state_method.state_method import get_kteam_aircraft_state_for_attention, get_kteam_global_ground_truth_state


class MultiHead_Attention_NN(nn.Module):
    def __init__(self, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 action_dims=dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
                 ground_truth_size_before_cat=(512, 512),
                 native_hidden_size=(512, 256),
                 policy_hidden_size=(256, 256, 128),
                 value_hidden_size=(256, 128),
                 state_token_embed_dim=100, state_token_num_heads=4, atten_depth=2,
                 activation='tanh', init_method='origin', aircraft_num=2,
                 re_distributed=True):  # re_distributed for training
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
        self.state_token_embed_dim = state_token_embed_dim
        self.state_token_num_heads = state_token_num_heads

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
        self.state_token_embed_layer = nn.Linear(state_token_dim, state_token_embed_dim)

        self.attn_layers = nn.ModuleList()
        self.w_k = nn.ModuleList()
        self.w_v = nn.ModuleList()
        self.w_q = nn.ModuleList()

        for _ in range(self.attn_depth):
            self.attn_layers.append(nn.MultiheadAttention(embed_dim=self.state_token_embed_dim,
                                                          num_heads=self.state_token_num_heads))
            self.w_k.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))
            self.w_v.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))
            self.w_q.append(nn.Linear(self.state_token_embed_dim, self.state_token_embed_dim))

        # for normalization #
        self.state_token_norm_layer = nn.LayerNorm([self.state_token_embed_dim])

        # ** 3.native hidden layers ** #
        self.native_hidden_layers = nn.ModuleList()
        last_native_dim = native_dim
        for dim in native_hidden_size:
            self.native_hidden_layers.append(nn.Linear(last_native_dim, dim))
            last_native_dim = dim

        # ** 4.cat native, tokens and global ** #
        flatten_dim_without_global = self.state_token_embed_dim * state_token_num + last_native_dim
        flatten_dim_with_global = self.state_token_embed_dim * state_token_num + last_native_dim + last_global_dim
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

        # init layers #
        set_init(self.global_hidden_layers)  # global part
        set_init([self.state_token_embed_layer])  # atten part
        set_init([self.state_token_norm_layer])
        set_init(self.w_q)
        set_init(self.w_k)
        set_init(self.w_v)
        set_init(self.native_hidden_layers, method=init_method)  # native part
        set_init(self.policy_affine_layers, method=init_method)  # cat part
        set_init(self.value_affine_layers, method=init_method)

        set_init(self.shoot_hiddens, method=init_method)  # output part
        set_init(self.target_hiddens, method=init_method)
        set_init(self.maneuver_hiddens, method=init_method)
        set_init([self.value_head_hidden], method=init_method)
        set_init(self.shoot_heads, method=init_method)
        set_init(self.target_heads, method=init_method)
        set_init(self.maneuver_heads, method=init_method)
        set_init([self.value_head], method=init_method)

    def forward(self, global_state, native_state, token_state):

        # *** attention forward *** #
        token_state = token_state.transpose(0, 1)  # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        token_embedding = self.state_token_embed_layer(token_state)

        for i in range(self.attn_depth):
            q_x = self.w_q[i](token_embedding)
            k_x = self.w_k[i](token_embedding)
            v_x = self.w_v[i](token_embedding)
            # print("new forward")

            tokens_out, weights = self.attn_layers[i](q_x, k_x, v_x)  #
            # print(tokens_out.size())  # todo problems here of dimention operation
            token_sum = tokens_out + token_embedding
            # print(token_sum.size())
            token_embedding = self.state_token_norm_layer(token_sum)

        tokens_out = token_embedding.transpose(0, 1)

        # *** native forward *** #
        for native_hidden_layer in self.native_hidden_layers:
            native_state = native_hidden_layer(native_state)
            native_state = self.activation(native_state)

        # *** global forward *** #
        for global_hidden_layers in self.global_hidden_layers:
            global_state = global_hidden_layers(global_state)
            global_state = self.activation(global_state)

        # *** flat and cat *** #
        token_out_flat = torch.flatten(tokens_out, start_dim=-2, end_dim=-1)

        policy_state = torch.cat([native_state, token_out_flat], dim=-1)
        value_state = torch.cat([global_state, native_state, token_out_flat], dim=-1)

        policy_state = policy_state.squeeze(0)
        for affine in self.policy_affine_layers:
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

        return maneuver_probs, shoot_probs, target_probs, v_head

    def select_action(self, x0, x1, x2, maneuver_masks: list, shoot_masks: list, target_masks: list):
        x0 = torch.FloatTensor(x0).unsqueeze(0)
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #

        maneuver_probs, shoot_probs, target_probs, _ = self.forward(x0, x1, x2)
        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        # maneuvers = []
        # shoots = []
        # targets = []
        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)
        maneuvers = (maneuver_probs * maneuver_masks + maneuver_masks * self.multinomial_protect).multinomial(1)

        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        shoots = (shoot_probs * shoot_masks + shoot_masks * self.multinomial_protect).multinomial(1)

        target_masks = torch.tensor(target_masks, dtype=torch.float32)
        targets = (target_probs * target_masks + target_masks * self.multinomial_protect).multinomial(1)

        return maneuvers, shoots, targets

    def select_action_after_target(self, x0, x1, x2, maneuver_masks: list, shoot_masks: list, target_masks: list):
        # shoot mask based on target, only use for choosing enemy as target and decide if shoot this target #
        x0 = torch.FloatTensor(x0).unsqueeze(0)
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #

        maneuver_probs, shoot_probs, target_probs, _ = self.forward(x0, x1, x2)
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

    def get_log_prob_and_values(self, x0, x1, x2, maneuvers, shoots, targets, maneuver_masks, shoot_masks, target_masks):

        maneuver_probs, shoot_probs, target_probs, value = self.forward(x0, x1, x2)

        maneuver_masks = maneuver_masks.transpose(0, 1)
        shoot_masks = shoot_masks.transpose(0, 1)
        target_masks = target_masks.transpose(0, 1)

        maneuver_probs_sum = torch.sum(maneuver_probs * maneuver_masks, -1, keepdim=True).detach()
        shoot_probs_sum = torch.sum(shoot_probs * shoot_masks, -1, keepdim=True).detach()
        target_probs_sum = torch.sum(target_probs * target_masks, -1, keepdim=True).detach()

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

        maneuver_probs = (maneuver_probs + Config.devide_protect) / (maneuver_probs_sum + Config.devide_protect)
        target_probs = (target_probs + Config.devide_protect) / (target_probs_sum + Config.devide_protect)
        shoot_probs = (shoot_probs + Config.devide_protect) / (shoot_probs_sum + Config.devide_protect)

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

    red_global_state = get_kteam_global_ground_truth_state(env, 0)
    red_atten_state = get_kteam_aircraft_state_for_attention(env, 0)

    action_dims = dict(maneuver_dim=12, shoot_dim=2, target_dim=4)

    net_nn = MultiHead_Attention_NN(len(red_global_state), len(red_atten_state[0]), len(red_atten_state[1][0]),
                                    len(red_atten_state[1]), action_dims=action_dims)

    a,b,c,d = net_nn.forward(torch.tensor(red_global_state).unsqueeze(0), torch.tensor(red_atten_state[0]).unsqueeze(0), torch.tensor(red_atten_state[1]).unsqueeze(0))
    print(a, b, c, d)

    # print(net_nn)




