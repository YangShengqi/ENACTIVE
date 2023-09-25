from utils.math import *
from train.config import Config
from state_method.state_method import get_kteam_aircraft_state_for_attention


class MultiHead_Attention_NN(nn.Module):
    def __init__(self, native_dim, token_state_dim, token_dim, action_dims=dict(maneuver_dim=0, shoot_dim=0, target_dim=0),
                 hidden_size=(256, 256, 128), native_hidden_size=256, token_embed_dim=100, num_heads=4,
                 activation='tanh', init_method='origin', aircraft_num=2, re_distributed=True):  # re_distributed for training
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

        self.token_embed_dim = token_embed_dim

        # for embedding #
        self.token_embed_layer = nn.Linear(token_state_dim, token_embed_dim)
        # for normalization #
        self.token_norm_layer = nn.LayerNorm([token_embed_dim])

        # start construction #
        # for native layers # 2 layers for native hidden #
        self.native_hidden_layers = nn.ModuleList()
        self.native_hidden_layers.append(nn.Linear(native_dim, native_hidden_size))
        self.native_hidden_layers.append(nn.Linear(native_hidden_size, native_hidden_size))

        # attention layers #
        self.embed_dim = token_embed_dim
        self.num_heads = num_heads
        self.attn_depth = 2
        self.attn_layers = nn.ModuleList()
        self.w_k = nn.ModuleList()
        self.w_v = nn.ModuleList()
        self.w_q = nn.ModuleList()

        for _ in range(self.attn_depth):
            self.attn_layers.append(nn.MultiheadAttention(embed_dim=self.token_embed_dim, num_heads=self.num_heads))
            self.w_k.append(nn.Linear(self.token_embed_dim, self.token_embed_dim))
            self.w_v.append(nn.Linear(self.token_embed_dim, self.token_embed_dim))
            self.w_q.append(nn.Linear(self.token_embed_dim, self.token_embed_dim))

        # after hidden layers for concatenation #
        flatten_dim = self.embed_dim * token_dim + native_hidden_size
        self.affine_layers = nn.ModuleList()
        last_dim = flatten_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # for separating different action probability calculation
        # add one layer after last hidden layer for each separate action heads and value heads
        action_hidden_size = int(hidden_size[0]/8)
        value_hidden_size = int(hidden_size[0]/4)

        self.maneuver_hiddens = nn.ModuleList()
        self.shoot_hiddens = nn.ModuleList()
        self.target_hiddens = nn.ModuleList()
        self.value_hidden = nn.Linear(last_dim, value_hidden_size)

        self.maneuver_heads = nn.ModuleList()
        self.shoot_heads = nn.ModuleList()
        self.target_heads = nn.ModuleList()

        for _ in range(self.aircraft_num):
            self.maneuver_hiddens.append(nn.Linear(last_dim,action_hidden_size))
            self.shoot_hiddens.append(nn.Linear(last_dim,action_hidden_size))
            self.target_hiddens.append(nn.Linear(last_dim,action_hidden_size))

            self.maneuver_heads.append(nn.Linear(action_hidden_size, action_dims["maneuver_dim"]))
            self.shoot_heads.append(nn.Linear(action_hidden_size, action_dims["shoot_dim"]))
            self.target_heads.append(nn.Linear(action_hidden_size, action_dims["target_dim"]))

        self.value_head = nn.Linear(value_hidden_size, 1)

        set_init(self.native_hidden_layers)
        set_init([self.token_norm_layer])
        set_init(self.w_q)
        set_init(self.w_k)
        set_init(self.w_v)
        # set_init(self.attn_layers)
        set_init(self.affine_layers, method=init_method)
        set_init(self.maneuver_hiddens, method=init_method)
        set_init(self.shoot_hiddens, method=init_method)
        set_init(self.target_hiddens, method=init_method)
        set_init([self.value_hidden], method=init_method)

        set_init(self.maneuver_heads, method=init_method)
        set_init(self.shoot_heads, method=init_method)
        set_init(self.target_heads, method=init_method)
        set_init([self.value_head], method=init_method)

    def forward(self, x1, x2):

        native = x1
        tokens = x2

        tokens = tokens.transpose(0, 1)  # in 2 vs 2 circumstance, tokens dim [6 * 17] # before 2020/04 simulator #
        # token to embedding #
        token_embedding = self.token_embed_layer(tokens)

        # attention forward #
        for i in range(self.attn_depth):

            q_x = self.w_q[i](token_embedding)
            k_x = self.w_k[i](token_embedding)
            v_x = self.w_v[i](token_embedding)
            # print("new forward")

            tokens_out, weights = self.attn_layers[i](q_x, k_x, v_x)  #
            # print(tokens_out.size())  # todo problems here of dimention operation
            token_sum = tokens_out + token_embedding
            # print(token_sum.size())
            token_embedding = self.token_norm_layer(token_sum)

        tokens = token_embedding.transpose(0, 1)

        for native_hidden_layer in self.native_hidden_layers:
            native = native_hidden_layer(native)
            native = self.activation(native)

        out_flat = torch.flatten(tokens, start_dim=-2, end_dim=-1)

        x = torch.cat([native, out_flat], dim=-1)

        x = x.squeeze(0)
        for affine in self.affine_layers:
            x = affine(x)
            x = self.activation(x)

        m_hiddens = [self.activation(self.maneuver_hiddens[i](x)) for i in range(self.aircraft_num)]
        s_hiddens = [self.activation(self.shoot_hiddens[i](x)) for i in range(self.aircraft_num)]
        t_hiddens = [self.activation(self.target_hiddens[i](x)) for i in range(self.aircraft_num)]

        v_hidden = self.activation(self.value_hidden(x))
        # m_hidden = self.activation(self.maneuver_hidden(x))
        # m_hidden = self.activation(self.shoot_hidden(x))
        # t_hidden = self.activation(self.target_hidden(x))
        # v_hidden = self.activation(self.value_hidden(x))
        maneuver_probs = torch.cat([torch.softmax(self.maneuver_heads[i](m_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        shoot_probs = torch.cat([torch.softmax(self.shoot_heads[i](s_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        target_probs = torch.cat([torch.softmax(self.target_heads[i](t_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        v_head = self.value_head(v_hidden)
        return maneuver_probs, shoot_probs, target_probs, v_head

    def select_action(self, x1, x2, maneuver_masks: list, shoot_masks: list, target_masks: list):
        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #

        maneuver_probs, shoot_probs, target_probs, _ = self.forward(x1, x2)
        maneuver_probs.squeeze(0)
        shoot_probs.squeeze(0)
        target_probs.squeeze(0)

        # maneuvers = []
        # shoots = []
        # targets = []
        maneuver_masks = torch.tensor(maneuver_masks, dtype=torch.float32)
        maneuvers = (maneuver_probs *maneuver_masks+maneuver_masks*self.multinomial_protect).multinomial(1)

        shoot_masks = torch.tensor(shoot_masks, dtype=torch.float32)
        shoots = (shoot_probs *shoot_masks+shoot_masks*self.multinomial_protect).multinomial(1)

        target_masks = torch.tensor(target_masks, dtype=torch.float32)
        targets = (target_probs *target_masks+target_masks*self.multinomial_protect).multinomial(1)

        return maneuvers, shoots, targets

    def select_action_after_target(self, x1, x2, maneuver_masks: list, shoot_masks: list, target_masks: list):
        # shoot mask based on target, only use for choosing enemy as target and decide if shoot this target #

        x1 = torch.FloatTensor(x1).unsqueeze(0)
        x2 = torch.FloatTensor(x2).unsqueeze(0)  # add one dim #

        maneuver_probs, shoot_probs, target_probs, _ = self.forward(x1, x2)
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

    def get_log_prob_and_values(self, x1, x2, maneuvers, shoots, targets, maneuver_masks, shoot_masks, target_masks):

        maneuver_probs, shoot_probs, target_probs, value = self.forward(x1, x2)

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

    # red_state = get_kteam_attention_LTR_state(env, 0, [-1, -1], [-1, -1], [-1, -1])
    red_state = get_kteam_aircraft_state_for_attention(env, 0)
    print(red_state)
    print(len(red_state), len(red_state[0]), len(red_state[1]))

    # print(red_state)
    # print("tokens", red_state[1])

    # # f_dim = len(red_state[1]) * 240 + len(red_state[0])  # token_num * embedding_dim + native_dim
    # #
    net_nn = MultiHead_Attention_NN(native_dim=len(red_state[0]), token_state_dim=len(red_state[1][0]), token_dim=len(red_state[1]), action_dims=dict(maneuver_dim=12, shoot_dim=2, target_dim=4))
    # #
    # # two_red_state_0 = [red_state[0], red_state[0]]
    # # two_red_state_1 = [red_state[1], red_state[1]]
    # #
    a,b,c,d = net_nn.forward(torch.tensor(red_state[0]).unsqueeze(0), torch.tensor(red_state[1]).unsqueeze(0))
    print(a, b, c, d)
    e,f,g = net_nn.select_action(red_state[0], red_state[1], [1] * 12, [1] * 2, [1] * 4)
    print(e, f, g)

    # h, i = net_nn.get_log_prob_and_values(torch.tensor([red_state[0]]), torch.tensor([red_state[1]]), e.unsqueeze(0), f.unsqueeze(0), g.unsqueeze(0))
    # print(h, i)



