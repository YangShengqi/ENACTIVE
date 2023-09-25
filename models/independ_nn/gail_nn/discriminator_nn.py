# with global input for value fitting, may solve coefficient problem of PG loss and value loss
from utils.math import *
from train.config import Config
from models.independ_nn.Commnet import CommNetWork


class Discriminator_NN(nn.Module):
    def __init__(self, id_dim, ground_truth_dim, native_dim, state_token_dim, state_token_num,
                 self_msl_token_dim, self_msl_token_num, bandit_msl_token_dim, bandit_msl_token_num,
                 action_dims=dict(horizontal_cmd_dim=0, vertical_cmd_dim=0, shoot_dim=0, target_dim=0, v_c_dim=0, nn_c_dim=0),
                 ground_truth_size_before_cat=(256, 256),
                 native_hidden_size=(256, 128),
                 policy_hidden_size=(128, 128, 64),
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
        #  --------------------------------- discriminator -----------------------------------------------------#
        action_onehot_dim = self.action_dims["horizontal_cmd_dim"] + self.action_dims["vertical_cmd_dim"] + \
                            self.action_dims["shoot_dim"] + self.action_dims["target_dim"] + self.action_dims["v_c_dim"] + \
                            self.action_dims["nn_c_dim"]
        flatten_dim = self.state_token_embed_dim * state_token_num + \
                      self.msl_token_embed_dim * (self_msl_token_num + bandit_msl_token_num) + \
                      last_native_dim + id_dim + last_global_dim + action_onehot_dim

        self.policy_commnet = CommNetWork(flatten_dim, policy_hidden_size, n_agents=self.aircraft_num)
        last_policy_dim = policy_hidden_size[-1]

        self.discrim_heads = nn.Linear(last_policy_dim, 1)

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

        set_init([self.discrim_heads], method=init_method)

    def macro_forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state,
                      ids_one_hot_state, action_one_hot):

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
        discrim_state = torch.cat(
            [ids_one_hot_state, global_state, native_state, state_tokens_out_flat, self_msl_tokens_out_flat,
             bandit_msl_tokens_out_flat, action_one_hot], dim=-1)

        policy_state = self.policy_commnet(discrim_state)
        policy_state = policy_state.squeeze(0)

        discrim_heads = self.discrim_heads(policy_state)
        discrim_probs = torch.sigmoid(discrim_heads)

        return discrim_probs

    def forward(self, global_state, native_state, token_state, self_msl_token_state, bandit_msl_token_state,
                ids_one_hot_state, action_one_hot):

        global_state = torch.tensor(global_state).unsqueeze(0)
        native_state = torch.tensor(native_state).unsqueeze(0)
        token_state = torch.tensor(token_state).unsqueeze(0)  # add one dim #
        self_msl_token_state = torch.tensor(self_msl_token_state).unsqueeze(0)
        bandit_msl_token_state = torch.tensor(bandit_msl_token_state).unsqueeze(0)
        ids_one_hot_state = torch.tensor(ids_one_hot_state).unsqueeze(0)
        action_one_hot = torch.tensor(action_one_hot).unsqueeze(0)

        discrim_probs = self.macro_forward(global_state,
                                           native_state,
                                           token_state,
                                           self_msl_token_state,
                                           bandit_msl_token_state,
                                           ids_one_hot_state,
                                           action_one_hot)

        discrim_probs.squeeze(0)

        return discrim_probs









