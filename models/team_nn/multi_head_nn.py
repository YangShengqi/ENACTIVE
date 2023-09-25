from utils.math import *
from train.config import Config


class MultiHeadNN(nn.Module):
    def __init__(self, state_dim, action_dims=dict(maneuver_dim=0, shoot_dim=0, target_dim=0), hidden_size=(256, 128),
                 activation='tanh', aircraft_num=2):
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
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        # for separating different action probability calculation
        # add one layer after last hidden layer for each separate action heads and value heads
        action_hidden_size = int(hidden_size[0]/8)
        value_hidden_size = int(hidden_size[0]/4)

        # self.maneuver_hidden = nn.Linear(last_dim, action_hidden_size)
        # self.shoot_hidden = nn.Linear(last_dim, action_hidden_size)
        # self.target_hidden = nn.Linear(last_dim, action_hidden_size)

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

        set_init(self.affine_layers)

        set_init(self.maneuver_hiddens)
        set_init(self.shoot_hiddens)
        set_init(self.target_hiddens)
        set_init([self.value_hidden])

        set_init(self.maneuver_heads)
        set_init(self.shoot_heads)
        set_init(self.target_heads)
        set_init([self.value_head])

    def forward(self, x):
        # print(x.shape)
        x = x.view(list(x.shape[:-2]) + [-1])
        x = x.squeeze(0)
        for affine in self.affine_layers:
            x = affine(x)
            x = self.activation(x)
        m_hiddens=[self.activation(self.maneuver_hiddens[i](x)) for i in range(self.aircraft_num)]
        s_hiddens=[self.activation(self.shoot_hiddens[i](x)) for i in range(self.aircraft_num)]
        t_hiddens=[self.activation(self.target_hiddens[i](x)) for i in range(self.aircraft_num)]
        v_hidden=self.activation(self.value_hidden(x))
        # m_hidden = self.activation(self.maneuver_hidden(x))
        # m_hidden = self.activation(self.shoot_hidden(x))
        # t_hidden = self.activation(self.target_hidden(x))
        # v_hidden = self.activation(self.value_hidden(x))
        maneuver_probs = torch.cat([torch.softmax(self.maneuver_heads[i](m_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        shoot_probs = torch.cat([torch.softmax(self.shoot_heads[i](s_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        target_probs = torch.cat([torch.softmax(self.target_heads[i](t_hiddens[i]), dim=-1).unsqueeze(0) for i in range(self.aircraft_num)],0)
        v_head = self.value_head(v_hidden)
        return maneuver_probs, shoot_probs, target_probs, v_head

    def select_action(self, x: list, maneuver_masks: list, shoot_masks: list, target_masks: list):
        maneuver_probs, shoot_probs, target_probs, _ = self.forward(torch.FloatTensor(x).unsqueeze(0))
        # maneuvers = []
        # shoots = []
        # targets = []
        maneuver_masks = torch.tensor(maneuver_masks,dtype=torch.float32)
        maneuvers = (maneuver_probs*maneuver_masks+maneuver_masks*self.multinomial_protect).multinomial(1)

        shoot_masks = torch.tensor(shoot_masks,dtype=torch.float32)
        shoots = (shoot_probs*shoot_masks+shoot_masks*self.multinomial_protect).multinomial(1)

        target_masks = torch.tensor(target_masks,dtype=torch.float32)
        targets = (target_probs*target_masks+target_masks*self.multinomial_protect).multinomial(1)



        #
        #     maneuvers.append(None)
        # if shoot_masks[i] is not None:
        #     for index, m in enumerate(shoot_masks[i]):
        #         if m == 0.0:
        #             shoot_probs[i][index] =shoot_probs[i][index]-shoot_probs[i][index]
        #         else:
        #             shoot_probs[i][index] = shoot_probs[i][index] + multinomial_protect
        #     #print("shoot_masks", shoot_masks[i])
        #     shoots.append(shoot_probs[i].multinomial(1))
        # else:
        #     shoots.append(None)
        # if target_masks[i] is not None:
        #     for index, m in enumerate(target_masks[i]):
        #         if m == 0.0:
        #             target_probs[i][index] = target_probs[i][index]-target_probs[i][index]
        #         else:
        #             target_probs[i][index] = target_probs[i][index] + multinomial_protect
        #     targets.append(target_probs[i].multinomial(1))
        # else:
        #     targets.append(None)

        # return maneuvers.tolist()[0],  shoots.tolist()[0],  targets.tolist()[0]

        return maneuvers, shoots, targets

    # def get_kl(self, x):
    #     action_prob1 = self.forward(x,None)#TODO
    #     action_prob0 = action_prob1.detach()
    #     kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
    #     return kl.sum(1, keepdim=True)

    def get_log_prob_and_values(self, x, maneuvers, shoots, targets):

        maneuver_probs, shoot_probs, target_probs, value = self.forward(x.unsqueeze(0))

        # maneuver_probs = torch.tensor(maneuver_probs,device=torch.device('cuda', index=0))
        #
        # shoot_probs = torch.tensor(shoot_probs,device=torch.device('cuda', index=0))
        #.
        # target_probs = torch.tensor(target_probs,device=torch.device('cuda', index=0))
        #value = torch.tensor(value,device=torch.device('cuda', index=0))
        #print(len(maneuver_probs), len(shoot_probs), len(target_probs), len(maneuvers), len(shoots), len(targets))
        #print("m device",maneuver_probs.device,"m device",maneuvers.device)
        #print(maneuver_probs[0].shape,torch.t(maneuvers).shape)
        maneuvers = torch.t(maneuvers)
        shoots = torch.t(shoots)
        targets = torch.t(targets)
        m = maneuvers.clone()
        m[torch.isnan(m)]=0
        m=m.unsqueeze(-1)
        s = shoots.clone()
        s[torch.isnan(s)]=0
        s=s.unsqueeze(-1)
        t = targets.clone()
        t[torch.isnan(t)]=0
        t=t.unsqueeze(-1)
        maneuver_probs=maneuver_probs.gather(-1,m.long())
        maneuver_probs[torch.isnan(maneuvers)]=1
        maneuver_probs.squeeze(-1)
        shoot_probs = shoot_probs.gather(-1,s.long())
        shoot_probs[torch.isnan(shoots)]=1
        shoot_probs.squeeze(-1)
        target_probs = target_probs.gather(-1,t.long())
        target_probs[torch.isnan(targets)]=1
        target_probs.squeeze(-1)

        ans = torch.log(torch.prod(maneuver_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(shoot_probs, 0) + self.log_protect) + \
              torch.log(torch.prod(target_probs, 0) + self.log_protect)

        # ans = torch.prod(maneuver_probs,0)*torch.prod(shoot_probs,0)*torch.prod(target_probs,0)
        #ans = torch.tensor(ans,device=torch.device('cuda', index=0) ).view(self.aircraft_num,-1)
        #ans = torch.t(ans.view(self.aircraft_num,-1))
        #torch.set_printoptions(edgeitems=10000)
        #print(ans)
        # ans = torch.log(ans + self.log_protect)
        #print("ans device",ans.device)
        return ans, value

    # def get_fim(self, x):
    #     action_prob = self.forward(x,None)#TODO
    #     M = action_prob.pow(-1).view(-1).detach()
    #     return M, action_prob, {}

