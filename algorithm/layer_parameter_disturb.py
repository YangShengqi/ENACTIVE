import torch


def get_tensor_disturb(tensor_input, std_scaling=0.05):
    tensor_std = tensor_input.std(-1)
    tensor_disturb = torch.randn(tensor_input.size())
    # print(tensor_disturb)
    # print(tensor_std)
    tensor_disturb = tensor_disturb * tensor_std.unsqueeze(-1) * std_scaling
    # print(tensor_disturb)
    return tensor_disturb


def tune_linear_layer(layer):
    # Linear type layer have parameter(layer.weight.data, layer.bias.data)
    tensor_w = layer.weight.data
    tensor_b = layer.bias.data
    tensor_w_disturb = get_tensor_disturb(tensor_w)
    tensor_b_disturb = get_tensor_disturb(tensor_b)
    if layer.weight.requires_grad:
        layer.weight.data = tensor_w + tensor_w_disturb
    if layer.bias.requires_grad:
        layer.bias.data = tensor_b + tensor_b_disturb


def tune_atten_layer(layer):
    # atten type layer have parameter(layer.in_proj_bias.data and layer.in_proj_weight.data)
    tensor_w = layer.in_proj_weight.data
    tensor_b = layer.in_proj_bias.data
    tensor_w_disturb = get_tensor_disturb(tensor_w)
    tensor_b_disturb = get_tensor_disturb(tensor_b)
    if layer.in_proj_weight.requires_grad:
        layer.in_proj_weight.data = tensor_w + tensor_w_disturb
    if layer.in_proj_bias.requires_grad:
        layer.in_proj_bias.data = tensor_b + tensor_b_disturb


def tune_LSTM_layer(layer):
    # LSTM type layer have parameter(bias_hh_l0.data, bias_ih_l0.data, weight_hh_l0.data, weight_ih_l0.data)
    tensor_bias_hh_l0 = layer.bias_hh_l0.data
    tensor_bias_ih_l0 = layer.bias_ih_l0.data
    tensor_weight_hh_l0 = layer.weight_hh_l0.data
    tensor_weight_ih_l0 = layer.weight_ih_l0.data
    tensor_bias_hh_l0_disturb = get_tensor_disturb(tensor_bias_hh_l0)
    tensor_bias_ih_l0_disturb = get_tensor_disturb(tensor_bias_ih_l0)
    tensor_weight_hh_l0_disturb = get_tensor_disturb(tensor_weight_hh_l0)
    tensor_weight_ih_l0_disturb = get_tensor_disturb(tensor_weight_ih_l0)
    if layer.bias_hh_l0.requires_grad:
        layer.bias_hh_l0.data = tensor_bias_hh_l0 + tensor_bias_hh_l0_disturb
    if layer.bias_ih_l0.requires_grad:
        layer.bias_ih_l0.data = tensor_bias_ih_l0 + tensor_bias_ih_l0_disturb
    if layer.weight_hh_l0.requires_grad:
        layer.weight_hh_l0.data = tensor_weight_hh_l0 + tensor_weight_hh_l0_disturb
    if layer.weight_ih_l0.requires_grad:
        layer.weight_ih_l0.data = tensor_weight_ih_l0 + tensor_weight_ih_l0_disturb


def tune_layer_parameters(layer):
    if type(layer) is torch.nn.ModuleList:
        for sub_layer in layer:
            tune_layer_parameters(sub_layer)
    elif type(layer) is torch.nn.Linear:
        tune_linear_layer(layer)
    elif type(layer) is torch.nn.MultiheadAttention:
        tune_atten_layer(layer)
    elif type(layer) is torch.nn.LSTM:
        tune_LSTM_layer(layer)
    else:
        print("layer form not implemented")
        exit(0)


if __name__ == "__main__":
    # a = torch.tensor([0.1, 0.2, 0.3], requires_grad=True, dtype=torch.float32)
    # layer = torch.nn.Linear(3, 5)
    # print("before tune", layer(a))
    # tune_linear_layer(layer)
    # print("after tune", layer(a))
    #
    # a_a = torch.tensor([[[0.1, 0.2, 0.3, 0.5], [0.2, 0.3, 0.4, 0.6]]])
    # a_b = torch.tensor([[[0.2, 0.3, 0.4, 0.6], [0.3, 0.2, 0.1, 0.5]]])
    # a_c = torch.tensor([[[0.3, 0.2, 0.1, 0.5], [0.1, 0.2, 0.3, 0.5]]])
    # layer_a = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2)
    # print("before tune", layer_a(a_a, a_b, a_c))
    # tune_atten_layer(layer_a)
    # print("after tune", layer_a(a_a, a_b, a_c))

    layer_l = torch.nn.LSTM(4, 2, num_layers=1, batch_first=True)
    a_l = torch.tensor([[[0.1, 0.3, 0.2, 0.5], [0.4, 0.2, 0.5, 0.1]]])
    h_0 = torch.tensor([[[0.2, 0.5]]])
    h_1 = torch.tensor([[[0.1, 0.4]]])
    state, nexth = layer_l(a_l, (h_0, h_1))
    print("before tune", state)
    print("before tune", nexth)
    tune_LSTM_layer(layer_l)
    state, nexth = layer_l(a_l, (h_0, h_1))
    print("after tune", state)
    print("after tune", nexth)

