import torch
import torch.nn.functional as F

mult = None

def get_model(num_dim_input, num_dim_output, config, args):
    global mult
    model = torch.nn.Sequential(
            torch.nn.Linear(num_dim_input, args.layer1, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(args.layer1, args.layer2, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(args.layer2, num_dim_output*num_dim_output, bias=False))

    if hasattr(config, 'get_xt_scale'):
        scale = config.get_xt_scale()
        mult = torch.diag(torch.from_numpy(scale))
    else:
        mult = None

    def forward(input):
        global mult
        output = model(input)
        output = output.view(input.shape[0], num_dim_output, num_dim_output)
        if mult is not None:
            mult = mult.type(input.type())
            output = torch.matmul(output, mult)
        return output
    return model, forward
