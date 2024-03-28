import torch
from torch import nn

#from Tetsuto Nagashima's implementation based off of AME 508 course work
class MLP(nn.Module):
    def __init__(self, input_dim=1, out_dim=1):
        super(MLP, self).__init__()

        self.width = 200
        self.depth = 4

        self.activation == nn.Tanh()

        # First hidden layer
        MLP_list = [nn.Linear(input_dim, self.width)]

        # Remaining hidden layers
        for _ in range(self.depth - 1):
            MLP_list.append(nn.Linear(self.width, self.width))

        # Output layer
        MLP_list.append(nn.Linear(self.width, out_dim))

        # Adding list of layers as modules
        self.model = nn.ModuleList(MLP_list)

        def init_weights(layer):
          if isinstance(layer, nn.Linear):
              nn.init.normal_(layer.weight, mean=0, std=1)
              if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0, std=1)

        self.model.apply(init_weights)
    
    def forward(self, x):  # no activation in last layer (from A2)
        for i, layer in enumerate(self.model):
          if i < len(self.model) - 1:  # Skip the last layer
            x = self.activation(layer(x))
          else:
            x = layer(x)  # No activatison in the final layer

#as defined by Raissi's paper
def AC_loss_function_Raissi(x_train, t_train, model):
    u = model(x_train)
    u_x = torch.autograd.grad(u, x_train, grad_outputs=torch.ones_like(x_train), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_train, grad_outputs=torch.ones_like(x_train), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_train, grad_outputs=torch.ones_like(t_train), create_graph=True)[0]

def AC_loss_function_Wang(x_train, model):
   
        
        