import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, dimsList, activation=None, initMethod=None, name="SimpleMLP"):
        super(SimpleMLP,self).__init__()
        if activation is None:
            activation = [nn.ReLU() for _ in range(len(dimsList)-2)]
            activation.append(nn.Tanh())
        assert(len(dimsList) == len(activation)+1)
        layerList = []
        self.name = name
        for no in range(len(activation)):
            layerList.append(nn.Linear(dimsList[no],dimsList[no+1]))
            if initMethod is not None:
                initMethod(layerList[-1].weight.data, layerList[-1].bias.data, no)
            if no == len(activation)-1 and activation[no] is None:
                continue
            layerList.append(activation[no])
        self.layerList = torch.nn.ModuleList(layerList)

    def forward(self,x):
        tmp = x
        for layer in self.layerList:
            tmp = layer(tmp)
        return tmp

class SimpleMLPreshape(SimpleMLP):
    def __init__(self, *args,**kwargs):
        # don't know why, this is the only way to do it,
        # otherwiese, numerical errors would be high for some tests.
        self.reshapeBack = kwargs.pop('reshapeBack', True)
        self.shape = kwargs.pop('shape', None)
        super(SimpleMLPreshape,self).__init__(*args,**kwargs)
    def forward(self,x):
        if self.shape is None:
            self.shape = x.shape
        x = x.reshape(x.shape[0],-1)
        x = super(SimpleMLPreshape,self).forward(x)
        if self.reshapeBack:
            return x.reshape(self.shape)
        else:
            return x
