from torch.nn import Module, ModuleList
from torch.nn import Linear
from torch.nn import functional


class WideComponent(Module):

    def __init__(self, architecture):
        super(WideComponent, self).__init__()
        self._layers = ModuleList()
        for in_features, out_features in architecture[:-1]:
            self._layers.append(Linear(in_features, out_features))
        self._output_layer = Linear(architecture[-1][0], architecture[-1][1], bias=False)

    def forward(self, features):
        for layer in self._layers:
            features = layer(features)
        predicts = self._output_layer(features)
        return predicts


class DeepComponent(Module):

    def __init__(self, architecture):
        super(DeepComponent, self).__init__()
        self._layers = ModuleList()
        for in_features, out_features in architecture[:-1]:
            self._layers.append(Linear(in_features, out_features))
        self._output_layer = Linear(architecture[-1][0], architecture[-1][1], bias=False)

    def forward(self, features):
        for layer in self._layers:
            features = functional.leaky_relu(layer(features))
        predicts = self._output_layer(features)
        return predicts
