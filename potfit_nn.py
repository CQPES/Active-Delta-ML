import os
import numpy as np


class Layer:
    def __init__(self) -> None:
        self.inp_size = 0
        self.out_size = 0
        self.weight   = []
        self.bias     = []

    def forward(self, X) -> float:
        return self.weight @ X + self.bias


class TanH():
    def forward(self, X) -> float:
        return np.tanh(X)


class PotfitNN:
    """PotfitNN
    
    Simple implementation for PIP-NN trained with MATLAB script potfit.m

    Attributes:
        p_min (np.ndarray): Scaling coefficient for PIP (min).
        p_max (np.ndarray): Scaling coefficient for PIP (max).
        V_min (np.ndarray): Scaling coefficient for potential energy (min),
            in eV.
        V_max (np.ndarray): Scaling coefficient for potential energy (max),
            in eV.
        layers (list): Layers in NN.
    """
    def __init__(
        self,
        weights_file: str,
        biases_file: str,
    ) -> None:
        assert os.path.exists(weights_file) and os.path.exists(biases_file), \
            "Parameter files do not exist!"

        self.layers = []

        fw = open(weights_file)
        inp_size, num_hid, out_size = [int(x) for x in fw.readline().split()]
        hid_sizes = [int(x) for x in fw.readline().split()]
        _, num_param = [int(x) for x in fw.readline().split()]

        pdela = np.array([float(x) for x in fw.readline().split()])
        pavga = np.array([float(x) for x in fw.readline().split()])

        self.p_min = -pdela[:-1] + pavga[:-1]
        self.p_max = pdela[:-1] + pavga[:-1]

        self.V_min = -pdela[-1] + pavga[-1]
        self.V_max = pdela[-1] + pavga[-1]

        for i in range(num_hid + 1):
            layer = Layer()
            if i == 0:
                layer.inp_size = inp_size
                layer.out_size = hid_sizes[0]
            elif i == num_hid:
                layer.inp_size = hid_sizes[i - 1]
                layer.out_size = out_size
            else:
                layer.inp_size = hid_sizes[i - 1]
                layer.out_size = hid_sizes[i]
        
            self.layers.append(layer)

            if i != num_hid:
                self.layers.append(TanH())

        fb = open(biases_file)
        for layer in self.layers:
            if isinstance(layer, TanH):
                continue
            
            for _ in range(layer.out_size * layer.inp_size):
                layer.weight.append(float(fw.readline()))

            for _ in range(layer.out_size):
                layer.bias.append(float(fb.readline()))

            layer.weight = np.array(layer.weight)
            layer.weight = layer.weight.reshape((layer.out_size, layer.inp_size))
            layer.bias   = np.array(layer.bias)

        fw.close()
        fb.close()

    @staticmethod
    def scale(t, t_min, t_max):
        t_scaled = 2 * (t - t_min) / (t_max - t_min) - 1
        np.nan_to_num(t_scaled, copy=False)
        return t_scaled

    @staticmethod
    def inverse_scale(t_scaled, t_min, t_max):
        t = (t_scaled + 1) * (t_max - t_min) / 2 + t_min
        return t

    def forward(
        self,
        p: np.ndarray,
    ) -> float:
        X = self.scale(p, self.p_min, self.p_max)

        for layer in self.layers:
            X = layer.forward(X)

        y = self.inverse_scale(X, self.V_min, self.V_max)

        return y
