import numpy as np
import math
import torch
import random

from micrograd.engine import Value

### The whole structure is created in such a way to resemble PyTorch Libraries - This uses the engine class 'Value'

class Module:
    """
        This is the Base Module class containing required functionality that needs to be inherited by all the child classes
        - zero_grad: will assign the value 0 to gradients of all the parameters
        - parameters: will return a list of parameters of that class object
    """
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron (Module):
    """
        This is a Neuron class which represents the simplest unit in a Neural Network, inherits from Module
        - Initialization: Done when the object is created
        - Forward Pass: Done when some parameter is passed in the object
        - Backward Pass: Done when .backward() is called through the object
    """
    
    def __init__(self, nin, nonlin = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    # Function is called when the class object is used as a function name and parameters are being passed into it
    # eg: Neuron n; n(x) -> will undergo Forward Pass basically
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh() if self.nonlin else act
        return out

    # Overrides the base class function
    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"

class Layer (Module):
    """
        This is a Layer class which represents the a combination of neurons in a Neural Network, inherits from Module
        - Initialization: Done when the object is created
        - Forward Pass: Done when some parameter is passed in the object
        - Backward Pass: Done when .backward() is called through the object
    """
    
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP (Module):
    """
        This is a MLP (Multi-Layer Perceptron) class which represents the a combination of layers in a Neural Network, inherits from Module
        - Initialization: Done when the object is created
        - Forward Pass: Done when some parameter is passed in the object
        - Backward Pass: Done when .backward() is called through the object
    """
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin = i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"