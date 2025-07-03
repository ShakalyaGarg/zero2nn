import numpy as np
import torch
import random
import math

class Value:
    """
        This class create a custom Data Structure which is used to replicate AutoGrad (PyTorch Functionality to calculate differentials AUTOMATICALLY)
        - It resembles autograd in its working as much as possible
        - Common Non Linearities are incorporated -> tanh and ReLU
    """
    def __init__(self, data, _children =(), _op = '', label = ''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # This function is automatically triggered when someone call the Value object and prints the representation mentioned
    def __repr__(self):
        return f"Value(data = {self.data}, grad = {self.grad})"

    # Function to undergo backward pass or backpropagation
    def backward(self):
        topo = []
        visited = set()

        # Here we make use of topological sort to get the required order in which backward pass should be called
        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # Function to replace operator '+' on Value objects automatically
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    # Function to have reverse add functionality if required
    def __radd__(self, other):
        return self + other

    # Function to replace operator '*' on Value objects automatically
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data 
        out._backward = _backward
        
        return out

    # Function to have reverse multiply functionality if required
    def __rmul__(self, other): # other * self
        return self*other

    # Function to calculate exponent for Value objects
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # Function to replace operator '**' on Value objects automatically
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports int/float values"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out

    # Function to replace operator '/' on Value objects automatically
    def __truediv__(self, other):
        return self * other**-1

    # Function to have reverse divide functionality if required
    def __rtruediv__(self, other):
        return other * self**-1

    # Function to replace operator '-' (Unary Operator) on Value objects automatically
    def __neg__(self):
        return self*-1

    # Function to replace operator '-' on Value objects automatically
    def __sub__(self, other):
        return self + (-other) # uses __neg__

    # Function to have reverse subtract functionality if required
    def __rsub__(self, other):
        return other + (-self)

    # Function to undergo ReLU non-linear transformation on Value object
    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    # Function to undergo tanh non-linear transformation on Value object
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out