import numpy as np
import random
from graphviz import Digraph
from micrograd.engine import Value

# Function to keep track of all the nodes and edges in a graph
def trace(root):
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

# Function to create a graphical representation of the expressions under consideration, often called COMPUTATION GRAPH
def draw_dot(root):
    dot = Digraph(format = 'svg', graph_attr = {'rankdir':'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}"%(n.label, n.data, n.grad), shape = 'record')
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot