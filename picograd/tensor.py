"""

Tensor class that can handle basic operations and support automatic differentiation

"""

import numpy as np

class Tensor:
    def __init__(self, 
                 data,                  # Numpy array holding the actual data of the tensor
                 requires_grad=False,   # Whether the tensor should track the gradients
                 creators=None,         # Parent tensors in computational graph
                 creation_op=None       # Operation that created this tensor
                ):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        if requires_grad:
            self.zero_grad()            # Initialize as an array of zeroes with the same shape as data
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}

        if creators is not None:
            for c in creators:
                # Update the children dictionary of the parent tensors
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    # In the case of reuse (i.e. of the same tensor)
                    c.children[self.id] += 1

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    # Override addition operation for tensors
    def __add__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(self.data + other.data, requires_grad=True, creators=[self, other], creation_op="add")
        return Tensor(self.data + other.data)

    # Override addition operation for tensors    
    def __mul__(self, other):
        if self.requires_grad and other.requires_grad:
            return Tensor(self.data * other.data, requires_grad=True, creators=[self, other], creation_op="mul")
        return Tensor(self.data * other.data)

    # Propagate the gradient backwards
    def backward(self, grad=None, grad_origin=None):
        # If for some reaason, you don't want to track the gradient
        if not self.requires_grad:
            return
        
        if grad is None:
            # Thus, return the derivative wrt self
            grad = Tensor(np.ones_like(self.data))
        
        if self.grad is None:
            self.grad = grad
        else:
            # Accumulate the gradient within the computational graph
            self.grad += grad
        
        # Prevent premature gradient propagation
        if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
            # Derivative wrt sum is 1
            if self.creation_op == "add":
                # Propagate the gradients
                self.creators[0].backward(self.grad, self.id)
                self.creators[1].backward(self.grad, self.id)
            # Derivative wrt mul is other
            elif self.creation_op == "mul":
                # Multiply the other gradient
                new_grad_0 = self.grad * self.creators[1].data
                new_grad_1 = self.grad * self.creators[0].data
                self.creators[0].backward(new_grad_0, self.id)
                self.creators[1].backward(new_grad_1, self.id)

    def all_children_grads_accounted_for(self):
        for id, count in self.children.items():
            if count != 0:
                return False
        return True