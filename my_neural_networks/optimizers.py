import logging
import torch
import math


class BaseOptimizer(object):
    """Optimzier Abstract Class
    """

    def __init__(self, init_learning_rate):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
        """
        self.learning_rate = init_learning_rate
    
    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        raise NotImplementedError

    def ahead(self, weights, biases):
        """Look ahead for weight updates

            This is for NesterovOptimizer, which requires to look ahead for future updates. For other optimizers, this just simply return the inputs.

        Args:
            weights: weights before the udpate
            biases: biases before the update

        Returns:
            weights and biases
        """
        return weights, biases


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent Optimizer
    """

    def __init__(self, init_learning_rate):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
        """
        super(SGDOptimizer, self).__init__(init_learning_rate)

    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        weights = [w - (self.learning_rate * g)
                    for w, g in zip(weights, W_grads)]
        biases = [b - (self.learning_rate * g.unsqueeze(1))
                    for b, g in zip(biases, b_grads)]
        return weights, biases


class MomentumOptimizer(BaseOptimizer):

    """Momentum Optimizer

    """

    def __init__(self, init_learning_rate, shape, gpu_id=-1, rho=0.9):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            rho: momentum hyperparameter
        """
        super(MomentumOptimizer, self).__init__(init_learning_rate)
        self.rho=rho;
        self.shape=shape;
        if gpu_id == -1:
            self.v = [torch.zeros(j, i)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.vb = [torch.zeros(i, 1)
                           for i in self.shape[1:]]
        else:
            self.v = [torch.zeros(j, i).cuda(gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.vb = [torch.zeros(i, 1).cuda(gpu_id)
                           for i in self.shape[1:]]


    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """

        self.v= [v*self.rho + (self.learning_rate * g)
                    for v, g in zip(self.v, W_grads)]

        self.vb = [vb*self.rho+ (self.learning_rate * g.unsqueeze(1))
                    for vb, g in zip(biases, b_grads)]

        weights = [w - bb
                   for w, bb in zip(weights, self.v)]
        biases = [b - vbb
                  for b, vbb in zip(biases, self.vb)]
        return weights, biases



class NesterovOptimizer(MomentumOptimizer):
    """Nesterov Optimizer
    """
    def __init__(self, init_learning_rate, shape, gpu_id=-1, rho=0.9):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            rho: momentum hyperparameter
        """

        super(NesterovOptimizer, self).__init__(init_learning_rate, shape, gpu_id, rho)

    def ahead(self, weights, biases):
        """Look ahead for weight updates

            This is for NesterovOptimizer, which looks ahead for future updates. For other optimizers, this simply returns the inputs.

            Think about how Nesterov Momentum requires for calculating gradients.

        Args:
            weights: weights before the udpate
            biases: biases before the update

        Returns:
            weights and biases
        """
        weights = [w + self.rho*bb
                   for w, bb in zip(weights, self.v)]
        biases = [b + self.rho*vbb
                  for b, vbb in zip(biases, self.vb)]
        return weights, biases


    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        self.v= [v*self.rho - (self.learning_rate * g)
                    for v, g in zip(self.v, W_grads)]

        self.vb = [vb*self.rho- (self.learning_rate * g.unsqueeze(1))
                    for vb, g in zip(self.vb, b_grads)]

        weights = [w + bb
                   for w, bb in zip(weights, self.v)]
        biases = [b + vbb
                  for b, vbb in zip(biases, self.vb)]
        return weights, biases


class AdamOptimizer(BaseOptimizer):
    """Adam Optimizer

    """
    def __init__(self, init_learning_rate, shape,
                 gpu_id=-1, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """Constructor

        Args:
            init_learning_rate: initial learning rate
            shape: network shape
            gpu_id: gpu ID if it is used
            beta1: adam hyperparameter
            beta2: adam hyperparameter
            epsilon: to avoid divided by zero
        """
        super(AdamOptimizer, self).__init__(init_learning_rate)
        self.beta1 = beta1
        self.beta2=beta2
        self.shape = shape
        self.t=0
        self.epsilon=epsilon
        if gpu_id == -1:
            self.m1w = [torch.zeros(j, i)
                      for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.m1b = [torch.zeros(i, 1)
                       for i in self.shape[1:]]
            self.m2w = [torch.zeros(j, i)
                      for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.m2b = [torch.zeros(i, 1)
                       for i in self.shape[1:]]

        else:
            self.m1w = [torch.zeros(j, i).cuda(gpu_id)
                      for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.m1b = [torch.zeros(i, 1).cuda(gpu_id)
                       for i in self.shape[1:]]
            self.m2w = [torch.zeros(j, i).cuda(gpu_id)
                      for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.m2b = [torch.zeros(i, 1).cuda(gpu_id)
                       for i in self.shape[1:]]



    def update(self, weights, biases, W_grads, b_grads):
        """Weight Updates

        Args:
            weights: weights before the udpate
            biases: biases before the update
            W_grads: gradients for the weights
            b_grads: gradients for the biases

        Returns:
            updated weights and biases
        """
        self.t=self.t+1
        self.m1w=[self.beta1*m1+(1-self.beta1)*g for m1,g in zip(self.m1w, W_grads)]
        self.m1b = [self.beta1 * m1 + (1 - self.beta1) *g.unsqueeze(1) for m1, g in zip(self.m1b, b_grads)]

        self.m2w = [self.beta2 * m2 + (1 - self.beta2) * g**2 for m2, g in zip(self.m2w, W_grads)]
        self.m2b = [self.beta2 * m2 + (1 - self.beta2) * g.unsqueeze(1)**2 for m2, g in zip(self.m2b, b_grads)]

        u1w = [m1/(1-pow(self.beta1, self.t)) for m1 in self.m1w]
        u1b = [m1 / (1 - pow(self.beta1, self.t)) for m1 in self.m1b]

        u2w = [m1/(1-pow(self.beta1, self.t)) for m1 in self.m2w]
        u2b = [m1 / (1 - pow(self.beta2, self.t)) for m1 in self.m2b]

        weights = [w - self.learning_rate*(u1/(u2.sqrt()+self.epsilon))
                   for w, u1, u2 in zip(weights, u1w, u2w)]
        biases = [b - self.learning_rate*(u1/(u2.sqrt()+self.epsilon))
                   for b, u1, u2 in zip(biases, u1b, u2b)]
        return weights, biases