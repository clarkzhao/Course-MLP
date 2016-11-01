# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate

class TimeDependentLearningRateScheduler(object):
    """scheduler interface which sets a exponential time-dependent learning rate."""

    def __init__(self, init_learning_rate, free_parameter):
        """Construct a new time-dependent learning rate scheduler object.

        Args:
            init_learning_rate: Learning rate to use in learning rule.
            free_parameter: free parameter governing how quickly the learning rate decays. 
        """
        self.learning_rate = init_learning_rate
        self.free_parameter = free_parameter
        
    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate * np.exp(-epoch_number/(1.0*self.free_parameter))
        
class MomentumCoefficientScheduler(object):
    def __init__(self, asy_mom_coeff, tau, gamma):
        self.asy_mom_coeff = asy_mom_coeff
        self.tau = tau
        self.gamma = gamma
    def update_learning_rule(self, learning_rate, epoch_number):
        """Update the hyperparameters of the momentum coefficient.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rate.mom_coeff = self.asy_mom_coeff * (1 - self.gamma / (epoch_number + self.tau) ) 
    

