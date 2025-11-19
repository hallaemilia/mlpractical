import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * np.sum(np.abs(parameter))
        

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * np.sign(parameter)
        

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient * 0.5 * np.sum(parameter ** 2)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient * parameter

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """L1 & L2 mix penalty.
    """

    def __init__(self, coefficient, mixing_term=0.5):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
            miving_term: Float in [0, 1] controlling mix between L1 and L2, mixing_term=1 gives pure L1 penalty.
        """
        assert coefficient > 0, 'Penalty coefficient must be positive'
        assert 0 <= mixing_term <= 1, 'Mixing term must be in [0, 1].'
        self.coefficient = coefficient
        self.mixing_term = mixing_term

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        L1_term = np.sum(np.abs(parameter))
        L2_term = 0.5 * np.sum(parameter ** 2)
        return self.coefficient * (self.mixing_term * L1_term + (1 - self.mixing_term) * L2_term)

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        L1_grad = np.sign(parameter)
        L2_grad = parameter
        return self.coefficient * (self.mixing_term * L1_grad + (1 - self.mixing_term) * L2_grad)

    def __repr__(self):
        return f'L1L2MixPenalty({0}_{1})'.format(self.coefficient, self.mixing_term)
