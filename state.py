from typing import Optional
import numpy as np 
import tensor
    
class State(tensor.Tensor):
    """Class state represents single and multi-qubit states.
    The State of two or more qubits is defined as their tensor product"""

    def __repr__(self) -> str:
        s = 'State('
        s += super().__str__().replace('\n','\n' + ' ' * len(s))
        s += ')'
        return s
    
    def __str__(self) -> str:
        s = f'{self.nbits}-qubit state.'
        s += ' Tensor:\n'
        s += super().__str__()
        return s
    
    def qubit(alpha: Optional[np.complexfloating] = None,
            beta: Optional[np.complexfloating] = None) -> State:
        """Produce a given state for a single qubit"""
        if alpha is None and beta is None:
            raise ValueError('alpha, or beta, or both are required')
        
        if beta is None: 
            beta = math.sqrt(1.0 - np.conj(alpha) * alpha)

        if alpha is None:
            alpha = math.sqrt(1.0 - np.conj(beta) * beta)
        
        if not math.is_close(np.conj(alpha) * alpha + np.conj(beta) * beta, 1.0):
            raise ValueError('Qubit probabilities do not add to 1')
        
        qb = np.zeros(2, dtype=tensor.tensor_type())
        qb[0] = alpha
        qb[1] = beta
        return State(qb)