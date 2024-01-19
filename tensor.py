from typing import Optional
import numpy as np
import math                                                               

tensor_width = 64

def tensor_type():
    if tensor_width == 64:
        return np.complex64
    return np.complex128

class Tensor(np.ndarray):
    """ Tensor is a numpy array representing a state or operator """
    def __new__(cls, input_arr):
        return np.asarray(input_arr, dtype=tensor_type()).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        
    def kron(self, arg):
         """Use the built in Kron function of NP to delegate for an instance
         Kron refers to Kronecker product, which is synonymous with tensor product, a product of two tensors.
         In tensor product, the first element of the first term is multiplied by the whole of the second term, and so on """
         return self.__class__(np.kron(self, arg))
    
    def __mul__(self, arg):
        # Override the multiplication operator 
        return self.kron(arg)
    
    def kpow(self, n: int):
        """ "Kapow" -- Returns the tensor product of this tensor n times, basically an exponent function but the operation is instead a tensor product
         Kronicker Power
         Often used to make a bigger tensor from a single tensor """
        
        if n == 0:
            return 1.0 # deal with base case, of course something raised to the zeroeth power is 1.0
        t = self
        for _ in range (n-1):
            t = np.kron(t, self)
        return self.__class__(t)  # return tensor type this way
    
    def is_close(self, arg) -> bool:
        """Returns whether or not the 1d or 2d tensor is numerically close to arg"""
        
        return np.allclose(self, arg, atol=1e-6)
    
    def is_hermitian(self) -> bool:
        """Checks if the tensor is hermitian, which implies that it is a square matrix that equals its transposed complex conjugate.
        This is denotated mathematically as ```U† = U```"""
        
        if len(self.shape) != 2:
            return False
        if self.shape[0] != self.shape[1]:
            return False
        return self.isclose(np.conj(self.transpose()))
    
    def is_unitary(self) -> bool:
        """Checks if the Tensor is Unitary, i.e. the tensor's complex conjugate transpose is equal to its inverse, where A*A† = I"""
        
        return Tensor(np.conj(self.transpose()) @ self).is_close(Tensor(np.eye(self.shape[0])))
    
    def is_permutation(self) -> bool:
        """Checks if the Tensor is a permutation matrix, which is when each row or column has exactly one 1 and nothing else. Placement doesn't matter past that"""
        x = self
        return (x.ndim == 2 and x.shape[0] == x.shape[1] and (x.sum(axis=0) == 1).all() and (x.sum(axis=1) == 1).all() ((x==1) or (x==0)).all())
    

    
    @property
    def nbits(self) -> int:
        """Compute  the number of qubits in the state."""

        return int(math.log2(self.shape[0]))