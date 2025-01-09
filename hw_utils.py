from typing import Tuple, Optional

import numpy as np


def reflection_coeff_to_polynomial_coeff(kr: np.ndarray):
        """
        Converts the reflection coefficients `r` to polynomial coefficients `a`

        :param kr: (np.array) the vector containing the reflection coefficients

        :return: (np.array) the vector of polynomial coefficients,
                (float) the final prediction error, e_final, based on the zero lag autocorrelation, R0 (default: 0.).
        """
        
        
        # p is the order of the prediction polynomial.
        p = kr.size
        # set a to be an actual polynomial
        a = np.array([1.0, kr[0]])
        # a (p)-size vector
        e = np.zeros(shape=(p,))
        
        # Set the e0 parameter equal to 0., by default
        e0 = 0.
        
        # Initial value
        e[0] = e0 * (1 - np.conj(kr[0]) * kr[0])
        
        # Recursive steps
        for k in range(1, p):
                a_, e_k_ = _levup(a, kr[k], e[k-1])
                
                a = a_
                e[k] = e_k_
        
        e_final = e[-1]
        
        return a, e_final



def polynomial_coeff_to_reflection_coeff(
        a: np.ndarray, 
        e_final: float = 0.
        ) -> np.ndarray:
        """
        Converts the polynomial coefficients `a` to the reflection coefficients `r`.
        If a[0] != 1, then the function normalizes the prediction polynomial by a[0]

        :param a: (np.ndarray) the vector containing the polynomial prediction coefficients
        :param e_final: (float) the final prediction error (default: 0.0)

        :return: (np.array) the reflection coefficients `r`.
        """
        if a.size <= 1:
                return np.array([])
        
        if a[0] == 0.:
                raise ValueError("Leading coefficient cannot be zero.")
        
        # Normalize by a[0]
        a = a / a[0]
        
        # The leading one does not count
        p  = a.size - 1
        e  = np.zeros(shape=(p,))
        kr = np.zeros(shape=(p,))
        
        e[-1]  = e_final
        kr[-1] = a[-1]
        
        for k in np.arange(p-2, -1, -1):
                a, e_k = _levdown(a, e[k+1])
                
                e[k]  = e_k
                kr[k] = a[-1]
        
        return kr

#########################################################################
#########################################################################
#########################################################################
      

def _levup(acur: np.ndarray, knxt: np.ndarray, ecur: float):
        
        # Drop the leading 1, it is not needed in the stepup
        acur = acur[1:]
        
        # Matrix formulation from Stoica is used to avoid looping
        acur_0     = np.append(arr=acur,       values=[0])
        acur_rev_1 = np.append(arr=acur[::-1], values=[1.])
        
        anxt = acur_0 + knxt * np.conj(acur_rev_1)
        
        enxt = (1.0 - np.dot(np.conj(knxt), knxt)) * ecur
        
        # Insert '1' at the beginning to make it an actual polynomial
        anxt = np.insert(anxt, 0, 1.0)
        
        return anxt, enxt


def _levdown(anxt: np.ndarray, enxt: Optional[float] = None) -> Tuple[np.ndarray, float]:
        
        
        # Drop the leading 1 (not needed in the step-down)
        anxt = anxt[1:]
        
        # Extract the (k+1)-th reflection coefficient
        knxt = anxt[-1]
        
        if knxt == 1.0:
                raise ValueError("At least one of the reflection coefficients is equal to one.\nThe algorithm fails for this case.")
        
        # A matrix formulation from Stoica is used to avoid looping
        acur = (anxt[:-1] - knxt * np.conj(anxt[::-1][1:])) / (1 - np.abs(knxt) ** 2)
                
        ecur = enxt / (1 - np.dot(np.conj(knxt).transpose(), knxt) ) if enxt is not None else None
        
        # Insert the constant 1 coefficient to make it a true polynomial
        acur = np.insert(acur, 0, 1)
        
        return acur, ecur
