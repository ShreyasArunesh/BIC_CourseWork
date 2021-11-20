
'''
 *
 * Name                   :   activation.py
 *
 * Description            :   it implements the following activation fuctions
 *
 *                            1. Null
 *                            2. Sigmoid 
 *                            3. Hyperbolic Tangent
 *                            4. Cosine
 *                            5. Gaussian
 *                            6. Identity
 *
 * Authors                :   Rémi Desmartin and Mohith Gowda Heggur Ramesh   
 *                                       
 *
'''


'''importing packages'''
from typing import TypeVar
import numpy as np

'''defining the input type for the activation functions'''
Param = TypeVar("Param", float, np.ndarray)


def null(x: Param) -> Param:
    '''
     *
     *  Summary : this block implements null function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns zero 
     *
     '''
    return 0



def sigmoid(x: Param) -> Param:
    
     '''
     *
     *  Summary : this block implements sigmoid function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns sigmoid of 'x' 
     *
     '''
     return 1/(1+np.exp(-x))

def hyperbolicTangent(x: Param) -> Param:
    '''
     *
     *  Summary : this block implements hyperbolic tangent function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns hyperbolic tangent of 'x' 
     *
     '''
    return np.tanh(x)

def cosine(x: Param) -> Param:
     '''
     *
     *  Summary : this block implements cosine function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns cosine of 'x' 
     *
     '''
     return np.cos(x)

def gaussian(x: Param) -> Param:
    '''
     *
     *  Summary : this block implements gaussian function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns gaussian of 'x' 
     *
     '''
    return np.exp(-(pow(x,2)/2))


def identity(x: Param) -> Param:
     '''
     *
     *  Summary : this block implements identity function
     *
     *  Args    : Param - float - np.ndarray
     *
     *  Returns : it returns 'x' itself
     *
     '''
     return x

