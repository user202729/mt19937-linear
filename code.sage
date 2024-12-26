##

import numpy
import numpy as np

def uint32_array_to_bits(a: np.ndarray)->np.ndarray:
    """
    sage: a=np.array([np.uint32(123)])
    sage: uint32_array_to_bits(a)
    array([0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)
    """
    # default bitorder of numpy is "big" so we follow that
    # view() appears to return in little endian
    return np.unpackbits(np.array(a, dtype=np.uint32).view(np.uint8))

def uint32_to_bits(a: int)->np.ndarray:
    return uint32_array_to_bits(np.array([a], dtype=np.uint32))

def uint32_array_from_bits(a: np.ndarray)->np.ndarray:
    """
    sage: a=np.array([np.uint32(123)])
    sage: uint32_array_from_bits(uint32_array_to_bits(a))
    array([123], dtype=uint32)
    """
    return np.packbits(a).view(np.uint32)

N = 624
M = 397
MATRIX_A = 0x9908B0DF
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7FFFFFFF

F=GF(2)
step=matrix(F, N*32)
for i in range(N-1):
    step[i*32:(i+1)*32, (i+1)*32:(i+2)*32]=matrix.identity(F, 32)

up=matrix(F, 32, N*32)
up[0:32,0:32]=matrix.identity(F, 32)

right=matrix(F, 32, N*32)
right[0:32,32:64]=matrix.identity(F, 32)

over=matrix(F, 32, N*32)
over[0:32,M*32:(M+1)*32]=matrix.identity(F, 32)


def select_bit(x: matrix, mask: int):
    mask=uint32_to_bits(mask)
    return x[[*np.argwhere(mask)[:,0]],:]

def function_to_matrix(f)->matrix:
    """
    sage: a=123456
    sage: a>>1
    61728
    sage: uint32_array_from_bits(np.array(
    ....:     function_to_matrix(lambda x: x>>1) * vector(F, [*uint32_to_bits(a)]),
    ....:     dtype=np.uint8))
    array([61728], dtype=uint32)
    """
    result=matrix(F, 32)
    for i in range(32):
        a=np.zeros(32, dtype=np.uint8)
        a[i]=1
        result[:,i]=matrix(F,[*uint32_array_to_bits(f(uint32_array_from_bits(a)))]).T
    return result

def and_var(x: matrix, mask: int):
    return function_to_matrix(lambda x: x&mask)*x

def repeat_32(a: matrix)->matrix:
    b=matrix(F, 32, a.ncols())
    for i in range(b.nrows()): b[i,:]=a
    return b

y = and_var(up, UPPER_MASK) + and_var(right, LOWER_MASK)
step[32*(N-1):,:] = over + function_to_matrix(lambda x: x>>1)*y + and_var(repeat_32(select_bit(y, 1)), MATRIX_A)

"""
sage: step.rank()
19937
"""

#%time step_inverse=step.pseudoinverse(algorithm="exact")  # unacceptable
step_powers=[step]
while 2^len(step_powers)<N: step_powers.append(step_powers[-1]^2)  # takes a while (but not too long)

def step_pow_n_multiply_vector(n: int, v: vector):
    """
    Returns ``step^n * v``.
    """
    global step_powers
    while n:
        b=(n&-n).bit_length()-1
        assert n&(1<<b)
        n-=1<<b
        while 2^len(step_powers)<=b: step_powers.append(step_powers[-1]^2)
        v=step_powers[b]*v
    return v

def internal_state_to_vector(a: tuple)->vector:
    match random.getstate():
        case [3r, [*state, i], None]: pass
        case _: assert False
    assert 1<=i<=N
    return step_pow_n_multiply_vector(i-1, vector(F, uint32_array_to_bits(state)))

def vector_to_internal_state(v: vector)->tuple:
    return (3r, tuple([int(x) for x in uint32_array_from_bits(np.array(v, dtype=np.uint8))] + [1r]), None)

import random
random.seed(1234r)

match random.getstate():
    case [3r, [*state1, 624r], None]: pass
    case _: assert False

output=[random.getrandbits(32) for i in range(N)]

match random.getstate():
    case [3r, [*state2, 624r], None]: pass
    case _: assert False

#
#%time assert nest(lambda v: step*v, N, vector(F, uint32_array_to_bits(state1))) == vector(F, uint32_array_to_bits(state2)) # 8s
#%time assert step_pow_n_multiply_vector(N, vector(F, uint32_array_to_bits(state1))) == vector(F, uint32_array_to_bits(state2)) # 200ms
#%time assert step^N * vector(F, uint32_array_to_bits(state1)) == vector(F, uint32_array_to_bits(state2)) # 90s?

#

v=matrix.identity(F, 32)
v=v+function_to_matrix(lambda x: x>>11)*v
v=v+and_var(function_to_matrix(lambda x: x<<7)*v, 0x9d2c5680)
v=v+and_var(function_to_matrix(lambda x: x<<15)*v, 0xefc60000)
v=v+function_to_matrix(lambda x: x>>18)*v
temper_matrix=v

for i in range(N):
    assert temper_matrix*vector(F, uint32_to_bits(state2[i])) == vector(F, uint32_to_bits(output[i]))

##

random.seed(1234r)
random.setstate(vector_to_internal_state(internal_state_to_vector(random.getstate())))
assert output==[random.getrandbits(32) for i in range(N)]


##


