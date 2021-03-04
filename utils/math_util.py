import math

def closest_pow2(x):
    """
    Return the closest power of 2 by checking whether 
    the second binary number is a 1.
    """
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2**(op(math.log(x,2)))