def compute_factorial(n):
    """Function to compute factorial of x (the product of an integer and all
    integers below)"""

    n=int(n)
    result = 1
    while n>1 :
        result=result*n
        n = n-1
    return result
