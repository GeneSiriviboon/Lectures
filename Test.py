def factorial(x):
    f = x
    for ii in reversed(range(x)[1:]):
        f = f*ii
    return f
