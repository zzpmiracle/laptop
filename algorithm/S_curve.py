# import numpy
def plot(r,b):
    x, y = [], []
    for s in range(1,10):
        x.append(0.1*s)
        y.append(1-(1-(0.1*s)**r)**b)
    print( y)
plot(5,50)