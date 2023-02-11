import numpy as np
import matplotlib.pyplot as plt
import math

#a
#this method is not a good way to evaluate exp(x) because compared to the method in 'b'
#this code does not reach to a lower value of absolute fractional error.

N = np.arange(1,61)

def S(x,N):
    sxn = 1
    for n in range(1, N):
        sxn = sxn + ((x**n) / (math.factorial(n)))
    return  sxn


print(S(-10,60))
print(S(-2,60))
print(S(2,60))
print(S(10,60))

x1 = 2
err1 = []
for n in N:
    err1.append(np.abs(S(x1,n) - math.exp(x1)) / math.exp(x1))

x2 = -2
err2 = []
for n in N:
    err2.append(np.abs(S(x2,n) - math.exp(x2)) / math.exp(x2))

x3 = -10
err3 = []
for n in N:
    err3.append(np.abs(S(x3,n) - math.exp(x3)) / math.exp(x3))
    
x4 = 10
err4 = []
for n in N:
    err4.append(np.abs(S(x4,n) - math.exp(x4)) / math.exp(x4))

#b
x1b = 2
err1b = []
for n in N:
    err1b.append(np.abs(S(x1b,n) - (1/math.exp(-x1b))) / (1/math.exp(-x1b)))
    
x2b = -2
err2b = []
for n in N:
    err2b.append(np.abs(S(x2b,n) - (1/math.exp(-x2b))) / (1/math.exp(-x2b)))
    
x3b = -10
err3b = []
for n in N:
    err3b.append(np.abs(S(x3b,n) - (1/math.exp(-x3b))) / (1/math.exp(-x3b)))
    
x4b = 10
err4b = []
for n in N:
    err4b.append(np.abs((S(x4b,n)) - (1/math.exp(-x4b))) / (1/math.exp(-x4b)))



#i'm trying to show the comparison of both methods here. Orange is with the identity,
#and blue is with exp(x). Some of the plots overlap each other though.
plt.xlabel('N')
plt.ylabel('Absolute Fractional Error for x=2')
plt.semilogy(N, err1, '.')
plt.semilogy(N, err1b, ':')
plt.show()

plt.xlabel('N')
plt.ylabel('Absolute Fractional Error for x=-2')
plt.semilogy(N, err2, '.')
plt.semilogy(N, err2b, ':')
plt.show()

plt.xlabel('N')
plt.ylabel('Absolute Fractional Error for x=-10')
plt.semilogy(N, err3, '.')
plt.semilogy(N, err3b, ':')
plt.show()

plt.xlabel('N')
plt.ylabel('Absolute Fractional Error for x=10')
plt.semilogy(N, err4, '.')
plt.semilogy(N, err4b, ':')
plt.show()
