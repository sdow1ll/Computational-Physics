import numpy as np
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
y = x**2

#a
#normal plot nothing too special here.
plt.plot(x,y)
plt.show()
#b 
#the plot from 'a' now is drawn with stars. Each star corresponds to the x elements
#in the array.
b = plt.plot(x,y,'*')
plt.show()
#c
#plots x, y with two different styles, the usual line style and then '+'s.
plt.plot(x,y,'-',x,y,'+')
plt.show()
#d (this one is weird because in the book it uses () but [] works)
#plots x,y with solid line, and then the next part plots a "+" on 2 only. The reason why 
#is because Python handles indices like so: x[start, stop, step]
plt.plot(x,y,'-',x[1:2:10],y[1:2:10],"+")
plt.show()
#e
#changes y axis so that it's log based.
plt.semilogy(x,y)
plt.show()
#f
#changes both x and y axis so that it's log based.
plt.loglog(x,y,'+')
plt.show()