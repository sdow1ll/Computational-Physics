import numpy as np

A = [[1, 2], [3, 4]]
B = np.array(A)

#a 
#appends the list A to itself.
print(A + A)
#b
#since B is a 2D array (matrix) B+B adds the values to each corresponding element
print(B + B)
#c
# outputs the same thing as "b" except now its list + array
print(A + B)
#d
#A-A doesn't work because Python recognizes A as a list. so ill try this:
#Aminus takes each corresponding element from the tuple in A, to subtract
#to its corresponding element in the other A.
Aminus =  [(a1 - a2) for a1, a2 in A]
print(Aminus)
#e 
#takes each element in matrix (2D array) and subtracts the elements
# to itself, thus making a zero matrix.
print(B - B)
#f
print(2 * A)
#g
print(2 * B)
#h
#The problem wants me to do: print(A * A) but it doesn't work.
#So, I'll try this:
#this calculation takes each element in the tuple and multiplies
#it to the other element in the other tuple.
Asquared = [(a1 * a2) for a1, a2 in A]
print(Asquared)
#i
#this is like matrix multipulcation 
print(B * B) 
#j
#this calculates the dot product for matrix B
#with itself
print(np.dot(B, B))
#k
#this is equivalent to B*B
print(B**2)
#l
#each element here is being divided by itself.
print(B/B)

