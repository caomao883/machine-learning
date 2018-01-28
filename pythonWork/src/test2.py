#coding:utf-8
import numpy as np


def quick_sort(list1):
   l,r = 0,len(list1)-1
   x = list1[(l+r)/2]
   stack = []
   stack.append((l,r))
   while len(stack) > 0:
       ll,rr = stack[len(stack)-1]
       l,r= ll,rr
       if l>=r:
           stack.pop()
       elif l+1 == r and list1[l] > list1[r]:
           stack.pop()
           list1[l],list1[r] = list1[r],list1[l]
       else:

           stack.pop()
           x = list1[(l+r)/2]
           list1[ll],list1[(l+r)/2]=list1[(l+r)/2],list1[ll]
           while l < r:
               if list1[l] <= x:
                   l +=1
               if list1[r] > x:
                   r -=1
               if l < r and list1[l] > list1[r]:
                   list1[l],list1[r] = list1[r],list1[l]
           if r >= ll and list1[r] < x:
               list1[ll],list1[r] = list1[r],list1[ll]
           if l-1 >= ll:
                stack.append((ll,l-1))
                stack.append((l,rr))


N =  100000
list1 = np.random.randint(1,N,N)
list1 = np.linspace(1,N,N,dtype=int)
print list1
quick_sort(list1)

print list1
#验证排序正确性
for i in list1:
    if i < len(list1)-1 and list1[i] > list1[i+1]:
        print "sort error",list1[i],",",list1[i+1]











