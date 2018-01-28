#coding:utf-8
list=[1,2,3,4]
print list
zw_list=[u'我',u'爱',u'中',u'国']
print zw_list
print zw_list[0]
#列表元素可以为任意数据类型
list1=[1,2,3,4]
list2=[1.,2.,3.,4.]
list3=[list1,list2]
print list3
#更新列表元素
list3[0]=[1,2,3,4,5,6]
print list3
# del list3[0]
# print list3
#计算列表长度
print len(list3)

print[1,2,3]+[4,5,6]#拼接
print[1,2,3]*3 #重复
list1=[1,2,3,4]
list2=[4,3,2,1]
print cmp(list1,list2)#比较

list1=[1,2,3,4]
list1.append(1)
print list1
list1.extend([1,2,3,4])
print list1
print list1.index(3)
list1.remove(1)
print list1
list1.sort()
print list1

tuple1=(1,2,3,4,5)
tuple2=(1,)

