#coding:utf-8
first_num=input(u'请输入第一个数字：')
opa=raw_input(u'请输入+-*/任意一个符号')
second_num=input(u'请输入第二个数字')

if opa=='+':
    print '%d+%d=%d'%(first_num,second_num,first_num+second_num)
elif opa=='-':
    print '{0}-{1}-{2}'.format(first_num,second_num,first_num-second_num)
elif opa=='*':
    print '%s*%s=%s'%(first_num,second_num,first_num/second_num)
elif opa=='/':
    if second_num==0:
        print u'除数不能为0'
    else:
        print '%s/%s=%s'%(first_num,second_num,first_num/second_num)