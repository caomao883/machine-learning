#coding:utf-8
import pandas as pd
import numpy as np
#读表
def load_sheef():
    file_path = "data/test.xlsx"
    data = pd.read_excel(file_path,header=None)
    print data
    return data

#写表
def write_sheet(data,header=None,index=None):
    file_path = "result/result.xlsx"
    data.to_excel(file_path,header=header,index=index)

#创建表
def create_sheet():
    file_path = "result/result.xlsx"
    names = ["id1","id2","id3","id4"]
    indexs = ["index1","index2"]
    data = pd.DataFrame(np.array([[1,2,3,4],[5,6,7,8]]),columns=names,index=indexs)
    write_sheet(data,True,False)


#计算每一行的和,并且加到最后一列
def sum_sheet():
    file_path = "data/test.xlsx"
    file_output = "result/result.xlsx"
    data = pd.read_excel(file_path,header=None)
    row_len,column_len = data.shape
    sum_column = []
    for i in range(row_len):
        sum_column.append(sum(data.iloc[i]))
    data[column_len] = sum_column
    data.to_excel(file_output,header=None,index=None)


#计算每一行的和,并且加到最后一列,使用循环的版本
def sum_sheet1():
    file_path = "data/test.xlsx"
    file_output = "result/result.xlsx"
    data = pd.read_excel(file_path,header=None)
    row_len,column_len = data.shape
    sum_column = []
    for i in range(row_len):
        sum_value = 0
        for j in range(column_len):
            sum_value += data.iloc[i][j]
        sum_column.append(sum_value)
    data[column_len] = sum_column
    data.to_excel(file_output,header=None,index=None)

#选取表的某几列，生成到第二个表中
def choose_some_columns():
    file_path = "data/test1.xlsx"
    file_output = "result/result.xlsx"
    data = pd.read_excel(file_path)
    names = data.columns
    names = [u"姓名",u"语文",u"英语",u"数学",u"政治",u"化学"]
    names_new = [u"姓名",u"语文",u"化学"]
    new_data = data[names_new]
    new_data.to_excel(file_output,sheet_name="Sheet2",index=None)


# def create_sheet2():
#     file_path = "data/test1.xlsx"
#     names = [u"姓名", u"语文", u"英语", u"数学", u"政治", u"化学"]
#     data = pd.DataFrame(np.array([[u"小李","99","98","98","97","95"],[u"小芳","95","96","97","98","99"]]),columns=names)
#     data.to_excel(file_path,index=None)
if __name__ == '__main__':
    print "送给最美最美业凤的礼物"
    #create_sheet()
    #sum_sheet1()
    #choose_some_columns()
    #create_sheet2()
    choose_some_columns()