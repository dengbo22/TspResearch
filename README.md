# TspSearch
2016.3.19
---
初次提交，整体代码可用，进入代码优化阶段

2016.5.30
---
修正代码可用，加入平均度衡量函数并且创建自定义函数集合用于进行规划测试

2016.6.5
---
加入.gitignore文件，删除.idea/workspace.xml文件

2016.6.6 —— Verion 1.3.0
---
1. 改动结果评判函数result_evaluation：  
`总长度 * AVERAGE + 方差 * (1 - AVERAGE)`改成`平均长度 + Weight * 方差(标准差)`
2. 修改loop_search()函数的返回值，如果最优解的值有所改动，则返回True，否则返回False。并在此基础上修改打印内容。
3. 去除第一轮信息素更新，只使用第二轮的信息素。
4. 尝试修改信息素更新过程，结果变差

2016.7.11 —— Version 1.3.1
改动整体框架，修改了部分函数的调用
