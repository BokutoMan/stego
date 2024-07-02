# 本项目为多媒体安全第二次作业

## 项目说明

### 目录img下为本次实验使用的图片和生产的图片

img/original中为原图

img/steg下为嵌入数据的图片

img/mask下的图片标记了嵌入位置

img/png中的图片是为了方便查看将pgm文件转为了png格式

### 目录embedding中有进行嵌入的代码

embedding/SRM.py为使用SRM选择嵌入位置的代码

embedding/local_variance.py为使用局部方差选择嵌入位置的代码

embedding/test.py为调用两个文件执行嵌入的代码，注解掉的为进行图片格式转换的代码

### 目录SRM_analysis中有使用SRM方法进行隐写分析的代码

其中的三个目录分别存放三类图片的特征文件

### 目录comatrix中的代码为使用共生矩阵进行隐写分析的代码


