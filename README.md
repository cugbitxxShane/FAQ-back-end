# FAQrobot

一个自动回复FAQ问题的聊天机器人。目前使用了简单词汇对比、词性权重、词向量3种相似度计算模式。输入符合格式的FAQ文本文件即可立刻使用。欢迎把无法正确区分的问题和FAQ文件发送到评论区。
 
## 程序版本和依赖库
使用 python3 运行  
jieba 分词使用的库  
gensim  词向量使用的库，如果使用词向量vec模式，则需要载入  

## 依赖的文件
如果使用词向量vec模式，需要下载3个文件：Word60.model，Word60.model.syn0.npy，Word60.model.syn1neg.npy  
下载地址：http://pan.baidu.com/s/1kURNutT 密码：1tq1  

## FAQ知识库文件
FAQ文件包含想要告知用户的问答内容。   
FAQ文件必须是UTF-8的无bom格式的文本文件。   
  
注释：注释文字由#开头。（整个一行都是注释内容）  
  
问答块格式如下：  
【问题】问题标题（可以有1或多个，至少有1个。必须由"【问题】"开头。）  
答案内容（可以有多行，必须紧跟着上面的【问题】，多行答案中间不能有空白的行。）  
多个问答块之间可以用空白行分割  
   
程序默认使用的是雷达问答FAQ文件。你可以载入自己编辑的FAQ文件。  
![展示效果](https://github.com/ofooo/FAQrobot/blob/master/doc/%E6%88%AA%E5%9B%BE%E5%B1%95%E7%A4%BA1.jpg)
 
## 主程序FAQrobot.py
直接运行该文件，即可对雷达问题进行问答。你可以载入自己的FAQ文件，请保证FAQ文件格式正确。  
robot.answer(inputtxt,'simple_POS') 可得出输入问题的返回答案。  
simType参数有如下模式：   
simple：简单的对比相同词汇数量，得到句子相似度  
simple_POS：简单的对比相同词汇数量,并对词性乘以不同的权重，得到句子相似度  
vec：用词向量计算相似度,并对词性乘以不同的权重，得到句子相似度  
all：调试模式，把以上几种模式的结果都显示出来，方便对比和调试  
  
inputtxt 可输入的特殊文本命令：  
-zsk 显示当前知识库  
-s -1 查看上一个问句的结果和中间参数  
-q -1 重复提问，把当一个问句当做输入  
-reload 重新载入QA知识库  
