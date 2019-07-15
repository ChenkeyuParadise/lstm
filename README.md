# 运行环境准备

### 安装anaconda


```
conda install scikit-learn

conda install pandas

conda install matplotlib
``````
# 相关文档

[keras手册](https://keras.io/zh/search.html?q=fit)

# 已做基础工作：已实现代码goodcar3days15min.py。该代码下的数据格式见表格"goodcar3days15min"。

该数据为一列，一个路段ID下的，共三天，每15min一个间隔的流量，考虑用两天的预测第三天的数据，并且采用窗口方法的回归，用前look_back的时刻流量预测下一个step的时刻流量。

# 问题：

我想实现8万多个ID的路段预测，数据名为“1128+1201”，简单来硕是从1到8万的实现。

处理数据为8万个ID.每个ID出一个结果，输出到excel中，并得到每个路段的MAE等指标。

我在后续工作中将会从路段等级、流量等方面分别分析预测的准确性等问题。
