# The-Repair-strategy-and-Invulenrability-Research-of-Complex-Networks
- Comunication network connectivity is essential to our daily life. This project we focus on the repair strategy and invulenrability of complex network, we provide alternative nodes geographical location information and connection methode when nodes are damaged seriously. This repair strategy can minimize the length of communication path and ensure the network connectivity effectively.
- First we calculate the geographical distance between each notes based on Great-circle formula, and designing shortest path connection scheme by Prim algortithm.
- In addition, transforming the original problem as the construction of Steiner tree, we use grid search and genetic algorithm (GA) to optimize the path length and providing the alternative nodes numbers, geographical location information as well as connection methode when the specified nodes failure. Simulation experiment results show that our repair strategy can ensure the network connectivity.
- Furthermore, relying on above research, we regard the minimum connected dominating set as the key nodes and add the edges amonge them to improve the connectivity of network. Then, with different number of failure nodes, we simulate the change of path lengh and connectivity of network with differnet number of backup nodes and edges. Finally, considering the path length and network connectivity, we construct an optimal communication network.

**You can download the full report from [here](https://github.com/PrideLee/The-Repair-strategy-and-Invulenrability-Research-of-Complex-Networks/blob/master/%E5%A4%8D%E6%9D%82%E9%80%9A%E4%BF%A1%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BF%AE%E5%A4%8D%E7%AD%96%E7%95%A5%E4%B8%8E%E6%8A%97%E6%94%BB%E5%87%BB%E6%80%A7%E7%A0%94%E7%A9%B6.pdf).**

## 摘要
随着科学技术的迅速发展，通信技术已成为国家竞争和社会进步的关键环节。日常生活中通信网络的可靠性保证对于信息的收集获取、消息的传递和日常生活的有序进行起着至关重要的作用。故本文对复杂通信网络的修复策略与鲁棒性进行研究，给出节点严重毁坏时备选节点的确定与连接方式，以保证在最短路径下恢复网络连通，同时给出高连通性网络设计方案，本文工作如下：

问题一要求给出网络的最短路径连接方案。对此本文首先基于Great-circle公式计算各城市节点间的球面距离，然后将问题转化为最小生成树问题，利用Prim算法求解。同时使用Python绘制网络的连接示意图，进行可视化展示。其中最短路径的计算结果为25343.943km。
问题二要求给出指定节点故障后，备份节点的位置、数目及连接方式使网络恢复连通。对此本文首先分析故障节点的边数，并将其分为边数为1，边数为2，边数大于2，三类情况进行讨论。当边数大于2时，该问题本质上为Steiner tree的构建问题，本文以最短路径为目标，讨论设置不同备选节点数量时的路径长度，利用实码加速遗传算法结合“先粗后精”搜索策略进行求解。最终计算结果显示上海、武汉、北京的备选节点数目均为1，其连接方式与问题一保持一致，且位置坐标分别为(121.46,31.24)，(114.77,30.76)，(115.79,39.21)。

问题三要求给出9个城市节点被同时损坏时，备份节点的数目、位置与连接方式，同时给出衡量网络连通性的指标。对此本文以问题二为基础，分析故障节点的位置及连接节点信息，发现该故障节点均集中于东经 ，北纬 这一区域内，且相互间存在较强关联。对此本文仍以启发式搜索算法求解备份节点数目为 时网络的最短路径长度。结果表明，当节点数目设置为7时网络路径最短，为2055.71km。针对网络的连通性评价，本文借鉴复杂网络的自然连通度指标进行描述。该指标具有区别网络结构分辨率高、可解释性好、严格单调等优点。

问题四要求在问题一构建网络的基础上设计“高可靠、短路径”的通信网，并模拟测试网络10%节点随机故障时其连通性变化。对此本文利用禁忌搜索与遗传算法相结合的策略寻求网络的最小支配集，然后以最小支配集中的顶点为关键节点，在其间增加连边，以提高网络的连通性。同时在不同关键节点数目下，探究增加的连边数目与自然连通度间的关系，并模拟仿真10%的节点遭遇故障时网络的连通度变化。实验表明当关键节点选择数目为25，边的添加数目为60时，网络的鲁棒性高且路径相对较短。

**关键词：复杂通信网络，鲁棒性，最短路径修复，自然连通度**

<div align=center><img width="800" height="400" src="https://github.com/PrideLee/The-Repair-strategy-and-Invulenrability-Research-of-Complex-Networks/blob/master/connectivity_simu.png"/></div>

<div align=center><img width="800" height="400" src="https://github.com/PrideLee/The-Repair-strategy-and-Invulenrability-Research-of-Complex-Networks/blob/master/networks.png"/></div>


