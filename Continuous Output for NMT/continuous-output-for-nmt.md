## Beyond BLEU: Training Neural Machine Translation with Semantic Similarity，ACL 2019

* BLEU 并不完全可信，他会给语义相似但词汇不同的句子以惩罚。本文介绍一种新的评价指标SIMILE。

* SIMILE是一种对句子相似度进行“连续”度量的指标，借鉴了专注于领域不可知的语义相似度的度量工作。

* 利用相似度评估句子，模型容易产生具有很多重复词或短语的“超长”句子（长度超过参考译文），为此SIM对“超长”句子进行惩罚：

  $LP(r,h) = e^{1-\frac{max(|r|,|h|)}{min()|r,|h||}} $

  其中$r$为参考译文，$h$为模型生成的译文。

* SIM的最终计算方式为：

  $SIMILE=LP(r,h)^{\alpha}SIM(r,h)$

  其中$\alpha \in \{0.25,0.5\}$，目的在于降低$LP(.)$的影响。

* 语义相似度并不能完全替代质量评估，但至少就最小风险训练而言，它是个不错的指标。

## Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs，ICLR 2019

* softmax的缺点：
  
  * 速度慢、需要较大的内存、词汇表大小固定，不利于推理OOV的词
  
* 因此此论文使用连续词嵌入层替换Softmax层

* 创新点：
  * 新的损失函数
  * 使用预先训练的词嵌入概率分布进行训练和推断的过程
  
* 训练：
  
  * 最小化模型输出的向量和参考译文词向量（来自预训练模型）的距离
  * 使用OpenNMT中标准序列到序列模型：编码器由一层双向LSTM组成，解码器由两层基于注意力的网络组成。
  
* 推断

  * 在目标词嵌入空间中搜索和当前输出词嵌入向量最近的向量，那么该词即是当前时刻预测的词

    $w_{predict}=argmin_w\{d(\hat{e},e(w))|w \in V\}$

    其中$e(w)$是目标语言词嵌入向量，$\hat{e}$是模型输出向量，$V$是词汇表。

  * 在NLLvMF中，选取和$\hat{e}$的vMF相似度最高的词做为输出词

  * 这种推断方式只能选取一个词，相当于**greedy search**

* LOSS计算

  * cosine

    * $Loss=1-\frac{\hat{e}e(w)}{||\hat{e}||\cdot ||e(w)||}$

  * Max Margin Loss

    * $Loss=\sum_{w'\in V,w'\neq w}max(0,\gamma +cos(\hat{e},e(w'))-cos(\hat{e},e(w)))$

    ​       其中$\gamma$是超参数，$w'$表示负的样本。

  * NLLvMF

    * $NLLvMF(\hat{e};e(w))=-log(C_m(||\hat{e}||))-\hat{e}^T e(w)$

    ​      其中$C_m(\cdot)$是正则项：$C_m(k)=\frac{k^{m/2-1}}{(2\pi)^{m/2}I_{m/2-1}(k)}$

  * Regularization of NLLvMF

    * $NLLvMF(\hat{e})_{reg1}=-logC_m(||\hat{e}||)-\hat{e}^Te(w)+\lambda_1||\hat{e}||$

    * $NLLvMF(\hat{e})_{reg2}=-logC_m(||\hat{e}||)-\lambda_2\hat{e}^Te(w)$

      其中$\lambda_1$和$\lambda_2$是scalar参数，且$\lambda_2<1$

## A Margin-based Loss with Synthetic Negative Samples for Continuous-output Machine Translation，EMNLP 2019

* 探究了margin-based loss在优化连续输出模型中的作用，并通过实验发现基于Synthetic负采样的syn-margin方法比vMF和标准margin-based losses更好
* 贡献：
  * syn-margin loss的公式
  * syn-margin loss的几何分析
  * 对NMT的连续输出模型的实验



