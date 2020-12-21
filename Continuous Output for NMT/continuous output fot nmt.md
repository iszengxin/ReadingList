* Beyond BLEU: Training Neural Machine Translation with Semantic Similarity，ACL 2019
  * BLEU 并不完全可信，他会给语义相似但词汇不同的句子以惩罚。本文介绍一种新的评价指标SIMILE。
  
  * SIMILE是一种对句子相似度进行“连续”度量的指标，借鉴了专注于领域不可知的语义相似度的度量工作。
  
  * 利用相似度评估句子，模型容易产生具有很多重复词或短语的“超长”句子（长度超过参考译文），为此SIM对“超长”句子进行惩罚：
  
    $LP(r,h) = e^{1-\frac{max(|r|,|h|)}{min()|r,|h||}} $
  
    其中$r$为参考译文，$h$为模型生成的译文。
  
  * SIM的最终计算方式为：
  
    $SIMILE=LP(r,h)^{\alpha}SIM(r,h)$
  
    其中$\alpha \in \{0.25,0.5\}$，目的在于降低$LP(.)$的影响。
  
  * 语义相似度并不能完全替代质量评估，但至少就最小风险训练而言，它是个不错的指标。
  
* Von Mises**-Fisher** Loss for Training Sequence to Sequence Models with Continuous Outputs，ICLR 2019

  * softmax的缺点：
    * 速度慢、需要较大的内存、词汇表大小固定，不利于推理OOV的词
  * 因此此论文使用连续词嵌入层替换Softmax层
  * 创新点：
    * 新的损失函数
    * 使用预先训练的词嵌入概率分布进行训练和推断的过程
  * 

* A Margin-based Loss with Synthetic Negative Samples for Continuous-output Machine Translation，EMNLP 2019

  * 不使用softmax而是训练词嵌入的模型，模型参数更少，训练是速度更快。



