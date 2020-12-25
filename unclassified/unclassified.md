## Bridging the Gap between Training and Inference for Neural Machine Translation，ACL best paper 2019

* 在模型训练中，其当前时刻预测的词依赖的context来自标准译文，而在推断阶段，当前时刻预测的词所依赖的context来自模型自己的推断。训练与推断的差别，将导致模型在推断时产生误差。因此这篇文章在模型训练阶段，对context中的词进行采样，使其既包括来自标准译文的词，又包括模型自己预测的词，以达到缓解训练和推断不一致。

