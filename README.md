# 实现了任务3架构变体 和 6工程实现

<!-- ![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png) -->

## 架构变体

### YARN
模型架构中改进ROPE为YARN (nanochat/gpt.py)，具体实现是修改了_precompute_rotary_embeddings 中计算freq的逻辑     
思路：保护高频信息，保留短距离信息精度；对于低频则缩小频率，增加长文本捕捉能力。

预期效果     
1 用长文本benchmark 比如 LongBench 相较之前显著提升，并且在基本任务上不掉点。  
2 在足够长文本训练上ppl能保持稳定下降，之前ROPE在同样情况下ppl会突然飙升。

没多余时间验证长文本能力了

### Gated-Attention
把selfCasualAttention改进为Gated Attention (nanochat/gpt.py)，具体是在attn计算得到输出，还未进行线性映射的时候，加入一个哈达玛积的Gate机制。思路来源于NIPS 2025 best paper (https://arxiv.org/abs/2505.06708)

思路：文章已经验证了Gate机制放在attn计算后的位置最佳，V计算后的位置次之。并且门控计算中sigmoid激活和哈达玛积的效果搭配最好。    

对于原论文的创新：我发现原架构在V的位置已经加入Gate机制，而且是采用加法运算，这种gate效果不如改进后的好，但我保留了这一部分，因为我觉得V后的门控和attn计算后的门控同时作用，比单一门控模型表现会更好一些。    
这个猜测还需要进一步设计消融实验去验证效果：①单一门控和双门控效果对比；②双门控中加法运算和哈达玛积运算的对比

遇到的问题：想要保留最后一个token的各层attention map但是模型的计算层层封装，要修改很多参数和实现才能保留下来，不够简洁高效。
此外我的计算资源有限，没有训练充分，观察效果可能不佳，就没去实现。

验证需要观察：     
1 输出各层gating score （直接通过print输出）             
2 推理一段文本，取最后一个token推理中的各层attention maps          
3 loss曲线      

预期效果：        
1 训练充分之后，gating score矩阵应该是非常稀疏的，数值聚集在0附近           
2 显著缓解attention sinks现象。即将序列最后一个token推理中的各层attention maps绘制出热力图，高分数应该聚集在对角线，而不是首个Token。从而让sliding window更加有效捕捉最近的语义信息，不用再考虑最初几个token的信息，减轻KV cache压力。           
3 训练更稳定，体现在loss曲线上就是毛刺更少          
4 在多个benchmark上效果提升        

## UI & Top-p采样

### UI
在 (nanochat/ui.html) 中加入滑块控制采样参数：温度、Top-k、Top-p，实时更新。

### Top-p采样
在 (/nano/engine.py) 加入top-p参数，新增top-p采样逻辑        
并在 /scripts/chat_xxx.py：所有推理都加上top-p参数        

遇到如何选择top-p top-k的问题，是选其一还是两者兼容。最终选取了二者兼容，即取top-p和top-k交集，理由如下：           
1 二者兼容的情况包含了只有其一生效的情况，如 Top-p=1.0，Top-k=50 本质上只有Top-k起作用。         
2 能够更细粒度地控制采样。

