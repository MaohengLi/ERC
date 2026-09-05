# 论文接口对齐版实现报告

新目录实现了最终推荐主线：

```text
Causal-ERC → DialogueGCN
+ A/V/T 三Token
+ 多模态 Peak-End 双系统路由
+ C1/C2 Prompt + 特征门控
```

## 与原始 Causal-ERC 保持一致的部分

- LLM输入仍然是 Causal Prompt 加 Audio、Visual、Text 三类连续特征Token。
- Prompt仍由 Instructions、History Context、Causal Prompt、Label Statement组成。
- LLM只接收三类模态投影，不接收 Graph Token 或 Causal Route Token。
- 输出仍为每个目标话语的六分类 emotion logits。
- Soft-HGR只作用于三类融合模态特征。
- LoRA和4B QLoRA加载方式保持一致。

## 改造部分

- 三路 DialogueRNN 替换为说话人关系感知 DialogueGCN。
- DialogueGCN默认采用过去向图，当前话语不读取未来话语。
- T/A/V内部路由探针用于下一轮 Peak-End 判定，不改变LLM输入接口。
- C1偏向图历史上下文，C2偏向局部融合特征；连续 `alpha_system2` 在投影前完成门控。
- 路由使用上一轮 T/A/V/LLM logits，不读取金标签。

## 重要实验建议

主实验应至少包含：

1. 原始 Causal-ERC：DialogueRNN + A/V/T 三Token。
2. 本模型：DialogueGCN + A/V/T 三Token + Dual-System Routing。
3. 本模型去掉 Peak-End。
4. 本模型去掉特征门控。
5. 本模型加 Graph/Route Token（仅作为额外扩展消融）。

这样可以分别识别 DialogueGCN、Peak-End 路由、特征门控和额外Token的贡献。

