1. 统一测试集：综合成本考虑，我们决定从Vbench的51条prompt中随机抽出10条作为测试集，命名为Vbench10，
依然放到/hy-tmp/work/Wan2.2/test_sets下，后续后续所有工作都在这个测试集上测试
2. 复现AdaCache结果：阅读原论文，选择speedup为 1.5x, 2x, 4x三组配置，在Vbench10上测试表现
3. 复现Taylorseer结果：阅读原论文，选择speedup为 1.5x, 2x, 4x三组配置，在Vbench10上测试表现，
注意这个任务需要另一台多卡机器来跑，所以我们的任务是写好推理代码，实验脚本，确保多卡推理能跑通
4. 使用seacache方法的timestep cache only实验需要在Vbench10上重新跑一遍，方便与merge做对照
5. sea-style merge方法目前表现不好，我们的目标是说明merge在一些情况下由于timestep cache only，
因此我们需要先想想怎么调整方法，然后在Vbench10上测试一下，并与timestep cache only的结果作比较
6. /hy-tmp/work/Wan2.2/adaptive_threshold_predictor目前的表现并不好，我们需要在Vnbench10上重新测试一下，
分析一下结果，看看问题出在了哪里