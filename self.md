fetal-ecg-synthetic-database-1.0.0/generate_dataset.py: 组织这个数据集成下面这样的结构，用以网络的使用(Organize the data in the folder structure (fecg_ground,mixture,mecg_ground) for the network.) 

init.py: 训练网络的选项列表(List of options used to train the network.) 

networks.py: 网络在模拟出来的数据集上测试的代码(The W-NETR architecture for FECG extraction testing on simulation dataset.)
networks_real.py: 网络在真实的数据集上测试的代码(The W-NETR architecture for FECG extraction testing on real dataset.)

test_simulation.py: (Runs the testing on simulation dataset.)
test_real.py: Runs the testing on real dataset.

1.调用关系理解（根目录下的代码文件）：
test.py,test_pcdb.py,test_real.py是最顶层的逻辑设计，它们是用了init.py,networks_real.py,networks.py模块

networks.py,networks_real.py是网络模型的设计，它们调用了init.py模块

init.py是训练网络的选项列表，是最基础的实现吧

2.调用关系理解（每个数据集（ADFECGDB、PCDB、fetal-ecg-synthetic-database-1.0.0）文件夹下的代码文件）：()
它们相对于根目录下的脚本是独立的自称体系的，主要目的只是为了处理数据集

3.调用关系理解（models目录下的文件：dynunet_block.py、PatchEmbeddingBlock.py、unetr_block.py、unetr_real.py、unetr.py、vit.py）：

4.调用关系理解（util目录下的文件：dataset_pcdb.py、dataset_real.py、dataset.py）
