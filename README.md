# MTSP_Diffusion
MTSP+pre_task(diffusion)

## 数据集
1.所有下载地址：https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy <br>
2.因为数据量的原因，目前只提供四个数据集供测试 <br>

## 操作（以exchange为例）
1.运行Pretrain_diffusion中的main文件，会执行pre_main.py，在pretrain_checkpoint中形成.pth文件保存模型参数（目前提供ill和exchange两个数据集）<br>
2.请直接运行Linear_main中的ill和exchange来测试 NLinear+ pre_diffusion <br>

## 实验数据
### Baseline：[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504.pdf):
![Nlinear](https://github.com/cure-lab/LTSF-Linear/blob/main/pics/Mul-results.png?raw=true](https://github.com/gedizxc/MT/blob/master/1024064503.png)https://github.com/gedizxc/MT/blob/master/1024064503.png)  
### Our work
#### Accurate test results
![Accurate test](https://github.com/gedizxc/MTSP_Diffusion/blob/master/baseline.png)

#### Ablation test results
![Ablation test](https://github.com/gedizxc/MTSP_Diffusion/blob/master/%20Ablation.png)

#### hyper-parameters test results
![period test](https://github.com/gedizxc/MTSP_Diffusion/blob/master/period.png)





