# MTSP_Diffusion(CCNU 1207)
MTSP+pre_task(diffusion)

## 数据集
1.所有下载地址：https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy <br>
2.因为数据量的原因，目前只提供四个数据集供测试 <br>

## 操作（以exchange为例）
1.请根据不同数据集修改data_loader中308行，period数值(ill是36，其他都用96了) <br>
2.运行Pretrain_diffusion中的main文件，会执行pre_main.py，在pretrain_checkpoint中形成.pth文件保存模型参数（目前提供ill和exchange两个数据集）<br>
3.请直接运行Linear_main中的ill和exchange来测试 NLinear+ pre_diffusion <br>
4.可以尝试在Linear_main中每个数据集main文件57行更换pred_len #96,192,336,720<br>

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

## 数据分析
### 原始数据集分析
![draft](https://github.com/gedizxc/MTSP_Diffusion/blob/master/draft.png)

### 训练策略分析
![ACF](https://github.com/gedizxc/MTSP_Diffusion/blob/master/ACF.png)

### 模型优化分析
![optim](https://github.com/gedizxc/MTSP_Diffusion/blob/master/optim.png)


## 致谢
https://github.com/cure-lab/LTSF-Linear





