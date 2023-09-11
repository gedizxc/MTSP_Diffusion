# MT
MTSP+pre_task(diffusion)

## 数据集
1.所有下载地址：https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy <br>
2.因为数据量的原因，目前只提供五个数据集供测试 <br>

## 操作
1.运行Pretrain_diffusion中的main文件，会执行pre_main.py，在pretrain_checkpoint中形成.pth文件保存模型参数（目前提供ill和exchange两个数据集）<br>
2.请直接运行Linear_main中的ill和exchange来测试 NLinear+ pre_diffusion <br>

## 问题
1.注意Unet.py中34，38提了两个问题 <br>
2.效率：感觉sample很花时间，但是我sample并不在预训练里呀 <br>
3.我在测试的时候加载了两次模型参数，一次是pretrain，一次是train，这样做有点奇怪 <br>

## 实验数据
1.待测


