#篮球定位器
1.使用python与YOLO实现目标检测
2.目的：读取验收视频，识别画面中的篮球，绘制检测框，读取类别 ，输出置信度，最终输出视频
#环境依赖
普通电脑，安装ultralytics与opencv-python
ultralytics:YOLO模型库，用来训练目标检测
opencv-python:OpenCv,读取视频，画检测框，保存输出视频
#运行方法
1.安装依赖
打开终端，确认安装ultralytics与opencv_python
2.模型微调训练
python train.py为启动模型微调训练的命令
解释（这个命令所做的事情）
1. 读取 dataset/data.yaml，加载篮球训练数据集
2. 加载YOLO预训练模型，用你的篮球标注图片继续微调
3. 训练过程不断调整模型参数，让模型学会识别篮球
4. 训练结束，把最好的模型权重保存：runs/detect/train/weights/best.pt
*教ai认识篮球
我使用vs code：
1.确认左侧文件栏，train.py在项目根目录
2.新建终端
3.确认终端路径(重要)
4.直接输入命令python train.py
注释：train.py代码在仓库中
3.视频推理
读取验收视频 evaluation/basketball_dribble_evaluation.mp4，逐帧检测篮球，
在画面绘制检测框、basketball类别名、置信度，最后在项目根目录输出 result.mp4。
#漏检情况：
问题：篮球被手、手臂、身体遮挡，只露出一小部分球面，模型缺少完整特征，识别不到篮球
解决办法:修改代码加载模型这一行，把普通检测变为追踪模式（对应代码25行）
# 原来代码
# results = model(frame)
# 改成跟踪代码
results = model.track(frame, persist=True)
#训练结果图resuits.png
第一行（训练集 train）
• train/box_loss：预测框位置损失，越小代表框越准
• train/cls_loss：分类损失，越小代表区分篮球/背景越好
• train/dfl_loss：辅助定位损失
• metrics/precision(B)：精确率，预测出来的目标真的是篮球的比例
• metrics/recall(B)：召回率，画面里篮球能被找出来的比例
第二行（验证集 val）
• val/box_loss、val/cls_loss、val/dfl_loss：验证集上的损失
• metrics/mAP50(B)：IOU=0.5时平均精度
• metrics/mAP50-95(B)：严格条件下综合精度
#验证集预测效果图val_pred.png
目的；
验证模型可以识别篮球，
用来分析失败案例：图里那些篮球被挡住、没有框的图片，是漏检例子。
