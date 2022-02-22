import torch
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.utils import register_module_hooks


config_file = 'configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'checkpoint/swin_base_patch244_window1677_sthv2.pth'

# 指定设备
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

# 测试单个视频并显示其结果
video = 'demo/demo.mp4'
labels = 'demo/label_map_k400.txt'
results = inference_recognizer(model, video, labels)

# 显示结果
print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])

cache = get_local.cache
print(list(cache.keys()))