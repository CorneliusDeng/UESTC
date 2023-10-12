import time
from tqdm import tqdm, trange

# 使用tqdm库设置进度条
for i in tqdm(range(100), desc="Training", unit="epoch"):
    time.sleep(0.01)

# trange是tqdm的range版本
for i in trange(100, desc="Testing", unit="epoch"):
    time.sleep(0.01)

# 使用rich库设置进度条
from rich.progress import Progress, BarColumn, TimeRemainingColumn
# 创建进度条
with Progress(
    BarColumn(),
    "[progress.description]{task.description}",
    TimeRemainingColumn(),
) as progress:
    # 添加任务
    task1 = progress.add_task("[green]Training", total=100)
    task2 = progress.add_task("[red]Testing", total=100)

    # 更新任务进度
    for i in range(100):
        progress.update(task1, advance=1)
        progress.update(task2, advance=1)
        time.sleep(0.01)