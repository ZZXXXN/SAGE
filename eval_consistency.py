import os
import sys
import glob
import numpy as np

# 1. 确保能引用到师姐的 RAFT 模块
sys.path.append('RAFT') 

# 2. 导入师姐的函数
# 注意：demo.py 里有 argparse，直接调用可能会跟主程序冲突
# 我们通过修改 sys.argv 来“骗”过 argparse，或者确保 demo.py 能在你的环境下跑通
try:
    from RAFT.demo import get_range_metrics
except ImportError:
    print("错误: 找不到 RAFT.demo。请确保 'RAFT' 文件夹在当前目录下。")
    sys.exit(1)

def calculate_consistency(frames_dir, stride=1):
    """
    计算一致性指标
    Args:
        frames_dir: 存放风格化结果图片的文件夹
        stride: 步长。1 = 短程 (Short-range), 10 = 长程 (Long-range)
    """
    # 获取所有帧，并按文件名排序 (000.png, 001.png...)
    # 支持 png 和 jpg
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")) + 
                    glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if len(frames) < 2:
        print(f"错误: 在 {frames_dir} 中没找到足够的图片。")
        return 0.0, 0.0

    print(f"正在计算一致性 (Stride={stride})... 共 {len(frames)} 帧")
    
    lpips_list = []
    rmse_list = []

    # 遍历视频帧
    # 如果是 Short-range (stride=1): 比较 Frame 0 vs 1, 1 vs 2...
    # 如果是 Long-range (stride=10): 比较 Frame 0 vs 10, 1 vs 11...
    for i in range(len(frames) - stride):
        img1_path = frames[i]
        img2_path = frames[i + stride]
        
        try:
            # 调用师姐的函数
            # 注意：师姐的代码内部每次都会加载模型，速度可能会有点慢，但这是最稳妥的调用方式
            # 这里的 get_range_metrics 会自动解析 argparse，我们不传额外参数让它用默认值
            # 关键：我们依靠 img1_path 和 img2_path 传参
            
            # 稍微处理一下 argparse 的冲突风险，把 sys.argv 临时清空
            original_argv = sys.argv
            sys.argv = ['demo.py'] 
            
            val_lpips, val_rmse = get_range_metrics(img1_path, img2_path)
            
            # 恢复 argv
            sys.argv = original_argv
            
            lpips_list.append(val_lpips)
            rmse_list.append(val_rmse)
            
            # 打印进度 (可选)
            print(f"Frame {i} -> {i+stride}: LPIPS={val_lpips:.4f}, RMSE={val_rmse:.4f}")
            
        except Exception as e:
            print(f"Frame {i} 出错: {e}")
            sys.argv = original_argv # 确保出错也能恢复

    if len(lpips_list) == 0:
        return 0.0, 0.0

    avg_lpips = np.mean(lpips_list)
    avg_rmse = np.mean(rmse_list)
    
    return avg_lpips, avg_rmse

if __name__ == "__main__":
    # ================= 配置路径 =================
    # 指向你的 StyleGaussian 生成结果的文件夹
    style_result_dir = "./output/truck-g/best/artistic-style_weight_5.2/default/train/0/renders" 
    
    # ================= 1. 计算短程一致性 (Short Range) =================
    # 定义：相邻帧之间的一致性，Stride = 1
    print("\n>>> 开始计算短程一致性 (Short Range Consistency)...")
    short_lpips, short_rmse = calculate_consistency(style_result_dir, stride=1)
    
    # ================= 2. 计算长程一致性 (Long Range) =================
    # 定义：间隔较远的帧，通常 Stride 取 5, 7 或 10，看你们领域的习惯
    # 比如 CoDeF 论文里有时用 stride=10
    print("\n>>> 开始计算长程一致性 (Long Range Consistency)...")
    long_lpips, long_rmse = calculate_consistency(style_result_dir, stride=10)

    # ================= 输出最终结果 =================
    print("\n" + "="*40)
    print(f"测试文件夹: {style_result_dir}")
    print("="*40)
    print(f"[Short-range (interval=1)]")
    print(f"  Mean LPIPS: {short_lpips:.5f}  (越小越好)")
    print(f"  Mean RMSE:  {short_rmse:.5f}  (越小越好)")
    print("-" * 40)
    print(f"[Long-range (interval=10)]")
    print(f"  Mean LPIPS: {long_lpips:.5f}  (越小越好)")
    print(f"  Mean RMSE:  {long_rmse:.5f}  (越小越好)")
    print("="*40)