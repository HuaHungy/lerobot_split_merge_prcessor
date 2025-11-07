#!/usr/bin/env python3
"""
数据集拆分与合并工具 - 修复版本
支持按帧数或episode数量拆分单个数据集，以及合并多个数据集
修复了task_index在episodes_stats.jsonl和tasks.jsonl中的一致性问题

使用示例:
# 拆分模式 - 按episode数量
python split_merge_dataset_fixed.py split \
   --input /mnt/nas/synnas/docker2/robocoin-datasets/leju_robot_pass_the_cleaner_b \
   --output /home/kemove/robotics-data-processor/lerobot/test \
   --max_episodes 200 \
   --fps 20 

# 拆分模式 - 按帧数
python split_merge_dataset_fixed.py split \
   --input /mnt/nas/synnas/docker2/robocoin-datasets/agilex_cobot_decoupled_magic_twist_bottle_cap \
   --output /home/kemove/robotics-data-processor/lerobot/test \
   --max_entries 1000 \
   --fps 20

# 合并模式
python split_merge_dataset_fixed.py merge \
   --sources /home/kemove/robotics-data-processor/lerobot/basket_storage_banana \
             /home/kemove/robotics-data-processor/lerobot/basket_storage_fruit \
             /home/kemove/robotics-data-processor/lerobot/plate_storage \
             /home/kemove/robotics-data-processor/lerobot/stack_baskets \
             /home/kemove/robotics-data-processor/lerobot/twist_bottle_cap \
             /home/kemove/robotics-data-processor/lerobot/zip_up_the_document_bag \
   --output /home/kemove/robotics-data-processor/lerobot/agilex_merged \
   --max_episodes 550 \
   --fps 20 \
   --max_dim 32

# 拆分：从第 100 个 episode 开始取 50 个
python split_merge_dataset.py split \
    --input /mnt/nas/synnas/docker2/robocoin-datasets/realman_rmc_aidal_box_up_down \
    --output /home/kemove/robotics-data-processor/lerobot/box_up_down \
    --start_episodes 2 \
    --max_episodes 300

# 拆分：从第 20,000 帧开始取 10,000 帧（按整 episode 对齐）
python split_merge_dataset.py split \
    --input /path/to/ds \
    --output /path/to/out \
    --start_frames 20000 \
    --max_frames 10000

# 合并：跨多源数据集，从整体第 5,000 帧后开始合并 500 个 episode
python split_merge_dataset.py merge \
    --sources /path/to/ds1 /path/to/ds2 /path/to/ds3 \
    --output /path/to/merged \
    --start_frames 5000 \
    --max_episodes 500
"""

import argparse
import json
import math
import os
import shutil
import traceback
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# 导入合并相关函数
from merge import (
    load_jsonl,
    save_jsonl,
    copy_videos,
    copy_data_files,
    merge_stats,
)


def get_info(path: str) -> Dict:
    """读取数据集的info.json文件"""
    info_path = os.path.join(path, "meta", "info.json")
    with open(info_path) as f:
        return json.load(f)


def get_chunks_size(info: Dict, default: int = 1000) -> int:
    """从info中获取chunks_size，如果没有则使用默认值"""
    return info.get("chunks_size", default)


def detect_folder_dim(folder: str, fallback_dim: int) -> int:
    """检测文件夹中数据的维度"""
    try:
        info = get_info(folder)
        state_shape = info.get("features", {}).get("observation.state", {}).get("shape", [fallback_dim])
        action_shape = info.get("features", {}).get("action", {}).get("shape", [fallback_dim])
        
        # 取两者中的最大值
        detected_dim = max(state_shape[0] if state_shape else fallback_dim,
                          action_shape[0] if action_shape else fallback_dim)
        print(f"检测到文件夹 {folder} 的维度: {detected_dim}")
        return detected_dim
    except Exception as e:
        print(f"检测文件夹 {folder} 维度失败，使用默认值 {fallback_dim}: {e}")
        return fallback_dim


def get_video_keys(info: Dict) -> List[str]:
    """从info.json中提取视频键名"""
    video_keys = []
    for feature_name, feature_info in info.get("features", {}).items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)
    return video_keys


def pad_episode_stats(stats: Dict, from_dim: int, to_dim: int) -> None:
    """对episode统计数据进行维度填充"""
    if from_dim >= to_dim:
        return
    
    padding_size = to_dim - from_dim
    zero_padding = [0.0] * padding_size
    
    for feature_name in ["observation.state", "action"]:
        if feature_name in stats.get("stats", {}):
            feature_stats = stats["stats"][feature_name]
            for stat_type in ["min", "max", "mean", "std"]:
                if stat_type in feature_stats and isinstance(feature_stats[stat_type], list):
                    feature_stats[stat_type].extend(zero_padding)


def recalc_merged_stats_with_episode_stats(merged_stats: Dict, all_stats_data: List[Dict], target_dim: int):
    """使用episode统计数据重新计算合并后的全局统计"""
    if not all_stats_data:
        return
    
    # 重新计算observation.state和action的统计
    for feature_name in ["observation.state", "action"]:
        if feature_name not in merged_stats:
            continue
            
        # 收集所有episode的统计数据
        all_mins = []
        all_maxs = []
        all_means = []
        all_counts = []
        
        for episode_stats in all_stats_data:
            if feature_name in episode_stats:
                stats = episode_stats[feature_name]
                if "min" in stats and "max" in stats and "mean" in stats and "count" in stats:
                    all_mins.append(stats["min"])
                    all_maxs.append(stats["max"])
                    all_means.append(stats["mean"])
                    all_counts.append(stats["count"][0] if isinstance(stats["count"], list) else stats["count"])
        
        if all_mins and all_maxs and all_means and all_counts:
            # 确保所有数据都有相同的维度
            for i, (mins, maxs, means) in enumerate(zip(all_mins, all_maxs, all_means)):
                if len(mins) < target_dim:
                    padding = [0.0] * (target_dim - len(mins))
                    all_mins[i] = mins + padding
                    all_maxs[i] = maxs + padding
                    all_means[i] = means + padding
            
            # 计算全局统计
            all_mins_array = np.array(all_mins)
            all_maxs_array = np.array(all_maxs)
            all_means_array = np.array(all_means)
            all_counts_array = np.array(all_counts)
            
            # 更新合并后的统计
            merged_stats[feature_name]["min"] = all_mins_array.min(axis=0).tolist()
            merged_stats[feature_name]["max"] = all_maxs_array.max(axis=0).tolist()
            
            # 加权平均计算全局均值
            total_count = all_counts_array.sum()
            if total_count > 0:
                weighted_means = (all_means_array * all_counts_array.reshape(-1, 1)).sum(axis=0) / total_count
                merged_stats[feature_name]["mean"] = weighted_means.tolist()
            
            merged_stats[feature_name]["count"] = [int(total_count)]
            
            print(f"重新计算了 {feature_name} 的全局统计，目标维度: {target_dim}")


def select_episodes(
    source_folders: List[str],
    max_entries: Optional[int],
    max_episodes: Optional[int],
    # 新增：起始偏移参数（优先按帧，其次按episode）
    start_entries: Optional[int] = None,
    start_episodes: Optional[int] = None,
) -> Tuple[
    List[Tuple[str, int, int]],  # episode_mapping: (folder, old_index, new_index)
    List[Dict],                   # all_episodes (updated episode_index)
    List[Dict],                   # all_episodes_stats (updated episode_index)
    Dict[int, int],               # episode_to_frame_index
    Dict[str, int],               # folder_dimensions
    Dict[str, Dict[int, int]],    # folder_task_mapping (old task_index -> new)
    List[Dict],                   # all_tasks (new task_index mapping)
    List[Dict],                   # all_stats_data (per-episode stats only)
    int,                          # total_frames
]:
    """在合并/拆分前，按限制选择 episode 并构建必要映射与统计辅助数据。"""
    episode_mapping = []
    all_episodes = []
    all_episodes_stats = []
    episode_to_frame_index = {}
    folder_dimensions = {}
    folder_task_mapping: Dict[str, Dict[int, int]] = {}
    all_stats_data = []

    # 统一任务索引：按任务描述去重并重新编号
    task_desc_to_new_index: Dict[str, int] = {}
    all_unique_tasks: List[Dict] = []

    total_frames = 0
    selected_total_episodes = 0

    # 起始偏移计数
    skipped_frames = 0
    skipped_episodes = 0
    # 将 start_episodes 按“从第 N 条开始读（1-based）”解释为跳过 N-1 条
    to_skip_episodes = max(start_episodes - 1, 0) if start_episodes is not None else 0

    # 基于第一个数据集提供的默认上限维度（若无法检测，则使用 32）
    first_info = get_info(source_folders[0])
    default_max_dim = int(first_info.get("features", {}).get("observation.state", {}).get("shape", [32])[0])

    # 计算每个源数据集的状态维度（如找不到则使用默认）
    for folder in source_folders:
        folder_dimensions[folder] = detect_folder_dim(folder, default_max_dim)

    for folder in source_folders:
        # 任务映射（旧->新）
        folder_task_mapping[folder] = {}
        tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
        folder_tasks = load_jsonl(tasks_path) if os.path.exists(tasks_path) else []
        for task in folder_tasks:
            desc = task["task"]
            old_idx = task["task_index"]
            if desc not in task_desc_to_new_index:
                new_idx = len(all_unique_tasks)
                task_desc_to_new_index[desc] = new_idx
                all_unique_tasks.append({"task_index": new_idx, "task": desc})
            folder_task_mapping[folder][old_idx] = task_desc_to_new_index[desc]

        # 读取 episode 与 stats
        episodes = load_jsonl(os.path.join(folder, "meta", "episodes.jsonl"))
        stats_path = os.path.join(folder, "meta", "episodes_stats.jsonl")
        episodes_stats = load_jsonl(stats_path) if os.path.exists(stats_path) else []
        stats_map = {s["episode_index"]: s for s in episodes_stats if "episode_index" in s}

        # 选择 episode
        for ep in episodes:
            # 优先按帧跳过
            if start_entries is not None and skipped_frames < start_entries:
                skipped_frames += ep.get("length", 0)
                skipped_episodes += 1
                continue
            # 其次按episode跳过（按1-based语义跳过 N-1 条）
            if start_entries is None and start_episodes is not None and skipped_episodes < to_skip_episodes:
                skipped_frames += ep.get("length", 0)
                skipped_episodes += 1
                continue

            # 限制逻辑：优先 max_entries（帧数），其次 max_episodes
            if max_entries is not None and total_frames >= max_entries and selected_total_episodes > 0:
                break
            if max_entries is None and max_episodes is not None and selected_total_episodes >= max_episodes:
                break

            old_index = ep["episode_index"]
            new_index = selected_total_episodes

            # 更新 episodes
            ep_copy = dict(ep)
            ep_copy["episode_index"] = new_index
            all_episodes.append(ep_copy)

            # 更新 episode_stats（如存在）
            if old_index in stats_map:
                stats = dict(stats_map[old_index])
                stats["episode_index"] = new_index
                
                # 更新 episodes_stats 中的 task_index
                if "stats" in stats and "task_index" in stats["stats"]:
                    # 获取原始的 task_index
                    original_task_index = stats["stats"]["task_index"]
                    if isinstance(original_task_index, dict) and "min" in original_task_index:
                        # 如果是统计格式，取min值作为原始task_index
                        old_task_idx = int(original_task_index["min"][0])
                    else:
                        # 如果是直接值
                        old_task_idx = int(original_task_index)
                    
                    # 查找新的task_index
                    if old_task_idx in folder_task_mapping[folder]:
                        new_task_idx = folder_task_mapping[folder][old_task_idx]
                        # 更新统计信息中的task_index
                        if isinstance(stats["stats"]["task_index"], dict):
                            stats["stats"]["task_index"]["min"] = [new_task_idx]
                            stats["stats"]["task_index"]["max"] = [new_task_idx]
                            stats["stats"]["task_index"]["mean"] = [float(new_task_idx)]
                        else:
                            stats["stats"]["task_index"] = new_task_idx
                
                # 零填充对齐维度到最终目标维度前，先记录原维度（安全起见统一按最大）
                all_episodes_stats.append(stats)
                if "stats" in stats:
                    all_stats_data.append(stats["stats"])

            # 记录该 episode 的起始帧索引
            episode_to_frame_index[new_index] = total_frames

            # 累加帧数
            episode_length = ep.get("length", 0)
            total_frames += episode_length
            selected_total_episodes += 1

            # 映射记录
            episode_mapping.append((folder, old_index, new_index))

    return (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_unique_tasks,
        all_stats_data,
        total_frames,
    )


def write_meta_and_copy(
    source_folders: List[str],
    output_folder: str,
    episode_mapping: List[Tuple[str, int, int]],
    all_episodes: List[Dict],
    all_episodes_stats: List[Dict],
    folder_dimensions: Dict[str, int],
    folder_task_mapping: Dict[str, Dict[int, int]],
    episode_to_frame_index: Dict[int, int],
    all_stats_data: List[Dict],
    all_tasks: List[Dict],
    total_frames: int,
    max_dim_cli: Optional[int],
    fps: int,
):
    """写 meta、合并统计并执行数据与视频拷贝。"""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    # 基于第一个数据集生成 info 基本框架
    base_info = get_info(source_folders[0])
    chunks_size = get_chunks_size(base_info)
    video_keys = get_video_keys(base_info)

    total_episodes = len(all_episodes)
    total_videos = len(video_keys) * total_episodes

    # 最终目标维度：优先使用 CLI 传入值，否则取源中的最大维度
    actual_max_dim = max_dim_cli or max(folder_dimensions.values())

    # 保存 episodes 与 episodes_stats（先对 episodes_stats 做维度对齐）
    aligned_episode_stats = []
    for stats in all_episodes_stats:
        # 找到该 episode 的源 folder 以取原维度（安全起见统一按最大）
        pad_episode_stats(stats, from_dim=actual_max_dim, to_dim=actual_max_dim)
        aligned_episode_stats.append(stats)

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(aligned_episode_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))
    
    # 过滤tasks.jsonl，只保留实际使用的task_index
    used_task_indices = set()
    for episode in all_episodes:
        # 从episode数据中获取task_index（如果存在）
        if "task_index" in episode:
            used_task_indices.add(episode["task_index"])
    
    # 如果没有从episodes中找到task_index，从episodes_stats中获取
    if not used_task_indices:
        for stats in aligned_episode_stats:
            if "stats" in stats and "task_index" in stats["stats"]:
                task_idx_info = stats["stats"]["task_index"]
                if isinstance(task_idx_info, dict) and "min" in task_idx_info:
                    used_task_indices.add(int(task_idx_info["min"][0]))
                else:
                    used_task_indices.add(int(task_idx_info))
    
    # 过滤all_tasks，只保留实际使用的任务
    filtered_tasks = []
    for task in all_tasks:
        if task["task_index"] in used_task_indices:
            filtered_tasks.append(task)
    
    save_jsonl(filtered_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    # 合并与重算 stats.json（全局统计）
    stats_list = []
    for folder in source_folders:
        stats_path = os.path.join(folder, "meta", "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats_list.append(json.load(f))
    if stats_list:
        merged_stats = merge_stats(stats_list)
        recalc_merged_stats_with_episode_stats(merged_stats, all_stats_data, target_dim=actual_max_dim)
        with open(os.path.join(output_folder, "meta", "stats.json"), "w") as f:
            json.dump(merged_stats, f, indent=4)

    # 更新 info.json
    base_info["total_episodes"] = total_episodes
    base_info["total_frames"] = total_frames
    base_info["total_tasks"] = len(filtered_tasks)  # 使用过滤后的任务数量
    base_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
    base_info["splits"] = {"train": f"0:{total_episodes}"}
    base_info["fps"] = fps

    # 对齐 observation.state 与 action 的 shape
    for feature_name in ["observation.state", "action"]:
        if feature_name in base_info.get("features", {}) and "shape" in base_info["features"][feature_name]:
            base_info["features"][feature_name]["shape"] = [actual_max_dim]

    base_info["total_videos"] = total_videos

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(base_info, f, indent=4)

    # 拷贝视频与数据文件
    copy_videos(source_folders, output_folder, episode_mapping)
    copy_data_files(
        source_folders=source_folders,
        output_folder=output_folder,
        episode_mapping=episode_mapping,
        max_dim=actual_max_dim,
        fps=fps,
        episode_to_frame_index=episode_to_frame_index,
        folder_task_mapping=folder_task_mapping,
        chunks_size=chunks_size,
    )

    print(f"Done: {total_episodes} episodes, {total_frames} frames, output={output_folder}")


def mode_merge(args: argparse.Namespace):
    """合并模式：将多个数据集合并为一个。"""
    source_folders = args.sources
    fps = args.fps if args.fps is not None else get_info(source_folders[0]).get("fps", 20)
    max_episodes = args.max_episodes
    max_dim_cli = args.max_dim
    # 新增：读取起始偏移
    start_entries = getattr(args, "start_entries", None)
    start_episodes = getattr(args, "start_episodes", None)

    (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_tasks,
        all_stats_data,
        total_frames,
    ) = select_episodes(source_folders, max_entries=None, max_episodes=max_episodes)

    write_meta_and_copy(
        source_folders=source_folders,
        output_folder=args.output,
        episode_mapping=episode_mapping,
        all_episodes=all_episodes,
        all_episodes_stats=all_episodes_stats,
        folder_dimensions=folder_dimensions,
        folder_task_mapping=folder_task_mapping,
        episode_to_frame_index=episode_to_frame_index,
        all_stats_data=all_stats_data,
        all_tasks=all_tasks,
        total_frames=total_frames,
        max_dim_cli=max_dim_cli,
        fps=fps,
    )


def mode_split(args: argparse.Namespace):
    """拆分模式：对单个数据集选择前 N 帧或前 N 个 episode 输出。"""
    input_folder = args.input
    fps = args.fps if args.fps is not None else get_info(input_folder).get("fps", 20)
    max_entries = args.max_entries
    max_episodes = args.max_episodes
    max_dim_cli = args.max_dim
    # 新增：读取起始偏移
    start_entries = getattr(args, "start_entries", None)
    start_episodes = getattr(args, "start_episodes", None)

    source_folders = [input_folder]

    (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_tasks,
        all_stats_data,
        total_frames,
    ) = select_episodes(
        source_folders,
        max_entries=max_entries,
        max_episodes=max_episodes,
        # 传入起始偏移参数
        start_entries=start_entries,
        start_episodes=start_episodes,
    )

    write_meta_and_copy(
        source_folders=source_folders,
        output_folder=args.output,
        episode_mapping=episode_mapping,
        all_episodes=all_episodes,
        all_episodes_stats=all_episodes_stats,
        folder_dimensions=folder_dimensions,
        folder_task_mapping=folder_task_mapping,
        episode_to_frame_index=episode_to_frame_index,
        all_stats_data=all_stats_data,
        all_tasks=all_tasks,
        total_frames=total_frames,
        max_dim_cli=max_dim_cli,
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser(description="数据集拆分与合并工具 - 修复版本")
    subparsers = parser.add_subparsers(dest="mode", help="操作模式")

    # 拆分模式
    split_parser = subparsers.add_parser("split", help="拆分数据集")
    split_parser.add_argument("--input", required=True, help="输入数据集路径")
    split_parser.add_argument("--output", required=True, help="输出数据集路径")
    split_parser.add_argument("--max_entries", type=int, help="最大帧数限制")
    split_parser.add_argument("--max_episodes", type=int, help="最大episode数量限制")
    split_parser.add_argument("--fps", type=int, help="帧率设置")
    split_parser.add_argument("--max_dim", type=int, help="最大维度设置")
    # 新增：起始偏移参数
    split_parser.add_argument("--start_entries", type=int, help="起始帧偏移（跳过前N帧）")
    split_parser.add_argument("--start_episodes", type=int, help="起始episode偏移（跳过前N个episode）")

    # 合并模式
    merge_parser = subparsers.add_parser("merge", help="合并数据集")
    merge_parser.add_argument("--sources", nargs="+", required=True, help="源数据集路径列表")
    merge_parser.add_argument("--output", required=True, help="输出数据集路径")
    merge_parser.add_argument("--max_episodes", type=int, help="最大episode数量限制")
    merge_parser.add_argument("--fps", type=int, help="帧率设置")
    merge_parser.add_argument("--max_dim", type=int, help="最大维度设置")
    # 新增：起始偏移参数
    merge_parser.add_argument("--start_entries", type=int, help="起始帧偏移（跳过前N帧）")
    merge_parser.add_argument("--start_episodes", type=int, help="起始episode偏移（跳过前N个episode）")
    args = parser.parse_args()

    if args.mode == "split":
        mode_split(args)
    elif args.mode == "merge":
        mode_merge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()