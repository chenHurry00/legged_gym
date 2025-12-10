#!/usr/bin/env python3
"""
模型清理脚本
用于清理日志文件夹中的旧epoch模型，只保留最新的模型。
支持单选、多选或全选任务进行清理。
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def get_all_tasks(log_dir: str) -> List[str]:
    """获取logs目录下的所有任务"""
    if not os.path.exists(log_dir):
        print(f"错误：日志目录 '{log_dir}' 不存在")
        return []

    tasks = []
    for item in os.listdir(log_dir):
        task_path = os.path.join(log_dir, item)
        if os.path.isdir(task_path):
            tasks.append(item)

    return sorted(tasks)

def get_training_sessions(task_dir: str) -> List[str]:
    """获取任务目录下的所有训练会话"""
    if not os.path.exists(task_dir):
        return []

    sessions = []
    for item in os.listdir(task_dir):
        session_path = os.path.join(task_dir, item)
        if os.path.isdir(session_path):
            # 检查是否包含模型文件
            model_files = [f for f in os.listdir(session_path) if f.startswith('model_') and f.endswith('.pt')]
            if model_files:
                sessions.append(item)

    return sorted(sessions)

def get_model_files(session_dir: str) -> List[Tuple[int, str]]:
    """获取会话目录下的所有模型文件，返回(epoch数, 文件名)的列表"""
    if not os.path.exists(session_dir):
        return []

    model_files = []
    pattern = re.compile(r'model_(\d+)\.pt')

    for item in os.listdir(session_dir):
        match = pattern.match(item)
        if match:
            epoch = int(match.group(1))
            model_files.append((epoch, item))

    # 按epoch数排序
    return sorted(model_files, key=lambda x: x[0])

def cleanup_session_models(session_dir: str, dry_run: bool = True) -> List[str]:
    """清理会话中的旧模型，只保留最新的一个"""
    model_files = get_model_files(session_dir)

    if len(model_files) <= 1:
        return []  # 没有需要删除的文件

    # 获取最新的模型
    latest_epoch, latest_model = model_files[-1]

    # 准备删除的文件列表（除了最新的）
    files_to_delete = []
    for epoch, filename in model_files[:-1]:
        file_path = os.path.join(session_dir, filename)
        files_to_delete.append(file_path)

    # 如果不是dry run，执行删除
    if not dry_run:
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"  已删除: {os.path.basename(file_path)}")

    return [os.path.basename(f) for f in files_to_delete]

def preview_cleanup(tasks: List[str], log_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """预览将要清理的文件"""
    preview_info = {}

    for task in tasks:
        task_dir = os.path.join(log_dir, task)
        sessions = get_training_sessions(task_dir)

        preview_info[task] = {}
        for session in sessions:
            session_dir = os.path.join(task_dir, session)
            files_to_delete = cleanup_session_models(session_dir, dry_run=True)
            if files_to_delete:
                preview_info[task][session] = files_to_delete

    return preview_info

def calculate_space_saved(preview_info: Dict[str, Dict[str, List[str]]], log_dir: str) -> int:
    """计算将要释放的磁盘空间（字节）"""
    total_size = 0

    for task, sessions in preview_info.items():
        for session, files in sessions.items():
            session_dir = os.path.join(log_dir, task, session)
            for filename in files:
                file_path = os.path.join(session_dir, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)

    return total_size

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    parser = argparse.ArgumentParser(description='清理旧epoch模型，只保留最新的模型')
    parser.add_argument('--log-dir', default='logs', help='日志目录路径 (默认: logs)')
    parser.add_argument('--tasks', nargs='+', help='指定要清理的任务名称')
    parser.add_argument('--all', action='store_true', help='清理所有任务')
    parser.add_argument('--execute', action='store_true', help='执行清理（默认只预览）')
    parser.add_argument('--list', action='store_true', help='列出所有可用的任务')

    args = parser.parse_args()

    # 获取所有任务
    all_tasks = get_all_tasks(args.log_dir)
    if not all_tasks:
        print("没有找到任何任务")
        return

    # 如果只是列出任务
    if args.list:
        print("\n可用的任务列表:")
        print("-" * 40)
        for i, task in enumerate(all_tasks, 1):
            task_dir = os.path.join(args.log_dir, task)
            sessions = get_training_sessions(task_dir)
            print(f"{i:2d}. {task} ({len(sessions)} 个训练会话)")
        return

    # 确定要处理的任务
    selected_tasks = []

    if args.all:
        selected_tasks = all_tasks
        print(f"\n已选择所有 {len(selected_tasks)} 个任务")
    elif args.tasks:
        # 验证指定的任务是否存在
        for task in args.tasks:
            if task not in all_tasks:
                print(f"警告：任务 '{task}' 不存在，跳过")
            else:
                selected_tasks.append(task)
        if selected_tasks:
            print(f"\n已选择 {len(selected_tasks)} 个任务: {', '.join(selected_tasks)}")
    else:
        # 交互式选择
        print("\n可用的任务列表:")
        print("-" * 60)
        for i, task in enumerate(all_tasks, 1):
            task_dir = os.path.join(args.log_dir, task)
            sessions = get_training_sessions(task_dir)
            print(f"{i:2d}. {task:<30} ({len(sessions)} 个训练会话)")
        print("-" * 60)

        while True:
            try:
                selection = input("\n请选择任务 (输入数字，多个用逗号分隔，或输入 'all' 选择全部): ").strip()

                if selection.lower() == 'all':
                    selected_tasks = all_tasks
                    break

                # 解析输入
                indices = []
                for part in selection.split(','):
                    part = part.strip()
                    if '-' in part:  # 处理范围，如 1-3
                        start, end = map(int, part.split('-'))
                        indices.extend(range(start, end + 1))
                    else:
                        indices.append(int(part))

                # 验证索引
                selected_tasks = []
                for idx in indices:
                    if 1 <= idx <= len(all_tasks):
                        selected_tasks.append(all_tasks[idx - 1])
                    else:
                        print(f"索引 {idx} 超出范围，跳过")

                if selected_tasks:
                    break
                else:
                    print("请至少选择一个有效的任务")

            except ValueError:
                print("输入格式错误，请重新输入")

    if not selected_tasks:
        print("\n没有选择任何任务")
        return

    # 预览将要删除的文件
    print("\n正在分析要删除的文件...")
    preview_info = preview_cleanup(selected_tasks, args.log_dir)

    # 统计信息
    total_sessions = sum(len(sessions) for sessions in preview_info.values())
    total_files = sum(len(files) for sessions in preview_info.values() for files in sessions)
    space_saved = calculate_space_saved(preview_info, args.log_dir)

    print(f"\n预览结果:")
    print("=" * 60)
    print(f"任务数: {len(selected_tasks)}")
    print(f"训练会话数: {total_sessions}")
    print(f"将删除的模型文件数: {total_files}")
    print(f"将释放的磁盘空间: {format_size(space_saved)}")

    # 显示详细删除列表
    if preview_info:
        print("\n详细删除列表:")
        print("-" * 60)
        for task, sessions in preview_info.items():
            if sessions:
                print(f"\n任务: {task}")
                for session, files in sessions.items():
                    print(f"  会话: {session}")
                    for file in files:
                        print(f"    - {file}")

    if not args.execute:
        if total_files > 0:
            print("\n" + "=" * 60)
            print("这是预览模式，没有实际删除文件")
            print("如需执行删除，请使用 --execute 参数")
            print(f"或运行: python cleanup_models.py --tasks {' '.join(selected_tasks)} --execute")
        else:
            print("\n没有需要删除的文件")
        return

    # 执行删除前的最后确认
    print("\n" + "!" * 60)
    print("警告：您即将删除模型文件！")
    print("这些操作是不可逆的！")
    print("!" * 60)

    if total_files == 0:
        print("\n没有需要删除的文件")
        return

    confirm = input(f"\n确认要删除 {total_files} 个模型文件吗？(输入 'yes' 确认): ").strip()

    if confirm.lower() != 'yes':
        print("操作已取消")
        return

    # 执行删除
    print("\n开始删除文件...")
    deleted_count = 0

    for task, sessions in preview_info.items():
        for session, files in sessions.items():
            session_dir = os.path.join(args.log_dir, task, session)
            print(f"\n任务 {task} - 会话 {session}:")

            for filename in files:
                file_path = os.path.join(session_dir, filename)
                try:
                    os.remove(file_path)
                    print(f"  ✓ 已删除: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {filename} - {e}")

    print(f"\n删除完成！共删除 {deleted_count} 个文件")
    print(f"释放磁盘空间: {format_size(space_saved)}")

if __name__ == "__main__":
    main()