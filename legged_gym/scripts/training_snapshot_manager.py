#!/usr/bin/env python3
"""
训练快照管理器 - Git stash版本
在执行训练脚本时自动创建整个仓库的当前状态到快照分支
使用git stash确保不影响当前状态
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path
import argparse


class TrainingSnapshotManager:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path).resolve()
        self.original_branch = None
        self.stash_commit = None

    def run_command(self, cmd, capture_output=True, check=True):
        """安全执行shell命令"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=capture_output,
                text=True,
                check=check,
                cwd=self.repo_path
            )
            return result.stdout.strip() if capture_output else ""
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败: {cmd}")
            if e.stdout:
                print(f"输出: {e.stdout}")
            if e.stderr:
                print(f"错误: {e.stderr}")
            return None if capture_output else False

    def get_current_git_state(self):
        """获取当前Git状态（完全不修改任何内容）"""
        # 获取当前分支
        self.original_branch = self.run_command("git branch --show-current")
        if not self.original_branch:
            self.original_branch = "HEAD"

        # 获取当前状态信息（只读操作）
        status_output = self.run_command("git status --porcelain")
        has_changes = bool(status_output and status_output.strip())

        # 获取所有已修改文件的状态
        staged_files = []
        modified_files = []
        untracked_files = []

        if status_output:
            for line in status_output.split('\n'):
                if not line.strip():
                    continue

                status_code = line[:2]
                file_path = line[3:]

                if status_code[0] in ['A', 'M', 'D', 'R', 'C']:
                    staged_files.append(file_path)
                if status_code[1] in ['M', 'D']:
                    modified_files.append(file_path)
                if status_code == '??':
                    untracked_files.append(file_path)

        return {
            "branch": self.original_branch,
            "has_changes": has_changes,
            "staged_files": staged_files,
            "modified_files": modified_files,
            "untracked_files": untracked_files,
            "status_output": status_output
        }

    def create_stash_based_snapshot(self, task_name=None):
        """创建基于stash的绝对安全快照"""
        print("🚀 开始创建基于stash的快照...")
        print("🛡️ 使用git stash，不影响当前状态")

        # 1. 获取当前Git状态
        git_state = self.get_current_git_state()
        print(f"📋 当前分支: {git_state['branch']}")
        print(f"📝 仓库状态: {'有修改' if git_state['has_changes'] else '干净'}")

        try:
            # 2. 生成时间戳和分支名
            timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            if not task_name:
                task_name = self.infer_task_name()

            snapshot_branch = f"train/{timestamp}_{task_name}"

            print(f"🌿 创建快照分支: {snapshot_branch}")

            # 3. 获取当前HEAD的commit hash
            current_commit = self.run_command("git rev-parse HEAD")
            if not current_commit:
                print("❌ 无法获取当前commit hash")
                return False, None

            # 4. 如果有修改，创建stash
            if git_state['has_changes']:
                print("💾 保存当前修改到stash...")

                # 创建带时间戳的stash消息
                stash_msg = f"Auto-stash for training snapshot - {timestamp} - {task_name}"

                # 使用git stash push保存所有修改（包括未跟踪文件）
                if self.run_command(f'git stash push -m "{stash_msg}" --include-untracked') is None:
                    print("❌ 创建stash失败")
                    return False, None

                # 获取stash的commit hash
                stash_list = self.run_command("git stash list | head -1")
                if stash_list:
                    # 提取stash的commit hash
                    self.stash_commit = self.run_command("git rev-parse stash@{0}")
                    print(f"✅ Stash已创建: {self.stash_commit[:8]}")

            # 5. 基于当前状态创建快照分支
            if git_state['has_changes'] and self.stash_commit:
                # 如果有stash，基于stash创建分支
                if self.run_command(f"git checkout -b {snapshot_branch} {self.stash_commit}") is None:
                    print("❌ 创建快照分支失败")
                    return False, None
            else:
                # 如果没有修改，基于当前HEAD创建分支
                if self.run_command(f"git checkout -b {snapshot_branch} {current_commit}") is None:
                    print("❌ 创建快照分支失败")
                    return False, None

            # 6. 创建快照提交
            commit_msg = f"Training snapshot: {timestamp}_{task_name}\n\n"
            commit_msg += f"Task: {task_name}\n"
            commit_msg += f"Original branch: {git_state['branch']}\n"
            commit_msg += f"Original commit: {current_commit}\n"
            commit_msg += f"Created at: {datetime.datetime.now().isoformat()}\n"

            if git_state['has_changes']:
                commit_msg += f"\nChanges summary:\n"
                commit_msg += f"- Staged files: {len(git_state['staged_files'])}\n"
                commit_msg += f"- Modified files: {len(git_state['modified_files'])}\n"
                commit_msg += f"- Untracked files: {len(git_state['untracked_files'])}\n"
                commit_msg += f"Stash commit: {self.stash_commit}\n"

                if git_state['staged_files']:
                    commit_msg += f"\nStaged files:\n"
                    for file in git_state['staged_files']:
                        commit_msg += f"  - {file}\n"

                if git_state['modified_files']:
                    commit_msg += f"\nModified files:\n"
                    for file in git_state['modified_files']:
                        commit_msg += f"  - {file}\n"

                if git_state['untracked_files']:
                    commit_msg += f"\nUntracked files:\n"
                    for file in git_state['untracked_files'][:10]:  # 只显示前10个
                        commit_msg += f"  - {file}\n"
                    if len(git_state['untracked_files']) > 10:
                        commit_msg += f"  ... and {len(git_state['untracked_files']) - 10} more\n"
            else:
                commit_msg += f"Status: Clean working directory - no changes"

            # 如果分支已经有内容（来自stash），创建amend提交
            if git_state['has_changes'] and self.stash_commit:
                self.run_command(f'git commit --amend -m "{commit_msg}"')
            else:
                # 创建空提交
                self.run_command(f'git commit --allow-empty -m "{commit_msg}"')

            print("✅ 快照提交成功")

            # 7. 收集快照信息
            snapshot_info = {
                "timestamp": timestamp,
                "task_name": task_name,
                "branch_name": snapshot_branch,
                "created_at": datetime.datetime.now().isoformat(),
                "original_branch": git_state["branch"],
                "original_commit": current_commit,
                "stash_commit": self.stash_commit,
                "repository_state": {
                    "has_changes": git_state["has_changes"],
                    "staged_files_count": len(git_state["staged_files"]),
                    "modified_files_count": len(git_state["modified_files"]),
                    "untracked_files_count": len(git_state["untracked_files"])
                }
            }

            print(f"🎉 基于stash的安全快照创建完成！")
            print(f"📍 分支: {snapshot_branch}")
            print(f"🛡️ 当前工作状态未受影响")

            return True, snapshot_info

        except Exception as e:
            print(f"❌ 创建快照时出错: {e}")
            return False, None

        finally:
            # 8. 确保切换回原分支
            if self.original_branch:
                print(f"🔄 切换回原分支: {self.original_branch}")
                self.run_command(f"git checkout {self.original_branch}")

            # 9. 恢复stash（如果创建了stash）
            if self.stash_commit:
                print("🔄 恢复stash...")
                if self.run_command("git stash pop") is None:
                    print("⚠️ 恢复stash失败，请手动恢复")
                else:
                    print("✅ Stash已恢复，当前状态完全恢复")

    def infer_task_name(self):
        """推断任务名称"""
        # 从分支名推断
        if self.original_branch:
            branch_lower = self.original_branch.lower()
            if 'go2' in branch_lower:
                return 'Go2'
            elif 'rough' in branch_lower:
                return 'rough'
            elif 'balance' in branch_lower:
                return 'balance'
            elif 'anymal' in branch_lower:
                return 'anymal'
            elif 'bio' in branch_lower:
                return 'bio'

        # 从环境变量推断
        task_env = os.environ.get('TRAINING_TASK')
        if task_env:
            return task_env

        # 从最近的提交推断
        last_commit = self.run_command("git log -1 --pretty=format:'%s'")
        if last_commit:
            commit_lower = last_commit.lower()
            if 'go2' in commit_lower:
                return 'Go2'
            elif 'rough' in commit_lower:
                return 'rough'
            elif 'balance' in commit_lower:
                return 'balance'

        return 'general'


def create_stash_based_snapshot(task_name=None):
    """便捷函数：创建基于stash的绝对安全仓库快照"""
    manager = TrainingSnapshotManager()
    success, snapshot_info = manager.create_stash_based_snapshot(task_name=task_name)
    return success, snapshot_info


if __name__ == "__main__":
    # 命令行使用
    parser = argparse.ArgumentParser(description="创建基于stash的绝对安全训练快照")
    parser.add_argument("--task", "-t", help="任务名称")
    parser.add_argument("--dry-run", action="store_true", help="仅测试，不实际创建快照")

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 干运行模式：仅测试快照创建流程")
        manager = TrainingSnapshotManager()
        git_state = manager.get_current_git_state()
        print(f"当前分支: {git_state['branch']}")
        print(f"有修改: {git_state['has_changes']}")
        print(f"暂存文件数: {len(git_state['staged_files'])}")
        print(f"修改文件数: {len(git_state['modified_files'])}")
        print(f"未跟踪文件数: {len(git_state['untracked_files'])}")
    else:
        success, snapshot_info = create_stash_based_snapshot(args.task)
        if success:
            print(f"✅ 快照创建成功: {snapshot_info['branch_name']}")
        else:
            print("❌ 快照创建失败")
            sys.exit(1)
