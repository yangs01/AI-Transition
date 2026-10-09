"""一键备份：timestamped zip 快照 + git 提交推送。

用法：
    python backup.py [目录]        # 默认当前目录
    双击运行亦可（备份所在目录）

做了什么：
  1. 把目录打成 backups/backup-YYYYMMDD-HHMMSS.zip
     （自动排除 .venv / __pycache__ / .git / backups 等）
  2. 如果是 git 仓库：git add -A → commit → push
"""
import os
import sys
import subprocess
import datetime
import zipfile

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "backups",
                ".mypy_cache", "node_modules", ".pytest_cache"}
EXCLUDE_EXTS = {".pyc"}


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def make_zip(src, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    # zip 快照不进 git
    gi = os.path.join(backup_dir, ".gitignore")
    if not os.path.exists(gi):
        open(gi, "w").write("*\n")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    zpath = os.path.join(backup_dir, f"backup-{stamp}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                if os.path.splitext(f)[1] in EXCLUDE_EXTS:
                    continue
                fp = os.path.join(dirpath, f)
                z.write(fp, os.path.relpath(fp, src))
    return zpath


def git_backup(src):
    code, _, _ = run(["git", "rev-parse", "--git-dir"], src)
    if code != 0:
        return "非 git 仓库，跳过推送（zip 快照已生成）"
    code, status, _ = run(["git", "status", "--porcelain"], src)
    if code != 0:
        return "git 状态读取失败"
    if status:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        run(["git", "add", "-A"], src)
        code, _, err = run(["git", "commit", "-m", f"backup {stamp}"], src)
        if code != 0:
            return f"commit 失败: {err}"
        print("已提交本次改动")
    else:
        print("工作区干净，无需提交")
    code, _, err = run(["git", "push"], src)
    if code != 0:
        return f"push 失败: {err}"
    return "git 推送完成"


def main():
    src = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    zpath = make_zip(src, os.path.join(src, "backups"))
    print(f"zip 快照: {zpath}")
    print(git_backup(src))


if __name__ == "__main__":
    main()
