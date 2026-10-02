#!/usr/bin/env python3
"""通过 GitHub Git Data API 推送本地 commit（绕过 github.com:443 与 SSH）。

适用场景：api.github.com 可达但 github.com:443 / SSH 22 被阻断。
流程：contents API 初始化仓库（空仓库对 git/blobs 返回 409）→
逐文件建 blob → 建全新 tree → 建 commit（parent=init）→
强制移动 main 到完整 commit（init 成为孤儿提交，最终历史只有一个 commit）。
"""
import base64
import json
import subprocess
import sys

REPO = "Hachiwa0/dot-agent"
REPO_DIR = "/home/hachiware/dot-agent"


def gh_api(endpoint: str, payload: dict | None = None, method: str | None = None) -> dict:
    method = method or ("POST" if payload is not None else "GET")
    cmd = ["gh", "api", "--method", method, f"repos/{REPO}/{endpoint}"]
    if payload is not None:
        cmd += ["--input", "-"]
    proc = subprocess.run(
        cmd, input=json.dumps(payload).encode() if payload is not None else b"",
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        print(f"gh api {method} {endpoint} 失败:\n{proc.stderr.decode()[:500]}", file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)


def git_out(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", REPO_DIR, *args], capture_output=True, text=True, check=True
    ).stdout


# 0. 探测远端 main：已存在 → 作为 parent 快进；不存在 → contents 初始化
init_sha = None
parent_sha = None
probe = subprocess.run(
    ["gh", "api", f"repos/{REPO}/git/ref/heads/main"], capture_output=True, text=True
)
if probe.returncode == 0:
    parent_sha = json.loads(probe.stdout)["object"]["sha"]
    print(f"远端 main 存在: {parent_sha[:8]}（新 commit 将快进追加）")
else:
    init = gh_api(
        "contents/.init",
        {"message": "init", "content": base64.b64encode(b"init\n").decode()},
        method="PUT",
    )
    init_sha = init["commit"]["sha"]
    print(f"仓库初始化 ✓（init commit {init_sha[:8]}，最终将成为孤儿提交）")

# 1. 本地 commit 元数据
commit_sha = git_out("rev-parse", "HEAD").strip()
message = git_out("log", "-1", "--format=%B")
author_name = git_out("log", "-1", "--format=%an").strip()
author_email = git_out("log", "-1", "--format=%ae").strip()
author_date = git_out("log", "-1", "--format=%aI").strip()
files = [f for f in git_out("ls-files").splitlines() if f]
print(f"本地 commit {commit_sha[:8]}，共 {len(files)} 个文件")

# 2. 逐文件创建 blob
tree_items = []
for path in files:
    with open(f"{REPO_DIR}/{path}", encoding="utf-8") as fh:
        content = fh.read()
    blob = gh_api("git/blobs", {"content": content, "encoding": "utf-8"})
    tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
print(f"  blobs ✓ ×{len(tree_items)}")

# 3. 全新 tree（不含 .init，仅本地文件）+ commit（parent 接续远端历史）
tree = gh_api("git/trees", {"tree": tree_items})
parents = [p for p in (init_sha, parent_sha) if p]
commit = gh_api(
    "git/commits",
    {
        "message": message,
        "tree": tree["sha"],
        "parents": parents,
        "author": {"name": author_name, "email": author_email, "date": author_date},
    },
)
print(f"远端 commit: {commit['sha']}")

# 4. 移动 main 到完整 commit
if init_sha or parent_sha:
    gh_api("git/refs/heads/main", {"sha": commit["sha"], "force": bool(init_sha)}, method="PATCH")
else:
    gh_api("git/refs", {"ref": "refs/heads/main", "sha": commit["sha"]})
print(f"已推送 main → https://github.com/{REPO}")
