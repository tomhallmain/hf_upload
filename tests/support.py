"""Shared fixtures for the upload_to_hf tests.

Nothing here touches the network, your git config, or your Hugging Face token:
- git runs with a throwaway global config (GIT_CONFIG_GLOBAL) and no system config;
- huggingface_hub / hf_xet are replaced by fakes (in sys.modules for in-process tests,
  first on PYTHONPATH for subprocess tests) and HF_HUB_OFFLINE / HF_HOME are set as a
  second line of defence in case the real package were ever imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = SCRIPT_DIR / "upload_to_hf.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REPO_ID = "example-user/fake-test-repo"  # never contacted: git is redirected, hub is faked
HUB_URL = f"https://huggingface.co/{REPO_ID}"
TOKEN = "hf_test_token_not_real"


def tempdir():
    # git writes read-only object files, which Windows refuses to delete: ignore those.
    return tempfile.TemporaryDirectory(prefix="hfup-test-", ignore_cleanup_errors=True)


def isolated_env(root: Path, extra_git_config: str = "") -> dict:
    """Environment for git and the script, sandboxed under `root`."""
    gitconfig = root / "gitconfig"
    gitconfig.write_text(
        "[init]\n\tdefaultBranch = master\n"  # match `git init` defaults that produced master
        "[user]\n\tname = Test\n\temail = test@example.invalid\n"
        "[commit]\n\tgpgsign = false\n" + extra_git_config,
        encoding="utf-8",
    )
    env = dict(os.environ)
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH"):
        env.pop(var, None)
    env.update(
        GIT_CONFIG_GLOBAL=str(gitconfig),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        HF_HOME=str(root / "hf_home"),
        HF_HUB_OFFLINE="1",
        PYTHONIOENCODING="utf-8",  # the script prints emoji; Windows consoles may not be UTF-8
    )
    return env


def hub_redirect_config(bare_repo: Path) -> str:
    """Git config that sends clone/fetch/push for HUB_URL to a local bare repo. Remotes keep
    HUB_URL as their configured URL (`git config remote.origin.url`), which is what the
    script checks."""
    return f'[url "{bare_repo.resolve().as_uri()}"]\n\tinsteadOf = {HUB_URL}\n'


def git(repo: Path, *args: str, env: dict) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def make_repo(path: Path, env: dict, files: dict | None = None) -> list[str]:
    """git init `path` and commit `files` one commit per file. Returns commit ids, oldest first."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", env=env)
    commits = []
    for name, content in (files or {"README.md": "# test\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        git(path, "add", name, env=env)
        git(path, "commit", "-q", "-m", f"add {name}", env=env)
        commits.append(git(path, "rev-parse", "HEAD", env=env))
    return commits


class BareHub:
    """A local bare repo playing the Hub repo, reached through hub_redirect_config().

    `commit_folder` does what an API upload does on the real Hub: it makes one new commit on
    main from a folder's files (adds and updates only, honouring ignore patterns).
    Paths derive from `root`, so another process can attach with BareHub(root, env).
    """

    def __init__(self, root: Path, env: dict):
        self.root = Path(root)
        self.bare = self.root / "hub.git"
        self.work = self.root / "hub_work"  # the "server side" checkout used to make commits
        self.env = env

    def create(self, files: dict) -> "BareHub":
        git(self.root, "init", "-q", "--bare", str(self.bare), env=self.env)
        git(self.bare, "symbolic-ref", "HEAD", "refs/heads/main", env=self.env)
        make_repo(self.work, self.env, files)
        git(self.work, "branch", "-q", "-M", "main", env=self.env)
        git(self.work, "remote", "add", "origin", HUB_URL, env=self.env)
        git(self.work, "push", "-q", "origin", "main", env=self.env)
        return self

    def head(self) -> str:
        return git(self.bare, "rev-parse", "main", env=self.env)

    def commits(self) -> list[str]:
        """Commit ids on main, newest first (what HfApi.list_repo_commits returns)."""
        return git(self.bare, "rev-list", "main", env=self.env).split()

    def clone(self, dest: Path) -> None:
        git(dest.parent, "clone", "-q", HUB_URL, str(dest), env=self.env)

    def commit_folder(self, folder: Path, message: str, ignore_patterns=()) -> str:
        import fnmatch
        import shutil

        git(self.work, "pull", "-q", "--ff-only", "origin", "main", env=self.env)
        for src in Path(folder).rglob("*"):
            rel = src.relative_to(folder)
            if ".git" in rel.parts or any(
                fnmatch.fnmatch(part, pat) for part in rel.parts for pat in ignore_patterns
            ):
                continue
            if src.is_file():
                (self.work / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, self.work / rel)
        git(self.work, "add", "-A", env=self.env)
        if git(self.work, "status", "--porcelain", env=self.env):
            git(self.work, "commit", "-q", "-m", message, env=self.env)
            git(self.work, "push", "-q", "origin", "HEAD:main", env=self.env)
        return self.head()

    def push_change(self, name: str, content: str) -> str:
        """Someone else changes the Hub repo (e.g. a web edit)."""
        git(self.work, "pull", "-q", "--ff-only", "origin", "main", env=self.env)
        (self.work / name).write_text(content, encoding="utf-8")
        git(self.work, "add", name, env=self.env)
        git(self.work, "commit", "-q", "-m", f"edit {name}", env=self.env)
        git(self.work, "push", "-q", "origin", "HEAD:main", env=self.env)
        return self.head()


def fake_sha(n: int) -> str:
    """A well-formed commit id that no local repo contains."""
    return f"{n:040x}"


# ----- in-process fakes ---------------------------------------------------------------------

class FakeHub:
    """State and call log behind the fake huggingface_hub module."""

    def __init__(self, commits=None, commits_error=None, repo_missing=False, saved_token=None):
        self.commits = list(commits or [])
        self.commits_error = commits_error
        self.repo_missing = repo_missing
        self.saved_token = saved_token
        self.uploads: list[dict] = []
        # When set, uploads commit to this BareHub and commits are read from it.
        self.bare_hub: BareHub | None = None
        self.after_upload = None  # optional callable, run after each upload

    def _upload(self, **kwargs):
        self.uploads.append(kwargs)
        result = None
        if self.bare_hub is not None:
            oid = self.bare_hub.commit_folder(
                Path(kwargs["folder_path"]), kwargs["commit_message"], kwargs.get("ignore_patterns", ())
            )
            result = types.SimpleNamespace(oid=oid)  # like huggingface_hub's CommitInfo
        if self.after_upload:
            self.after_upload()
        return result

    def modules(self, with_xet: bool = True) -> dict:
        """sys.modules entries; use with mock.patch.dict(sys.modules, ...)."""
        hub = types.ModuleType("huggingface_hub")
        utils = types.ModuleType("huggingface_hub.utils")

        class RepositoryNotFoundError(Exception):
            pass

        state = self

        class Commit:
            def __init__(self, commit_id):
                self.commit_id = commit_id

        class HfApi:
            def repo_info(self, repo_id, token=None):
                if state.repo_missing:
                    raise RepositoryNotFoundError(repo_id)
                return object()

            def list_repo_commits(self, repo_id, token=None):
                if state.commits_error:
                    raise state.commits_error
                commits = state.bare_hub.commits() if state.bare_hub else state.commits
                return [Commit(c) for c in commits]

        hub.__version__ = "1.30.0"
        hub.HfApi = HfApi
        hub.get_token = lambda: state.saved_token
        hub.upload_folder = state._upload
        hub.utils = utils
        utils.RepositoryNotFoundError = RepositoryNotFoundError

        xet = types.ModuleType("hf_xet")
        xet.__version__ = "1.2.0"
        return {
            "huggingface_hub": hub,
            "huggingface_hub.utils": utils,
            "hf_xet": xet if with_xet else None,  # None makes `import hf_xet` raise ImportError
        }


# ----- subprocess fakes ---------------------------------------------------------------------

_FAKE_HUB_PACKAGE = '''
import json, os
__version__ = "1.30.0"

def _state():
    with open(os.environ["FAKE_HF_STATE"], encoding="utf-8") as f:
        return json.load(f)

def _log(event, **data):
    with open(os.environ["FAKE_HF_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, **data}) + "\\n")

def get_token():
    return _state().get("saved_token")

def _bare_hub():
    root = _state().get("bare_hub_root")
    if not root:
        return None
    from pathlib import Path
    from tests.support import BareHub  # the script's folder is on sys.path
    return BareHub(Path(root), dict(os.environ))

def upload_folder(**kwargs):
    _log("upload_folder", **kwargs)
    hub = _bare_hub()
    if hub is None:
        return None
    from pathlib import Path
    from types import SimpleNamespace
    oid = hub.commit_folder(Path(kwargs["folder_path"]), kwargs["commit_message"],
                            kwargs.get("ignore_patterns", ()))
    return SimpleNamespace(oid=oid)

class _Commit:
    def __init__(self, commit_id):
        self.commit_id = commit_id

class HfApi:
    def repo_info(self, repo_id, token=None):
        from huggingface_hub.utils import RepositoryNotFoundError
        if _state().get("repo_missing"):
            raise RepositoryNotFoundError(repo_id)
        return object()

    def list_repo_commits(self, repo_id, token=None):
        state = _state()
        if state.get("commits_error"):
            raise ConnectionError(state["commits_error"])
        hub = _bare_hub()
        commits = hub.commits() if hub else state.get("commits", [])
        return [_Commit(c) for c in commits]
'''


class FakeHubProcess:
    """A fake huggingface_hub (and hf_xet) package on disk, for running the script as a CLI."""

    def __init__(self, root: Path, with_xet: bool = True):
        self.root = root
        self.site = root / "fake_site"
        self.state_file = root / "fake_hf_state.json"
        self.log_file = root / "fake_hf_log.jsonl"
        pkg = self.site / "huggingface_hub"
        (pkg / "utils").mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(textwrap.dedent(_FAKE_HUB_PACKAGE), encoding="utf-8")
        (pkg / "utils" / "__init__.py").write_text(
            "class RepositoryNotFoundError(Exception):\n    pass\n", encoding="utf-8"
        )
        xet = self.site / "hf_xet"
        xet.mkdir(exist_ok=True)
        # Without xet, shadow any real hf_xet with a package that refuses to import.
        (xet / "__init__.py").write_text(
            "__version__ = '1.2.0'\n" if with_xet else "raise ImportError('hf_xet hidden by test')\n",
            encoding="utf-8",
        )
        self.set_state()

    def set_state(self, **state):
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def env(self, base_env: dict) -> dict:
        env = dict(base_env)
        env.update(
            PYTHONPATH=os.pathsep.join([str(self.site), str(SCRIPT_DIR)]),
            FAKE_HF_STATE=str(self.state_file),
            FAKE_HF_LOG=str(self.log_file),
        )
        return env

    def calls(self, event: str | None = None) -> list[dict]:
        if not self.log_file.exists():
            return []
        rows = [json.loads(line) for line in self.log_file.read_text(encoding="utf-8").splitlines()]
        return [r for r in rows if event is None or r["event"] == event]


def run_script(args: list[str], env: dict, stdin: str = "", cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        timeout=120,
    )
