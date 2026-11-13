"""End-to-end tests: run upload_to_hf.py as a subprocess, the way it is used.

A fake huggingface_hub package is first on PYTHONPATH and records every upload_folder call;
git clone/fetch/push go to a local bare repo through insteadOf. No network, no real token.
Run from the hf_upload folder:  python -m unittest -v   (or: python -m pytest)
"""

from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path

from tests.support import (
    HUB_URL,
    REPO_ID,
    TOKEN,
    BareHub,
    FakeHubProcess,
    fake_sha,
    git,
    hub_redirect_config,
    isolated_env,
    make_repo,
    run_script,
    tempdir,
)


class CliTestCase(unittest.TestCase):
    with_xet = True

    def setUp(self):
        tmp = tempdir()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "repo"
        self.bare = self.root / "hub.git"
        self.git_env = isolated_env(self.root, hub_redirect_config(self.bare))
        self.fake = FakeHubProcess(self.root, with_xet=self.with_xet)
        self.env = self.fake.env(self.git_env)
        self.env["HF_TOKEN"] = TOKEN

    def git(self, *args, repo=None):
        return git(repo or self.repo, *args, env=self.git_env)

    def run_cli(self, *args, stdin="", cwd=None):
        result = run_script([REPO_ID, *args], self.env, stdin=stdin, cwd=cwd)
        self.output = result.stdout + result.stderr
        return result

    def assertExit(self, result, code):
        self.assertEqual(result.returncode, code, f"exit {result.returncode}, output:\n{self.output}")

    def uploads(self):
        return self.fake.calls("upload_folder")


class HubUploadCliTests(CliTestCase):
    def setUp(self):
        super().setUp()
        self.commits = make_repo(self.repo, self.git_env, {"README.md": "# m\n", "predict.py": "x = 1\n"})

    def test_upload_update_to_existing_repo(self):
        self.fake.set_state(commits=[self.commits[-1], fake_sha(1)])  # local has Hub's latest
        result = self.run_cli("-p", str(self.repo), "-m", "update files", "--delete-missing")
        self.assertExit(result, 0)
        (call,) = self.uploads()
        self.assertEqual(call["repo_id"], REPO_ID)
        self.assertEqual(call["token"], TOKEN)
        self.assertEqual(call["commit_message"], "update files")
        self.assertEqual(call["delete_patterns"], ["*"])
        self.assertNotIn("use_xet", call)
        self.assertEqual(self.git("config", "--get", "remote.origin.url"), HUB_URL)
        self.assertNotIn("xet", self.git("config", "--list", "--local"))
        self.assertFalse((self.repo / ".gitignore").exists())

    def test_auto_discovers_repo_from_subfolder(self):
        sub = self.repo / "model"
        sub.mkdir()
        result = self.run_cli(cwd=sub)
        self.assertExit(result, 0)
        self.assertEqual(Path(self.uploads()[0]["folder_path"]), self.repo.resolve())

    def test_unrelated_local_git_blocks_upload(self):
        self.fake.set_state(commits=[fake_sha(3), fake_sha(2), fake_sha(1)])
        result = self.run_cli("-p", str(self.repo))
        self.assertExit(result, 1)
        self.assertEqual(self.uploads(), [])
        self.assertIn("shares no history", self.output)
        self.assertIn("has 3 commits", self.output)
        self.assertTrue((self.repo / ".git").is_dir())  # never deleted
        self.assertEqual(self.git("remote"), "")  # nothing changed locally

    def test_ignore_local_git_overrides(self):
        self.fake.set_state(commits=[fake_sha(2), fake_sha(1)])
        result = self.run_cli("-p", str(self.repo), "--ignore-local-git")
        self.assertExit(result, 0)
        self.assertEqual(len(self.uploads()), 1)

    def test_behind_hub_blocks_upload(self):
        self.fake.set_state(commits=[fake_sha(5), *reversed(self.commits)])
        result = self.run_cli("-p", str(self.repo))
        self.assertExit(result, 1)
        self.assertEqual(self.uploads(), [])
        self.assertIn("git pull origin main", self.output)

    def test_fresh_hub_repo_allows_first_upload(self):
        self.fake.set_state(commits=[fake_sha(1)])
        result = self.run_cli("-p", str(self.repo))
        self.assertExit(result, 0)
        self.assertIn("fresh repo", self.output)

    def test_hub_repo_missing(self):
        self.fake.set_state(repo_missing=True)
        result = self.run_cli("-p", str(self.repo), "--method", "hub")
        self.assertExit(result, 1)
        self.assertEqual(self.uploads(), [])

    def test_folder_without_git(self):
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "file.txt").write_text("x", encoding="utf-8")
        # Answers "Continue anyway?" (not a git repository) twice.
        result = self.run_cli("-p", str(plain), stdin="y\ny\n")
        self.assertExit(result, 0)
        self.assertEqual(Path(self.uploads()[0]["folder_path"]), plain.resolve())
        self.assertFalse((plain / ".git").exists())


class SetRemoteOnlyCliTests(CliTestCase):
    def test_sets_remote_and_uploads_nothing(self):
        make_repo(self.repo, self.git_env)
        result = self.run_cli("-p", str(self.repo), "--set-remote-only")
        self.assertExit(result, 0)
        self.assertEqual(self.git("config", "--get", "remote.origin.url"), HUB_URL)
        self.assertEqual(self.fake.calls(), [])

    def test_repairs_token_in_url(self):
        make_repo(self.repo, self.git_env)
        self.git("remote", "add", "origin", f"https://{TOKEN}@huggingface.co/{REPO_ID}")
        result = self.run_cli("-p", str(self.repo), "--set-remote-only")
        self.assertExit(result, 0)
        self.assertEqual(self.git("config", "--get", "remote.origin.url"), HUB_URL)
        self.assertNotIn(TOKEN, self.output)

    def test_not_a_git_repo(self):
        plain = self.root / "plain"
        plain.mkdir()
        result = self.run_cli("-p", str(plain), "--set-remote-only")
        self.assertExit(result, 1)
        self.assertIn("not a git repository", self.output)


class NoXetCliTests(CliTestCase):
    with_xet = False  # shadows any real hf_xet with one that fails to import

    def setUp(self):
        super().setUp()
        make_repo(self.repo, self.git_env)

    def test_use_xet_requires_hf_xet(self):
        result = self.run_cli("-p", str(self.repo), "--method", "hub", "--use-xet")
        self.assertExit(result, 1)
        self.assertIn("pip install -U hf-xet", self.output)
        self.assertEqual(self.uploads(), [])

    def test_upload_works_without_hf_xet(self):
        result = self.run_cli("-p", str(self.repo), "--method", "hub")
        self.assertExit(result, 0)
        self.assertEqual(len(self.uploads()), 1)


class SyncCliTests(CliTestCase):
    """Repeated API uploads from a clone: each run leaves local git on the new Hub commit,
    so the next run passes the history check without a manual pull."""

    def setUp(self):
        super().setUp()
        self.bare_hub = BareHub(self.root, self.git_env).create({"README.md": "v1\n"})
        self.bare_hub.clone(self.repo)
        self.fake.set_state(bare_hub_root=str(self.root))

    def edit_and_upload(self, text, *extra):
        (self.repo / "README.md").write_text(text, encoding="utf-8")
        return self.run_cli("-p", str(self.repo), "-m", f"README {text.strip()}", *extra)

    def test_consecutive_uploads(self):
        for text in ("v2\n", "v3\n"):
            with self.subTest(text=text):
                result = self.edit_and_upload(text)
                self.assertExit(result, 0)
                self.assertNotIn("is behind", self.output)
                self.assertEqual(self.git("rev-parse", "HEAD"), self.bare_hub.head())
                self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(
            self.git("log", "--format=%s").splitlines()[:2], ["README v3", "README v2"]
        )

    def test_no_sync_git_then_next_upload_is_blocked(self):
        self.assertExit(self.edit_and_upload("v2\n", "--no-sync-git"), 0)
        self.assertNotEqual(self.git("rev-parse", "HEAD"), self.bare_hub.head())
        result = self.edit_and_upload("v3\n")
        self.assertExit(result, 1)
        self.assertIn("is behind", self.output)
        self.assertEqual(len(self.uploads()), 1)


class GitMethodCliTests(CliTestCase):
    def setUp(self):
        super().setUp()
        git(self.root, "init", "-q", "--bare", str(self.bare), env=self.git_env)

    def bare_branches(self):
        return git(self.bare, "branch", "--format=%(refname:short)", env=self.git_env).split()

    def test_git_push_to_main(self):
        make_repo(self.repo, self.git_env)
        result = self.run_cli("-p", str(self.repo), "--method", "git")
        self.assertExit(result, 0)
        self.assertEqual(self.bare_branches(), ["main"])
        self.assertEqual(self.uploads(), [])

    @unittest.skipIf(os.name == "nt", "uses a POSIX shell script as a stand-in for git-lfs")
    def test_large_untracked_file_menu_abort(self):
        make_repo(self.repo, self.git_env, {"model.safetensors": b"\0" * (2 * 1024 * 1024)})
        stub_dir = self.root / "bin"
        stub_dir.mkdir()
        stub = stub_dir / "git-lfs"
        stub.write_text('#!/bin/sh\n[ "$1" = version ] && echo git-lfs/3.5.0\nexit 0\n', encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.env["PATH"] = f"{stub_dir}{os.pathsep}{self.env['PATH']}"
        result = self.run_cli("-p", str(self.repo), "--method", "git", "--threshold", "1", stdin="3\n")
        self.assertExit(result, 1)
        self.assertIn("model.safetensors", self.output)
        self.assertIn("Upload aborted", self.output)
        self.assertNotIn("Use Xet", self.output)
        self.assertEqual(self.bare_branches(), [])


if __name__ == "__main__":
    unittest.main()
