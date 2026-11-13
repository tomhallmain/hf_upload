"""In-process tests for HFUploadManager and helpers in upload_to_hf.py.

Uses real git on temporary repos and a fake huggingface_hub injected into sys.modules.
Run from the hf_upload folder:  python -m unittest -v   (or: python -m pytest)
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests.support import (
    HUB_URL,
    REPO_ID,
    TOKEN,
    FakeHub,
    fake_sha,
    git,
    isolated_env,
    make_repo,
    push_redirect_config,
    tempdir,
)

import upload_to_hf as up


class ScriptTestCase(unittest.TestCase):
    """Temp dir, isolated git/HF environment, and a fake huggingface_hub for each test."""

    hub_kwargs: dict = {}

    def setUp(self):
        tmp = tempdir()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "repo"
        self.env = isolated_env(self.root, self.extra_git_config())
        env_patch = mock.patch.dict(os.environ, self.env, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.hub = FakeHub(**self.hub_kwargs)
        modules_patch = mock.patch.dict(sys.modules, self.hub.modules())
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def extra_git_config(self) -> str:
        return ""

    def manager(self, token=TOKEN) -> up.HFUploadManager:
        return up.HFUploadManager(self.repo, REPO_ID, token)

    def call(self, fn, *args, stdin: str = "", **kwargs):
        """Run fn with stdout captured and input() fed from `stdin`. Returns (result, output)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("sys.stdin", io.StringIO(stdin)):
            result = fn(*args, **kwargs)
        return result, out.getvalue()

    def git(self, *args) -> str:
        return git(self.repo, *args, env=self.env)

    def remote_url(self):
        return self.git("config", "--get", "remote.origin.url")


class TokenResolutionTests(ScriptTestCase):
    def setUp(self):
        super().setUp()
        self.repo.mkdir()

    def test_explicit_token_wins(self):
        os.environ["HF_TOKEN"] = "from-env"
        self.assertEqual(self.manager(token="explicit").token, "explicit")

    def test_hf_token_env(self):
        os.environ["HF_TOKEN"] = "from-hf-token"
        os.environ["HUGGINGFACE_TOKEN"] = "from-legacy"
        self.assertEqual(self.manager(token=None).token, "from-hf-token")

    def test_legacy_huggingface_token_env(self):
        os.environ["HUGGINGFACE_TOKEN"] = "from-legacy"
        self.assertEqual(self.manager(token=None).token, "from-legacy")

    def test_saved_login_token(self):
        self.hub.saved_token = "from-hf-auth-login"
        self.assertEqual(self.manager(token=None).token, "from-hf-auth-login")

    def test_no_token_and_no_huggingface_hub(self):
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertIsNone(self.manager(token=None).token)

    def test_missing_repo_path_raises(self):
        with self.assertRaises(ValueError):
            up.HFUploadManager(self.root / "does-not-exist", REPO_ID)


class EnsureRemoteTests(ScriptTestCase):
    def setUp(self):
        super().setUp()
        make_repo(self.repo, self.env)

    def test_adds_missing_remote_without_token(self):
        ok, out = self.call(self.manager().ensure_remote)
        self.assertTrue(ok)
        self.assertEqual(self.remote_url(), HUB_URL)
        self.assertNotIn(TOKEN, self.remote_url())
        self.assertIn("Remote added", out)

    def test_idempotent(self):
        self.call(self.manager().ensure_remote)
        ok, out = self.call(self.manager().ensure_remote)
        self.assertTrue(ok)
        self.assertIn("already set", out)
        self.assertEqual(self.git("remote").split(), ["origin"])

    def test_strips_embedded_token(self):
        self.git("remote", "add", "origin", f"https://hf_SECRET@huggingface.co/{REPO_ID}.git")
        ok, out = self.call(self.manager().ensure_remote)
        self.assertTrue(ok)
        self.assertEqual(self.remote_url(), HUB_URL)
        self.assertNotIn("hf_SECRET", out)

    def test_accepts_git_suffix_and_trailing_slash(self):
        for url in (f"{HUB_URL}.git", f"{HUB_URL}/"):
            with self.subTest(url=url):
                if self.git("remote"):
                    self.git("remote", "remove", "origin")
                self.git("remote", "add", "origin", url)
                ok, _ = self.call(self.manager().ensure_remote)
                self.assertTrue(ok)
                self.assertEqual(self.remote_url(), url)  # equivalent: left as is

    def test_leaves_foreign_remote_alone(self):
        self.git("remote", "add", "origin", "https://github.com/someone/else.git")
        ok, out = self.call(self.manager().ensure_remote)
        self.assertFalse(ok)
        self.assertEqual(self.remote_url(), "https://github.com/someone/else.git")
        self.assertIn("leaving it unchanged", out)


class LocalGitVsHubTests(ScriptTestCase):
    """check_local_git_matches_hub: the guard against uploading from an unrelated/stale .git."""

    def setUp(self):
        super().setUp()
        self.c1, self.c2 = make_repo(self.repo, self.env, {"a.txt": "a\n", "b.txt": "b\n"})

    def check(self, method="hub"):
        head_before = self.git("rev-parse", "HEAD")
        ok, out = self.call(self.manager().check_local_git_matches_hub, method)
        # Never modifies the local repository.
        self.assertEqual(self.git("rev-parse", "HEAD"), head_before)
        self.assertEqual(self.git("remote"), "")
        return ok, out

    def test_up_to_date(self):
        self.hub.commits = [self.c2, self.c1]
        self.assertTrue(self.check()[0])

    def test_local_ahead(self):
        self.hub.commits = [self.c1]
        self.assertTrue(self.check()[0])

    def test_behind_fails_with_pull_instructions(self):
        self.hub.commits = [fake_sha(9), self.c2, self.c1]
        ok, out = self.check()
        self.assertFalse(ok)
        self.assertIn("is behind", out)
        self.assertIn("git pull origin main", out)

    def test_unrelated_with_hub_history_fails_with_both_workflows(self):
        self.hub.commits = [fake_sha(2), fake_sha(1)]
        for method in ("hub", "git"):
            with self.subTest(method=method):
                ok, out = self.check(method)
                self.assertFalse(ok)
                self.assertIn("shares no history", out)
                self.assertIn("has 2 commits", out)
                self.assertIn(f"git clone {HUB_URL}", out)
                self.assertIn("--ignore-local-git", out)
        self.assertTrue((self.repo / ".git").is_dir())

    def test_unrelated_fresh_repo_allowed_for_hub_method(self):
        self.hub.commits = [fake_sha(1)]
        ok, out = self.check("hub")
        self.assertTrue(ok)
        self.assertIn("fresh repo", out)

    def test_unrelated_fresh_repo_fails_for_git_method(self):
        self.hub.commits = [fake_sha(1)]
        ok, out = self.check("git")
        self.assertFalse(ok)
        self.assertIn("already exists", out)

    def test_hub_unreachable_skips_check(self):
        self.hub.commits_error = ConnectionError("offline")
        ok, out = self.check()
        self.assertTrue(ok)
        self.assertIn("skipping the history check", out)

    def test_no_huggingface_hub_skips_check(self):
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            ok, out = self.check()
        self.assertTrue(ok)
        self.assertIn("skipping", out)

    def test_folder_without_own_git_is_not_checked(self):
        # A subfolder of a git repo: rev-parse succeeds, but the history isn't this folder's.
        sub = self.repo / "sub"
        sub.mkdir()
        self.hub.commits = [fake_sha(2), fake_sha(1)]
        manager = up.HFUploadManager(sub, REPO_ID, TOKEN)
        ok, _ = self.call(manager.check_local_git_matches_hub, "hub")
        self.assertTrue(ok)


class HubUploadTests(ScriptTestCase):
    def setUp(self):
        super().setUp()
        make_repo(self.repo, self.env)

    def test_upload_kwargs(self):
        ok, out = self.call(self.manager().upload_with_huggingface_hub, "my message")
        self.assertTrue(ok, out)
        (kwargs,) = self.hub.uploads
        self.assertEqual(kwargs["repo_id"], REPO_ID)
        self.assertEqual(kwargs["token"], TOKEN)
        self.assertEqual(Path(kwargs["folder_path"]), self.repo.resolve())
        self.assertEqual(kwargs["commit_message"], "my message")
        self.assertIn(".git", kwargs["ignore_patterns"])
        self.assertNotIn("use_xet", kwargs)  # not an upload_folder parameter
        self.assertNotIn("delete_patterns", kwargs)

    def test_default_commit_message(self):
        self.call(self.manager().upload_with_huggingface_hub)
        self.assertEqual(self.hub.uploads[0]["commit_message"], "Upload from script")

    def test_delete_missing(self):
        manager = self.manager()
        manager.delete_missing = True
        self.call(manager.upload_with_huggingface_hub)
        self.assertEqual(self.hub.uploads[0]["delete_patterns"], ["*"])

    def test_sets_remote_after_upload(self):
        self.call(self.manager().upload_with_huggingface_hub)
        self.assertEqual(self.remote_url(), HUB_URL)

    def test_writes_no_xet_leftovers(self):
        self.call(self.manager().upload_with_huggingface_hub)
        self.assertFalse((self.repo / ".gitignore").exists())
        self.assertFalse((self.repo / ".xet").exists())
        result = self.git("config", "--list", "--local")
        self.assertNotIn("xet", result)

    def test_works_without_hf_xet(self):
        with mock.patch.dict(sys.modules, {"hf_xet": None}):
            ok, out = self.call(self.manager().upload_with_huggingface_hub)
        self.assertTrue(ok)
        self.assertIn("hf-xet", out)
        self.assertEqual(len(self.hub.uploads), 1)

    def test_missing_hub_repo(self):
        self.hub.repo_missing = True
        ok, out = self.call(self.manager().upload_with_huggingface_hub)
        self.assertFalse(ok)
        self.assertEqual(self.hub.uploads, [])
        self.assertIn("create it first", out)

    def test_no_token(self):
        manager = self.manager(token=None)  # no env token, no saved login
        self.assertIsNone(manager.token)
        ok, _ = self.call(manager.upload_with_huggingface_hub)
        self.assertFalse(ok)
        self.assertEqual(self.hub.uploads, [])


class GitPushTests(ScriptTestCase):
    """push_to_huggingface against a local bare repo standing in for the Hub (via pushInsteadOf)."""

    def extra_git_config(self):
        self.bare = self.root / "hub.git"
        return push_redirect_config(self.bare)

    def setUp(self):
        super().setUp()
        git(self.root, "init", "-q", "--bare", str(self.bare), env=self.env)
        make_repo(self.repo, self.env)

    def bare_branches(self):
        return git(self.bare, "branch", "--format=%(refname:short)", env=self.env).split()

    def test_pushes_master_to_main(self):
        self.assertEqual(self.git("branch", "--show-current"), "master")
        ok, out = self.call(self.manager().push_to_huggingface)
        self.assertTrue(ok, out)
        self.assertEqual(self.bare_branches(), ["main"])
        self.assertEqual(self.remote_url(), HUB_URL)

    def test_rejected_push_explains_unrelated_history(self):
        other = self.root / "other"
        make_repo(other, self.env, {"hub-only.txt": "x\n"})
        git(other, "push", "-q", self.bare.resolve().as_uri(), "HEAD:main", env=self.env)
        ok, out = self.call(self.manager().push_to_huggingface)
        self.assertFalse(ok)
        self.assertIn("no shared history", out)
        self.assertIn("--allow-unrelated-histories", out)

    def test_foreign_remote_stops_before_push(self):
        self.git("remote", "add", "origin", "https://github.com/someone/else.git")
        ok, _ = self.call(self.manager().push_to_huggingface)
        self.assertFalse(ok)
        self.assertEqual(self.bare_branches(), [])

    def test_commits_uncommitted_changes_when_asked(self):
        (self.repo / "new.txt").write_text("new\n", encoding="utf-8")
        ok, out = self.call(self.manager().push_to_huggingface, stdin="y\nadd new file\n")
        self.assertTrue(ok, out)
        self.assertEqual(self.git("log", "-1", "--format=%s"), "add new file")
        self.assertEqual(git(self.bare, "log", "-1", "--format=%s", "main", env=self.env), "add new file")


class LargeFileTests(ScriptTestCase):
    def setUp(self):
        super().setUp()
        make_repo(
            self.repo,
            self.env,
            {"small.txt": "x\n", "model/model.safetensors": b"\0" * (2 * 1024 * 1024)},
        )

    def test_detect_large_files(self):
        files = self.manager().detect_large_files(threshold_mb=1)
        self.assertEqual([f["path"] for f in files], ["model/model.safetensors"])
        self.assertEqual(files[0]["extension"], ".safetensors")

    def test_lfs_status_by_pattern(self):
        manager = self.manager()
        files = manager.detect_large_files(threshold_mb=1)
        for patterns, tracked in (
            ([], False),
            (["*.safetensors"], True),
            (["*.bin"], False),
            (["model/*"], True),
            (["model/model.safetensors"], True),
        ):
            with self.subTest(patterns=patterns), mock.patch.object(
                manager, "get_git_lfs_patterns", return_value=patterns
            ):
                t, u = manager.check_lfs_status(files)
                self.assertEqual((len(t), len(u)), (1, 0) if tracked else (0, 1))

    def test_storage_menu(self):
        manager = self.manager()
        untracked = manager.detect_large_files(threshold_mb=1)
        with mock.patch.object(manager, "offer_lfs_setup", return_value=True) as lfs:
            self.assertTrue(self.call(manager.offer_storage_setup, untracked, stdin="1\n")[0])
            lfs.assert_called_once_with(untracked)
        self.assertTrue(self.call(manager.offer_storage_setup, untracked, stdin="2\n")[0])
        ok, out = self.call(manager.offer_storage_setup, untracked, stdin="3\n")
        self.assertFalse(ok)
        self.assertNotIn("Xet", out)
        self.assertTrue(self.call(manager.offer_storage_setup, [])[0])


class FindGitRepoTests(ScriptTestCase):
    def test_walks_up_from_nested_dir_and_file(self):
        make_repo(self.repo, self.env)
        nested = self.repo / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "f.txt").write_text("x", encoding="utf-8")
        self.assertEqual(up.find_git_repo(nested), self.repo.resolve())
        self.assertEqual(up.find_git_repo(nested / "f.txt"), self.repo.resolve())


if __name__ == "__main__":
    unittest.main()
