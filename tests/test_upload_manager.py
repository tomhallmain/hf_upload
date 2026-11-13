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
    BareHub,
    FakeHub,
    fake_sha,
    git,
    hub_redirect_config,
    isolated_env,
    make_repo,
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

    def test_delete_missing_with_nothing_missing(self):
        manager = self.manager()
        manager.delete_missing = True
        self.call(manager.upload_with_huggingface_hub)
        self.assertNotIn("delete_patterns", self.hub.uploads[0])  # the (empty) Hub lacks nothing

    def test_delete_missing_when_hub_files_unknown(self):
        # Without the Hub's file list nothing is left out of the upload, so "*" is safe.
        self.hub.tree_error = ConnectionError("offline")
        manager = self.manager()
        manager.delete_missing = True
        ok, out = self.call(manager.upload_with_huggingface_hub)
        self.assertTrue(ok)
        self.assertEqual(self.hub.uploads[0]["delete_patterns"], ["*"])
        self.assertIn("Could not list the Hub's files", out)

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
    """push_to_huggingface against a local bare repo standing in for the Hub (via insteadOf)."""

    def extra_git_config(self):
        self.bare = self.root / "hub.git"
        return hub_redirect_config(self.bare)

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


class BareHubTestCase(ScriptTestCase):
    """The "Hub" is a local bare repo; the fake upload_folder commits the folder to it and
    returns the new commit id, and self.repo is a real clone of it."""

    hub_files = {"README.md": "v1\n", "model/weights.bin": "w1\n", "notes.txt": "n\n"}

    def extra_git_config(self):
        return hub_redirect_config(self.root / "hub.git")

    def setUp(self):
        super().setUp()
        self.bare_hub = BareHub(self.root, self.env).create(self.hub_files)
        self.bare_hub.clone(self.repo)
        self.hub.bare_hub = self.bare_hub
        self.start = self.git("rev-parse", "HEAD")

    def snapshot(self):
        return {
            p.relative_to(self.repo).as_posix(): p.read_bytes()
            for p in self.repo.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(self.repo).parts
        }

    def upload(self, message="update", manager=None):
        ok, out = self.call((manager or self.manager()).upload_with_huggingface_hub, message)
        self.assertTrue(ok, out)  # sync problems never fail the upload
        return out

    def head(self):
        return self.git("rev-parse", "HEAD")


class SyncAfterUploadTests(BareHubTestCase):
    """After an API upload, local git moves to the commit the upload created."""

    def test_local_moves_to_uploaded_commit(self):
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        (self.repo / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        before = self.snapshot()
        out = self.upload("add requirements")
        self.assertNotEqual(self.bare_hub.head(), self.start)
        self.assertEqual(self.head(), self.bare_hub.head())
        self.assertEqual(self.git("rev-parse", "origin/main"), self.bare_hub.head())
        self.assertEqual(self.git("log", "-1", "--format=%s"), "add requirements")
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(self.snapshot(), before)  # working tree untouched
        self.assertIn("Local git now at", out)

    def test_lists_remaining_differences(self):
        # Without --delete-missing a local deletion is not uploaded, so it stays a difference.
        (self.repo / "notes.txt").unlink()
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        out = self.upload()
        self.assertEqual(self.head(), self.bare_hub.head())
        self.assertEqual(self.git("status", "--porcelain"), "D notes.txt")
        self.assertIn("notes.txt", out)

    def test_nothing_changed(self):
        out = self.upload()
        self.assertEqual(self.head(), self.start)
        self.assertEqual(self.bare_hub.head(), self.start)
        self.assertIn("already matches", out)

    def test_local_only_commits_are_not_moved(self):
        (self.repo / "local.txt").write_text("x\n", encoding="utf-8")
        self.git("add", "local.txt")
        self.git("commit", "-q", "-m", "local only")
        local_head = self.head()
        out = self.upload()
        self.assertEqual(self.head(), local_head)
        self.assertNotEqual(self.bare_hub.head(), self.start)  # the upload itself happened
        self.assertIn("Local git not synced", out)
        self.assertIn("git reset origin/main", out)

    def test_fetch_failure_leaves_local_unchanged(self):
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        self.hub.after_upload = lambda: self.bare_hub.bare.rename(self.root / "hub-gone.git")
        out = self.upload()
        self.assertEqual(self.head(), self.start)
        self.assertIn("git fetch failed", out)
        self.assertIn("git reset origin/main", out)

    def test_hub_changed_again_after_upload(self):
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        self.hub.after_upload = lambda: self.bare_hub.push_change("web-edit.txt", "x\n")
        out = self.upload()
        self.assertEqual(self.head(), self.start)
        self.assertIn("another change landed", out)

    def test_no_sync_option(self):
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        manager = self.manager()
        manager.sync_git = False
        out = self.upload(manager=manager)
        self.assertEqual(self.head(), self.start)
        self.assertEqual(self.git("rev-parse", "origin/main"), self.start)  # not even fetched
        self.assertNotIn("Syncing", out)

    def test_foreign_origin_is_not_synced(self):
        self.git("remote", "set-url", "origin", "https://github.com/someone/else.git")
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        out = self.upload()
        self.assertEqual(self.head(), self.start)
        self.assertIn("origin is not the Hub repo", out)

    def test_folder_without_git(self):
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "README.md").write_text("plain\n", encoding="utf-8")
        out = self.upload(manager=up.HFUploadManager(plain, REPO_ID, TOKEN))
        self.assertNotIn("Syncing", out)
        self.assertFalse((plain / ".git").exists())

    def test_upload_without_commit_id(self):
        # Older huggingface_hub versions may not return a CommitInfo.
        self.hub.bare_hub = None
        out = self.upload()
        self.assertEqual(self.head(), self.start)
        self.assertIn("did not report its commit id", out)


class UnchangedLargeFileTests(BareHubTestCase):
    """Large (LFS/Xet) files already on the Hub with the same SHA-256 are not sent again."""

    WEIGHTS = bytes(range(256)) * 16  # 4 KB, above BareHub.LFS_MIN_SIZE: an LFS file
    hub_files = {
        "README.md": "v1\n",
        "model/weights.bin": WEIGHTS,
        "data[1].bin": WEIGHTS[::-1],  # glob characters in the name
        "notes.txt": "n\n",
    }

    def hub_paths(self):
        return {e.path for e in self.bare_hub.tree()}

    def test_unchanged_large_files_not_sent(self):
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        out = self.upload()
        self.assertEqual(self.hub.sent[0], ["README.md", "notes.txt"])
        self.assertIn("Unchanged on the Hub, not uploading: data[1].bin, model/weights.bin", out)
        self.assertEqual(self.hub_paths(), set(self.hub_files))  # still on the Hub
        self.assertEqual(self.git("status", "--porcelain"), "")  # and sync still works

    def test_changed_large_file_same_size_is_sent(self):
        (self.repo / "model" / "weights.bin").write_bytes(self.WEIGHTS[::-1])  # same size
        self.upload()
        self.assertIn("model/weights.bin", self.hub.sent[0])
        self.assertNotIn("data[1].bin", self.hub.sent[0])

    def test_changed_large_file_new_size_is_sent(self):
        (self.repo / "model" / "weights.bin").write_bytes(self.WEIGHTS + b"x")
        self.upload()
        self.assertIn("model/weights.bin", self.hub.sent[0])

    def test_delete_missing_deletes_only_missing_files(self):
        # With a "*" delete pattern, upload_folder would also delete the unchanged large
        # files left out of the upload.
        (self.repo / "notes.txt").unlink()
        manager = self.manager()
        manager.delete_missing = True
        out = self.upload(manager=manager)
        self.assertEqual(self.hub.uploads[0]["delete_patterns"], ["notes.txt"])
        self.assertEqual(self.hub_paths(), {"README.md", "model/weights.bin", "data[1].bin"})
        self.assertIn("Deleting from the Hub (missing locally): notes.txt", out)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_missing_file_kept_and_reported_without_delete_missing(self):
        (self.repo / "notes.txt").unlink()
        (self.repo / "README.md").write_text("v2\n", encoding="utf-8")
        out = self.upload()
        self.assertNotIn("delete_patterns", self.hub.uploads[0])
        self.assertIn("notes.txt", self.hub_paths())
        self.assertIn("On the Hub but not in the folder, so kept: notes.txt", out)
        self.assertEqual(self.git("status", "--porcelain"), "D notes.txt")
        self.assertIn("rerun with\n   --delete-missing", out)  # in the post-sync summary too

    def test_hub_file_list_unavailable_sends_everything(self):
        self.hub.tree_error = ConnectionError("offline")
        out = self.upload()
        self.assertIn("model/weights.bin", self.hub.sent[0])
        self.assertIn("large files will be sent even if unchanged", out)


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
