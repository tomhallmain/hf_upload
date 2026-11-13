#!/usr/bin/env python3
"""
Hugging Face Repository Upload Script
Uploads with the huggingface_hub API (recommended; it uses Xet storage automatically
when the hf_xet package is installed), or falls back to git push with Git LFS.
Can be run from anywhere by specifying the repository path
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
import glob
import hashlib
from datetime import datetime
import tempfile
import shutil

class HFUploadManager:
    def __init__(self, repo_path: Union[str, Path], repo_id: str, token: Optional[str] = None):
        """
        Initialize the upload manager
        
        Args:
            repo_path: Path to the repository (can be absolute or relative)
            repo_id: Hugging Face repository ID
            token: Hugging Face access token
        """
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            try:  # token saved by `hf auth login` / `huggingface-cli login`
                from huggingface_hub import get_token
                self.token = get_token()
            except ImportError:
                pass
        self.original_cwd = Path.cwd()
        self.git_lfs_patterns = []
        self.large_files = []
        self.tracked_large_files = []
        self.untracked_large_files = []
        self.delete_missing = False
        self.sync_git = True

    def _run_git_command(self, cmd: List[str], check: bool = True,
                         no_prompt: bool = False) -> subprocess.CompletedProcess:
        """Run a git command in the repository context.

        no_prompt: fail instead of asking for credentials on the terminal (git would otherwise
        wait for input that captured output hides).
        """
        full_cmd = ["git", "-C", str(self.repo_path)] + cmd
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"} if no_prompt else None
        return subprocess.run(full_cmd, check=check, capture_output=True, text=True, env=env)

    def _git_output(self, *args: str) -> Optional[str]:
        """stdout of a git command, or None if it fails."""
        result = self._run_git_command(list(args), check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    @property
    def remote_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    def ensure_remote(self, name: str = "origin") -> bool:
        """Point remote `name` at the Hugging Face repo. Safe to call on every run.

        Adds the remote if missing, and strips a token embedded in an existing URL (older
        versions of this script wrote one into .git/config). The URL never carries a token:
        for git auth use `hf auth login` and accept storing it as a git credential.
        Leaves a remote that points somewhere else untouched and returns False.
        """
        # The configured URL, not `git remote get-url`, which applies url.*.insteadOf rewrites.
        result = self._run_git_command(["config", "--get", f"remote.{name}.url"], check=False)
        if result.returncode != 0:
            try:
                self._run_git_command(["remote", "add", name, self.remote_url])
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to add remote {name}: {e.stderr.strip()}")
                return False
            print(f"✓ Remote added: {name} -> {self.remote_url}")
            return True

        current = result.stdout.strip()
        without_creds = re.sub(r"^(https?://)[^@/]+@", r"\1", current)
        normalized = without_creds.rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        if normalized != self.remote_url:
            print(f"⚠️  Remote {name} points to {without_creds}, not {self.remote_url}; leaving it unchanged.")
            return False
        if without_creds != current:
            self._run_git_command(["remote", "set-url", name, self.remote_url])
            print(f"✓ Removed the access token embedded in remote {name}'s URL")
        else:
            print(f"✓ Remote {name} already set: {self.remote_url}")
        return True

    def check_environment(self) -> Dict[str, bool]:
        """Check if required tools are available"""
        env_status = {
            "git": False,
            "git_lfs": False,
            "huggingface_hub": False,
            "hf_xet": False,
            "git_repo": False,
            "token_available": bool(self.token),
            "repo_exists": self.repo_path.exists(),
            "repo_is_dir": self.repo_path.is_dir()
        }
        
        # Check Git
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            env_status["git"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        # Check Git LFS
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
            env_status["git_lfs"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        

        # Check huggingface_hub
        try:
            import huggingface_hub
            env_status["huggingface_hub"] = True
            
            # Check for Xet support via hf_xet (the correct import name)
            try:
                import hf_xet
                env_status["hf_xet"] = True
            except ImportError:
                env_status["hf_xet"] = False
        except ImportError:
            pass

        # Check if path is a git repo
        if self.repo_path.exists() and self.repo_path.is_dir():
            try:
                self._run_git_command(["rev-parse", "--git-dir"], check=True)
                env_status["git_repo"] = True
            except subprocess.CalledProcessError:
                pass

        return env_status

    def check_xet_availability(self) -> Tuple[bool, str]:
        """Check if Xet is available and return status with message"""
        # The correct PyPI package name is hf-xet, and the import name is hf_xet
        try:
            import hf_xet
            version = getattr(hf_xet, '__version__', 'unknown')
            return True, f"hf_xet {version} available"
        except ImportError:
            pass

        # Check huggingface_hub version to give a helpful message
        try:
            import huggingface_hub
            version = getattr(huggingface_hub, '__version__', '0.0.0')

            try:
                # Proper version comparison (not string comparison)
                from packaging.version import Version
                hf_version = Version(version)
                if hf_version >= Version("0.30.0"):
                    return False, (
                        f"huggingface_hub {version} supports Xet, but hf-xet is not installed. "
                        f"Install with: pip install hf-xet"
                    )
                else:
                    return False, (
                        f"huggingface_hub {version} is too old (need >=0.30.0). "
                        f"Upgrade with: pip install --upgrade huggingface-hub"
                    )
            except ImportError:
                # packaging not available, fall back to simple check
                return False, f"huggingface_hub {version} installed, but hf_xet not importable"
        except ImportError:
            return False, "huggingface_hub not installed"

    def get_git_lfs_patterns(self) -> List[str]:
        """Get currently tracked Git LFS patterns"""
        try:
            result = self._run_git_command(["lfs", "track"])
            patterns = []
            for line in result.stdout.strip().split('\n'):
                if line and ")" in line:
                    # Extract pattern from "path/to/file (pattern)"
                    pattern = line.split(")")[1].strip() if ")" in line else line
                    if pattern:
                        patterns.append(pattern)
            return patterns
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def detect_large_files(self, threshold_mb: int = 50) -> List[Dict]:
        """Detect large files in the repository"""
        large_files = []
        
        # Get all files tracked by git
        try:
            result = self._run_git_command(["ls-files"])
            files = result.stdout.strip().split('\n')
            
            for file_path in files:
                if not file_path:
                    continue
                    
                full_path = self.repo_path / file_path
                if not full_path.exists():
                    continue
                    
                size_mb = full_path.stat().st_size / (1024 * 1024)
                if size_mb >= threshold_mb:
                    large_files.append({
                        "path": file_path,
                        "size_mb": round(size_mb, 2),
                        "extension": Path(file_path).suffix.lower() or "no_extension"
                    })
        except subprocess.CalledProcessError:
            pass
            
        return large_files
    
    def check_lfs_status(self, large_files: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Check which large files are tracked by Git LFS"""
        lfs_patterns = self.get_git_lfs_patterns()
        tracked = []
        untracked = []
        
        for file_info in large_files:
            # Check if file matches any LFS pattern
            is_tracked = False
            file_path = file_info["path"]
            
            for pattern in lfs_patterns:
                if pattern.startswith("*."):
                    # Extension pattern
                    extension = file_info["extension"]
                    if extension and extension[1:] == pattern[2:]:
                        is_tracked = True
                        break
                elif "*" in pattern:
                    # Wildcard pattern - simple check
                    pattern_regex = pattern.replace("*", ".*")
                    if re.search(pattern_regex, file_path):
                        is_tracked = True
                        break
                elif pattern == file_path:
                    is_tracked = True
                    break
                    
            if is_tracked:
                tracked.append(file_info)
            else:
                untracked.append(file_info)
                
        return tracked, untracked
    
    def get_current_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = self._run_git_command(["branch", "--show-current"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def get_hub_commits(self) -> Optional[List[str]]:
        """Commit ids on the Hub repo's main branch, newest first. None if they can't be read
        (huggingface_hub missing, repo not found, no network, ...)."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("⚠️  huggingface_hub not installed: skipping the local git vs Hub history check")
            return None
        try:
            commits = HfApi().list_repo_commits(self.repo_id, token=self.token)
        except Exception as e:
            print(f"⚠️  Could not read Hub commits ({type(e).__name__}): skipping the history check")
            return None
        return [c.commit_id for c in commits]

    def check_local_git_matches_hub(self, method: str) -> bool:
        """Fail when the local .git does not track the existing Hub repo.

        Local git is "valid" for this upload when it contains the Hub's latest commit (up to
        date or ahead). Otherwise it is either behind (shares some commits) or unrelated (shares
        none, e.g. `git init` in a folder that was uploaded through the API). An unrelated local
        repo is tolerated only for an API upload into a fresh Hub repo (just its creation
        commit). Never modifies the local repository.
        """
        if not (self.repo_path / ".git").is_dir():
            return True  # no git here (or it belongs to a parent folder): nothing to compare
        hub = self.get_hub_commits()
        if not hub:
            return True
        result = self._run_git_command(["rev-list", "--all"], check=False)
        local = set(result.stdout.split()) if result.returncode == 0 else set()

        if hub[0] in local:
            return True  # local has the Hub's latest commit

        url = self.remote_url
        if any(c in local for c in hub):
            print(f"\n❌ Local git in {self.repo_path} is behind {url}:")
            print(f"   the Hub's latest commit {hub[0][:8]} is not in your local history.")
            print("   It has commits made since your last pull, for example uploads through the")
            print("   API. Uploading now could overwrite those changes. Update first:")
            print("     git pull origin main")
            print("   then rerun this script.")
            return False

        if method == "hub" and len(hub) <= 1:
            print(f"\nℹ️  {url} is a fresh repo. Your local git history will not be linked to it;")
            print("   after this upload, either keep using API uploads (and move the local .git out")
            print(f"   of the folder), or work from a fresh clone: git clone {url}")
            return True

        what = f"has {len(hub)} commits" if len(hub) > 1 else "already exists"
        print(f"\n❌ The local git repository in {self.repo_path} shares no history with")
        print(f"   {url}, which {what}.")
        print("   Its commits, branches and remotes describe a different history from the Hub,")
        print("   so git status/log/push here are misleading, and git push will be rejected.")
        print("\n   Pick one workflow for this model:")
        print("   A) API uploads only (simplest; the Hub keeps the history):")
        print(f"        move {self.repo_path / '.git'} out of the folder, or delete it")
        print("        (a renamed copy left inside the folder would be uploaded), then rerun")
        print("        this script with --method hub.")
        print("   B) git workflow (local history that matches the Hub):")
        print("        git lfs install")
        print(f"        git clone {url} <new folder>")
        print("        copy your changed files into <new folder>, commit, and push from there.")
        print("        If you mix in API uploads later, run git pull before each push.")
        print("\n   To upload anyway, rerun with --ignore-local-git.")
        return False

    def check_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes"""
        try:
            result = self._run_git_command(["status", "--porcelain"])
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return True
    
    def offer_storage_setup(self, untracked_files: List[Dict]) -> bool:
        """git push method only: offer Git LFS for large files committed as plain git objects"""
        if not untracked_files:
            return True

        print("\n⚠️  Large files not tracked by Git LFS:")
        for file_info in untracked_files:
            print(f"  - {file_info['path']} ({file_info['size_mb']} MB)")

        print("\nThe Hub rejects plain git files over 10 MB on git push. Options:")
        print("1. Use Git LFS")
        print("2. Continue anyway (the push will likely be rejected)")
        print("3. Abort the upload process")
        print("\n💡 Or skip git entirely: --method hub uploads large files with no LFS setup.")

        choice = input("\nChoose option (1/2/3): ").strip()

        if choice == "1":  # Git LFS
            return self.offer_lfs_setup(untracked_files)

        elif choice == "2":  # Continue
            return True
            
        else:  # Abort
            print("\n❌ Upload aborted.")
            return False
    
    def offer_lfs_setup(self, untracked_files: List[Dict]) -> bool:
        """Set up Git LFS for untracked large files"""
        if not untracked_files:
            return True
            
        print("\n🔧 Configuring Git LFS...")
        # Get unique extensions
        extensions = set()
        for file_info in untracked_files:
            if file_info["extension"] != "no_extension":
                extensions.add(file_info["extension"])
        
        for ext in extensions:
            pattern = f"*.{ext[1:]}"
            try:
                self._run_git_command(["lfs", "track", pattern])
                print(f"  ✓ Added LFS tracking for {pattern}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to add LFS tracking for {pattern}: {e.stderr}")
                return False
                
        # Commit .gitattributes
        try:
            self._run_git_command(["add", ".gitattributes"])
            self._run_git_command(["commit", "-m", "Configure Git LFS for large files"])
            print("  ✓ Committed LFS configuration")
            print("  ℹ️  This only affects files added from now on. Files already in earlier")
            print("     commits stay plain git objects until history is rewritten, e.g.:")
            print("       git lfs migrate import --include=\"<pattern>\" --everything")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to commit LFS configuration: {e.stderr}")
            return False
    
    def push_to_huggingface(self, commit_message: Optional[str] = None) -> bool:
        """Push the repository to Hugging Face"""
        if not commit_message:
            commit_message = f"Upload from script - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Check for uncommitted changes
        if self.check_uncommitted_changes():
            print("\n⚠️  Uncommitted changes detected.")
            choice = input("Do you want to commit them before pushing? (y/n): ").strip().lower()
            
            if choice == 'y':
                message = input("Enter commit message (optional, press Enter for default): ").strip()
                if not message:
                    message = commit_message
                    
                try:
                    self._run_git_command(["add", "."])
                    self._run_git_command(["commit", "-m", message])
                    print("✓ Changes committed")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to commit changes: {e.stderr}")
                    return False
            else:
                print("⚠️  Proceeding with uncommitted changes. This may cause issues.")
        
        print("\n🔗 Checking Hugging Face remote...")
        if not self.ensure_remote():
            return False

        # Push to Hugging Face. The Hub's default branch is always main, whatever the local
        # branch is called (git init may have named it master).
        branch = self.get_current_branch() or "HEAD"
        print(f"\n🚀 Pushing local {branch} to Hugging Face main...")
        
        try:
            result = self._run_git_command(["push", "-u", "origin", "HEAD:main"], check=False)

            if result.returncode != 0:
                print("\n❌ Push failed:")
                print(result.stderr)

                stderr = result.stderr.lower()
                if "rejected" in stderr and ("fetch first" in stderr or "non-fast-forward" in stderr):
                    print("\nℹ️  The Hub repo has commits your local branch lacks, for example from")
                    print("   uploads through the huggingface_hub API. Those have no shared history")
                    print("   with your local git, so git push cannot update the repo. Either keep")
                    print("   uploading with the API (--method hub), or merge the Hub history first:")
                    print("     git fetch origin")
                    print("     git merge origin/main --allow-unrelated-histories")
                if "10 mib" in stderr or "larger than" in stderr or "pre-receive hook" in stderr:
                    print("\nℹ️  The Hub rejects files over 10 MB committed as plain git objects.")
                    print("   Files like model.safetensors must be committed through Git LFS before")
                    print("   a git push will work. The API upload (--method hub) avoids this.")

                # Check if error is about LFS
                if "lfs" in result.stderr.lower():
                    print("\n📦 Git LFS error detected.")
                    print("Try running: git lfs push --all origin")
                    choice = input("Attempt LFS push? (y/n): ").strip().lower()
                    
                    if choice == 'y':
                        print("\n🔄 Attempting Git LFS push...")
                        result = self._run_git_command(["lfs", "push", "--all", "origin"], check=False)
                        
                        if result.returncode == 0:
                            print("✅ LFS push successful!")
                            return True
                        else:
                            print("❌ LFS push failed:")
                            print(result.stderr)
                            return False
                return False
            
            print("✅ Successfully pushed to Hugging Face!")
            print(f"📍 Your model is available at: https://huggingface.co/{self.repo_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Push failed: {e.stderr}")
            return False
    
    def upload_with_huggingface_hub(self, commit_message: Optional[str] = None) -> bool:
        """Upload through the huggingface_hub API (no git involved).

        Large files need no LFS setup. Xet storage is used automatically when the hf_xet
        package is installed; there is nothing to enable per repository.
        """
        try:
            from huggingface_hub import upload_folder, HfApi
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError:
            print("❌ huggingface_hub library not installed.")
            print("Install it with: pip install huggingface-hub")
            return False
        
        if not self.token:
            print("❌ Hugging Face token not available.")
            return False
        
        print("\n📤 Uploading to Hugging Face using huggingface_hub...")
        
        xet_available, xet_status = self.check_xet_availability()
        if xet_available:
            print(f"✅ {xet_status}: large files upload through Xet (changed chunks only)")
        else:
            print(f"ℹ️  {xet_status}. Upload still works; hf-xet makes large-file uploads faster.")

        try:
            # Check if repository exists
            api = HfApi()
            try:
                api.repo_info(repo_id=self.repo_id, token=self.token)
                print(f"✓ Repository exists: {self.repo_id}")
            except RepositoryNotFoundError:
                print(f"ℹ️  Repository {self.repo_id} not found.")
                print("You need to create it first:")
                print(f"  https://huggingface.co/new?name={self.repo_id.split('/')[-1]}")
                return False
            
            # Prepare upload kwargs
            upload_kwargs = {
                "folder_path": str(self.repo_path),
                "repo_id": self.repo_id,
                "token": self.token,
                "commit_message": commit_message or "Upload from script",
                "ignore_patterns": [
                    ".git",
                    ".gitattributes",
                    "*.pyc",
                    "__pycache__",
                    ".DS_Store",
                ]
            }

            # Safe to repeat for updates: each call makes one new Hub commit. huggingface_hub
            # sends every large (LFS/Xet) file before it notices the file is unchanged, so
            # unchanged large files are found here first and left out of the upload.
            unchanged, remote_only = self.plan_upload(api)
            if unchanged:
                upload_kwargs["ignore_patterns"] += [glob.escape(p) for p in unchanged]

            # Local deletions are mirrored only with --delete-missing, and then only for the
            # exact files missing locally: a "*" pattern would also delete the unchanged
            # files just left out above.
            if self.delete_missing:
                if remote_only is None:  # Hub file list unknown; nothing was left out above
                    upload_kwargs["delete_patterns"] = ["*"]
                elif remote_only:
                    upload_kwargs["delete_patterns"] = [glob.escape(p) for p in remote_only]
                    print(f"🗑️  Deleting from the Hub (missing locally): {', '.join(remote_only)}")
            elif remote_only:
                print(f"ℹ️  On the Hub but not in the folder, so kept: {', '.join(remote_only)}")
                print("   Rerun with --delete-missing to delete them from the Hub.")

            has_git = (self.repo_path / ".git").is_dir()
            sync = has_git and self.sync_git
            if sync:  # recorded before uploading, to know whether local was up to date
                hub_commits = self.get_hub_commits()
                hub_head_before = hub_commits[0] if hub_commits else None
                local_head_before = self._git_output("rev-parse", "--verify", "-q", "HEAD")

            commit_info = upload_folder(**upload_kwargs)
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False

        print("✅ Successfully uploaded to Hugging Face!")
        # The API upload does not use git. Record the remote anyway so `git remote -v`,
        # `git fetch` etc. know where the repo lives.
        if has_git:
            remote_ok = self.ensure_remote()
            if sync:
                if not remote_ok:
                    print("ℹ️  Not syncing local git: origin is not the Hub repo.")
                else:
                    try:
                        self.sync_local_git(hub_head_before, local_head_before,
                                            getattr(commit_info, "oid", None))
                    except Exception as e:  # never turn a successful upload into a failure
                        print(f"⚠️  Could not sync local git: {e}")
                        self._print_manual_sync()
        print(f"📍 Your model is available at: https://huggingface.co/{self.repo_id}")
        return True

    def _local_files(self) -> Dict[str, Path]:
        """Files in the folder by repo path (posix), excluding .git and the HF cache folder.

        Deliberately broader than upload_folder's own filtering: used to decide what is
        missing locally, so erring towards "present" can only mean fewer deletions.
        """
        files = {}
        for path in self.repo_path.rglob("*"):
            rel = path.relative_to(self.repo_path).as_posix()
            if ".git" in rel.split("/") or rel.startswith(".cache/huggingface/"):
                continue
            if path.is_file():
                files[rel] = path
        return files

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def plan_upload(self, api) -> Tuple[List[str], Optional[List[str]]]:
        """Compare the folder with the Hub's current files.

        Returns (unchanged_large_files, hub_only_files):
        - unchanged_large_files: local files stored as LFS/Xet on the Hub whose SHA-256 matches
        - hub_only_files: files on the Hub that are not in the folder (never .gitattributes),
          or None if the Hub's file list could not be read
        """
        try:
            entries = list(api.list_repo_tree(self.repo_id, recursive=True, token=self.token))
        except Exception as e:
            print(f"⚠️  Could not list the Hub's files ({type(e).__name__}); "
                  "large files will be sent even if unchanged.")
            return [], None
        remote = {e.path: e for e in entries if hasattr(e, "blob_id")}  # files, not folders
        local = self._local_files()

        def lfs_field(lfs, name):
            return lfs.get(name) if isinstance(lfs, dict) else getattr(lfs, name, None)

        candidates = []
        for rel, path in local.items():
            lfs = getattr(remote.get(rel), "lfs", None)
            if lfs and lfs_field(lfs, "size") == path.stat().st_size:
                candidates.append((rel, path, lfs_field(lfs, "sha256")))
        unchanged = []
        if candidates:
            print(f"🔍 Checking {len(candidates)} large file(s) against the Hub...")
            for rel, path, remote_sha in candidates:
                if self._sha256(path) == remote_sha:
                    unchanged.append(rel)
        if unchanged:
            print(f"✓ Unchanged on the Hub, not uploading: {', '.join(sorted(unchanged))}")

        hub_only = sorted(p for p in remote if p not in local and p != ".gitattributes")
        return sorted(unchanged), hub_only

    @staticmethod
    def _print_manual_sync():
        print("   To move your local branch to the Hub's latest commit (files are not changed):")
        print("     git fetch origin")
        print("     git reset origin/main")

    def sync_local_git(self, hub_head_before: Optional[str], local_head_before: Optional[str],
                       new_commit: Optional[str]) -> bool:
        """After an API upload, point local HEAD and index at the commit the upload created.

        The working tree is what was just uploaded, so after `git reset --mixed` git status
        shows it as committed. Runs only when local HEAD was the Hub's latest commit before
        the upload; local-only commits are never moved away from. The working tree is never
        modified.
        """
        print("\n🔄 Syncing local git with the new Hub commit...")
        if not new_commit:
            print("ℹ️  The upload did not report its commit id; local git not synced.")
            self._print_manual_sync()
            return False
        if new_commit == local_head_before:
            print("✓ Nothing changed on the Hub; local git already matches it.")
            return True
        if not hub_head_before or local_head_before != hub_head_before:
            print("ℹ️  Local git not synced: before the upload, local HEAD was not the Hub's")
            print("   latest commit (local-only commits, or history not compared).")
            print("   Commit or push those changes yourself, or, to drop them from your branch:")
            self._print_manual_sync()
            return False

        fetch = self._run_git_command(
            ["fetch", "--quiet", "origin", "+refs/heads/main:refs/remotes/origin/main"],
            check=False, no_prompt=True,
        )
        if fetch.returncode != 0:
            print(f"⚠️  git fetch failed: {fetch.stderr.strip()}")
            print("   (A private repo needs git credentials: `hf auth login`, and store the")
            print("    token as a git credential.)")
            self._print_manual_sync()
            return False

        hub_main = self._git_output("rev-parse", "--verify", "-q", "refs/remotes/origin/main")
        if hub_main != new_commit:
            print(f"ℹ️  Hub main is now {str(hub_main)[:8]}, not the uploaded commit "
                  f"{new_commit[:8]} (another change landed); local git not synced.")
            print("   Review with `git log HEAD..origin/main`, then pull.")
            return False
        is_child = self._run_git_command(
            ["merge-base", "--is-ancestor", local_head_before, new_commit], check=False
        )
        if is_child.returncode != 0:
            print("ℹ️  The uploaded commit does not follow local HEAD; local git not synced.")
            return False

        self._run_git_command(["reset", "--quiet", "--mixed", new_commit])
        print(f"✓ Local git now at {new_commit[:8]}, the commit this upload created.")
        status = self._git_output("status", "--short") or ""
        if status:
            lines = status.splitlines()
            print("   Remaining local differences (files the upload skipped or left on the Hub):")
            for line in lines[:10]:
                print(f"     {line}")
            if len(lines) > 10:
                print(f"     ... and {len(lines) - 10} more (git status)")
            if any("D" in line[:2] for line in lines):
                print("   Files deleted locally (D) are still on the Hub: rerun with")
                print("   --delete-missing to delete them there.")
        return True


def find_git_repo(start_path: Path) -> Optional[Path]:
    """Find the nearest git repository by walking up the directory tree"""
    current = start_path.resolve()
    
    # If the start path is a file, start from its parent
    if current.is_file():
        current = current.parent
    
    while current != current.parent:  # Stop at root
        git_dir = current / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return current
        current = current.parent
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Upload a folder to Hugging Face (huggingface_hub API, or git push with LFS)",
        epilog="Can be run from anywhere by specifying the repository path"
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--path", "-p",
        help="Path to the repository (default: current directory or nearest git repo)"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token (overrides HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--method",
        choices=["git", "hub", "auto"],
        default="auto",
        help="Upload method: git push, huggingface_hub, or auto (recommended)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Size threshold in MB for detecting large files (default: 50)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip interactive prompts and use default actions"
    )
    parser.add_argument(
        "--no-auto-discover",
        action="store_true",
        help="Don't auto-discover git repository, use exact path or current directory"
    )
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="hub method: fail unless hf_xet is installed (Xet is used automatically when it is)"
    )
    parser.add_argument(
        "--set-remote-only",
        action="store_true",
        help="Only add/repair the 'origin' remote for repo_id, then exit (no upload)"
    )
    parser.add_argument(
        "--delete-missing",
        action="store_true",
        help="hub method: also delete files from the Hub repo that no longer exist locally"
    )
    parser.add_argument(
        "--no-sync-git",
        action="store_true",
        help="hub method: don't move local git to the new Hub commit after uploading"
    )
    parser.add_argument(
        "--ignore-local-git",
        action="store_true",
        help="Upload even if the folder's .git does not match the Hub repo's history"
    )
    parser.add_argument(
        "--message", "-m",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    print("🔄 Hugging Face Upload Manager")
    print("=" * 50)
    
    # Determine repository path
    repo_path = None
    
    if args.path:
        # Use explicitly provided path
        repo_path = Path(args.path).resolve()
        if not repo_path.exists():
            print(f"❌ Specified path does not exist: {repo_path}")
            sys.exit(1)
    else:
        # Try to find repository
        if not args.no_auto_discover:
            # Check current directory first
            current_path = Path.cwd()
            repo_path = find_git_repo(current_path)
            
            if repo_path:
                print(f"🔍 Auto-discovered git repository at: {repo_path}")
            else:
                # Fall back to current directory
                repo_path = current_path
                print(f"ℹ️  No git repository found, using current directory: {repo_path}")
        else:
            # Use current directory
            repo_path = Path.cwd()
    
    # Check if path is a directory
    if repo_path.is_file():
        print(f"ℹ️  Path is a file, looking for git repository...")
        parent_repo = find_git_repo(repo_path)
        if parent_repo:
            repo_path = parent_repo
            print(f"🔍 Found git repository at: {repo_path}")
        else:
            print(f"❌ Path is a file and not inside a git repository: {repo_path}")
            sys.exit(1)
    
    print(f"📂 Repository path: {repo_path}")
    
    # Check if it's a git repository
    git_dir = repo_path / ".git"
    if not (git_dir.exists() and git_dir.is_dir()):
        if args.set_remote_only:
            print(f"❌ {repo_path} is not a git repository")
            sys.exit(1)
        print(f"⚠️  {repo_path} is not a git repository")
        choice = input("Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            sys.exit(1)
    
    try:
        # Initialize manager
        manager = HFUploadManager(repo_path, args.repo_id, args.token)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    manager.delete_missing = args.delete_missing
    manager.sync_git = not args.no_sync_git

    if args.set_remote_only:
        sys.exit(0 if manager.ensure_remote() else 1)

    # Check environment
    print("\n🔍 Checking environment...")
    env_status = manager.check_environment()
    
    for tool, available in env_status.items():
        if tool in ["repo_exists", "repo_is_dir"]:
            continue
        status = "✅" if available else "❌"
        print(f"  {status} {tool.replace('_', ' ').title()}")
    
    if not env_status["git_repo"]:
        print("\n⚠️  Not a Git repository. Some features may be limited.")
        
        if not args.force:
            choice = input("Continue anyway? (y/n): ").strip().lower()
            if choice != 'y':
                sys.exit(1)
    
    # Determine upload method
    method = args.method
    if method == "auto":
        if env_status["huggingface_hub"] and manager.token:
            method = "hub"
            print("\nℹ️  Using huggingface_hub API upload")
        else:
            method = "git"
            print("\nℹ️  Using git push method (huggingface_hub or token not available)")
    if method == "hub" and args.use_xet and not env_status["hf_xet"]:
        print("\n❌ --use-xet given but hf_xet is not installed. Install with:")
        print("  pip install -U hf-xet")
        sys.exit(1)

    # A local .git that doesn't track the existing Hub repo: stop before uploading anything.
    if env_status["git_repo"] and not args.ignore_local_git:
        print("\n🔍 Comparing local git history with the Hub repo...")
        if not manager.check_local_git_matches_hub(method):
            sys.exit(1)

    # Large files only matter for git push: the API upload handles them itself.
    if method == "git":
        print(f"\n📊 Detecting large files (>{args.threshold} MB)...")
        large_files = manager.detect_large_files(args.threshold)

        if large_files:
            print(f"Found {len(large_files)} large files:")
            for file_info in large_files:
                print(f"  - {file_info['path']} ({file_info['size_mb']} MB)")

            # Check LFS tracking status
            tracked, untracked = manager.check_lfs_status(large_files)
            manager.tracked_large_files = tracked
            manager.untracked_large_files = untracked

            # Handle untracked files
            if untracked and env_status["git_repo"]:
                if env_status["git_lfs"]:
                    if not manager.offer_storage_setup(untracked):
                        sys.exit(1)
                else:
                    print("\n⚠️  Large files not tracked and Git LFS is not installed.")
                    print("  - Git LFS: https://git-lfs.github.com/")
                    print("  - Or upload with --method hub, which needs no LFS")

                    if not args.force:
                        choice = input("Continue without handling large files? (y/n): ").strip().lower()
                        if choice != 'y':
                            sys.exit(1)
        else:
            print("✅ No large files detected.")

    # Perform upload
    success = False
    
    if method == "hub":
        success = manager.upload_with_huggingface_hub(args.message)
        if not success and args.method == "auto":
            print("\n🔄 Falling back to git push method...")
            success = manager.push_to_huggingface(args.message)
    else:
        success = manager.push_to_huggingface(args.message)
    
    if success:
        print("\n✨ Upload complete!")
        sys.exit(0)
    else:
        print("\n❌ Upload failed.")
        print("\nTroubleshooting tips:")
        print("1. Check your Hugging Face token: echo $HF_TOKEN")
        print("2. Ensure you have write permissions to the repository")
        print("3. Try uploading with huggingface_hub: pip install huggingface-hub")
        print("4. For better performance with large files:")
        print("   - Install Xet: pip install huggingface_hub[hf_xet]")
        print("   - Or use Git LFS: git lfs version")
        print("5. Verify remote URL: git remote -v")
        print("6. Make sure the repository exists on Hugging Face")
        sys.exit(1)

if __name__ == "__main__":
    main()
