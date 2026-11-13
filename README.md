# hf_upload

`upload_to_hf.py` uploads a local folder, such as a trained model run, to an existing
Hugging Face model repository. Use it for the first upload and for every update after that.

It has two upload methods:

- **`hub` (recommended):** uploads through the `huggingface_hub` API. No git is involved,
  and large files such as `model.safetensors` need no Git LFS setup. If the `hf-xet` package
  is installed, large files use Xet storage automatically: only changed chunks are sent.
  There is nothing to enable per repository.

  Large files that are already on the Hub unchanged are not uploaded at all. The script
  compares each one's SHA-256 with the hash the Hub stores before uploading. This is needed
  because `huggingface_hub` sends every large file before it checks whether the file
  changed, and without this check Xet would still transfer part of an unchanged file.
- **`git`:** commits and runs `git push` to the Hub's `main` branch. This needs a local clone
  whose history matches the Hub repo, and needs Git LFS for files over 10 MB.

`auto` (the default) picks `hub` when `huggingface_hub` is installed and a token is
available, and `git` otherwise.

## Requirements

- Python 3.10+ and git
- `huggingface_hub` for the `hub` method: `pip install -U huggingface_hub hf-xet`
- Git LFS (`git lfs install`) for the `git` method, if the repo has large files

## Authentication

The token is taken from the first of these that is set:

1. `--token`
2. the `HF_TOKEN` environment variable
3. the `HUGGINGFACE_TOKEN` environment variable
4. a token saved by `hf auth login`

The script never writes the token into a git remote URL. If an older version of this
script wrote one into `.git/config`, `--set-remote-only` or any successful upload removes it.

## Usage

The Hub repo must already exist (create it at https://huggingface.co/new).

```
# Upload or update a folder
python upload_to_hf.py <user>/<repo> -p path/to/model_folder -m "Describe the change"

# Also delete Hub files that no longer exist locally
python upload_to_hf.py <user>/<repo> -p path/to/model_folder --delete-missing

# Only add or repair the 'origin' remote of a local git repo, without uploading
python upload_to_hf.py <user>/<repo> -p path/to/model_folder --set-remote-only
```

Without `-p`, the script uses the nearest git repository above the current directory, or
the current directory itself.

| Option | Meaning |
|---|---|
| `-p, --path` | Folder to upload |
| `-m, --message` | Commit message (default: "Upload from script") |
| `--method {auto,hub,git}` | Upload method (default: `auto`) |
| `--delete-missing` | `hub` method: delete files from the Hub that are missing locally (never `.gitattributes`). Without it, such files are kept, and listed before uploading |
| `--set-remote-only` | Set `origin` to `https://huggingface.co/<repo_id>`, then exit |
| `--ignore-local-git` | Skip the local-git-vs-Hub history check (see below) |
| `--no-sync-git` | `hub` method: don't move local git to the new Hub commit after uploading |
| `--use-xet` | `hub` method: fail unless `hf_xet` is installed |
| `--threshold MB` | `git` method: size above which files count as large (default: 50) |
| `--token` | Hugging Face token |
| `--force` | Skips only two prompts: the second "not a Git repository" question, and the large-files question when Git LFS isn't installed. All other prompts still ask |
| `--no-auto-discover` | Use the current directory instead of searching for a git repo |

Exit code: 0 on success, 1 on failure.

## Local git and the Hub

An API upload creates its commit on the Hub, not in your local `.git`.

**Sync after upload.** When the folder is a clone of the Hub repo and local `HEAD` was the
Hub's latest commit, the script then runs `git fetch origin` and `git reset --mixed` to the
commit the upload created. Your files are not changed: they are exactly what was uploaded,
so `git status` comes out clean, apart from files the upload skipped (such as
`.gitattributes`) or local deletions not uploaded (without `--delete-missing`).
The sync is skipped, with the manual commands printed, when:

- local has commits the Hub doesn't (they are never moved away from)
- the fetch fails (a private repo needs git credentials: `hf auth login`, then store the
  token as a git credential)
- the Hub changed again right after the upload
- `origin` points somewhere else, or `--no-sync-git` is given

Sync manually with `git fetch origin` then `git reset origin/main` (never `--hard`).

Before uploading, the script compares local commits with the Hub repo's commits and stops
when they don't match. It never modifies or deletes your local `.git`.

| Local `.git` | Result |
|---|---|
| None, or contains the Hub's latest commit | Uploads |
| Shares commits with the Hub but lacks its latest one (behind) | Stops: run `git pull origin main` first |
| Shares no commits, and the Hub repo has history | Stops and explains the two workflows below |
| Shares no commits, and the Hub repo is brand new | `hub`: uploads with a warning. `git`: stops |
| Hub commits can't be read (offline etc.) | Warns and skips the check |

Pick one workflow per model:

- **API uploads only (simplest).** Keep no `.git` in the model folder; the Hub holds the
  history. If one exists, move it out of the folder or delete it. A renamed copy left
  inside the folder would be uploaded, because only `.git` itself is excluded.
- **Git workflow.** Work from a clone of the Hub repo:
  ```
  git lfs install
  git clone https://huggingface.co/<user>/<repo>
  ```
  Commit and push from the clone, or upload from it with this script: the sync above keeps
  the clone on the Hub's latest commit. After API uploads made without the sync
  (`--no-sync-git`, another machine, the website), run `git pull` before your next push.
  To attach Hub history to an existing folder without cloning again, see
  [docs/TODO.md](docs/TODO.md), item 2.

## Tests

Run from this folder:

```
python -m unittest -v      # or: python -m pytest
```

- `tests/test_upload_manager.py`: in-process tests of `HFUploadManager` (token lookup,
  remote handling, the history check, upload arguments, syncing after upload, git push,
  large-file handling).
- `tests/test_cli.py`: runs the script as a subprocess for end-to-end cases.
- `tests/support.py`: shared isolation helpers and fakes.

The tests use no network, no real token and none of your git settings:

- git runs with a temporary global config;
- `huggingface_hub` and `hf_xet` are replaced by fakes that record calls, with
  `HF_HUB_OFFLINE` and a temporary `HF_HOME` set as a backup;
- clones, fetches and pushes to the Hub URL go to a local bare repository through
  `insteadOf`, and the fake `upload_folder` commits to that repository, as the real Hub does.

The Git LFS menu test uses a shell-script stand-in for `git-lfs`, so it is skipped on Windows.
