# GitHub Push Notes

The local Git repository is initialized at:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train
```

Branch:

```bash
main
```

Large runtime files are intentionally excluded by `.gitignore`:

- `.env`
- `assets/models/`
- `latents/`
- `outputs/`
- checkpoints and logs

After creating an empty GitHub repository, push with:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train
git remote add origin git@github.com:<OWNER>/<REPO>.git
git push -u origin main
```

Or with HTTPS:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

Before pushing, confirm the staged repository is lightweight:

```bash
git status --short --ignored
git ls-files | wc -l
git count-objects -vH
```

Expected: `assets/models/` appears only as ignored, not tracked.
