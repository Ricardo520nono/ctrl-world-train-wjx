# GitHub 推送说明

本地 Git 仓库位置：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train
```

分支：

```bash
main
```

以下大文件或运行产物不会提交到 GitHub，已经由 `.gitignore` 排除：

- `.env`
- `assets/models/`
- `latents/`
- `outputs/`
- checkpoint 和 log

如果需要重新设置远端仓库，可以执行：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train
git remote add origin git@github.com:<OWNER>/<REPO>.git
git push -u origin main
```

也可以使用 HTTPS：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

推送前建议确认仓库仍然很轻：

```bash
git status --short --ignored
git ls-files | wc -l
git count-objects -vH
```

预期结果：`assets/models/` 只会显示为 ignored，不会出现在 tracked 文件里。
