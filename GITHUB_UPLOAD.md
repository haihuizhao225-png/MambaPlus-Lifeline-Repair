# 将本项目发布至 GitHub

推荐先创建**私有仓库**，防止误上传真实震后修复记录、模型权重或包含敏感信息的实验结果。压缩包仅包含源码、文档与模拟数据。

## Windows PowerShell + GitHub CLI

1. 安装 Git 和 [GitHub CLI](https://cli.github.com/)；登录自己的GitHub账号。
2. 解压 `MambaPlus_GitHub_Repository.zip`，在解压后的 `MambaPlus_GitHub_Repository` 文件夹打开PowerShell。
3. 按顺序运行：

```powershell
gh auth login
git init
git branch -M main
git add .
git commit -m "Initial MambaPlus paper implementation"
gh repo create MambaPlus-PostEarthquake-Lifelines --private --source . --remote origin --push
```

如果 `git commit` 报错缺少用户名/邮箱，先使用您自己的信息配置：`git config --global user.name "Your Name"`、`git config --global user.email "you@example.com"`，然后重试提交。

## 注意

- `.gitignore` 默认排除本地真实 CSV/XLSX、检查点、BERT缓存和实验结果；不要使用 `git add -f` 强行发布私人数据。
- `--private` 可以在GitHub后续设置中改为公开，但公开前建议确认作者署名、开源许可证、研究数据授权及代码论文一致性。
- 该脚本**不会自动运行**，也不会替您登录或创建云端仓库；只有最后一条命令成功后，才会有真实GitHub链接。
