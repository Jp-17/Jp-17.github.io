# 项目进度记录

---

## 2026-02-22 17:00

**做了什么**：对网站进行全面重构，包括以下工作：
1. 检查并同步本地仓库与远程（`git pull`，拉取了 `.github/workflows/jekyll.yml`）
2. 将原有 `sproogen/resume-theme` 全部内容存档至 `references/original_site_content.md`
3. 重写 `_config.yml`，移除远程主题，改为自定义 Jekyll 配置
4. 新建 `_layouts/default.html`（基础布局：导航栏 + 页脚）
5. 新建 `_layouts/post.html`（博客文章布局，含 CN/EN 切换按钮）
6. 新建 `assets/css/style.css`（全站自定义样式，简洁科技风）
7. 新建 `assets/js/lang-toggle.js`（CN/EN 语言切换 + localStorage 持久化）
8. 重写 `index.html`（About Me 主页，包含简介/兴趣/教育/学术经历/创业/当前状态）
9. 新建 `blog.html`（博客列表页，路由 `/blog/`）
10. 新建 `_posts/2025-01-20-spike-foundation-models.md`（双语博客：Spike 神经基础模型综述，含完整中英文内容）
11. 更新 `CLAUDE.md`，补充网站设计风格规范和博客文章规范

**效果如何**：所有文件创建/修改成功，待 push 后通过 GitHub Actions 验证部署效果。

**是否遇到问题**：`pdftoppm` 未安装导致无法直接读取 CV PDF，安装 `poppler` 后通过 `pdftotext` 提取了文本内容。

**如何解决**：使用 `brew install poppler` 安装后，改用 `pdftotext` 命令提取 PDF 文本。

**还要做什么**：
- 验证 GitHub Actions 构建是否成功
- 检查部署后的网站效果（https://jp-17.github.io/）
- 按需调整样式细节和内容
