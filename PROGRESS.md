# 项目进度记录

---

## 2026-05-11 18:22

**做了什么**：

1. 检查用户对 `index.html` 的最新修改，确认改动集中在 About 文案和 `NeuroHorizon` 条目 bullet
2. 使用 `ruby@3.3` 环境执行 `bundle exec jekyll build`，重新生成 `_site`
3. 检查 `_site/index.html`，确认新文案 `Neural Foundation Model`、`across brain areas, sessions, and subjects` 和 `scaling and generalization` 已生成

**效果如何**：本地 Jekyll 构建成功，首页静态产物已更新。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次主页内容修改，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 18:08

**做了什么**：

1. 调整 `index.html` 中 `Entrepreneurship` 部分的 `Co-Founder & Algorithm Lead` 条目
2. 将原来的单段正文改为与上方经历条目一致的 `<ul>` 列表格式
3. 列表内容压缩为三条：休学创业背景、实时交互数字人系统方向、负责的 3DGS 与音频驱动算法 pipeline
4. 使用 `ruby@3.3` 环境执行 `bundle exec jekyll build`，确认 `_site/index.html` 中已生成新的列表结构

**效果如何**：`Co-Founder & Algorithm Lead` 条目现在和上方经历条目在字体大小、结构和列举方式上保持一致。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次格式调整，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 17:59

**做了什么**：

1. 调整 `index.html` 中 `NeuroHorizon` 学术经历条目的正文结构
2. 将原来的三段长段落改为与 `Multi-task Neural Representation Research` 一致的 `<ul>` 列表格式
3. 保留标题和 `Submitted to NeurIPS 2026 · Under review` 状态行，列表压缩为三条核心内容
4. 使用 `ruby@3.3` 环境执行 `bundle exec jekyll build`，确认 `_site/index.html` 中已生成新的列表结构

**效果如何**：`NeuroHorizon` 条目现在和上一条学术经历在字体大小、结构和列举方式上保持一致，页面更简洁。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次格式调整，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 17:53

**做了什么**：

1. 更新 `index.html` 的 `Academic Experience` 部分，在 `2022 – 2024 Multi-task Neural Representation Research` 下方新增一条 2026 学术工作记录
2. 新增论文条目 `NeuroHorizon: Long-Horizon Forward Prediction of Neural Population Activity via Autoregressive Decoding with Hierarchical Memory`
3. 在条目中标注 `Submitted to NeurIPS 2026 · Under review`，并根据用户提供摘要整理为适合 About 页面阅读的三段简介
4. 使用 `ruby@3.3` 环境执行 `bundle exec jekyll build`，确认 `_site/index.html` 已生成新条目

**效果如何**：About 页面 `Academic Experience` 时间线中已新增 `NeuroHorizon` 条目，内容包含投稿状态、方法概述与实验结论。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次主页学术经历更新，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 16:56

**做了什么**：

1. 为 `_posts/2026-02-22-spike-foundation-models.md` 新增中英文说明字段 `author_note_cn` / `author_note_en`
2. 更新 `_layouts/post.html`，在博客日期下方按语言渲染作者说明小字
3. 更新 `assets/css/style.scss`，为日期说明添加更小号的次级文本样式
4. 使用 `ruby@3.3` 环境执行 `bundle exec jekyll build`，并检查 `_site/blog/2026/02/22/spike-foundation-models/index.html`，确认中英文说明均已生成

**效果如何**：`Spike Neural Foundation Models` 这篇双语博客在日期下方新增了一行说明，中文视图显示“作者与 AI 交流并审核后的内容”，英文视图显示对应英文说明。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次博客说明文字更新，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 16:50

**做了什么**：

1. 更新 `CLAUDE.md` 中的本地构建说明，补充当前实际可用的 Ruby/Bundler 环境
2. 在“项目概览”中新增本地构建环境说明：优先使用 Homebrew `ruby@3.3`
3. 重写“常用命令”部分，明确记录 `export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"`、`bundle exec jekyll build`、`bundle exec jekyll serve --host 127.0.0.1 --port 4000`
4. 补充系统 Ruby 2.6 误用时的排查方法（`which ruby` / `which bundle`）

**效果如何**：`CLAUDE.md` 已与当前本机可用的构建方式保持一致，后续执行本地编译时不再依赖隐含环境假设。

**是否遇到问题**：无。

**还要做什么**：提交并 push 本次文档更新，随后验证 GitHub Actions 部署是否成功。

## 2026-05-11 16:45

**做了什么**：

1. 阅读 `CLAUDE.md` 和 `PROGRESS.md`，确认本项目使用 Jekyll，推送到 `master` 后由 GitHub Actions 自动部署
2. 检查仓库状态，确认本次待发布内容来自用户已修改的 `index.html`
3. 使用 Homebrew 已安装的 `ruby@3.3` 环境执行 `bundle exec jekyll build`，成功重新生成 `_site`
4. 启动本地预览服务 `bundle exec jekyll serve --host 127.0.0.1 --port 4000`
5. 通过 `curl -I http://127.0.0.1:4000/` 验证本地首页可正常访问（HTTP 200）

**效果如何**：本地构建成功，预览服务已启动，可通过 `http://127.0.0.1:4000/` 查看修改后的主页效果。

**是否遇到问题**：直接使用系统 Ruby 2.6 执行 `bundle exec jekyll build` 时失败，原因是项目锁文件要求 `bundler 2.7.1`，而系统环境不匹配。

**如何解决**：切换到机器上已安装的 `ruby@3.3` 与对应 `bundle 2.7.1` 后重新构建，问题解决。

**还要做什么**：提交并 push 本次主页修改，随后验证 GitHub Actions 部署是否成功。

## 2026-02-23 10:30

**做了什么**：修复博客文章目录（TOC）三个问题：

1. `_layouts/post.html`：将 TOC 容器从 `<nav>` 改为 `<div>`，消除全局 `nav { position: sticky }` 的样式继承（导致 TOC 粘在顶部遮挡内容）
2. `assets/css/style.scss`：目录字体调大（`.toc-item a` 0.85→0.925rem，`.toc-h3 a` 0.825→0.875rem），行距从 1.45 调至 1.5
3. `assets/css/style.scss`：博客正文各级标题字体和间距调大：h2 1.1→1.2rem、h3 0.975→1.05rem；h2 上边距 1.75→2.25rem、h3 上边距 1.25→1.75rem；段落间距 0.85→1rem
4. 提交并推送（commit `de974de`），部署成功

**效果如何**：目录正常随页面滚动，不再遮挡内容；字体和间距更舒适易读。

**是否遇到问题**：根本原因是用 `<nav>` 标签触发了全局 nav 的 sticky 定位规则。

**��要做什么**：无。

---

## 2026-02-23 10:00

**做了什么**：为博客文章页面添加自动生成目录（TOC）功能：

1. `_layouts/post.html`：在 post-header 与 post-body 之间插入 `<nav id="post-toc">` 容器，并添加内联 JS 脚本自动提取当前激活语言块中的 h2/h3 标题生成目录链接
2. `assets/js/lang-toggle.js`：在 `applyLang` 函数末尾添加 `rebuildTOC()` 调用，切换 CN/EN 时目录同步刷新
3. `assets/css/style.scss`：添加 `.post-toc`、`.post-toc-label`、`.post-toc-list`、`.toc-item`、`.toc-h3` 等样式，风格与网站整体保持一致（浅灰背景 + 细边框，无动画）
4. 提交并推送（commit `0e5a58c`）

**效果如何**：TOC 自动从文章中提取 h2/h3 生成可点击目录，标题少于 2 个时隐藏；双语文章切换语言后目录标签（"目录"/"Contents"）和条目同步更新。

**是否遇到问题**：无。

**还要做什么**：等待 GitHub Actions 部署成功后验证效果。

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

- ~~验证 GitHub Actions 构建是否成功~~（已完成，见下次记录）
- ~~检查部署后的网站效果~~（已完成）
- 按需调整样式细节和内容

---

## 2026-02-22 18:30

**做了什么**：修复 GitHub Actions 构建失败问题，具体：

1. 删除 `assets/main.scss`（`sproogen/resume-theme` 残留，导致"File to import not found: modern-resume-theme"错误）
2. 将 `assets/css/style.css` 重命名为 `assets/css/style.scss`，添加 Jekyll front matter（`---`），并加 `@charset "UTF-8";`——覆盖 `jekyll-theme-primer 0.6.0` 自带的 `style.scss`，解决 SASS 3.7.4 的 US-ASCII 编码错误
3. 删除 `_config.yml` 中冗余配置，将 `vendor/` 加入 exclude 列表
4. 删除 `.claude/worktrees/` 目录（Claude Code 并行任务产生的临时文件，无用）
5. 提交并推送（commit `386c2b9`）

**效果如何**：GitHub Actions 构建成功，网站正常部署。About 页面显示 Brain Foundation Model 等研究兴趣、教育/经历时间轴；Blog 页面正确列出 Spike 神经基础模型博文（中英文双语）。

**是否遇到问题**：

1. Jekyll 扫描 `vendor/bundle/` 目录，尝试处理 gem 内部的 `.markdown.erb` 文件，导致日期解析错误
2. `assets/main.scss` 导入 `modern-resume-theme`（已移除），构建失败
3. `jekyll-theme-primer 0.6.0` 提供 `assets/css/style.scss`，其 SCSS 包含 UTF-8 字符，而 SASS 3.7.4 默认 US-ASCII 编码，导致编码错误

**如何解决**：

1. 在 `_config.yml` 的 `exclude` 列表中加入 `vendor/` 和 `vendor/bundle/`
2. `git rm assets/main.scss`
3. 将 `style.css` 改为 `style.scss`（含 front matter 和 `@charset "UTF-8"`），覆盖 theme 的同名文件

**还要做什么**：

- 按需优化样式细节和页面内容

---

## 2026-02-22 22:00

**做了什么**：

1. 将 `references/蒋鹏-cv.pdf` 添加到 `.gitignore`，避免 PDF 被 git 追踪
2. 为博客 post 布局（`_layouts/post.html`）添加条件式 MathJax 加载（front matter 中设 `math: true` 即启用）
3. 将博客文件从 `_posts/2025-01-20-spike-foundation-models.md` 重命名为 `_posts/2026-02-22-spike-foundation-models.md`，front matter 日期同步更新
4. 中文博客内容全部替换为 `references/spike_foundation_models_blog_cn.md`（更完整、含参考文献和公式）
5. 英文博客内容全部替换为 `references/spike_foundation_models_blog_en.md`（对应完整英文版）
6. 更新主页 `index.html`：
   - About 部分新增一段，说明当前研究方向（Brain Foundation Model、brain encoding/decoding、real-time interactive digital humans）
   - Now 部分展开 Brain Foundation Model 内容：encoding 侧理解多模态神经编码机制，decoding 侧用通用神经表征完成视觉解码（image/video reconstruction）
7. 分三次提交并 push 到 `origin master`

**效果如何**：所有修改成功提交，GitHub Pages 自动构建中（1-2 分钟后生效）。博客 URL 将变为 `https://jp-17.github.io/blog/2026/02/22/spike-foundation-models/`。

**是否遇到问题**：新博客内容含大量 LaTeX 公式，原布局无 MathJax，需额外添加。

**如何解决**：在 `_layouts/post.html` 中添加条件式 MathJax 3 CDN 加载，front matter 加 `math: true` 的博文才会载入。

**还要做什么**：无（所有任务已完成）。

---

## 2026-02-22 23:30

**做了什么**：

1. 排查 GitHub Pages 部署失败问题：发现 run #6 起 build job 因 Ruby 3.1.6 gem 安装失败而报错，deploy 被跳过，网站内容未更新
2. 修复 `.github/workflows/jekyll.yml`：升级 Ruby 3.1 → 3.3、更新 `ruby/setup-ruby` 到 v1、`cache-version` 0 → 1 强制清缓存
3. 在 `_config.yml` 添加 `future: true`，防止 UTC 时区导致当日文章被 Jekyll 判定为"未来文章"
4. 更新 `CLAUDE.md`：
   - 新增"Push 后验证部署"规范（必须确认 GitHub Actions build + deploy 都成功）
   - 更新 `post.html` 和博客 front matter 的文档（MathJax / `math: true`）

**效果如何**：run #10（commit 08f50f8）55 秒完成，build + deploy 均成功。博客页面 `/blog/2026/02/22/spike-foundation-models/` 和主页均正常显示更新内容。

**是否遇到问题**：

1. GitHub Actions runs #6-#9 的 build job 全部失败（Ruby 3.1.6 gem 安装错误），但 Actions 列表页面显示 ✓ 有误导性
2. 快速连续 push 4 次导致中间 commits 的 workflow 被 concurrency 规则取消，仅最后一次 run 实际执行

**如何解决**：

1. 升级 Ruby 3.1 → 3.3（与本地一致）+ 清除 bundler 缓存（cache-version 递增）
2. 在 CLAUDE.md 中加入部署验证规范，避免今后遗漏

**还要做什么**：无。

---

## 2026-02-23 01:00

**做了什么**：

1. 修复博客 LaTeX 行内公式渲染问题：将所有行内 `$..` 改为 `$$..$$`（Kramdown 只识别 `$$`，不识别单 `$`；行内文本中的 `$$..$$` 会转为 `\(...\)` 行内数学，独立行的 `$$..$$` 转为 `\[...\]` 显示数学）
2. 将数学块内的 `\_` 替换为 `_`，消除渲染时多余的反斜杠（Kramdown 在 `$$` 块内不处理 markdown，所以 `_` 不需要转义）
3. 主页 Now 部分 "real-time interactive digital humans" 加粗显示（`<strong>` 标签）
4. 本地 Jekyll 构建验证：确认行内公式正确转为 `\(...\)`、显示公式正确转为 `\[...\]`、无 `\_` 残留
5. 提交 push 并验证部署成功（commit `9a5dc92`）

**效果如何**：部署成功，博客页面所有 LaTeX 公式（行内和显示）均可被 MathJax 正确渲染，无 `\_` 残留，无 `<em>` 标签混入数学公式。主页加粗显示正常。

**是否遇到问题**：Kramdown + GFM 模式下，单 `$` 不被识别为数学定界符，其中的 `_..._` 被当作 emphasis 处理（转为 `<em>` 标签），导致公式被破坏。

**如何解决**：通过 Python 脚本批量替换：(1) 所有单 `$` 改为 `$$`；(2) `$$` 块内的 `\_` 改为 `_`。Kramdown 会保护 `$$` 块内的内容不做 markdown 处理。

**还要做什么**：无。
