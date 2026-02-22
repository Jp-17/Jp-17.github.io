# CLAUDE.md — 项目 AI 助手指令文件

本文件供 Claude Code 在每次对话中自动加载，用于了解项目背景、规范和工作要求。

---

## 项目概览

- **项目类型**：个人学术主页（GitHub Pages 静态网站）
- **技术栈**：Jekyll + 自定义主题（无第三方远程主题）
- **部署方式**：推送到 `master` 分支后，GitHub Actions 自动构建并发布到 `https://jp-17.github.io/`
- **主要文件结构**：
  - `_config.yml` — Jekyll 核心配置（permalink / plugins 等）
  - `_layouts/default.html` — 基础布局（导航栏 + 页脚）
  - `_layouts/post.html` — 博客文章布局（含 CN/EN 切换；`math: true` 时加载 MathJax）
  - `assets/css/style.scss` — 全站样式（自定义，不依赖外部框架）
  - `assets/js/lang-toggle.js` — 博客 CN/EN 语言切换逻辑
  - `index.html` — About Me 主页
  - `blog.html` — 博客列表页（permalink: /blog/）
  - `_posts/` — 博客文章（文件名格式：`YYYY-MM-DD-slug.md`）
  - `images/` — 图片资源
  - `references/` — 本地参考资料（不部署到网站，已在 `_config.yml` 中 exclude）

## 常用命令

```bash
# 本地预览（需要提前安装 Ruby & Bundler）
bundle exec jekyll serve

# 安装依赖
bundle install
```

---

## 网站设计风格（必须遵守）

网站整体风格定位：**简洁、科技感、学术**。具体要求：

- **不使用炫酷动画**：无旋转、无弹跳、无复杂过渡，hover 效果仅限颜色过渡（`transition: 0.12–0.15s`）
- **不使用鲜艳配色**：主色调为白底深字（`#0f172a` 文字，`#ffffff` 背景），accent 色使用低饱和蓝色（`#2563eb`），辅助色使用灰色（`#64748b`）
- **极简布局**：最大宽度 740px，大量留白，无复杂网格或多列布局
- **字体**：使用系统字体栈（`system-ui, -apple-system, ...`），等宽字体用于日期/代码（`SF Mono, Fira Code, ...`）
- **边框**：细线（1px）、低对比度（`#e2e8f0`），无阴影或仅极轻阴影
- **不引入 CSS 框架**（Bootstrap、Tailwind 等），保持样式文件的精简

修改样式时，**优先编辑 `assets/css/style.scss`**，避免内联样式扩散。

---

## 博客文章规范

新增博客文章时：

1. 文件放在 `_posts/` 目录，命名格式：`YYYY-MM-DD-slug.md`
2. Front matter 必须包含以下字段：

```yaml
---
layout: post
title: "英文标题（作为 slug 和 SEO 用）"
title_cn: "中文标题"
title_en: "English Title"
date: YYYY-MM-DD
bilingual: true    # 若为双语文章则设为 true；纯中文/英文文章可省略
math: true         # 若文章含 LaTeX 公式则设为 true，会加载 MathJax
---
```

1. 双语文章内容结构：

```markdown
<div class="lang-cn" markdown="1">
（中文内容，支持完整 Markdown）
</div>

<div class="lang-en" markdown="1">
（English content）
</div>
```

1. 语言切换由 `assets/js/lang-toggle.js` 自动处理，默认显示 CN，用户选择会保存到 `localStorage`。

---

## 工作规范

### 0. 执行任务前（必须遵守）

**每次开始执行任务前，必须先阅读 `PROGRESS.md`**，了解之前的工作记录、遗留问题和下一步计划，避免重复劳动或遗漏上下文。

### 3. 任务完成后检查 CLAUDE.md（必须遵守）

每次任务完成后，**检查本文件"项目概览"部分是否已过时**——包括文件结构、技术栈、注意事项等，若有变化则立即更新，保持 CLAUDE.md 与实际项目状态一致。

### 1. 进度文档更新（必须遵守）

每完成一个小任务（或任务阶段），**必须立即更新进度文档** `PROGRESS.md`（若不存在则创建）。

进度文档格式要求：

- 使用 Markdown 格式
- 语言：中文
- 每条记录包含：
  - **日期-时间**（格式：`YYYY-MM-DD HH:MM`）
  - **做了什么**：具体操作描述
  - **效果如何**：结果/状态（成功/失败/待验证）
  - **是否遇到问题**：记录遇到的报错、异常、困难或不确定点；若无则写"无"
  - **如何解决**：针对上述问题的解决方法或绕过方案；若无问题则省略此项
  - **还要做什么**：下一步计划（若任务已全部完成则写"无"）

示例格式：

```markdown
## 2026-02-22 15:30

**做了什么**：更新 `index.html` 中的 About Me 段落，补充最新研究方向描述。

**效果如何**：修改成功，本地预览正常显示。

**是否遇到问题**：YAML front matter 缩进错误导致 Jekyll 构建失败。

**如何解决**：将 front matter 中多行字段的缩进统一改为 2 个空格，问题解决。

**还要做什么**：添加最新论文到 Publications 部分。
```

### 2. Git 提交规范（必须遵守）

每完成一个小任务后，**必须执行 `git add & commit & push`**，以保持细粒度的提交记录。

- 提交频率：每个独立小任务完成后立即提交，不要积累多个任务再一起提交
- Commit message 格式（中文）：简短描述本次改动，例如：
  - `更新 About Me 研究方向描述`
  - `新增博客：Spike 基础模型综述`
  - `修复博客 CN/EN 切换按钮样式`
- Push 目标：`origin master`

```bash
git add <具体文件>   # 优先指定文件，避免误提交
git commit -m "简短中文描述"
git push origin master
```

> 注意：每次 push 后 GitHub Pages 会自动重新构建部署，通常 1-2 分钟后生效。

### 4. Push 后验证部署（必须遵守）

每次 push 到 `origin master` 后，**必须验证 GitHub Actions 部署是否成功**，部署成功后才算任务完成：

1. 等待约 60 秒后，访问 `https://github.com/Jp-17/Jp-17.github.io/actions` 查看最新 workflow run 状态
2. 确认 build job **实际成功**（注意：run 列表显示 ✓ 不一定代表真正成功，需检查 run 详情中 build 和 deploy 两个 job 是否都成功；耗时 < 20s 的 run 通常是失败/跳过）
3. 确认部署生效：访问 `https://jp-17.github.io/` 检查内容是否更新
4. 如果部署失败，**必须排查原因并修复**，直到部署成功后才能结束任务

---

## 注意事项

- `references/` 目录已在 `_config.yml` 的 `exclude` 中，**不会被部署到网站**，用于存放本地资料
- 图片请放在 `images/` 目录下，引用路径为 `images/xxx.jpg`（在 HTML 中用 `{{ '/images/xxx.jpg' | relative_url }}`）
- 不要随意修改 `Gemfile` 或 `Gemfile.lock`，除非明确需要更新依赖
- 本项目无测试套件，修改后通过本地 `bundle exec jekyll serve` 预览验证
- 修改 `_layouts/` 或 `assets/css/style.scss` 时，需重启 `jekyll serve` 才能生效（CSS 热重载，但 layout 变更需重启）
