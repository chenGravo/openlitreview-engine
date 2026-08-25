# OpenLitReview

OpenLitReview 是一个按需启动、证据优先、可审计的中文学术文献综述引擎。每次任务动态确定学科与方向，重点检索外文学术文献；程序负责文献身份、去重、排序、开放全文、引用和质量审计，大模型只处理检索词扩展、证据提取、跨文献综合和中文写作。

## 首版边界

- 运行环境：私人 GitHub 仓库中的 GitHub Actions，手动启动，结束后自动停止。
- 学术来源：匿名访问 Crossref、Semantic Scholar 和 Europe PMC；后续仅增加无需账户身份凭据的合法开放来源。
- 数据范围：默认只处理公开元数据和合法开放全文。
- 语言：检索外文文献，输出中文普通叙述性综述。
- 引用：正文使用 Pandoc citation keys，最终通过固定版本的 CSL 样式生成 GB/T 7714—2025 顺序编码制。
- 模型：DeepSeek、Kimi、豆包通过统一适配层调用，主模型由盲测决定。
- 预算：三家供应商每自然月 80 元预警、90 元内部停止，保留 10 元外部账单缓冲。
- 质量闸门：默认至少 30 条保留记录、15 篇证据文献、5 篇合法全文核验文献和
  12 篇正文实际引用文献；未达标时只交付审计材料，不生成“合格”成品。
- 双模型复核：第二模型判定需修订时自动返修一次并再次复核，仍未通过则阻断文档渲染。

## 本地或云端使用

```bash
python -m pip install -e .
corepack pnpm@11.19.0 --dir node install --frozen-lockfile --prod
openlitreview validate examples/task.example.yml
openlitreview search examples/task.example.yml --output runs/demo
```

正式使用由配套的私人 workspace 模板执行。模板不会配置定时任务或常驻服务，只有用户手动触发时才消耗 GitHub Actions 分钟和模型 API 额度。

每次任务可以在 YAML 中直接填写 `user_requirements`，也可以通过
`writing_requirements_file` 读取同一任务目录内的 Markdown、TXT、DOCX 或 PDF 要求文件。
研究方向、关键词、年份、排除词、篇幅和质量门槛均按任务动态设置，不绑定任何固定学科。

## 安全原则

- 文献检索请求不附带用户姓名、邮箱、电话号码、GitHub 用户名、仓库名或账户关联型学术 API 密钥；明显包含邮箱、手机号或身份证号的检索词会被拒绝。
- 文献网站仍会收到完成检索所必需的学术检索词、DOI/文献网址、请求时间和 GitHub 托管运行器的公网 IP；这些是联网检索不可避免的协议数据。
- OpenAlex 当前要求账户关联型 API 密钥，因此在匿名模式中停用；系统不会读取或发送 OpenAlex、Semantic Scholar 的个人 API 密钥。
- 模型不能创建正式文献记录；参考文献必须来自检索库或用户合法导入。
- 不抓取或绕过受限数据库、验证码、账号和付费墙。
- 不在日志中输出 API 密钥、全文、完整提示词和模型完整响应。
- 正文不自动插入模型名称或技术说明；私人审计记录与稿件分开保存。
- 不提供 AI 检测规避、法定标识删除或违反适用披露规则的功能。
- 价格只采用供应商官方页面核验的快照；详见 `PRICING_SOURCES.md`。未知模型直接拒绝调用。

## 许可证

核心代码按 AGPL-3.0-or-later 发布。第三方依赖、模型和 CSL 样式各自适用其原许可证，详见 `THIRD_PARTY_NOTICES.md`。
