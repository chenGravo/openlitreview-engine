# Third-party notices

This file records the Phase-0 dependency baseline. Direct and transitive Python versions and the
Node dependency graph are locked in repository files; release provenance must retain those locks.

| Component | Intended use | License |
|---|---|---|
| httpx | HTTP client | BSD-3-Clause |
| Pydantic | Typed validation | MIT |
| PyYAML | Task configuration | MIT |
| pypdf | Lawfully obtained PDF text extraction | BSD-3-Clause |
| python-docx | DOCX post-processing and fallback | MIT |
| Pandoc | Document conversion as a separate executable | GPL-2.0-or-later |
| citeproc-js 2.4.63 | CSL-M citation rendering as a separate Node.js process | CPAL-1.0-or-later OR AGPL-3.0-or-later |
| zotero-chinese/styles | GB/T 7714—2025 CSL-M asset | CC BY-SA 3.0 |
| BGE-M3 | Optional semantic retrieval model | MIT |
| BGE Reranker v2 M3 | Optional reranking model | Apache-2.0 |
| Noto CJK | Cloud DOCX/PDF Chinese font rendering; installed by the workspace, not bundled here | SIL Open Font License 1.1 |

The bundled GB/T style is copied from `zotero-chinese/styles` only after pinning an upstream commit. Its attribution and original license must remain adjacent to the asset.
The citeproc-js upstream notice is retained at `node/CITEPROC_NOTICE.txt`.
