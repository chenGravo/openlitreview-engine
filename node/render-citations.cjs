"use strict";

const fs = require("fs");
const path = require("path");
const CSL = require("citeproc");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(2);
}

if (process.argv.length !== 7) {
  fail(
    "Usage: node render-citations.cjs INPUT.md REFERENCES.json STYLE.csl ASSET_DIR OUTPUT.md"
  );
}

const [, , markdownPath, referencesPath, stylePath, assetDirectory, outputPath] =
  process.argv;
const markdown = fs.readFileSync(markdownPath, "utf8");
const items = JSON.parse(fs.readFileSync(referencesPath, "utf8"));
const style = fs.readFileSync(stylePath, "utf8");
const itemMap = Object.fromEntries(items.map((item) => [String(item.id), item]));

const sys = {
  retrieveLocale(language) {
    const requested = path.join(assetDirectory, `locales-${language}.xml`);
    const fallback = path.join(assetDirectory, "locales-zh-CN.xml");
    return fs.readFileSync(fs.existsSync(requested) ? requested : fallback, "utf8");
  },
  retrieveItem(id) {
    const item = itemMap[String(id)];
    if (!item) {
      throw new Error(`Unknown citation key: ${id}`);
    }
    return item;
  },
};

const engine = new CSL.Engine(sys, style);
engine.opt.development_extensions.wrap_url_and_doi = true;
const citationPattern = /\[((?:[^\]\n]*@[A-Za-z0-9_.:-]+[^\]\n]*))\]/g;
const matches = [...markdown.matchAll(citationPattern)];
const renderedCitations = [];
const prior = [];

for (let index = 0; index < matches.length; index += 1) {
  const ids = [...matches[index][1].matchAll(/@([A-Za-z0-9_.:-]+)/g)].map(
    (match) => match[1]
  );
  if (ids.length === 0) {
    renderedCitations[index] = matches[index][0];
    continue;
  }
  for (const id of ids) {
    if (!itemMap[id]) {
      fail(`Unknown citation key: ${id}`);
    }
  }
  const citationId = `CITATION-${index + 1}`;
  const citation = {
    citationID: citationId,
    citationItems: ids.map((id) => ({ id })),
    properties: { noteIndex: index + 1 },
  };
  const updates = engine.processCitationCluster(citation, prior, [])[1];
  for (const [citationIndex, text] of updates) {
    renderedCitations[citationIndex] = htmlToText(text);
  }
  prior.push([citationId, index + 1]);
}

let citationIndex = 0;
let renderedMarkdown = markdown.replace(citationPattern, () => {
  const rendered = renderedCitations[citationIndex];
  citationIndex += 1;
  return rendered;
});

if (matches.length > 0) {
  const bibliography = engine.makeBibliography();
  if (!bibliography || !Array.isArray(bibliography[1])) {
    fail("citeproc-js did not produce a bibliography");
  }
  const entries = bibliography[1]
    .map((entry) => htmlToText(entry).replace(/\s+/g, " ").trim())
    .filter(Boolean);
  renderedMarkdown = renderedMarkdown.trimEnd();
  renderedMarkdown += "\n\n## 参考文献\n\n";
  renderedMarkdown += `${entries.join("\n\n")}\n`;
}

fs.writeFileSync(outputPath, renderedMarkdown, "utf8");

function htmlToText(value) {
  return String(value)
    .replace(/<br\s*\/?\s*>/gi, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#([0-9]+);/g, (_, number) => String.fromCodePoint(parseInt(number, 10)))
    .trim();
}

