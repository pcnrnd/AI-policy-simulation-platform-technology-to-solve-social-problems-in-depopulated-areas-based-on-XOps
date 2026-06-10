// 보고서 Export 모듈 — 무의존성으로 실제 .docx / .xlsx(OOXML) 및 HWP 호환 HTML을 생성한다.
// 문서 모델(blocks)을 받아 각 포맷으로 직렬화. app/ exporters 의 "라이브러리 없는 OOXML 직접 생성" 방식과 동일.

import { createZip, escapeXml } from "./zip.js";

const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const XLSX_MIME =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

const XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>';

/* ------------------------------------------------------------------ *
 * 문서 모델
 *   block = { type: "heading" | "paragraph" | "list", level?, text?, items? }
 * ------------------------------------------------------------------ */

/* --------------------------- DOCX --------------------------- */

function docxParagraph(text, { bold = false, size = 22, spacing = 120 } = {}) {
  const runProps = bold ? "<w:rPr><w:b/><w:sz w:val=\"" + size + "\"/></w:rPr>" : `<w:rPr><w:sz w:val="${size}"/></w:rPr>`;
  return (
    `<w:p><w:pPr><w:spacing w:before="${spacing}" w:after="${spacing}"/></w:pPr>` +
    `<w:r>${runProps}<w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r></w:p>`
  );
}

function blockToDocx(block) {
  if (block.type === "heading") {
    const size = block.level === 1 ? 36 : 28;
    return docxParagraph(block.text, { bold: true, size, spacing: 180 });
  }
  if (block.type === "list") {
    return (block.items || [])
      .map((item) => docxParagraph("•  " + item, { size: 22, spacing: 60 }))
      .join("");
  }
  return docxParagraph(block.text || "", { size: 22 });
}

/**
 * @param {{ title: string, blocks: object[] }} doc
 * @returns {Blob} 실제 Word가 여는 .docx
 */
export function buildDocx({ title, blocks }) {
  const body =
    docxParagraph(title, { bold: true, size: 40, spacing: 240 }) +
    blocks.map(blockToDocx).join("");

  const documentXml =
    XML_DECL +
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">' +
    `<w:body>${body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>` +
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>';

  const contentTypes =
    XML_DECL +
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
    '<Default Extension="xml" ContentType="application/xml"/>' +
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' +
    "</Types>";

  const rels =
    XML_DECL +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>' +
    "</Relationships>";

  return createZip(
    [
      { name: "[Content_Types].xml", content: contentTypes },
      { name: "_rels/.rels", content: rels },
      { name: "word/document.xml", content: documentXml }
    ],
    DOCX_MIME
  );
}

/* --------------------------- XLSX --------------------------- */

function columnLetter(index) {
  let n = index;
  let letter = "";
  do {
    letter = String.fromCharCode(65 + (n % 26)) + letter;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return letter;
}

function xlsxCell(ref, value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `<c r="${ref}"><v>${value}</v></c>`;
  }
  return `<c r="${ref}" t="inlineStr"><is><t xml:space="preserve">${escapeXml(value)}</t></is></c>`;
}

/**
 * @param {{ sheetName?: string, rows: (string|number)[][] }} sheet
 * @returns {Blob} 실제 Excel이 여는 .xlsx
 */
export function buildXlsx({ sheetName = "Sheet1", rows }) {
  const rowsXml = rows
    .map((cells, r) => {
      const cellsXml = cells
        .map((value, c) => xlsxCell(`${columnLetter(c)}${r + 1}`, value))
        .join("");
      return `<row r="${r + 1}">${cellsXml}</row>`;
    })
    .join("");

  const sheetXml =
    XML_DECL +
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">' +
    `<sheetData>${rowsXml}</sheetData></worksheet>`;

  const workbookXml =
    XML_DECL +
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ' +
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' +
    `<sheets><sheet name="${escapeXml(sheetName).slice(0, 31)}" sheetId="1" r:id="rId1"/></sheets></workbook>`;

  const workbookRels =
    XML_DECL +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>' +
    "</Relationships>";

  const contentTypes =
    XML_DECL +
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
    '<Default Extension="xml" ContentType="application/xml"/>' +
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' +
    "</Types>";

  const rels =
    XML_DECL +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>' +
    "</Relationships>";

  return createZip(
    [
      { name: "[Content_Types].xml", content: contentTypes },
      { name: "_rels/.rels", content: rels },
      { name: "xl/workbook.xml", content: workbookXml },
      { name: "xl/_rels/workbook.xml.rels", content: workbookRels },
      { name: "xl/worksheets/sheet1.xml", content: sheetXml }
    ],
    XLSX_MIME
  );
}

/* --------------------------- HWP (Hancom HTML) --------------------------- */

function blockToHtml(block) {
  if (block.type === "heading") {
    const tag = block.level === 1 ? "h1" : "h2";
    return `<${tag}>${escapeXml(block.text)}</${tag}>`;
  }
  if (block.type === "list") {
    return "<ul>" + (block.items || []).map((i) => `<li>${escapeXml(i)}</li>`).join("") + "</ul>";
  }
  return `<p>${escapeXml(block.text || "")}</p>`;
}

/**
 * 한컴오피스(아래아한글)가 직접 여는 HTML 기반 HWP 호환 문서.
 * 진짜 바이너리 HWP 포맷은 사양이 방대하여, app/ 의 "HWP via Hancom-HTML" 전략을 따른다.
 * @returns {Blob}
 */
export function buildHwpHtml({ title, blocks }) {
  const bodyHtml = blocks.map(blockToHtml).join("\n");
  const html =
    "<!DOCTYPE html><html lang=\"ko\"><head><meta charset=\"UTF-8\">" +
    `<title>${escapeXml(title)}</title>` +
    "<style>body{font-family:'맑은 고딕',sans-serif;line-height:1.6;color:#111;}" +
    "h1{font-size:18pt;}h2{font-size:14pt;color:#1a3a6b;}p,li{font-size:11pt;}</style></head><body>" +
    `<h1>${escapeXml(title)}</h1>${bodyHtml}</body></html>`;
  return new Blob([html], { type: "application/x-hwp" });
}

/* --------------------------- 공통 다운로드 --------------------------- */

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
