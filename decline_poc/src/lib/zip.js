// 무의존성 ZIP 작성기 (store-only / 무압축).
// 외부 라이브러리 없이 브라우저에서 유효한 .docx/.xlsx(OOXML) 컨테이너를 생성하기 위한 모듈.
// app/ 의 exporters 철학(python-docx/openpyxl 없이 zipfile+OOXML 직접 생성)을 클라이언트로 이식.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

const encoder = new TextEncoder();

function toBytes(content) {
  return content instanceof Uint8Array ? content : encoder.encode(content);
}

// ZIP 로컬 헤더/중앙 디렉터리에 쓰이는 DOS 시간·날짜 (고정값으로 결정성 유지: 2026-01-01 00:00).
const DOS_TIME = 0;
const DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1;

/**
 * 파일 엔트리 목록을 store(무압축) 방식의 ZIP Blob으로 직렬화한다.
 * @param {{ name: string, content: string | Uint8Array }[]} files
 * @param {string} mimeType - 결과 Blob MIME (docx/xlsx 등)
 * @returns {Blob}
 */
export function createZip(files, mimeType = "application/zip") {
  const localParts = [];
  const centralParts = [];
  let offset = 0;

  files.forEach((file) => {
    const nameBytes = toBytes(file.name);
    const dataBytes = toBytes(file.content);
    const crc = crc32(dataBytes);
    const size = dataBytes.length;

    const local = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true); // local file header signature
    lv.setUint16(4, 20, true); // version needed
    lv.setUint16(6, 0, true); // flags
    lv.setUint16(8, 0, true); // method = store
    lv.setUint16(10, DOS_TIME, true);
    lv.setUint16(12, DOS_DATE, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, size, true); // compressed size
    lv.setUint32(22, size, true); // uncompressed size
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true); // extra length
    local.set(nameBytes, 30);

    localParts.push(local, dataBytes);

    const central = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(central.buffer);
    cv.setUint32(0, 0x02014b50, true); // central dir signature
    cv.setUint16(4, 20, true); // version made by
    cv.setUint16(6, 20, true); // version needed
    cv.setUint16(8, 0, true); // flags
    cv.setUint16(10, 0, true); // method
    cv.setUint16(12, DOS_TIME, true);
    cv.setUint16(14, DOS_DATE, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, size, true);
    cv.setUint32(24, size, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true); // extra len
    cv.setUint16(32, 0, true); // comment len
    cv.setUint16(34, 0, true); // disk number
    cv.setUint16(36, 0, true); // internal attrs
    cv.setUint32(38, 0, true); // external attrs
    cv.setUint32(42, offset, true); // local header offset
    central.set(nameBytes, 46);
    centralParts.push(central);

    offset += local.length + dataBytes.length;
  });

  const centralSize = centralParts.reduce((sum, p) => sum + p.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true); // EOCD signature
  ev.setUint16(8, files.length, true); // entries on disk
  ev.setUint16(10, files.length, true); // total entries
  ev.setUint32(12, centralSize, true); // central dir size
  ev.setUint32(16, offset, true); // central dir offset
  ev.setUint16(20, 0, true); // comment len

  return new Blob([...localParts, ...centralParts, eocd], { type: mimeType });
}

/** XML/텍스트의 특수문자를 이스케이프한다. */
export function escapeXml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}
