import test from "node:test";
import assert from "node:assert/strict";
import { focusZoom, LABEL_H, labelWidth, placeLabels } from "./labels.js";

test("고른 곳으로 옮길 확대 단계: 원이 범례에 걸리지 않는 가장 큰 단계, 없으면 null", () => {
  // 단계 z에서 1도 = 2^z px인 단순 투영, 화면 가운데(200, 150)에 target, 범례는 왼쪽 아래 (0,200)~(100,300)
  const project = (p, z) => ({ x: p.lng * 2 ** z, y: -p.lat * 2 ** z });
  const opts = { zooms: [6, 5, 4, 3], spots: [[200, 150]], project, blocked: [{ x: 0, y: 200, w: 100, h: 100 }] };
  const target = { lat: 0, lng: 0 };
  // 왼쪽 아래 2도 떨어진 원: 6단계(64px/도)면 (72, 278)로 범례 안, 5단계(32px/도)면 (136, 214)로 범례 오른쪽 밖
  const sw = { lat: -2, lng: -2, r: 5 };
  assert.deepEqual(focusZoom(target, [sw], opts), { zoom: 5, x: 200, y: 150 });
  assert.equal(focusZoom(target, [sw], { ...opts, zooms: [6] }), null);
  // 둘째 자리(오른쪽으로 100px)에 두면 6단계에서도 원이 (172, 278) — 범례 밖
  assert.deepEqual(focusZoom(target, [sw], { ...opts, spots: [[200, 150], [300, 150]] }), { zoom: 6, x: 300, y: 150 });
  assert.deepEqual(focusZoom(target, [], opts), { zoom: 6, x: 200, y: 150 });
});

const box = (it, p) => ({ x: it.x + p.dx - it.w / 2, y: it.y + p.dy - it.h / 2, w: it.w, h: it.h });
const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

test("라벨 폭: 숫자·쉼표 7px, '명' 14px, 여백 8px", () => {
  assert.equal(labelWidth("839명"), 3 * 7 + 14 + 8);
  assert.equal(labelWidth("1,234명"), 5 * 7 + 14 + 8);
});

test("라벨 자리: 큰 값이 가운데, 겹치면 위·아래·옆으로, 자리가 없으면 뺀다, 결과끼리 안 겹친다", () => {
  const it = (id, x, y, rank) => ({ id, x, y, r: 10, w: 40, h: LABEL_H, rank });
  const items = [it("a", 100, 100, 1), it("b", 105, 100, 9), it("c", 300, 300, 5)];
  const out = placeLabels(items);
  const by = Object.fromEntries(out.map((p) => [p.id, p]));
  assert.deepEqual([by.b.dx, by.b.dy], [0, 0]); // 큰 값 먼저, 가운데
  assert.deepEqual([by.a.dx, by.a.dy], [0, -(10 + LABEL_H / 2 + 2)]); // 겹쳐서 위로
  assert.deepEqual([by.c.dx, by.c.dy], [0, 0]);
  const boxes = out.map((p) => box(items.find((i) => i.id === p.id), p));
  boxes.forEach((a, i) => boxes.slice(i + 1).forEach((b) => assert.ok(!overlaps(a, b))));
  // 가림 영역(범례 등)과 겹치는 자리는 쓰지 않고, 사방이 막히면 뺀다
  assert.deepEqual(placeLabels([it("z", 50, 50, 1)], [{ x: 0, y: 0, w: 200, h: 200 }]), []);
  const crowd = Array.from({ length: 8 }, (_, i) => it(`k${i}`, 100, 100, i));
  assert.equal(placeLabels(crowd).length, 3); // 가운데·위·아래(옆 자리는 가운데 라벨과 겹침)
});
