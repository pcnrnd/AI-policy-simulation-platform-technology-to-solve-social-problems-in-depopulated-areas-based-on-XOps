// 지도 숫자 라벨 자리 잡기와 목록에서 고른 곳으로 옮길 확대 단계(표시용). 원이 숫자·범례에 가리지 않게 한다.

// 라벨 글자 폭(px) — styles.css .pop-label(고정폭 11px)과 맞춘다. 한글 '명'은 숫자 두 칸으로 센다.
const DIGIT_PX = 7;
const PAD_PX = 8;
export const LABEL_H = 16;
export const labelWidth = (text) => [...text].reduce((w, ch) => w + (ch === "명" ? 2 * DIGIT_PX : DIGIT_PX), PAD_PX);

const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/**
 * 값이 큰 라벨부터 원 가운데 → 위 → 아래 → 오른쪽 → 왼쪽 순으로 놓아 보고, 이미 놓인 라벨이나 가림 영역(범례·조건 카드)과
 * 겹치지 않는 첫 자리를 쓴다. 다섯 자리가 모두 겹치면 그 라벨은 뺀다.
 * items = [{ id, x, y, r, w, h, rank }] (지도 컨테이너 px, x·y = 원 중심, r = 원 반지름), blocked = [{ x, y, w, h }]
 * 반환 = [{ id, dx, dy }] (라벨 중심의 원 중심 대비 이동 px)
 */
export function placeLabels(items, blocked = [], gap = 2) {
  const taken = [...blocked];
  const placed = [];
  for (const it of [...items].sort((a, b) => b.rank - a.rank)) {
    const dy = it.r + it.h / 2 + gap;
    const dx = it.r + it.w / 2 + gap;
    const spot = [[0, 0], [0, -dy], [0, dy], [dx, 0], [-dx, 0]]
      .map(([ox, oy]) => ({ ox, oy, box: { x: it.x + ox - it.w / 2, y: it.y + oy - it.h / 2, w: it.w, h: it.h } }))
      .find((s) => !taken.some((t) => overlaps(s.box, t)));
    if (!spot) continue;
    taken.push(spot.box);
    placed.push({ id: it.id, dx: spot.ox, dy: spot.oy });
  }
  return placed;
}

/**
 * 목록에서 고른 곳(target)으로 옮길 때의 확대 단계와 화면 자리: zooms(큰 것부터)마다 spots([[x, y]], 앞이 우선)에 target을 두어 보고,
 * 어떤 원(circles = [{ lat, lng, r }])도 가림 영역(blocked = [{ x, y, w, h }], 범례)에 걸리지 않는 첫 조합 { zoom, x, y }. 없으면 null.
 * project(점, 단계) = { x, y } 세계 px(Leaflet map.project와 같은 꼴).
 */
export function focusZoom(target, circles, { zooms, spots, project, blocked }) {
  for (const zoom of zooms) {
    const t = project(target, zoom);
    const pts = circles.map((c) => ({ ...project(c, zoom), r: c.r }));
    const spot = spots.find(([x, y]) => !pts.some((p) => {
      const box = { x: p.x - t.x + x - p.r, y: p.y - t.y + y - p.r, w: 2 * p.r, h: 2 * p.r };
      return blocked.some((b) => overlaps(box, b));
    }));
    if (spot) return { zoom, x: spot[0], y: spot[1] };
  }
  return null;
}
