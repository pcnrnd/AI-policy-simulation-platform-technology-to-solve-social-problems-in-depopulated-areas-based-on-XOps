import { useEffect, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet } from "../lib/api.js";

const BASE = "/api/v3/public-stats";
const EMPTY = { population: null };

/**
 * 정책 추천 그래프의 공표 통계 — 지역 생활인구 분기 값을 플랫폼 API에서 읽는다.
 * 데모 표시·실데이터 표시 모두 같은 경로(시드에 공표 값 없음). 조회 실패·빈 결과는 null이고,
 * 그래프 자리에는 기존 행동 안내가 나온다.
 * @param {string|undefined} regionName 지역 이름(예: 전북특별자치도 남원시)
 * @returns {{ population: object|null }}
 */
export function usePublicTrends(regionName) {
  const [trends, setTrends] = useState(EMPTY);
  useEffect(() => {
    let alive = true;
    setTrends(EMPTY);
    // 실패 원문은 따로 남기지 않는다 — 화면은 빈 그래프 자리의 행동 안내로 상태를 보여 준다.
    const read = (path, params, token) => apiGet(`${BASE}/${path}`, { token, params }).then((res) => (res?.status === "ok" ? res.data : null), () => null);
    (async () => {
      const token = await getRealdataToken().catch(() => null);
      if (!token || !alive) return;
      const population = regionName ? await read("living-population", { region: regionName }, token) : null;
      if (!alive) return;
      setTrends({ population: population?.points?.length ? population : null });
    })();
    return () => { alive = false; };
  }, [regionName]);
  return trends;
}

/**
 * 지역 월평균 체류인구(시설 추천 계산의 지역 규모)와 그 기간 — 플랫폼 공표 통계 API에서 읽는다(시드에 값 없음).
 * 받지 못하면 null이고, 추천 구역은 같은 자리에 자료 등록 안내를 보인다.
 * @param {string|undefined} regionName
 * @returns {{ value: number, period_from: string, period_to: string }|null}
 */
export function useRegionStay(regionName) {
  const [stay, setStay] = useState(null);
  useEffect(() => {
    let alive = true;
    setStay(null);
    (async () => {
      const token = await getRealdataToken().catch(() => null);
      if (!token || !alive || !regionName) return;
      const res = await apiGet(`${BASE}/stay-average`, { token, params: { region: regionName } }).catch(() => null);
      if (alive && res?.status === "ok" && res.data?.value > 0) setStay({ value: res.data.value, period_from: res.data.period_from, period_to: res.data.period_to });
    })();
    return () => { alive = false; };
  }, [regionName]);
  return stay;
}
