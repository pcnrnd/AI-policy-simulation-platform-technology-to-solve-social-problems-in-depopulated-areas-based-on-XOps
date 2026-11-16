import { useEffect, useState } from "react";
import { getRealdataToken } from "../components/realdata/realdataClient.jsx";
import { apiGet } from "../lib/api.js";

const BASE = "/api/v3/public-stats";
const EMPTY = { population: null, spots: null };

/**
 * 정책 추천 그래프의 공표 통계 — 지역 생활인구 분기 값과 관광지점 월별 입장객을 플랫폼 API에서 읽는다.
 * 데모 표시·실데이터 표시 모두 같은 경로(시드에 공표 값 없음). 조회 실패·빈 결과는 null이고,
 * 그래프 자리에는 기존 행동 안내가 나온다.
 * @param {string|undefined} regionName 지역 이름(예: 전북특별자치도 남원시)
 * @param {string[]} spots 통계표 관광지점 이름
 * @returns {{ population: object|null, spots: Record<string, {ym: string, value: number}[]>|null }}
 */
export function usePublicTrends(regionName, spots) {
  const [trends, setTrends] = useState(EMPTY);
  const spotKey = spots.join("|");
  useEffect(() => {
    let alive = true;
    setTrends(EMPTY);
    // 실패 원문은 따로 남기지 않는다 — 화면은 빈 그래프 자리의 행동 안내로 상태를 보여 준다.
    const read = (path, params, token) => apiGet(`${BASE}/${path}`, { token, params }).then((res) => (res?.status === "ok" ? res.data : null), () => null);
    (async () => {
      const token = await getRealdataToken().catch(() => null);
      if (!token || !alive) return;
      const names = spotKey ? spotKey.split("|") : [];
      const [population, visitors] = await Promise.all([
        regionName ? read("living-population", { region: regionName }, token) : null,
        names.length ? read("spot-visitors", { spot: names }, token) : null
      ]);
      if (!alive) return;
      setTrends({
        population: population?.points?.length ? population : null,
        spots: visitors ? Object.fromEntries(visitors.spots.map((s) => [s.spot_name, s.points])) : null
      });
    })();
    return () => { alive = false; };
  }, [regionName, spotKey]);
  return trends;
}
