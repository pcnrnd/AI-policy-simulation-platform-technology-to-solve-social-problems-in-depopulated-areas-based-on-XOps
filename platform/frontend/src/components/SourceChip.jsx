import { useAppState } from "../context/AppStateContext.jsx";
import { sourceChipText } from "../lib/dataMode.js";

// 카드·값 단위 출처 칩 — 현재 모드와 값의 전달 경로(dataSource)로 "데모 / 데모 포함 / 실데이터"를 보인다.
export default function SourceChip({ dataSource, sourceLabel }) {
  const { mockDataVisible } = useAppState();
  const text = sourceChipText(mockDataVisible, dataSource, sourceLabel);
  if (!text) return null;
  const tone = text.startsWith("실데이터") ? "real" : "demo";
  return <span className={`source-chip is-${tone}`}>{text}</span>;
}
