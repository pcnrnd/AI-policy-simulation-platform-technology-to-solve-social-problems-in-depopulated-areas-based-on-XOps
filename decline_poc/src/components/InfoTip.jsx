// 회색 물음표 아이콘 + 커스텀 툴팁 — hover/focus 시 설명을 표시한다.
// 네이티브 title의 지연·OS 스타일을 피하고, 키보드 포커스(전자정부 표준 접근성)에도 대응.
export default function InfoTip({ text, label = "설명 보기" }) {
  return (
    <span className="infotip" tabIndex={0} role="note" aria-label={`${label}: ${text}`}>
      <i className="fa-solid fa-circle-question" aria-hidden="true"></i>
      <span className="infotip-bubble" role="tooltip">
        {text}
      </span>
    </span>
  );
}
