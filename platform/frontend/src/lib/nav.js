const GROUP_EXPANDED_PREFIX = "nav-group-expanded:";

/**
 * 사이드바 섹션에서 그룹 헤더를 빼고 실제 화면 탭만 순서대로 반환한다.
 * 패널 렌더·키보드 이동·헤더 제목이 같은 순서를 쓰도록 App과 Sidebar가 공유한다.
 */
export function flattenNavTabs(sections) {
  return sections.flatMap((section) => section.tabs);
}

/**
 * 접힌 그룹의 자식은 빼고, 지금 사이드바에 보이는 탭만 반환한다.
 */
export function visibleNavTabs(sections, expanded) {
  return sections.flatMap((section) => {
    if (section.label && expanded[section.id] === false) return [];
    return section.tabs;
  });
}

/**
 * 그룹 펼침 여부를 localStorage에서 읽는다. 값이 없으면 기본은 펼침이다.
 */
export function readNavGroupExpanded(sectionId) {
  if (typeof window === "undefined") return true;
  return window.localStorage.getItem(GROUP_EXPANDED_PREFIX + sectionId) !== "false";
}

/**
 * 그룹 펼침 여부를 localStorage에 저장한다.
 */
export function writeNavGroupExpanded(sectionId, isExpanded) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(GROUP_EXPANDED_PREFIX + sectionId, String(isExpanded));
}
