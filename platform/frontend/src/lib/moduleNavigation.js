const MODULE_IDS = { "tab-overview": "dashboard", "tab-reporter": "reporting" };
const validPath = (value) => typeof value === "string" && /^\/(?:[A-Za-z0-9_.-]+\/)+$/.test(value)
  && !/(?:\/\.\.?\/|\/\/)/.test(value) && !/^\/(?:api|assets|authelia)\//.test(value);

export function parseIntegrationConfig(raw) {
  if (raw?.schema !== 1 || raw.source !== "ingress-env" || !validPath(raw.appPath)) throw new Error("모듈 연결 설정을 확인하지 못했습니다.");
  const modules = {};
  const paths = [];
  for (const id of ["dashboard", "reporting"]) {
    const module = raw.modules?.[id];
    if (typeof module?.configured !== "boolean" || typeof module.apiConfigured !== "boolean"
      || ![module.path, module.apiPath].every((path) => validPath(path) && path.startsWith(raw.appPath) && path !== raw.appPath)
      || !Array.isArray(module.supportedModes) || module.supportedModes.some((mode) => !["mock", "real"].includes(mode))
      || new Set(module.supportedModes).size !== module.supportedModes.length) throw new Error("모듈 연결 설정을 확인하지 못했습니다.");
    for (const path of [module.path, module.apiPath]) {
      if (paths.some((other) => path.startsWith(other) || other.startsWith(path))) throw new Error("모듈 경로가 겹칩니다.");
      paths.push(path);
    }
    modules[id] = { configured: module.configured, apiConfigured: module.apiConfigured, path: module.path,
      apiPath: module.apiPath, supportedModes: [...module.supportedModes] };
  }
  return { appPath: raw.appPath, modules };
}

export function moduleNavSections(sections, config, isMock) {
  return sections.map((section) => ({ ...section, tabs: section.tabs.map((tab) => {
    const module = config?.modules[MODULE_IDS[tab.id]];
    return module?.configured ? { ...tab, href: `${module.path}?mode=${isMock ? "mock" : "real"}` } : tab;
  }) }));
}

export function firstInternalTab(sections) {
  return sections.flatMap((section) => section.tabs).find((tab) => !tab.href)?.id;
}
