import assert from "node:assert/strict";
import { parseIntegrationConfig, moduleNavSections, firstInternalTab } from "./moduleNavigation.js";
const sections = [{ tabs: [{ id: "tab-overview" }, { id: "tab-dataops" }] },
  { tabs: [{ id: "tab-simulator" }, { id: "tab-reporter" }] }];
const raw = { schema: 1, source: "ingress-env", appPath: "/rnd/population/", modules: {
  dashboard: { configured: false, apiConfigured: false, path: "/rnd/population/dashboard/", apiPath: "/rnd/population/dashboard-api/", supportedModes: [] },
  reporting: { configured: false, apiConfigured: false, path: "/rnd/population/reporting/", apiPath: "/rnd/population/reporting-api/", supportedModes: [] }
} };
const inactive = parseIntegrationConfig(raw);
assert.deepEqual(moduleNavSections(sections, inactive, true), sections);
assert.deepEqual(moduleNavSections(sections, null, false), sections);
assert.equal(firstInternalTab(sections), "tab-overview");
const active = parseIntegrationConfig({ ...raw, modules: Object.fromEntries(Object.entries(raw.modules).map(([id, module]) => [id, { ...module, configured: true, supportedModes: ["mock", "real"] }])) });
const on = moduleNavSections(sections, active, true);
const off = moduleNavSections(sections, active, false);
assert.equal(on[0].tabs[0].href, "/rnd/population/dashboard/?mode=mock");
assert.equal(off[1].tabs[1].href, "/rnd/population/reporting/?mode=real");
assert.strictEqual(on[0].tabs[1], sections[0].tabs[1]);
assert.strictEqual(on[1].tabs[0], sections[1].tabs[0]);
assert.equal(firstInternalTab(on), "tab-dataops");
// Configured but support-unconfirmed still goes to the module's readiness route.
const unsupported = parseIntegrationConfig({ ...raw, modules: { ...raw.modules, dashboard: { ...raw.modules.dashboard, configured: true } } });
assert.equal(moduleNavSections(sections, unsupported, true)[0].tabs[0].href, "/rnd/population/dashboard/?mode=mock");
assert.equal(moduleNavSections(sections, unsupported, true)[1].tabs[1].href, undefined);
for (const path of ["https://partner.invalid/", "//partner.invalid/", "/api/", "/rnd/population/../evil/", "/rnd/population/dashboard/?mode=real", "/rnd/population/dashboard/#fragment"]) {
  assert.throws(() => parseIntegrationConfig({ ...raw, modules: { ...raw.modules, dashboard: { ...raw.modules.dashboard, path } } }));
}
assert.throws(() => parseIntegrationConfig({ ...raw, modules: { ...raw.modules, reporting: { ...raw.modules.reporting, path: raw.modules.dashboard.path } } }));
assert.throws(() => parseIntegrationConfig({ ...raw, modules: { ...raw.modules, dashboard: { ...raw.modules.dashboard, configured: "true" } } }));
assert.throws(() => parseIntegrationConfig({ ...raw, modules: { ...raw.modules, dashboard: { ...raw.modules.dashboard, supportedModes: ["unknown"] } } }));
assert.throws(() => parseIntegrationConfig({ ...raw, schema: 2 }));
console.log("moduleNavigation: unset internal tabs, configured two links + ON/OFF mode, unsupported readiness path, origin/prefix validation PASS");
