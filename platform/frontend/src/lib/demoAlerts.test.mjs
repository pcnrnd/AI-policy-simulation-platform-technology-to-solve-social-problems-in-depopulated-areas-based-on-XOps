import assert from "node:assert/strict";
import { retainRealAlerts, retainModeIndependentAlerts, notificationFromAlert, unreadNotifications, detectDemoDrift, runDemoDriftCheck } from "./demoAlerts.js";
import { setDataMode, dataModeEpoch } from "./dataMode.js";

const demo = { id: 1, source: "demo", title: "드리프트", unread: true };
const real = { id: 2, source: "real", title: "실측 경보", unread: true };
const read = { id: 3, source: "real", title: "읽은 알림", unread: false };
const session = { id: 4, system: true, title: "다시 로그인", unread: true };
const retained = retainRealAlerts([demo, real, read, session]);
assert.deepEqual(retained, [real, read, session]);
assert.equal(unreadNotifications(retained), 2);
assert.equal(unreadNotifications(retained.map((entry) => ({ ...entry, unread: false }))), 0);
assert.deepEqual(retainRealAlerts(retained), retained);
assert.deepEqual(retainRealAlerts([demo]), []);

// 실제 재로그인 팝업과 같은 생성 경로가 벨 이력에도 세션·시스템 분류를 넘긴다.
const sessionPopup = {
  id: 10, source: "session", system: true,
  title: "다시 로그인했습니다", message: "세션 안내"
};
const sessionNotice = { id: 10, unread: true, ...notificationFromAlert(sessionPopup) };
assert.equal(sessionNotice.system, true);
assert.equal(sessionNotice.source, "session");
assert.equal(sessionNotice.severity, "info");
const readSystemNotice = { id: 11, unread: false, ...notificationFromAlert({ system: true, title: "시스템 안내" }) };
const sessionBySource = { id: 12, source: "session", unread: false };
const systemBySource = { id: 13, source: "system", unread: true };
const expectedNotices = [sessionNotice, readSystemNotice, sessionBySource, systemBySource];
const mockNotice = { id: 14, source: "mock", unread: true };
const unclassifiedDataNotice = { id: 15, title: "데이터 결과", unread: true };
setDataMode("mock");
let popups = [sessionPopup, demo, mockNotice];
let notices = [...expectedNotices, demo, mockNotice, unclassifiedDataNotice];
setDataMode("real");
popups = retainModeIndependentAlerts(popups);
notices = retainModeIndependentAlerts(notices);
assert.deepEqual(popups, [sessionPopup]);
assert.deepEqual(notices, expectedNotices);
assert.equal(unreadNotifications(notices), 2);
assert.strictEqual(notices[0], sessionNotice);
assert.strictEqual(notices[1], readSystemNotice);
// real 결과가 info 라도 세션 알림으로 분류하지 않는다.
notices.push({ ...real, severity: "info" }, read);
popups.push(real);
setDataMode("mock");
popups = retainModeIndependentAlerts(popups);
notices = retainModeIndependentAlerts(notices);
assert.deepEqual(popups, [sessionPopup]);
assert.deepEqual(notices, expectedNotices);
assert.equal(unreadNotifications(notices), 2);
assert.equal(notices[1].unread, false);
const allRead = notices.map((entry) => ({ ...entry, unread: false }));
setDataMode("real");
assert.equal(unreadNotifications(retainModeIndependentAlerts(allRead)), 0);
assert.deepEqual(retainModeIndependentAlerts(), []);
console.log("mode notifications: ON→OFF→ON 세션·시스템 및 읽음 상태 보존, demo/mock/real 결과 제거 PASS");

const calculated = { source: "seed", drifted: true, psi: 0.4, kl: 0.2, psi_threshold: 0.2, kl_threshold: 0.1 };
const detected = [];
await detectDemoDrift({ loadDrift: async () => calculated, isCurrent: () => true, onDetected: (_, alert) => detected.push(alert) });
assert.equal(detected.length, 1);
assert.equal(detected[0].source, "demo");
assert.match(detected[0].message, /PSI 0.4.*임계 0.2/);
await detectDemoDrift({ loadDrift: async () => ({ ...calculated, psi: 0.01, kl: 0.01 }), isCurrent: () => true, onDetected: () => assert.fail("임계 미만 경보") });
await assert.rejects(detectDemoDrift({ loadDrift: async () => ({ source: null }), isCurrent: () => true, onDetected: () => assert.fail("판정 없는 경보") }));
setDataMode("mock");
const epoch = dataModeEpoch();
let finish;
const pending = detectDemoDrift({ loadDrift: () => new Promise((resolve) => { finish = resolve; }),
  isCurrent: () => epoch === dataModeEpoch(), onDetected: () => assert.fail("이전 모드 응답이 경보/로그를 복원") });
setDataMode("real");
setDataMode("mock");
finish(calculated);
await pending;
await detectDemoDrift({ loadDrift: () => assert.fail("real 모드에서 mock 요청"), isCurrent: () => false, onDetected: () => assert.fail("real 경보") });
console.log("demoAlerts: 데모 OFF에서 데모 경보만 제거·실측 경보와 읽음 상태 보존 PASS");

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
// Sidebar/Settings는 모드 전환 때 유지된다. 실제 Context가 사용하는 guard를 같은 ref로 계속 호출한다.
for (const oldOutcome of ["success", "failure"]) {
  setDataMode("mock");
  let enabled = true, checking = false, injected = false, requests = 0;
  const requestEpochRef = { current: null };
  const alerts = [], errors = [];
  const invoke = (response) => {
    const epoch = dataModeEpoch();
    return runDemoDriftCheck({ requestEpochRef, epoch,
      isCurrent: () => enabled && epoch === dataModeEpoch(),
      onChecking: (value) => { checking = value; },
      loadDrift: () => { requests += 1; return response.promise; },
      onDetected: (_, alert) => { injected = true; alerts.push(alert); },
      onError: (error) => errors.push(error)
    });
  };
  const oldResponse = deferred();
  const oldRequest = invoke(oldResponse);
  assert.equal(checking, true);
  assert.equal(requests, 1);
  await invoke(deferred());
  assert.equal(requests, 1, "동일 epoch 연타는 중복 요청을 시작하지 않는다");
  // 기존 Context의 mode toggle과 같은 reset. 유지된 Settings에는 별도 boolean 잠금이 없다.
  setDataMode("real"); enabled = false; requestEpochRef.current = null; checking = false;
  await invoke(deferred());
  assert.equal(requests, 1, "OFF에서 요청하지 않는다");
  setDataMode("mock"); enabled = true;
  const newResponse = deferred();
  const newRequest = invoke(newResponse);
  const newEpoch = dataModeEpoch();
  assert.equal(requests, 2, "이전 요청이 지연 중이어도 새 ON epoch 요청은 시작한다");
  assert.equal(checking, true);
  if (oldOutcome === "success") oldResponse.resolve(calculated);
  else oldResponse.reject(new Error("old epoch failure"));
  await oldRequest;
  assert.equal(requestEpochRef.current, newEpoch, "이전 finally가 새 잠금을 해제하지 않는다");
  assert.equal(checking, true);
  assert.equal(injected, false);
  assert.deepEqual(alerts, []);
  assert.deepEqual(errors, []);
  await invoke(deferred());
  assert.equal(requests, 2, "이전 finally 뒤에도 새 epoch 중복 요청을 차단한다");
  newResponse.resolve(calculated);
  await newRequest;
  assert.equal(requestEpochRef.current, null);
  assert.equal(checking, false);
  assert.equal(injected, true);
  assert.equal(alerts.length, 1);
  assert.equal(alerts[0].source, "demo");
  assert.deepEqual(errors, []);
}
console.log("PASS: persistent Context epoch guard, ON→OFF→ON, same-epoch duplicate and late success/error/finally isolation");
