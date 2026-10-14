import assert from "node:assert/strict";
import { retainRealAlerts, unreadNotifications } from "./demoAlerts.js";

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
console.log("demoAlerts: 데모 OFF에서 데모 경보만 제거·실측 경보와 읽음 상태 보존 PASS");
