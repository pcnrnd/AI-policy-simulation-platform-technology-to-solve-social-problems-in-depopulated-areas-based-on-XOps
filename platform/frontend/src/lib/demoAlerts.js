export function retainRealAlerts(entries = []) {
  return entries.filter((entry) => entry.source !== "demo");
}

export function unreadNotifications(entries = []) {
  return entries.filter((entry) => entry.unread).length;
}
