#!/bin/sh
set -eu
task_script="$(dirname "$0")/../docker/40-integration-config.sh"
task_tmp=$(mktemp -d /tmp/phase-d-env.XXXXXX)
passed=0
reject() {
  if env -i PATH="$PATH" "$1=$2" sh "$task_script" --validate-only > "$task_tmp/error.log" 2>&1; then
    printf 'FAIL: accepted invalid %s\n' "$1" >&2; exit 1
  fi
  grep -q 'Integration environment invalid:' "$task_tmp/error.log"
  passed=$((passed + 1))
}
env -i PATH="$PATH" sh "$task_script" --validate-only > /dev/null
reject DASHBOARD_UPSTREAM 'https://user:placeholder@dashboard.invalid'
reject DASHBOARD_UPSTREAM 'http://dashboard.invalid/path/'
reject DASHBOARD_UPSTREAM 'http://dashboard.invalid?x=1'
reject DASHBOARD_UPSTREAM 'http://dashboard.invalid#x'
reject DASHBOARD_UPSTREAM 'http://dashboard.invalid;return 200'
reject DASHBOARD_UPSTREAM 'http://dashboard.invalid:65536'
task_newline='
'
reject DASHBOARD_UPSTREAM "http://dashboard.invalid${task_newline}"
reject REPORTING_API_UPSTREAM "http://reporting.invalid${task_newline}injected"
reject DASHBOARD_UPSTREAM_BASE_PATH '/../../'
reject DASHBOARD_UPSTREAM_BASE_PATH '/api/;return 200/'
reject DASHBOARD_UPSTREAM_BASE_PATH '/%2e%2e/'
reject APP_PUBLIC_PATH '/api/'
reject DASHBOARD_PUBLIC_PATH '/assets/'
reject REPORTING_API_PUBLIC_PATH '/authelia/'
reject DASHBOARD_PUBLIC_PATH '/rnd/population/'
reject REPORTING_PUBLIC_PATH '/rnd/population/dashboard/'
reject REPORTING_PUBLIC_PATH '/rnd/population/dashboard/nested/'
reject DASHBOARD_PUBLIC_PATH '/other/dashboard/'
reject DASHBOARD_SUPPORTED_MODES 'mock,unknown'
reject DASHBOARD_SUPPORTED_MODES 'mock,mock'
reject AUTH_PUBLIC_URL 'https://elsewhere.invalid/authelia/'
reject SITE_DOMAIN 'https://platform.invalid'
reject SITE_DOMAIN 'platform.invalid/path'
reject SITE_DOMAIN 'platform.invalid?query'
reject SITE_DOMAIN 'platform.invalid#fragment'
reject SITE_DOMAIN 'platform..invalid'
reject SITE_DOMAIN "platform.invalid${task_newline}"
env -i PATH="$PATH" SITE_DOMAIN=platform.invalid AUTH_PUBLIC_URL=https://platform.invalid/authelia/ \
  APP_PUBLIC_PATH=/demo/app/ DASHBOARD_PUBLIC_PATH=/demo/app/dash/ DASHBOARD_API_PUBLIC_PATH=/demo/app/dash-api/ \
  REPORTING_PUBLIC_PATH=/demo/app/report/ REPORTING_API_PUBLIC_PATH=/demo/app/report-api/ \
  DASHBOARD_UPSTREAM=https://dashboard.invalid:443 DASHBOARD_UPSTREAM_BASE_PATH=/ui/v1/ \
  DASHBOARD_SUPPORTED_MODES=mock sh "$task_script" --validate-only > /dev/null
env -i PATH="$PATH" INTEGRATION_CONF_DIR="$task_tmp/conf" INTEGRATION_HTML_DIR="$task_tmp/html" \
  sh "$task_script" > /dev/null
grep -q '"configured":false' "$task_tmp/html/integration-config.json"
if grep -q 'proxy_pass' "$task_tmp/conf/routes.conf"; then printf 'FAIL: unconfigured fallback\n'; exit 1; fi
env -i PATH="$PATH" DASHBOARD_UPSTREAM=http://private-dashboard.invalid:80 DASHBOARD_SUPPORTED_MODES=mock \
  INTEGRATION_CONF_DIR="$task_tmp/configured" INTEGRATION_HTML_DIR="$task_tmp/public" sh "$task_script" > /dev/null
if grep -q 'private-dashboard' "$task_tmp/public/integration-config.json"; then printf 'FAIL: private origin exposed\n'; exit 1; fi
if env -i PATH="$PATH" REPORTING_API_UPSTREAM='http://reporting.invalid/invalid/' \
  INTEGRATION_CONF_DIR="$task_tmp/invalid" INTEGRATION_HTML_DIR="$task_tmp/invalid-html" sh "$task_script" > /dev/null 2>&1; then
  printf 'FAIL: invalid startup accepted\n'; exit 1
fi
[ ! -e "$task_tmp/invalid" ] && [ ! -e "$task_tmp/invalid-html" ]
printf 'integration env: %s rejection cases, valid custom prefixes, no unset fallback/private origin leak, validation before writes PASS\n' "$passed"
