#!/bin/sh
# Only these non-secret integration variables are consumed. Never print values.
set -eu
nl='
'
cr=$(printf '\r')
fail() { printf 'Integration environment invalid: %s (%s)\n' "$1" "$2" >&2; exit 1; }
read_env() {
  env_value=$(printenv "$1" 2>/dev/null || true; printf '.')
  env_value=${env_value%.}
  env_value=${env_value%"$nl"}
}
clean() {
  case "$2" in *"$nl"*|*"$cr"*) fail "$1" 'newline';; esac
}
path() {
  clean "$1" "$2"
  printf '%s' "$2" | grep -Eq '^/([A-Za-z0-9_.-]+/)*$' || fail "$1" 'absolute trailing-slash path required'
  case "$2" in *'/../'*|*'/./'*|*'//'*) fail "$1" 'path traversal';; esac
}
public_path() {
  path "$1" "$2"
  case "$2" in /|/api/*|/assets/*|/authelia/*) fail "$1" 'reserved prefix';; esac
}
origin() {
  clean "$1" "$2"
  [ -n "$2" ] || return 0
  printf '%s' "$2" | grep -Eq '^https?://[A-Za-z0-9][A-Za-z0-9_.-]*(:[0-9]{1,5})?$' || fail "$1" 'scheme/host/port only; no userinfo/query/fragment/path'
  authority=${2#*://}
  case "$authority" in *:*) port=${authority##*:}; [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || fail "$1" 'port range';; esac
}
read_env APP_PUBLIC_PATH; app_path=${env_value:-/rnd/population/}
public_path APP_PUBLIC_PATH "$app_path"
read_env SITE_DOMAIN; site_domain=$env_value
clean SITE_DOMAIN "$site_domain"
if [ -n "$site_domain" ]; then
  printf '%s' "$site_domain" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$' || fail SITE_DOMAIN 'hostname only'
  case "$site_domain" in *'..'*|*'.-'*|*'-.'*) fail SITE_DOMAIN 'invalid domain label';; esac
fi
read_env AUTH_PUBLIC_URL; auth_url=$env_value
clean AUTH_PUBLIC_URL "$auth_url"
if [ -n "$auth_url" ]; then
  [ -n "$site_domain" ] || fail AUTH_PUBLIC_URL 'SITE_DOMAIN required'
  [ "${auth_url%/}" = "https://$site_domain/authelia" ] || fail AUTH_PUBLIC_URL 'same SITE_DOMAIN HTTPS /authelia portal required'
fi
records=''
public_paths=''
for module in dashboard reporting; do
  prefix=$(printf '%s' "$module" | tr '[:lower:]' '[:upper:]')
  read_env "${prefix}_UPSTREAM"; fe_origin=$env_value
  read_env "${prefix}_API_UPSTREAM"; api_origin=$env_value
  read_env "${prefix}_UPSTREAM_BASE_PATH"; fe_base=${env_value:-/}
  read_env "${prefix}_API_UPSTREAM_BASE_PATH"; api_base=${env_value:-/}
  read_env "${prefix}_PUBLIC_PATH"; fe_path=${env_value:-${app_path}${module}/}
  read_env "${prefix}_API_PUBLIC_PATH"; api_path=${env_value:-${app_path}${module}-api/}
  read_env "${prefix}_SUPPORTED_MODES"; modes=$env_value
  origin "${prefix}_UPSTREAM" "$fe_origin"
  origin "${prefix}_API_UPSTREAM" "$api_origin"
  path "${prefix}_UPSTREAM_BASE_PATH" "$fe_base"
  path "${prefix}_API_UPSTREAM_BASE_PATH" "$api_base"
  clean "${prefix}_SUPPORTED_MODES" "$modes"
  case "$modes" in ''|mock|real|mock,real|real,mock) :;; *) fail "${prefix}_SUPPORTED_MODES" 'explicit mock/real list required';; esac
  for candidate in "$fe_path" "$api_path"; do
    public_path "${prefix}_PUBLIC_PATH" "$candidate"
    case "$candidate" in "$app_path"?*) :;; *) fail "${prefix}_PUBLIC_PATH" 'must be below APP_PUBLIC_PATH';; esac
    for previous in $public_paths; do
      case "$candidate" in "$previous"*) fail "${prefix}_PUBLIC_PATH" 'prefix collision';; esac
      case "$previous" in "$candidate"*) fail "${prefix}_PUBLIC_PATH" 'prefix collision';; esac
    done
    public_paths="$public_paths $candidate"
  done
  records="${records}${module}|${fe_origin}|${fe_base}|${fe_path}|${api_origin}|${api_base}|${api_path}|${modes}${nl}"
done
if [ "${1:-}" = --validate-only ]; then printf 'Integration environment validation PASS\n'; exit 0; fi

conf_dir=${INTEGRATION_CONF_DIR:-/etc/nginx/integration}
html_dir=${INTEGRATION_HTML_DIR:-/usr/share/nginx/html}
mkdir -p "$conf_dir" "$html_dir"
cat > "$conf_dir/maps.conf.tmp" <<'EOF'
map "$http_x_data_mode|$arg_mode" $integration_request_mode {
  default invalid;
  "~^\\|$" unspecified;
  "~^\\|mock$" mock;
  "~^\\|real$" real;
  "~^mock\\|$" mock;
  "~^real\\|$" real;
  "~^mock\\|mock$" mock;
  "~^real\\|real$" real;
}
map $args $integration_query_invalid {
  default 0;
  "~(?:^|&)mode=[^&]*(?:&[^&]*)*&mode=" 1;
  "~(?:^|&)[^=&]*%[0-9A-Fa-f]{2}[^=&]*=" 1;
  "~(?:^|&)(?:M[oO][dD][eE]|mO[dD][eE]|moD[eE]|modE)=" 1;
  "~(?:^|&)mode(?:&|$)" 1;
}
EOF
cat > "$conf_dir/routes.conf.tmp" <<EOF
resolver 127.0.0.11 valid=10s ipv6=off;
location = /integration-config.json {
  root $html_dir;
  add_header Cache-Control "no-store" always;
  add_header X-Content-Type-Options nosniff always;
}
location = ${app_path%/} { return 308 $app_path\$is_args\$args; }
location ^~ $app_path {
  root $html_dir;
  try_files \$uri /index.html;
}
EOF
printf '{"schema":1,"source":"ingress-env","appPath":"%s","modules":{' "$app_path" > "$html_dir/integration-config.json.tmp"
printf '%s' "$records" | while IFS='|' read -r module fe_origin fe_base fe_path api_origin api_base api_path modes; do
  [ -n "$module" ] || continue
  [ "$module" = dashboard ] && label='종합 대시보드' || label='자동화 리포팅'
  fe_configured=false; [ -z "$fe_origin" ] || fe_configured=true
  api_configured=false; [ -z "$api_origin" ] || api_configured=true
  case "$modes" in '') modes_json='[]';; mock) modes_json='["mock"]';; real) modes_json='["real"]';; *) modes_json='["mock","real"]';; esac
  [ "$module" = dashboard ] || printf ',' >> "$html_dir/integration-config.json.tmp"
  printf '"%s":{"configured":%s,"apiConfigured":%s,"path":"%s","apiPath":"%s","supportedModes":%s}' "$module" "$fe_configured" "$api_configured" "$fe_path" "$api_path" "$modes_json" >> "$html_dir/integration-config.json.tmp"
  cat > "$html_dir/integration-ready-$module.html" <<EOF
<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>$label | 준비 중</title><style>
body{margin:0;background:#0c1220;color:#e5eaf4;font:16px/1.7 system-ui,sans-serif}header{padding:22px 6vw;border-bottom:1px solid #253048;color:#94a3b8}main{max-width:780px;margin:10vh auto;padding:36px;border:1px solid #253048;border-radius:16px;background:#131d2e}h1{font-size:28px}a{color:#77b7ff}p{color:#bdc9dd}.tag{color:#9fb3d2;font-size:14px}a:focus-visible{outline:3px solid #77b7ff;outline-offset:5px}
</style></head><body><header>인구감소 R&amp;D 통합 플랫폼</header><main><p class="tag">외부 모듈 연결 준비</p><h1>$label 준비 중</h1><p>모듈 연결 주소 또는 선택한 데이터 모드의 지원이 아직 확인되지 않았습니다.</p><p>연결 설정과 데이터 모드를 확인한 뒤 다시 접속하세요.</p><a href="$app_path">플랫폼으로 돌아가기</a></main></body></html>
EOF
  cat >> "$conf_dir/maps.conf.tmp" <<EOF
map \$cookie_population_${module}_mode \$${module}_cookie_mode { default invalid; "" real; "~^mock\$" mock; "~^real\$" real; }
map \$integration_request_mode \$${module}_mode { unspecified \$${module}_cookie_mode; default \$integration_request_mode; }
map \$${module}_mode \$${module}_mode_allowed { default 0;
EOF
  case ",$modes," in *,mock,*) printf '  mock 1;\n' >> "$conf_dir/maps.conf.tmp";; esac
  case ",$modes," in *,real,*) printf '  real 1;\n' >> "$conf_dir/maps.conf.tmp";; esac
  printf '}\n' >> "$conf_dir/maps.conf.tmp"
  cat >> "$conf_dir/maps.conf.tmp" <<EOF
map \$${module}_mode \$${module}_cookie_header {
  default "";
  mock "population_${module}_mode=mock; Path=$app_path; Secure; HttpOnly; SameSite=Lax";
  real "population_${module}_mode=real; Path=$app_path; Secure; HttpOnly; SameSite=Lax";
}
EOF
  cat >> "$conf_dir/routes.conf.tmp" <<EOF
location = /integration-ready-$module.html {
  internal;
  root $html_dir;
  try_files /integration-ready-$module.html =500;
  add_header Cache-Control "no-store" always;
  add_header X-Integration-State \$${module}_state always;
  add_header X-Content-Type-Options nosniff always;
  add_header Content-Security-Policy "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'self'" always;
}
EOF
  for kind in fe api; do
    if [ "$kind" = fe ]; then route_origin=$fe_origin; route_base=$fe_base; route_path=$fe_path; else route_origin=$api_origin; route_base=$api_base; route_path=$api_path; fi
    cat >> "$conf_dir/routes.conf.tmp" <<EOF
location = ${route_path%/} { return 308 $route_path\$is_args\$args; }
location ^~ $route_path {
  set \$${module}_state unconfigured;
  error_page 418 =200 /integration-ready-$module.html;
EOF
    if [ -z "$route_origin" ]; then
      printf '  return 418;\n}\n' >> "$conf_dir/routes.conf.tmp"
      continue
    fi
    escaped_origin=$(printf '%s' "$route_origin" | sed 's/\./\\./g')
    escaped_base=$(printf '%s' "$route_base" | sed 's/\./\\./g')
    escaped_path=$(printf '%s' "$route_path" | sed 's/\./\\./g')
    cat >> "$conf_dir/maps.conf.tmp" <<EOF
map \$request_uri \$${module}_${kind}_upstream_uri {
  default "";
  "~^${escaped_path}(.*)\$" "$route_base\$1";
}
map \$upstream_http_location \$${module}_${kind}_redirect {
  default "";
  "~*(?:\\.\\.|%2e|%2f|%5c)" "";
  "~^${escaped_origin}${escaped_base}(.*)\$" "$route_path\$1";
  "~^${escaped_base}(.*)\$" "$route_path\$1";
}
EOF
    cat >> "$conf_dir/routes.conf.tmp" <<EOF
  if (\$integration_query_invalid = 1) { return 400 "Ambiguous mode query"; }
  if (\$integration_request_mode = invalid) { return 400 "Invalid or conflicting data mode"; }
EOF
    if [ "$kind" = api ]; then printf '  if ($integration_request_mode = unspecified) { return 400 "X-Data-Mode or mode query required"; }\n' >> "$conf_dir/routes.conf.tmp"; fi
    cat >> "$conf_dir/routes.conf.tmp" <<EOF
  set \$${module}_state unsupported_mode;
  if (\$${module}_mode_allowed = 0) { return 418; }
  if (\$${module}_${kind}_upstream_uri = "") { return 400 "Invalid module path encoding"; }
  set \$${module}_${kind}_origin "$route_origin";
  proxy_pass \$${module}_${kind}_origin\$${module}_${kind}_upstream_uri;
  proxy_set_header Host \$proxy_host;
  proxy_set_header X-Real-IP \$remote_addr;
  proxy_set_header X-Forwarded-Host \$host;
  proxy_set_header X-Forwarded-Proto \$http_x_forwarded_proto;
  proxy_set_header X-Forwarded-Prefix "$route_path";
  proxy_set_header Remote-User \$http_remote_user;
  proxy_set_header Remote-Groups \$http_remote_groups;
  proxy_set_header Remote-Email \$http_remote_email;
  proxy_set_header Remote-Name \$http_remote_name;
  proxy_set_header X-Data-Mode \$${module}_mode;
  proxy_set_header Cookie "";
  proxy_set_header Authorization "";
  proxy_set_header If-None-Match "";
  proxy_set_header If-Modified-Since "";
  proxy_ssl_server_name on;
  proxy_ssl_verify on;
  proxy_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
  proxy_connect_timeout 5s;
  proxy_read_timeout 30s;
  proxy_intercept_errors off;
  proxy_redirect off;
  proxy_ignore_headers Set-Cookie Cache-Control Expires X-Accel-Redirect;
  proxy_hide_header Set-Cookie;
  proxy_hide_header Clear-Site-Data;
  proxy_hide_header X-Accel-Redirect;
  proxy_hide_header Cache-Control;
  proxy_hide_header Expires;
  proxy_hide_header ETag;
  proxy_hide_header Last-Modified;
  proxy_hide_header Location;
  proxy_hide_header Refresh;
  proxy_hide_header Content-Location;
  add_header Location \$${module}_${kind}_redirect always;
  add_header Cache-Control "private, no-store" always;
  add_header Vary X-Data-Mode always;
  add_header X-Data-Mode \$${module}_mode always;
  add_header X-Integration-State configured always;
  add_header X-Content-Type-Options nosniff always;
  add_header Referrer-Policy same-origin always;
  add_header X-Frame-Options SAMEORIGIN always;
EOF
    if [ "$kind" = fe ]; then printf '  add_header Set-Cookie $%s_cookie_header always;\n' "$module" >> "$conf_dir/routes.conf.tmp"; fi
    printf '}\n' >> "$conf_dir/routes.conf.tmp"
  done
done
printf '}}\n' >> "$html_dir/integration-config.json.tmp"
mv "$conf_dir/maps.conf.tmp" "$conf_dir/maps.conf"
mv "$conf_dir/routes.conf.tmp" "$conf_dir/routes.conf"
mv "$html_dir/integration-config.json.tmp" "$html_dir/integration-config.json"
printf 'Integration routing configuration generated\n'
