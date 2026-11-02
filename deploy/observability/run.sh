#!/usr/bin/env bash
# Prometheus + Grafana as plain binaries inside WSL (no Docker; Build Guide Phase 13).
#
#   one-time:  download the CURRENT versions (P54) and record them below
#     Prometheus LTS linux-amd64 tarball: https://prometheus.io/download/        -> ~/tools/prometheus
#     Grafana OSS standalone tarball:     https://grafana.com/grafana/download?edition=oss -> ~/tools/grafana
#   usage:     deploy/observability/run.sh        (Ctrl-C stops both)
set -euo pipefail

PROMETHEUS_VERSION="<fill in after download>"
GRAFANA_VERSION="<fill in after download>"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROM_DIR="${PROM_DIR:-$HOME/tools/prometheus}"
GRAFANA_DIR="${GRAFANA_DIR:-$HOME/tools/grafana}"
OBS="$REPO_DIR/deploy/observability"

# the dashboards provider needs an absolute path: render it into a temp provisioning dir
PROV="$(mktemp -d)"
cp -r "$OBS/grafana/provisioning/." "$PROV/"
sed -i "s#REPO_DIR#$REPO_DIR#g" "$PROV/dashboards/turingserve.yaml"

cleanup() { kill "${PROM_PID:-}" "${GRAF_PID:-}" 2>/dev/null || true; rm -rf "$PROV"; }
trap cleanup EXIT INT TERM

"$PROM_DIR/prometheus" --config.file="$OBS/prometheus.yml" \
  --storage.tsdb.path="$HOME/tools/prometheus-data" --web.listen-address=:9090 &
PROM_PID=$!

GF_PATHS_PROVISIONING="$PROV" "$GRAFANA_DIR/bin/grafana" server --homepath "$GRAFANA_DIR" &
GRAF_PID=$!

echo "Prometheus http://localhost:9090   Grafana http://localhost:3000 (first login admin/admin)"
echo "versions: prometheus=$PROMETHEUS_VERSION grafana=$GRAFANA_VERSION"
wait
