#!/bin/bash
# Default matrix download = irregular / scale-free SuiteSparse suite
# (web, social, citation, circuit hubs, MAWI). Cap: 5 GiB per .mtx.
#
# For the older FE/science set, use git history; for this suite:
#   ./scripts/download_irregular_suite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/download_irregular_suite.sh"
