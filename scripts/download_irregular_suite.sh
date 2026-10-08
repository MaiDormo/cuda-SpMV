#!/bin/bash
# Irregular / scale-free SuiteSparse graphs for SpMV (power-law, web, social,
# citation, circuit). Refuses any unpacked .mtx above 5 GiB.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/data"
MAX_MTX_BYTES=${MAX_MTX_BYTES:-$((5 * 1024 * 1024 * 1024))}
MAX_GZ_BYTES=${MAX_GZ_BYTES:-$((1200 * 1024 * 1024))}

mkdir -p "$DATA_DIR"

# name|url
MATRICES=(
  # already classic irregular (re-download skipped if present)
  "mawi_201512020330|https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz"
  "webbase-1M|https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz"
  "scircuit|https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz"
  "mac_econ_fwd500|https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz"
  # SNAP web / social / citation
  "web-Google|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz"
  "web-BerkStan|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz"
  "web-NotreDame|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-NotreDame.tar.gz"
  "web-Stanford|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Stanford.tar.gz"
  "soc-LiveJournal1|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-LiveJournal1.tar.gz"
  "soc-Pokec|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Pokec.tar.gz"
  "cit-Patents|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/cit-Patents.tar.gz"
  "email-EuAll|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-EuAll.tar.gz"
  "amazon0312|https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz"
  # LAW / Gleich crawls
  "hollywood-2009|https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz"
  "ljournal-2008|https://suitesparse-collection-website.herokuapp.com/MM/LAW/ljournal-2008.tar.gz"
  "wb-edu|https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wb-edu.tar.gz"
  "wikipedia-20070206|https://suitesparse-collection-website.herokuapp.com/MM/Gleich/wikipedia-20070206.tar.gz"
  # large irregular circuits
  "circuit5M|https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz"
  "Freescale1|https://suitesparse-collection-website.herokuapp.com/MM/Freescale/Freescale1.tar.gz"
)

# Convert MatrixMarket "pattern" (no values) → real with value 1.0
pattern_to_real() {
  local mtx="$1"
  local header
  header=$(head -1 "$mtx")
  if [[ "$header" != *"pattern"* ]]; then
    return 0
  fi
  echo "  converting pattern → real (fill=1.0): $mtx"
  local tmp="${mtx}.real"
  {
    echo "%%MatrixMarket matrix coordinate real general"
    # skip comment/% lines after banner until dimensions, then rewrite entries
    awk '
      BEGIN { seen_dims=0 }
      /^%%/ { next }
      /^%/ { print; next }
      !seen_dims {
        print; seen_dims=1; next
      }
      NF>=2 { print $1, $2, 1.0 }
    ' "$mtx"
  } > "$tmp"
  mv "$tmp" "$mtx"
}

for entry in "${MATRICES[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  mtx="$DATA_DIR/$name/$name.mtx"

  if [[ -f "$mtx" ]]; then
    pattern_to_real "$mtx"
    size=$(stat -c%s "$mtx")
    if (( size > MAX_MTX_BYTES )); then
      echo "ERROR: existing $mtx is too large — removing"
      rm -rf "$DATA_DIR/$name"
    else
      echo "SKIP $name (present, $(awk -v s="$size" 'BEGIN{printf "%.2f GiB", s/1024/1024/1024}'))"
      continue
    fi
  fi

  echo "----------------------------------------"
  echo "Fetching $name"
  gz_bytes=$(curl -sI -L --max-time 30 "$url" | tr -d '\r' | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {print $2}' | tail -1)
  gz_bytes=${gz_bytes:-0}
  if (( gz_bytes <= 0 )); then
    echo "ERROR: no size for $url — skip"
    continue
  fi
  if (( gz_bytes > MAX_GZ_BYTES )); then
    echo "SKIP $name: gz too large"
    continue
  fi
  echo "Remote: $(awk -v s="$gz_bytes" 'BEGIN{printf "%.1f MiB", s/1024/1024}')"

  tmp="$DATA_DIR/${name}.tar.gz"
  wget -q --show-progress -O "$tmp" "$url"
  tar -xzf "$tmp" -C "$DATA_DIR"
  rm -f "$tmp"

  if [[ ! -f "$mtx" ]]; then
    found=$(find "$DATA_DIR/$name" -name '*.mtx' -type f ! -name '*_b.mtx' | head -1 || true)
    if [[ -n "${found:-}" ]]; then
      mkdir -p "$(dirname "$mtx")"
      mv "$found" "$mtx"
    fi
  fi
  if [[ ! -f "$mtx" ]]; then
    echo "ERROR: missing $mtx"
    rm -rf "$DATA_DIR/$name"
    continue
  fi

  pattern_to_real "$mtx"
  size=$(stat -c%s "$mtx")
  echo "Unpacked: $(awk -v s="$size" 'BEGIN{printf "%.2f GiB", s/1024/1024/1024}')"
  if (( size > MAX_MTX_BYTES )); then
    echo "REJECT >5 GiB — delete"
    rm -rf "$DATA_DIR/$name"
    continue
  fi
  chmod -R a+rX "$DATA_DIR/$name"
done

echo "========================================"
du -sh "$DATA_DIR"/*/ 2>/dev/null | sort -h
echo "data/ total: $(du -sh "$DATA_DIR" | awk '{print $1}')"
