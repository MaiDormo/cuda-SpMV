# Shared irregular / scale-free SuiteSparse list (sourced by run scripts).
# Split for edu-short 5-minute jobs.

# Compact set for single-job scripts (cpu / adaptive / cusparse) that must
# finish within the 5-minute partition.
IRREGULAR_CORE=(
  "web-Google/web-Google.mtx"
  "webbase-1M/webbase-1M.mtx"
  "soc-LiveJournal1/soc-LiveJournal1.mtx"
  "cit-Patents/cit-Patents.mtx"
  "circuit5M/circuit5M.mtx"
  "mawi_201512020330/mawi_201512020330.mtx"
)

IRREGULAR_BATCH1=(
  "email-EuAll/email-EuAll.mtx"
  "web-NotreDame/web-NotreDame.mtx"
  "mac_econ_fwd500/mac_econ_fwd500.mtx"
  "scircuit/scircuit.mtx"
  "web-Stanford/web-Stanford.mtx"
  "amazon0312/amazon0312.mtx"
  "webbase-1M/webbase-1M.mtx"
  "web-Google/web-Google.mtx"
  "web-BerkStan/web-BerkStan.mtx"
  "cit-Patents/cit-Patents.mtx"
)

IRREGULAR_BATCH2=(
  "Freescale1/Freescale1.mtx"
  "wikipedia-20070206/wikipedia-20070206.mtx"
  "hollywood-2009/hollywood-2009.mtx"
  "wb-edu/wb-edu.mtx"
  "soc-LiveJournal1/soc-LiveJournal1.mtx"
)

IRREGULAR_BATCH3=(
  "ljournal-2008/ljournal-2008.mtx"
  "circuit5M/circuit5M.mtx"
  "soc-Pokec/soc-Pokec.mtx"
  "mawi_201512020330/mawi_201512020330.mtx"
)
