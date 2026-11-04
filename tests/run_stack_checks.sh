#!/bin/bash
# Checks that need the real training stack (Python 3.6, torch 0.4, gym, ic3net-envs).
# Run from the repo root inside the ic3 env:
#
#   conda activate ic3
#   bash tests/run_stack_checks.sh            # all checks
#   bash tests/run_stack_checks.sh smoke      # or: gates | layout | determinism | parity
#
# PY overrides the interpreter (default: python). LEGACY_REF overrides the legacy ref
# (default: legacy-aamas2027). Each check prints PASS/FAIL; the exit code is the number
# of failed checks.

set -u
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-python}
LEGACY_REF=${LEGACY_REF:-legacy-aamas2027}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ic3checks.XXXXXX")
FAILS=0
ONLY=${1:-all}

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
want() { [[ "$ONLY" == "all" || "$ONLY" == "$1" ]]; }

PP_EASY="--env_name predator_prey --nagents 3 --dim 5 --max_steps 20 --vision 0"
TJ_BASE="--env_name traffic_junction --vision 0 --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 0 --curr_end 0"
TINY="--hid_size 16 --detach_gap 10 --lrate 0.001 --recurrent --num_epochs 2 --epoch_size 1 --batch_size 40"

ARCHS="ic iric commnet phase1 phase2 phase3 phase4 phase5"
arch_flags() {  # (portable: macOS ships bash 3.2, which has no associative arrays)
  case $1 in
    ic) echo "" ;;
    iric) echo "--mean_ratio 0" ;;
    commnet) echo "--commnet" ;;
    phase1|phase2|phase3|phase4|phase5) echo "--ic3net --phase ${1#phase}" ;;
  esac
}

# Strip things that legitimately differ between runs (timings) and lines whose meaning
# changed on purpose (gate logging), leaving reward/success/steps.
filter_log() { grep -E '^(Epoch|Success|Steps-taken|Add-Rate)' "$1" | sed -E 's/\tTime .*//'; }

# ---------------------------------------------------------------- gate unit tests
if want gates; then
  if $PY tests/test_gates.py > "$WORK/gates.log" 2>&1; then pass "gate consistency (tests/test_gates.py)"
  else fail "gate consistency (see $WORK/gates.log)"; fi
fi

# ---------------------------------------------------------------- smoke runs
if want smoke; then
  for env in pp tj; do
    for a in $ARCHS; do
      if [[ $env == pp ]]; then cfg="$PP_EASY"; else cfg="$TJ_BASE --nagents 5 --dim 6 --max_steps 20 --difficulty easy"; fi
      out="$WORK/smoke_${env}_${a}"
      if $PY main.py $cfg $TINY $(arch_flags $a) --nprocesses 2 --seed 1 --save "$out.pt" > "$out.log" 2>&1 \
         && $PY -c "import torch,sys; d=torch.load(sys.argv[1]); assert 'args' in d and 'meta' in d and d['meta']['seed']==1" "$out.pt" 2>>"$out.log"; then
        pass "smoke $env $a"
      else
        fail "smoke $env $a (see $out.log)"
      fi
    done
  done
fi

# ---------------------------------------------------------------- TJ layout reaches the env
# Expected encoder input sizes: 2 + vocab, vocab = 1 + BASE + 2, BASE = 2d (easy), 4d (medium), 8d (hard).
if want layout; then
  for spec in "easy 6 5 17" "medium 14 10 61" "hard 18 20 149"; do
    set -- $spec; diff=$1; dim=$2; nag=$3; obs=$4
    out="$WORK/layout_$diff"
    $PY main.py $TJ_BASE --nagents $nag --dim $dim --max_steps 20 --difficulty $diff $TINY --commnet \
        --num_epochs 1 --nprocesses 1 --seed 1 > "$out.log" 2>&1
    if grep -q "Linear(in_features=$obs," "$out.log"; then pass "TJ $diff layout (obs dim $obs)"
    else fail "TJ $diff layout: expected obs dim $obs (see $out.log)"; fi
  done
fi

# ---------------------------------------------------------------- determinism with workers
if want determinism; then
  for a in phase1 phase3; do
    for k in 1 2; do
      $PY main.py $PP_EASY $TINY $(arch_flags $a) --nprocesses 3 --num_epochs 3 --seed 7 > "$WORK/det_${a}_$k.log" 2>&1
    done
    if [[ -s "$WORK/det_${a}_1.log" ]] && diff <(filter_log "$WORK/det_${a}_1.log") <(filter_log "$WORK/det_${a}_2.log") > /dev/null \
       && [[ -n "$(filter_log "$WORK/det_${a}_1.log")" ]]; then
      pass "determinism $a (same seed, 3 processes)"
    else
      fail "determinism $a (see $WORK/det_${a}_*.log)"
    fi
  done
fi

# ---------------------------------------------------------------- exact parity with legacy
# Baselines must be unchanged; phases 2-5 must be unchanged under --legacy_gate_resample.
# The legacy code never seeded numpy/random, so a wrapper seeds them the way main.py now does.
if want parity; then
  LEG="$WORK/legacy"; mkdir -p "$LEG"
  git -C "$REPO" archive "$LEGACY_REF" ':(exclude)*.whl' | tar -x -C "$LEG"
  for env in pp tj; do
    if [[ $env == pp ]]; then cfg="$PP_EASY"; else cfg="$TJ_BASE --nagents 5 --dim 6 --max_steps 20 --difficulty easy"; fi
    for a in $ARCHS; do
      args="$cfg $TINY $(arch_flags $a) --nprocesses 1 --num_epochs 3 --seed 3"
      (cd "$LEG" && $PY -c "
import sys, random, numpy
numpy.random.seed(3); random.seed(3)
sys.argv = ['main.py'] + sys.argv[1:]
exec(compile(open('main.py').read(), 'main.py', 'exec'))
" $args > "$WORK/par_${env}_${a}_legacy.log" 2>&1)
      $PY main.py $args --legacy_gate_resample > "$WORK/par_${env}_${a}_new.log" 2>&1
      if [[ -n "$(filter_log "$WORK/par_${env}_${a}_legacy.log")" ]] \
         && diff <(filter_log "$WORK/par_${env}_${a}_legacy.log") <(filter_log "$WORK/par_${env}_${a}_new.log") > /dev/null; then
        pass "legacy parity $env $a"
      else
        fail "legacy parity $env $a (see $WORK/par_${env}_${a}_*.log)"
      fi
    done
  done
fi

echo
echo "Logs in $WORK"
echo "$FAILS check(s) failed."
exit $FAILS
