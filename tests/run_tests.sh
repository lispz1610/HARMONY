#!/usr/bin/env bash
# Runs HARMONY's compiled C++ tests across 1 and 2 MPI ranks.
#
#   ./run_tests.sh
#   MPIRUN="mpirun --oversubscribe" ./run_tests.sh
#   RANKS_LIST="1 2" ./run_tests.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

MPIRUN=${MPIRUN:-"mpirun --oversubscribe"}
RANKS_LIST=${RANKS_LIST:-"1 2"}

if [ "$(id -u)" -eq 0 ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

shopt -s nullglob
TEST_BINS=(bin/test_*)
shopt -u nullglob

if [ ${#TEST_BINS[@]} -eq 0 ]; then
    echo "error: no test binaries found in $PWD/bin (run ./build.sh tests first)" >&2
    exit 1
fi

failed=0
executed=0

for exe in "${TEST_BINS[@]}"; do
    [ -x "$exe" ] || continue
    name=$(basename "$exe")
    for np in $RANKS_LIST; do
        executed=$((executed + 1))
        log=$(mktemp)
        $MPIRUN -np "$np" "./$exe" > "$log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "PASS  $name (-np $np)"
        elif [ "$rc" -eq 77 ]; then
            echo "SKIP  $name (-np $np)"
        else
            echo "FAIL  $name (-np $np, exit $rc)"
            cat "$log"
            failed=1
        fi
        rm -f "$log"
    done
done

if [ "$executed" -eq 0 ]; then
    echo "error: no executable test binaries found in $PWD/bin" >&2
    exit 1
fi

exit "$failed"
