#!/usr/bin/env bash
# Build the APPL SARSOP `pomdpsol` binary from source.
#
# Why this exists:
#   bcgt-v2.0 milestone C.2 uses SARSOP (Kurniawati, Hsu, Lee 2008) for offline
#   POMDP solving on the multi-hypothesis belief space. Our Python project uses
#   pomdp_py, which provides a clean subprocess wrapper at
#     pomdp_py.utils.interfaces.solvers.sarsop()
#   but the wrapper needs an external `pomdpsol` binary from the APPL toolkit
#   (https://github.com/AdaCompNUS/sarsop).
#
#   APPL's upstream Makefile predates modern gcc and trips on multiple-definition
#   linker errors. The single patch here (add -fcommon to CFLAGS) restores the
#   old loose-linking behavior and the binary builds cleanly.
#
# Usage:
#   bash scripts/build_pomdpsol.sh                   # builds to vendor/sarsop/
#   bash scripts/build_pomdpsol.sh /custom/prefix    # builds to /custom/prefix
#
# After build, set POMDPSOL_PATH or pass the binary path to
# pomdp_py.utils.interfaces.solvers.sarsop(pomdpsol_path=...).
#
# Build time: ~90 seconds on a 6-core workstation. Disk: ~50 MB build tree,
# ~18 MB stripped binary.
#
# Sources patched:
# - upstream commit: jan 2020 (HEAD of github.com/AdaCompNUS/sarsop)
# - patch: 1 line in src/Makefile, adds -fcommon to CFLAGS so the parser-
#   generated globals (IP, IR) link cleanly against modern gcc/ld.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${1:-$REPO_ROOT/vendor/sarsop}"
BUILD_DIR="$PREFIX/build"
SRC_URL="https://github.com/AdaCompNUS/sarsop.git"

mkdir -p "$PREFIX"
mkdir -p "$BUILD_DIR"

if [ ! -d "$BUILD_DIR/.git" ]; then
  echo "==> Cloning APPL SARSOP -> $BUILD_DIR"
  git clone --depth 1 "$SRC_URL" "$BUILD_DIR"
fi

cd "$BUILD_DIR/src"

if ! grep -q -- '-fcommon' Makefile; then
  echo "==> Patching Makefile: add -fcommon to CFLAGS"
  sed -i \
    's|^CFLAGS        = -g -w -O3 \$(INCDIR) -msse2  -mfpmath=sse \$(CYGWIN_CFLAGS)|CFLAGS        = -g -w -O3 -fcommon $(INCDIR) -msse2  -mfpmath=sse $(CYGWIN_CFLAGS)|' \
    Makefile
fi

echo "==> Building (single-threaded; the Makefile has racy deps with -j)"
make clean >/dev/null 2>&1 || true
make

cp pomdpsol "$PREFIX/pomdpsol"
cp pomdpsim "$PREFIX/pomdpsim" 2>/dev/null || true
cp pomdpeval "$PREFIX/pomdpeval" 2>/dev/null || true

echo
echo "==> Done."
echo "    Binary: $PREFIX/pomdpsol"
echo "    Use with:"
echo "      from pomdp_py.utils.interfaces.solvers import sarsop"
echo "      policy = sarsop(agent, pomdpsol_path='$PREFIX/pomdpsol',"
echo "                      timeout=60, precision=0.01)"
echo
