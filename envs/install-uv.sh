#!/usr/bin/env bash
# Install the pinned uv release if it isn't already on this host.
#
#   envs/install-uv.sh            # installs to ~/.local/bin (override with UV_INSTALL_DIR)
#
# Downloads the official GitHub release tarball and checks it against the sha256 pinned
# here, not against a checksum fetched alongside it. Does nothing if `uv` in the install
# dir already reports the pinned version. Prints the path of the uv binary.
set -euo pipefail

version=0.13.9
target=uv-x86_64-unknown-linux-gnu
sha256=a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5

dest=${UV_INSTALL_DIR:-$HOME/.local/bin}
if [[ -x $dest/uv ]] && [[ $("$dest/uv" --version) == "uv $version "* ]]; then
  echo "$dest/uv"
  exit 0
fi
[[ $(uname -sm) == "Linux x86_64" ]] || { echo "unsupported platform: $(uname -sm)" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sSfL -o "$tmp/uv.tar.gz" \
  "https://github.com/astral-sh/uv/releases/download/$version/$target.tar.gz"
echo "$sha256  $tmp/uv.tar.gz" | sha256sum --check --quiet
tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
mkdir -p "$dest"
install -m 755 "$tmp/$target/uv" "$tmp/$target/uvx" "$dest/"
echo "$dest/uv"
