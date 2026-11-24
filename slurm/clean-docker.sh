#!/usr/bin/env bash
# Remove this project's Docker leftovers from one node, as a job per node:
#
#   slurm/submit.sh -w <node> --cpus-per-task=1 -- slurm/clean-docker.sh [--apply]
#
# Lists by default; --apply removes containers of dead jobs (hnc-<job>-<port>, from
# serving.py), the compose-era leftovers (projects hnc-smoke-<job>, image
# hnc-runner:local), dangling images and unused build cache (the last two node-wide:
# Docker can't tell whose). Pulled images are never touched. Refuses while another
# hnc job runs on the node.
set -euo pipefail

apply=false
case ${1:-} in
  --apply) apply=true ;;
  '') ;;
  *) echo "usage: slurm/clean-docker.sh [--apply]" >&2; exit 1 ;;
esac

node=${SLURMD_NODENAME:-$(hostname -s)}
running=$(squeue -h -w "$node" -u "$USER" -t RUNNING -o '%i %j' | awk '$2 ~ /^hnc/ {print $1}')
others=$(grep -vx "${SLURM_JOB_ID:-}" <<<"$running" || true)
if [[ -n $others ]]; then
  echo "refusing: other hnc jobs run on $node: $(echo $others)" >&2
  exit 1
fi

# id<TAB>what, one line per object.
containers=$(
  docker ps -a --format '{{.ID}}	{{.Names}} ({{.State}})' |
    awk -F'\t' '$2 ~ /^hnc-([0-9]+|local)-[0-9]+ /'
  docker ps -a --filter label=com.docker.compose.project \
    --format '{{.ID}}	{{.Label "com.docker.compose.project"}}/{{.Names}} ({{.State}})' |
    awk -F'\t' '$2 ~ /^hnc-smoke-/'
)
networks=$(
  docker network ls --filter label=com.docker.compose.project \
    --format '{{.ID}}	{{.Name}}' | awk -F'\t' '$2 ~ /^hnc-smoke-/'
)
images=$(docker image ls --filter reference='hnc-runner' --format '{{.Repository}}:{{.Tag}}	{{.ID}} {{.Size}}')
dangling=$(docker image ls --filter dangling=true --format '{{.ID}}	{{.Size}}')

show() { # title, lines
  echo "$1: $(grep -c . <<<"$2" || true)"
  [[ -n $2 ]] && sed 's/^/  /' <<<"$2"
  return 0
}
echo "node $node ($($apply && echo apply || echo "dry run; --apply removes"))"
show "leftover containers" "$containers"
show "leftover networks" "$networks"
show "hnc images" "$images"
show "dangling images" "$dangling"
docker system df

$apply || exit 0
# rm -f also stops a container that's still running; no job owns one (checked above).
[[ -z $containers ]] || docker rm -f $(cut -f1 <<<"$containers")
[[ -z $networks ]] || docker network rm $(cut -f1 <<<"$networks")
# By tag, without -f: an image some container still uses is refused, not removed.
[[ -z $images ]] || docker image rm $(cut -f1 <<<"$images")
docker image prune --force
docker builder prune --all --force
echo "after:"
docker system df
