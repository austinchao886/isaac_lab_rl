SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
echo ${SCRIPT_DIR}
/isaac-lab/isaaclab.sh -p -m tensorboard.main --logdir ${SCRIPT_DIR}/logs
