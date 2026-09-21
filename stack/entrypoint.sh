#!/bin/bash
# Sets up SSH access (RunPod's startSsh convention), then execs into the autostop watchdog as
# PID 1 so pod runtime stays bounded regardless of what training work runs interactively over
# SSH in the meantime.
set -u

: "${RUNPOD_POD_ID:?set RUNPOD_POD_ID so autostop knows which pod to stop}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY so autostop can call the RunPod API}"

# RunPod's startSsh convention: it injects PUBLIC_KEY but expects the image itself to configure
# and start sshd.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd
    echo "[entrypoint] sshd started"
fi

echo "[entrypoint] $(date -u +%FT%TZ) handing off to autostop watchdog"
exec python3 /app/stack/autostop.py
