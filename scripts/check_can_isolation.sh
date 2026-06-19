#!/bin/bash
# Check whether the CAN adapter (gs_usb) shares a USB hub with any camera.
#
# motor "loss communication" with CAN bus-errors=0 is caused by the camera's
# high-bandwidth USB traffic delaying CAN frame transport when they share a
# USB hub. This script tells you if they're still on the same hub.
#
# Run it AFTER moving the wrist camera to a different USB port:
#   bash scripts/check_can_isolation.sh

set -e

# --- CAN adapter USB port path (e.g. 3-5.3) ---
can_glob=(/sys/bus/usb/drivers/gs_usb/*-*)
if [ ! -e "${can_glob[0]}" ]; then
    echo "❌ gs_usb (CAN adapter) not found under /sys/bus/usb/drivers/gs_usb/"
    echo "   Is the CAN adapter plugged in?"
    exit 2
fi
can_dev=$(basename "${can_glob[0]}" | sed 's/:.*//')   # strip :1.0 interface
can_hub="${can_dev%.*}"                                  # parent hub = drop last .N
echo "CAN (gs_usb)   USB port = $can_dev   parent hub = $can_hub"

# --- each camera USB port path ---
shared=0
for v in /sys/bus/usb/drivers/uvcvideo/*-*; do
    [ -e "$v" ] || continue
    vdev=$(basename "$v" | sed 's/:.*//')
    vhub="${vdev%.*}"
    # only print one interface per physical camera
    case " $seen " in *" $vdev "*) continue;; esac
    seen="$seen $vdev"
    if [ "$vhub" = "$can_hub" ]; then
        echo "  camera       USB port = $vdev   parent hub = $vhub   ❌ SHARES CAN's hub"
        shared=1
    else
        echo "  camera       USB port = $vdev   parent hub = $vhub   ✓ separate"
    fi
done

echo ""
if [ "$shared" -eq 1 ]; then
    echo "❌ FAIL: a camera still shares the CAN adapter's USB hub."
    echo "   → Move that camera to a USB port on a DIFFERENT controller (blue USB-3 port)."
    exit 1
else
    echo "✓ PASS: CAN adapter has its USB hub to itself. motor timing should be stable now."
    echo "   Next: bash scripts/launch_isolated.sh ... and/or python scripts/diagnose_motors.py --duration 120 --motion"
    exit 0
fi
