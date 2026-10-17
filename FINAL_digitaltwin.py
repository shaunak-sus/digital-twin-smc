"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DIGITAL TWIN  —  Single-Station SMC Virtual Machining Center  v3          ║
║   Based on: IIT Indore Thesis (V. Purushothaman, V. Pandhare)               ║
║                                                                              ║
║   Changes from v2 (per project spec PDF):                                   ║
║     1. Actuator detection: displacement-from-rest REMOVED.                  ║
║        Replaced with ROI pixel-difference motion detection (Method B).      ║
║        "Not moving = Idle" — fixes the timer-never-stops bug.               ║
║     2. Compact HUD: 260px left panel, no large working-zone overlay.        ║
║     3. draw_zone() calls for object_zone removed from render loop.          ║
║     4. CSV now logs: cycle_number, cycle_start_time, cycle_end_time,        ║
║        total_cycle_time, piston_active_time, actuator_active_time,          ║
║        emergency_events.                                                     ║
║     5. Piston keeps displacement-from-rest (correct per spec).              ║
║     6. live_machine_status.json still written each frame.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import json
import math
import os
import time
import pandas as pd
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# ═══════════════════════════ USER CONFIG ════════════════════════════════════
MODEL_PATH      = r"weights/machine.pt"
VEST_MODEL_PATH = r"weights/vest.pt"
VIDEO_SOURCE = 0          # webcam index (int) OR video file path (str)
OUTPUT_CSV   = r"cycle_log.csv"
LIVE_STATUS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "live_machine_status.json",
)

# ── Vest Detection Config ────────────────────────────────────────────────────
VEST_CONF_THRESH    = 0.1   # min confidence for vest model detections
# PPE model class IDs (best (4).pt)
PPE_CLASS_PERSON    = 5
PPE_CLASS_VEST      = 1
PPE_CLASS_NO_VEST   = 4
# Fraction of person bbox that must overlap with vest/no-vest box to assign
VEST_IOU_THRESH     = 0.25
# Cooldown: once non-compliant, keep alert for N seconds even if detection lost
VEST_ALERT_COOLDOWN = 3.0

FRAME_W, FRAME_H = 1280, 720   # resize every input frame to this

# ── Calibration ──────────────────────────────────────────────────────────────
CALIBRATION_FRAMES = 60        # frames to average for rest positions

# ── Zone & Activity Thresholds ───────────────────────────────────────────────
ZONE_PADDING        = 50       # px padding around machine bboxes for zones
OBJECT_ZONE_PADDING = 80       # px padding around the combined working area

# Piston: displacement from REST needed to declare ACTIVE (px)
PISTON_ACTIVE_THRESH   = 8

# Actuator: ROI pixel-difference threshold to declare ACTIVE
# Tune this value — lower = more sensitive, higher = less sensitive
ACTUATOR_MOTION_THRESH = 15    # mean absolute pixel difference per pixel

# Minimum consecutive frames a condition must hold before FSM transitions
MIN_FRAMES_IN_STATE = 5

# Cooldown: after last ACTIVE frame, component stays marked active for N sec
PISTON_COOLDOWN_SEC   = 1.2
ACTUATOR_COOLDOWN_SEC = 0.4   # kept short because ROI method reacts quickly

# Object memory: keep position for N frames after YOLO loses the detection
MAX_LOST_FRAMES = 25

# HUD width (compact)
HUD_W = 260
# ════════════════════════════════════════════════════════════════════════════


# ── MODELS ───────────────────────────────────────────────────────────────────
model      = YOLO(MODEL_PATH)
vest_model = YOLO(VEST_MODEL_PATH)
device     = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
vest_model.to(device)
print(f"[INIT] Machine YOLO on {device.upper()}")
print(f"[INIT] Vest/PPE YOLO on {device.upper()}")

if isinstance(VIDEO_SOURCE, int):
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(f"Camera res: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

cv2.namedWindow("Digital Twin v3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Digital Twin v3", FRAME_W, FRAME_H)


# ═══════════════════════════ STATE MACHINE DEFINITION ═══════════════════════
STATES = [
    "IDLE",
    "OBJECT_PRESENT",
    "PISTON_EXTENDING",
    "PISTON_RETRACTING",
    "ACTUATOR_ACTIVE",
    "CYCLE_COMPLETE",
    "EMERGENCY",
]

STATE_COLOR = {
    "IDLE":               (160, 160, 160),
    "OBJECT_PRESENT":     (0,   165, 255),
    "PISTON_EXTENDING":   (0,   255, 255),
    "PISTON_RETRACTING":  (255, 200,   0),
    "ACTUATOR_ACTIVE":    (128, 255,   0),
    "CYCLE_COMPLETE":     (0,   255,   0),
    "EMERGENCY":          (0,     0, 255),
}

STATE_DESC = {
    "IDLE":               "Waiting for workpiece",
    "OBJECT_PRESENT":     "Workpiece in zone",
    "PISTON_EXTENDING":   "Piston extending",
    "PISTON_RETRACTING":  "Piston retracting",
    "ACTUATOR_ACTIVE":    "Actuator operating",
    "CYCLE_COMPLETE":     "Cycle done",
    "EMERGENCY":          "HALTED — green light off",
}


# ═════════════════════════ CALIBRATION STORAGE ══════════════════════════════
calib = {
    "done":           False,
    "piston_boxes":   [],
    "actuator_boxes": [],
    # Filled after calibration:
    "piston_rest":    None,   # (cx, cy)
    "actuator_rest":  None,
    "piston_zone":    None,   # [x1,y1,x2,y2]
    "actuator_zone":  None,
    "object_zone":    None,
    "actuator_roi":   None,   # [x1,y1,x2,y2] fixed ROI for pixel-diff
    "end_zone":       None,   # [x1,y1,x2,y2] cycle-complete trigger zone
}

# Offset of END_ZONE from actuator_zone — TUNE THIS to your conveyor U-turn point
END_ZONE_OFFSET_X = -120  # px from actuator left edge; negative = left of actuator
END_ZONE_WIDTH     = 160  # px wide

def compute_zones():
    pb = calib["piston_boxes"]
    ab = calib["actuator_boxes"]

    avg_p = [int(sum(c) / len(c)) for c in zip(*pb)]
    avg_a = [int(sum(c) / len(c)) for c in zip(*ab)]

    def cx_cy(box):
        return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2

    def expand(box, pad):
        return [
            max(0,       box[0] - pad),
            max(0,       box[1] - pad),
            min(FRAME_W, box[2] + pad),
            min(FRAME_H, box[3] + pad),
        ]

    pz = expand(avg_p, ZONE_PADDING)
    az = expand(avg_a, ZONE_PADDING)

    oz = [
        max(0,       min(pz[0], az[0]) - OBJECT_ZONE_PADDING),
        max(0,       min(pz[1], az[1]) - OBJECT_ZONE_PADDING),
        min(FRAME_W, max(pz[2], az[2]) + OBJECT_ZONE_PADDING),
        min(FRAME_H, max(pz[3], az[3]) + OBJECT_ZONE_PADDING),
    ]

    # Actuator ROI = tighter box around actuator avg position for pixel-diff
    roi_pad = ZONE_PADDING + 10
    actuator_roi = expand(avg_a, roi_pad)

    calib["piston_rest"]   = cx_cy(avg_p)
    calib["actuator_rest"] = cx_cy(avg_a)
    calib["piston_zone"]   = pz
    calib["actuator_zone"] = az
    calib["object_zone"]   = oz
    calib["actuator_roi"]  = actuator_roi
    # END_ZONE: placed to the LEFT of actuator_zone (toward conveyor U-turn).
    # Adjust END_ZONE_OFFSET_X negative to move left, positive to move right.
    ez_x1 = max(0,       az[0] + END_ZONE_OFFSET_X)
    ez_x2 = min(FRAME_W, az[0] + END_ZONE_OFFSET_X + END_ZONE_WIDTH)
    ez_y1 = max(0,       az[1] - 30)
    ez_y2 = min(FRAME_H, az[3] + 30)
    calib["end_zone"] = [ez_x1, ez_y1, ez_x2, ez_y2]
    print(f"  End zone (cycle-complete trigger): {calib['end_zone']}")
    calib["done"]          = True

    print("[CALIB] Complete.")
    print(f"  Piston   rest: {calib['piston_rest']},  zone: {pz}")
    print(f"  Actuator rest: {calib['actuator_rest']}, zone: {az}")
    print(f"  Actuator ROI (pixel-diff): {actuator_roi}")
    print(f"  Object zone: {oz}")


# ═════════════════════════ RUNTIME STATE VARS ═══════════════════════════════
fsm_state        = "IDLE"
prev_fsm_state   = "IDLE"
state_frame_cnt  = 0
cycle_log        = []
cycle_start      = None
cycle_start_time = None   # wall-clock string for CSV
cycle_number     = 0
emergency_events = 0
s_time           = {}

# Object tracking
last_obj_pos  = None
last_obj_id   = None
obj_lost_cnt  = 0
prev_obj_in_zone = False
cycle_obj_id  = None

# Piston direction
piston_history   = deque(maxlen=10)
piston_direction = "NONE"

# Cooldown trackers
piston_last_active   = 0.0
actuator_last_active = 0.0

# ROI motion detection — keeps the previous grayscale ROI crop
prev_actuator_roi_gray = None
# New motion tracking for workpiece stability
prev_obj_center = None  # (x, y) of previous object centroid
stationary_time = 0.0   # seconds the object has remained stationary
OBJECT_MOTION_THRESH = 5  # pixel distance threshold to consider movement
STATIONARY_WAIT_SEC = 0.5   # required stationary duration before cycle complete
actuator_motion_score  = 0.0   # for HUD display

prev_frame_time             = time.time()
last_loop_time              = time.time()
current_cycle_piston_time   = 0.0
current_cycle_actuator_time = 0.0
total_piston_time           = 0.0
total_actuator_time         = 0.0
waiting_for_next_part       = False
piston_seen                 = False
actuator_seen               = False
part_removed_while_waiting  = False
end_zone_frame_count        = 0    # consecutive frames object seen in END_ZONE
# Vest / PPE safety state
vest_last_noncompliant = 0.0   # timestamp of last non-compliant frame
vest_workers_total     = 0     # persons detected this frame
vest_workers_compliant = 0     # persons with Safety Vest
vest_workers_violation = 0     # persons with NO-Safety Vest
vest_status            = "NO_WORKER"  # NO_WORKER | COMPLIANT | VIOLATION


# ════════════════════════ HELPER FUNCTIONS ══════════════════════════════
# Computes intersection area / area of box1 (used for person<→label matching)
def bbox_iou_area(b1, b2):
    """Return intersection_area / b1_area."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area1 = max(1, (b1[2]-b1[0]) * (b1[3]-b1[1]))
    return inter / area1


def run_vest_detection(frame_bgr):
    """
    Run the PPE model on frame_bgr.
    Returns (vest_status, workers_total, workers_compliant, workers_violation)
      vest_status: 'NO_WORKER' | 'COMPLIANT' | 'VIOLATION'
    Also draws coloured bboxes on frame_bgr in-place.
    """
    res = vest_model(frame_bgr, imgsz=640, conf=VEST_CONF_THRESH, verbose=False)[0]

    persons    = []   # list of [x1,y1,x2,y2]
    vest_boxes = []   # list of (cls_id, [x1,y1,x2,y2])

    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if cls_id == PPE_CLASS_PERSON:
                persons.append([x1,y1,x2,y2])
            elif cls_id in (PPE_CLASS_VEST, PPE_CLASS_NO_VEST):
                vest_boxes.append((cls_id, [x1,y1,x2,y2]))

    compliant = 0
    violation = 0
    for pb in persons:
        best_cls  = None
        best_iou  = 0.0
        for vcls, vb in vest_boxes:
            iou = bbox_iou_area(pb, vb)
            if iou > best_iou:
                best_iou = iou
                best_cls = vcls
        if best_iou >= VEST_IOU_THRESH:
            if best_cls == PPE_CLASS_VEST:
                compliant += 1
                # green outline on person
                cv2.rectangle(frame_bgr, (pb[0],pb[1]), (pb[2],pb[3]), (0,220,0), 2)
                cv2.putText(frame_bgr, "VEST OK", (pb[0], pb[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
            else:
                violation += 1
                # red outline on person
                cv2.rectangle(frame_bgr, (pb[0],pb[1]), (pb[2],pb[3]), (0,0,255), 2)
                cv2.putText(frame_bgr, "NO VEST!", (pb[0], pb[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        else:
            # no vest label near this person — treat as unknown, draw white
            cv2.rectangle(frame_bgr, (pb[0],pb[1]), (pb[2],pb[3]), (180,180,180), 1)

    if len(persons) == 0:
        status = "NO_WORKER"
    elif violation > 0:
        status = "VIOLATION"
    else:
        status = "COMPLIANT"

    return status, len(persons), compliant, violation

def inside(pt, zone):
    if pt is None or zone is None:
        return False
    x, y = pt
    return zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]

def displacement(pt, rest):
    if pt is None or rest is None:
        return 0.0
    return math.hypot(pt[0] - rest[0], pt[1] - rest[1])

def draw_zone(img, zone, color, label):
    if zone is None:
        return
    cv2.rectangle(img, (zone[0], zone[1]), (zone[2], zone[3]), color, 2)
    cv2.putText(img, label, (zone[0] + 5, zone[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def draw_rest_marker(img, rest, color, label):
    if rest is None:
        return
    cv2.drawMarker(img, rest, color, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(img, f"{label} REST", (rest[0] + 12, rest[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_bar(img, value, max_val, threshold, x, y, w, h, color, label):
    """Horizontal bar: value vs max_val, threshold line at threshold/max_val."""
    ratio  = min(value / max(max_val, 1), 1.0)
    bar_w  = int(w * ratio)
    bar_col = (0, 0, 220) if value > threshold else (0, 180, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + bar_w, y + h), bar_col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (130, 130, 130), 1)
    tx = x + int(w * threshold / max(max_val, 1))
    cv2.line(img, (tx, y), (tx, y + h), (255, 255, 0), 1)
    cv2.putText(img, f"{label}: {value:.1f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

def log_event(event, extra=None):
    entry = {"timestamp": time.strftime("%H:%M:%S"), "event": event, "extra": extra}
    cycle_log.append(entry)
    print(f"  [LOG] {entry}")

def write_live_status(status_dict):
    try:
        with open(LIVE_STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
    except OSError as e:
        print(f"[STATUS] Write failed: {e}")

def transition(new_state):
    global fsm_state, prev_fsm_state, state_frame_cnt, s_time
    if new_state != fsm_state:
        prev_fsm_state  = fsm_state
        fsm_state       = new_state
        state_frame_cnt = 0
        s_time[new_state] = time.time()
        print(f"  [FSM] {prev_fsm_state} → {new_state}")

def duration_since(state_name):
    return round(time.time() - s_time.get(state_name, time.time()), 2)


# ════════════════════════════ MAIN LOOP ═════════════════════════════════════
print("[START] Running.  q=quit  r=reset  s=skip state  d=force recalib")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[END] Video finished or read error.")
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    now   = time.time()

    # ── YOLO INFERENCE ────────────────────────────────────────────────────────
    results = model.track(
        frame,
        imgsz=640,
        conf=0.15,
        persist=True,
        verbose=False,
    )

    raw_piston   = None
    raw_actuator = None
    object_dets  = {}
    green_light  = False

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label  = results[0].names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if label == "green light":
                green_light = True
            elif label == "piston" and raw_piston is None:
                raw_piston   = (x1, y1, x2, y2, cx, cy)
            elif label == "actuator" and raw_actuator is None:
                raw_actuator = (x1, y1, x2, y2, cx, cy)
            elif label == "object":
                tid = int(box.id[0]) if box.id is not None else 0
                object_dets[tid] = (cx, cy, x1, y1, x2, y2)

    # ── VEST / PPE DETECTION ──────────────────────────────────────────────────
    vest_status, vest_workers_total, vest_workers_compliant, vest_workers_violation = \
        run_vest_detection(frame)
    if vest_status == "VIOLATION":
        vest_last_noncompliant = now
    # Keep status as VIOLATION for cooldown seconds after last violation frame
    if (vest_status != "VIOLATION" and
            (now - vest_last_noncompliant) < VEST_ALERT_COOLDOWN):
        vest_status = "VIOLATION"

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    if not calib["done"]:
        if raw_piston:
            calib["piston_boxes"].append(raw_piston[:4])
        if raw_actuator:
            calib["actuator_boxes"].append(raw_actuator[:4])

        if (len(calib["piston_boxes"])   >= CALIBRATION_FRAMES and
                len(calib["actuator_boxes"]) >= CALIBRATION_FRAMES):
            compute_zones()

    # ── PISTON: DISPLACEMENT FROM REST ───────────────────────────────────────
    piston_disp = 0.0
    if calib["done"] and raw_piston:
        piston_disp = displacement(
            (raw_piston[4], raw_piston[5]), calib["piston_rest"])

    # ── ACTUATOR: ROI PIXEL-DIFFERENCE MOTION DETECTION ──────────────────────
    # "Not moving = Idle" — fixes the infinite-active bug from v2.
    # We compare the current grayscale ROI crop against the previous frame's.
    # Mean absolute pixel difference > ACTUATOR_MOTION_THRESH → moving.
    actuator_motion_score = 0.0
    if calib["done"] and calib["actuator_roi"] is not None:
        roi = calib["actuator_roi"]
        roi_crop = frame[roi[1]:roi[3], roi[0]:roi[2]]
        if roi_crop.size > 0:
            roi_gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
            if prev_actuator_roi_gray is not None and \
               prev_actuator_roi_gray.shape == roi_gray.shape:
                diff = cv2.absdiff(roi_gray, prev_actuator_roi_gray)
                actuator_motion_score = float(np.mean(diff))
            prev_actuator_roi_gray = roi_gray.copy()

    # ── ACTIVITY FLAGS + COOLDOWNS ───────────────────────────────────────────
    # Piston: cooldown-based (displacement signal is clean)
    if piston_disp > PISTON_ACTIVE_THRESH:
        piston_last_active = now
    piston_active = (now - piston_last_active) < PISTON_COOLDOWN_SEC

    # Actuator: motion-score based with short cooldown
    # NOTE: actuator_last_active only refreshes when there IS actual motion.
    # This is the fix — the old code refreshed on displacement (always true
    # when extended), causing the timer to never stop.
    if actuator_motion_score > ACTUATOR_MOTION_THRESH:
        actuator_last_active = now
    actuator_active = (now - actuator_last_active) < ACTUATOR_COOLDOWN_SEC

    # ── CYCLE TIMING ─────────────────────────────────────────────────────────
    dt = max(0.0, now - last_loop_time)
    if cycle_start is not None:
        if piston_active:
            current_cycle_piston_time   += dt
            total_piston_time           += dt
        if actuator_active:
            current_cycle_actuator_time += dt
            total_actuator_time         += dt
    last_loop_time = now

    # ── PISTON DIRECTION ─────────────────────────────────────────────────────
    if raw_piston:
        piston_history.append(raw_piston[4])
    if len(piston_history) >= 4:
        xs    = list(piston_history)
        trend = xs[-1] - xs[0]
        if piston_active:
            piston_direction = ("EXTENDING"  if trend >  2 else
                                "RETRACTING" if trend < -2 else "HOLDING")
        else:
            piston_direction = "IDLE"
    else:
        piston_direction = "IDLE"
    # ── OBJECT TRACKING ───────────────────────────────────────────────────────
    obj_pos = None
    obj_id = None
    if calib["done"]:
        # Restrict to object_zone normally; once waiting for cycle completion,
        # extend the search region to include end_zone so the same object
        # continues to be tracked as it travels to the END_ZONE trigger point.
        search_zone = calib["object_zone"]
        if fsm_state == "ACTUATOR_ACTIVE" and waiting_for_next_part:
            oz, ez = calib["object_zone"], calib["end_zone"]
            search_zone = [min(oz[0], ez[0]), min(oz[1], ez[1]),
                           max(oz[2], ez[2]), max(oz[3], ez[3])]
        best_tid = None
        for tid, det in object_dets.items():
            if inside((det[0], det[1]), search_zone):
                best_tid = tid
                break
        if best_tid is not None:
            obj_pos      = (object_dets[best_tid][0], object_dets[best_tid][1])
            obj_id       = best_tid
            last_obj_pos = obj_pos
            last_obj_id  = obj_id
            obj_lost_cnt = 0
        elif last_obj_pos is not None:
            if obj_lost_cnt < MAX_LOST_FRAMES:
                obj_pos       = last_obj_pos
                obj_id        = last_obj_id
                obj_lost_cnt += 1
            else:
                last_obj_pos = None
                last_obj_id  = None
                obj_lost_cnt = 0
    elif object_dets:
        first_tid = list(object_dets.keys())[0]
        first   = object_dets[first_tid]
        obj_pos = (first[0], first[1])
        obj_id = first_tid

    obj_in_zone = obj_pos is not None

    # Runtime trackers for seen components in the current cycle
    
    if calib["done"]:
        if piston_active:
            piston_seen = True
        if actuator_active:
            actuator_seen = True
    inside_end_zone = False
    # ── FINITE STATE MACHINE ─────────────────────────────────────────────────
    def debounce(condition):
        global state_frame_cnt
        if condition:
            state_frame_cnt += 1
        else:
            state_frame_cnt = 0
        return state_frame_cnt >= MIN_FRAMES_IN_STATE

    # ── END_ZONE CYCLE COMPLETION (primary cycle-end signal) ─────────────────
    # Rule: object detected in END_ZONE for MIN_FRAMES_IN_STATE consecutive
    # frames while a cycle is running → log cycle → reset → back to IDLE.
    # This runs independently of all FSM sub-states.
    if (calib["done"]
            and calib["end_zone"] is not None
            and cycle_start is not None
            and fsm_state not in ("IDLE", "CYCLE_COMPLETE", "EMERGENCY")
            and obj_pos is not None):
        if inside(obj_pos, calib["end_zone"]):
            end_zone_frame_count += 1
        else:
            end_zone_frame_count = 0
        inside_end_zone = end_zone_frame_count > 0

        if end_zone_frame_count >= MIN_FRAMES_IN_STATE:
            end_zone_frame_count = 0
            # Log and close the cycle
            cycle_end_time = time.strftime("%H:%M:%S")
            total = round(now - cycle_start, 2)
            log_event("CYCLE_COMPLETE",
                      extra={
                          "cycle_number":         cycle_number,
                          "cycle_start_time":     cycle_start_time,
                          "cycle_end_time":       cycle_end_time,
                          "total_cycle_time":     total,
                          "piston_active_time":   round(current_cycle_piston_time,   2),
                          "actuator_active_time": round(current_cycle_actuator_time, 2),
                          "emergency_events":     emergency_events,
                      })
            # Reset for next cycle
            cycle_start                 = None
            cycle_start_time            = None
            current_cycle_piston_time   = 0.0
            current_cycle_actuator_time = 0.0
            waiting_for_next_part       = False
            piston_seen                 = False
            actuator_seen               = False
            transition("IDLE")
    else:
        end_zone_frame_count = 0

    # ── NORMAL FSM CHAIN ─────────────────────────────────────────────────────
    if not calib["done"]:
        pass

    elif not green_light and fsm_state not in ("IDLE", "EMERGENCY"):
        if debounce(True):
            transition("EMERGENCY")
            emergency_events += 1
            log_event("EMERGENCY_STOP",
                      extra={"at_state": prev_fsm_state,
                             "elapsed":  duration_since(prev_fsm_state)})
            cycle_start = None

    elif fsm_state == "EMERGENCY":
        if green_light and debounce(True):
            transition("IDLE")

    elif fsm_state == "IDLE":
        if cycle_start is None and obj_in_zone:
            cycle_number    += 1
            cycle_start      = now
            cycle_start_time = time.strftime("%H:%M:%S")
            current_cycle_piston_time   = 0.0
            current_cycle_actuator_time = 0.0
            piston_seen                 = False
            actuator_seen               = False
            waiting_for_next_part       = False
            part_removed_while_waiting  = False
            cycle_obj_id                = obj_id
        if debounce(obj_in_zone):
            transition("OBJECT_PRESENT")

    elif fsm_state == "OBJECT_PRESENT":
        if debounce(piston_active and piston_direction == "EXTENDING"):
            transition("PISTON_EXTENDING")

    elif fsm_state == "PISTON_EXTENDING":
        if debounce(piston_active and piston_direction == "RETRACTING"):
            transition("PISTON_RETRACTING")
        elif debounce(not piston_active):
            transition("PISTON_RETRACTING")

    elif fsm_state == "PISTON_RETRACTING":
        if debounce(not piston_active and actuator_active):
            transition("ACTUATOR_ACTIVE")

    elif fsm_state == "ACTUATOR_ACTIVE":
        if not actuator_active:
            waiting_for_next_part = True

    # ── RENDERING ─────────────────────────────────────────────────────────────
    vis = frame.copy()

    # Required Debugging instrumentation
    cv2.putText(
        vis,
        f"WAIT={waiting_for_next_part} "
        f"REMOVED={part_removed_while_waiting} "
        f"OBJ={obj_in_zone}",
        (280, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # Machine zones (piston + actuator only; object working zone hidden)
    draw_zone(vis, calib["piston_zone"],   (0, 255, 255), "PISTON")
    draw_zone(vis, calib["actuator_zone"], (255, 128,   0), "ACTUATOR")
    # Actuator ROI outline (motion detection window)
    if calib["actuator_roi"] is not None:
        roi = calib["actuator_roi"]
        roi_col = (0, 80, 220) if actuator_active else (60, 60, 100)
        cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), roi_col, 1)
        cv2.putText(vis, "ROI", (roi[0] + 3, roi[1] + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, roi_col, 1)

    # Rest markers
    draw_rest_marker(vis, calib["piston_rest"],   (0, 255, 255), "P")
    draw_rest_marker(vis, calib["actuator_rest"], (255, 128,   0), "A")

    # Object dot
    if obj_pos:
        cv2.circle(vis, obj_pos, 10, (0, 255, 0), 2)
        cv2.putText(vis, "OBJ", (obj_pos[0] + 12, obj_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # END_ZONE overlay — always drawn once per frame after calibration
    draw_zone(vis, calib["end_zone"], (0, 220, 255), "END ZONE")
    dbg_obj = f"{obj_pos}" if obj_pos else "None"
    cv2.putText(vis, f"obj={dbg_obj}  in_end_zone={inside_end_zone}  fsm={fsm_state}",
                (10, FRAME_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    # YOLO bounding boxes
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label  = results[0].names[cls_id]
            conf   = float(box.conf[0])
            col = (128, 128, 128)
            if label == "green light":
                col = (0, 220, 0) if green_light else (0, 0, 220)
            elif label == "piston":
                col = (0, 255, 255) if piston_active else (60, 80, 60)
            elif label == "actuator":
                col = (255, 128, 0) if actuator_active else (80, 60, 60)
            elif label == "object":
                col = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # ── COMPACT HUD (260px left panel) ───────────────────────────────────────
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (HUD_W, FRAME_H), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
    cv2.line(vis, (HUD_W, 0), (HUD_W, FRAME_H), (60, 60, 60), 1)

    y     = 26
    pad   = 10
    st_col = STATE_COLOR[fsm_state]

    # State
    cv2.putText(vis, "STATE", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1); y += 20
    cv2.putText(vis, fsm_state, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, st_col, 2);          y += 18
    cv2.putText(vis, STATE_DESC[fsm_state], (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1); y += 16
    cv2.line(vis, (pad, y), (HUD_W - pad, y), (55, 55, 55), 1);     y += 10

    # Green light
    gl_col  = (0, 210, 0) if green_light else (0, 0, 210)
    gl_txt  = "GREEN LIGHT  ON" if green_light else "GREEN LIGHT  OFF"
    cv2.putText(vis, gl_txt, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, gl_col, 1);          y += 18
    cv2.line(vis, (pad, y), (HUD_W - pad, y), (55, 55, 55), 1);     y += 10

    # Piston displacement bar
    bar_w = HUD_W - 2 * pad
    draw_bar(vis, piston_disp, PISTON_ACTIVE_THRESH * 3,
             PISTON_ACTIVE_THRESH, pad, y, bar_w, 12,
             (0, 255, 255), "Piston disp px");                        y += 20

    # Actuator motion-score bar
    draw_bar(vis, actuator_motion_score, ACTUATOR_MOTION_THRESH * 3,
             ACTUATOR_MOTION_THRESH, pad, y, bar_w, 12,
             (255, 128, 0), "Actuator motion");                       y += 20

    # Piston direction
    dir_col = ((0, 255, 128) if piston_direction == "EXTENDING" else
               (0, 128, 255) if piston_direction == "RETRACTING" else
               (130, 130, 130))
    cv2.putText(vis, f"Piston: {piston_direction}", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, dir_col, 1);         y += 16

    # Object
    obj_col = (0, 255, 0) if obj_in_zone else (110, 110, 110)
    obj_txt = "Object: DETECTED" if obj_in_zone else "Object: absent"
    if obj_lost_cnt > 0:
        obj_txt += f" (mem {obj_lost_cnt}f)"
    cv2.putText(vis, obj_txt, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, obj_col, 1);         y += 16

    # Debounce bar
    db_ratio = min(state_frame_cnt / max(MIN_FRAMES_IN_STATE, 1), 1.0)
    cv2.rectangle(vis, (pad, y), (pad + bar_w, y + 8), (35, 35, 35), -1)
    cv2.rectangle(vis, (pad, y), (pad + int(bar_w * db_ratio), y + 8),
                  (70, 70, 180), -1)
    cv2.putText(vis, f"Debounce {state_frame_cnt}/{MIN_FRAMES_IN_STATE}",
                (pad, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                (120, 120, 180), 1);                                  y += 16
    cv2.line(vis, (pad, y), (HUD_W - pad, y), (55, 55, 55), 1);     y += 10

    # Cycle stats
    completed = sum(1 for e in cycle_log if e["event"] == "CYCLE_COMPLETE")
    emerg_cnt = sum(1 for e in cycle_log if e["event"] == "EMERGENCY_STOP")
    elapsed   = round(now - cycle_start, 1) if cycle_start else 0.0

    for txt, col in [
        (f"Cycles done:  {completed}",                         (200, 200, 200)),
        (f"Emergencies:  {emerg_cnt}",                         (200, 100, 100)),
        (f"Cycle timer:  {elapsed}s",                          (200, 200, 200)),
        (f"Piston use:   {current_cycle_piston_time:.1f}s",    (0,   210, 210)),
        (f"Actuator use: {current_cycle_actuator_time:.1f}s",  (255, 160,  80)),
    ]:
        cv2.putText(vis, txt, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1);          y += 16

    cv2.line(vis, (pad, y), (HUD_W - pad, y), (55, 55, 55), 1);     y += 10

    # ── VEST / PPE STATUS (HUD) ───────────────────────────────────────────
    vest_col = (
        (0, 0, 220)   if vest_status == "VIOLATION"  else
        (0, 200, 0)   if vest_status == "COMPLIANT"  else
        (130, 130, 130)
    )
    cv2.putText(vis, "SAFETY VEST", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1); y += 16
    cv2.putText(vis, vest_status, (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, vest_col, 2);         y += 18
    cv2.putText(vis,
                f"Workers: {vest_workers_total}  OK:{vest_workers_compliant}  ! :{vest_workers_violation}",
                (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)
    y += 16
    cv2.line(vis, (pad, y), (HUD_W - pad, y), (55, 55, 55), 1);     y += 10

    # Full-frame VIOLATION banner (flashing red bar across top of image area)
    if vest_status == "VIOLATION":
        flash = int(now * 3) % 2 == 0   # toggles ~3 Hz
        banner_col = (0, 0, 200) if flash else (0, 0, 140)
        cv2.rectangle(vis, (HUD_W, 0), (FRAME_W, 44), banner_col, -1)
        cv2.putText(vis,
                    f"⚠  SAFETY VIOLATION  —  WORKER WITHOUT VEST  ({vest_workers_violation} person(s))",
                    (HUD_W + 12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Calibration status
    if not calib["done"]:
        pf   = len(calib["piston_boxes"])
        af   = len(calib["actuator_boxes"])
        prog = min(100, int(max(pf, af) / CALIBRATION_FRAMES * 100))
        cv2.putText(vis, f"CALIBRATING {prog}%", (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1); y += 18
        cv2.putText(vis, f"P:{pf}/{CALIBRATION_FRAMES} A:{af}/{CALIBRATION_FRAMES}",
                    (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (140, 140, 140), 1);                                y += 16
    else:
        cv2.putText(vis, "CAL: LOCKED", (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 90), 1);  y += 16

    # FPS
    fps_val         = 1.0 / max(now - prev_frame_time, 1e-6)
    prev_frame_time = now
    cv2.putText(vis, f"FPS: {fps_val:.1f}", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    # Key hints
    cv2.putText(vis, "q quit  r reset  s skip  d recalib",
                (pad, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1)

    # ── LIVE STATUS JSON ─────────────────────────────────────────────────────
    write_live_status({
        "timestamp":                    time.strftime("%Y-%m-%d %H:%M:%S"),
        "state":                        fsm_state,
        "state_description":            STATE_DESC[fsm_state],
        "object_present":               obj_in_zone,
        "piston_state":                 "ACTIVE" if piston_active   else "IDLE",
        "actuator_state":               "ACTIVE" if actuator_active else "IDLE",
        "green_light_state":            "ON"     if green_light     else "OFF",
        "piston_displacement_px":       round(piston_disp, 2),
        "actuator_motion_score":        round(actuator_motion_score, 2),
        "actuator_motion_threshold":    ACTUATOR_MOTION_THRESH,
        "piston_direction":             piston_direction,
        "cycle_number":                 cycle_number,
        "cycle_elapsed_s":              elapsed,
        "current_cycle_piston_time":    round(current_cycle_piston_time,   2),
        "current_cycle_actuator_time":  round(current_cycle_actuator_time, 2),
        "total_piston_time":            round(total_piston_time,   2),
        "total_actuator_time":          round(total_actuator_time, 2),
        "completed_cycles":             completed,
        "emergency_events":             emergency_events,
        "calibrated":                   calib["done"],
        "fps":                          round(fps_val, 2),
        "vest_status":                  vest_status,
        "vest_workers_total":           vest_workers_total,
        "vest_workers_compliant":        vest_workers_compliant,
        "vest_workers_violation":        vest_workers_violation,
    })

    cv2.imshow("Digital Twin v3", vis)

    # ── KEYBOARD ──────────────────────────────────────────────────────────────
    prev_obj_in_zone = obj_in_zone
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        fsm_state                   = "IDLE"
        state_frame_cnt             = 0
        cycle_start                 = None
        cycle_start_time            = None
        current_cycle_piston_time   = 0.0
        current_cycle_actuator_time = 0.0
        last_obj_pos                = None
        obj_lost_cnt                = 0
        piston_history.clear()
        prev_actuator_roi_gray      = None
        end_zone_frame_count        = 0
        print("[RESET] Full reset.")
    elif key == ord('s'):
        idx             = STATES.index(fsm_state)
        fsm_state       = STATES[(idx + 1) % (len(STATES) - 1)]
        state_frame_cnt = 0
        print(f"[SKIP] → {fsm_state}")
    elif key == ord('d'):
        calib.update({
            "done": False, "piston_boxes": [], "actuator_boxes": [],
            "piston_rest": None, "actuator_rest": None,
            "piston_zone": None, "actuator_zone": None,
            "object_zone": None, "actuator_roi": None,
        })
        fsm_state                   = "IDLE"
        cycle_start                 = None
        cycle_start_time            = None
        current_cycle_piston_time   = 0.0
        current_cycle_actuator_time = 0.0
        prev_actuator_roi_gray      = None
        print("[RECALIB] Cleared — recalibrating now.")

cap.release()
cv2.destroyAllWindows()


# ── SAVE CSV ──────────────────────────────────────────────────────────────────
if cycle_log:
    rows = []
    for e in cycle_log:
        row = {"timestamp": e["timestamp"], "event": e["event"]}
        if isinstance(e.get("extra"), dict):
            row.update(e["extra"])
        rows.append(row)

    # Ensure the required columns appear (fill NaN where absent)
    required_cols = [
        "timestamp", "event", "cycle_number",
        "cycle_start_time", "cycle_end_time", "total_cycle_time",
        "piston_active_time", "actuator_active_time", "emergency_events",
    ]
    df = pd.DataFrame(rows)
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols + [c for c in df.columns if c not in required_cols]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] Log saved → {OUTPUT_CSV}")
    print(df.to_string())
else:
    print("\n[DONE] No events logged.")