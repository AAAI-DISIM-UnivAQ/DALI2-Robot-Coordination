"""
Metrics collector for the DALI2 Multi-Robot Search-and-Rescue demo.

Subscribes to the Redis LINDA channel and records timestamps of key events.
When all 3 victims are delivered (or timeout is reached), writes a summary
to stdout and appends a row to metrics.csv.

Usage:
    python metrics_collector.py [--redis-host localhost] [--redis-port 6379]
                                [--timeout 300] [--victims 3]
                                [--output metrics.csv]

Run this BEFORE launching the demo (or at least before the bridge starts
publishing events).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass, field

import redis

LINDA_RE = re.compile(r"^([^:]+):(.*):([^:]+)$")

NUM_VICTIMS = 3


@dataclass
class RunMetrics:
    start_time: float = 0.0
    end_time: float = 0.0
    # Discovery
    first_sighting_t: float = 0.0  # time of first victim_in_range or vision victim
    sightings: list = field(default_factory=list)       # (t, victim_id, source)
    # Delivery
    deliveries: list = field(default_factory=list)      # (t, victim_id, robot)
    # Sync (heavy lift)
    ready_to_lift_events: list = field(default_factory=list)  # (t, robot, victim)
    lift_now_events: list = field(default_factory=list)       # (t, victim)
    # Vision LLM
    vision_calls_total: int = 0       # screenshots taken
    vision_calls_skipped: int = 0     # pre-filter skipped (result=clear immediately)
    vision_calls_llm: int = 0         # actually sent to LLM
    vision_results: list = field(default_factory=list)  # (t, robot, result)


def parse_linda(payload: str):
    m = LINDA_RE.match(payload)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main():
    p = argparse.ArgumentParser(description="Collect metrics from a rescue run")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--timeout", type=int, default=300,
                   help="Max seconds per run (default: 300 = 5 min)")
    p.add_argument("--victims", type=int, default=3,
                   help="Number of victims expected")
    p.add_argument("--output", default="metrics.csv",
                   help="CSV file to append results to")
    args = p.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port,
                    decode_responses=True)
    ps = r.pubsub(ignore_subscribe_messages=True)
    ps.subscribe("LINDA")

    m = RunMetrics()
    delivered_set: set[str] = set()
    mission_complete = False

    print(f"[metrics] Listening on LINDA channel (timeout {args.timeout}s)...")
    print(f"[metrics] Waiting for events... (will stop after {args.victims} deliveries)")

    m.start_time = time.time()

    try:
        for raw in ps.listen():
            elapsed = time.time() - m.start_time
            if elapsed > args.timeout:
                print(f"[metrics] Timeout ({args.timeout}s) reached.")
                break

            if raw["type"] != "message":
                continue
            payload = raw["data"]
            parsed = parse_linda(payload)
            if parsed is None:
                continue
            to, content, frm = parsed
            t = time.time()

            # --- victim_in_range (proximity detection) ---
            if "victim_in_range(" in content and frm == "sim":
                # victim_in_range(victim_2,-4.000,-3.000,heavy)
                m.sightings.append((elapsed, to, "proximity"))
                if not m.first_sighting_t:
                    m.first_sighting_t = elapsed
                print(f"  [{elapsed:6.1f}s] SIGHTING (proximity): {to} -> {content}")

            # --- vision_result (VLM detection) ---
            elif "vision_result(" in content and frm == "sim":
                m.vision_calls_total += 1
                if "victim(" in content:
                    m.vision_calls_llm += 1
                    m.vision_results.append((elapsed, to, content))
                    m.sightings.append((elapsed, to, "vlm"))
                    if not m.first_sighting_t:
                        m.first_sighting_t = elapsed
                    print(f"  [{elapsed:6.1f}s] SIGHTING (VLM): {to} -> {content}")
                elif "obstacle" in content:
                    m.vision_calls_llm += 1
                    m.vision_results.append((elapsed, to, content))
                elif "clear" in content:
                    # Could be pre-filter skip or LLM saying clear
                    m.vision_calls_skipped += 1
                    m.vision_results.append((elapsed, to, content))

            # --- victim_seen (reported to coordinator) ---
            elif "victim_seen(" in content and to == "coordinator":
                print(f"  [{elapsed:6.1f}s] REPORTED to coordinator: {content} from {frm}")

            # --- ready_to_lift ---
            elif "ready_to_lift(" in content and to == "coordinator":
                m.ready_to_lift_events.append((elapsed, frm, content))
                print(f"  [{elapsed:6.1f}s] READY_TO_LIFT: {frm} -> {content}")

            # --- lift_now ---
            elif "lift_now(" in content and frm == "coordinator":
                m.lift_now_events.append((elapsed, content))
                print(f"  [{elapsed:6.1f}s] LIFT_NOW: -> {to}: {content}")

            # --- delivered ---
            elif "delivered(" in content and frm == "sim":
                # Extract victim id
                vid_match = re.search(r"delivered\((\w+)\)", content)
                vid = vid_match.group(1) if vid_match else content
                if vid not in delivered_set:
                    delivered_set.add(vid)
                    m.deliveries.append((elapsed, vid, to))
                    print(f"  [{elapsed:6.1f}s] DELIVERED: {vid} by {to}")
                    if len(delivered_set) >= args.victims:
                        mission_complete = True
                        m.end_time = time.time()
                        print(f"\n[metrics] MISSION COMPLETE in {elapsed:.1f}s")
                        break

    except KeyboardInterrupt:
        print("\n[metrics] Interrupted by user.")

    # --- Summary ---
    total_time = (m.end_time - m.start_time) if m.end_time else (time.time() - m.start_time)
    sync_latency = None
    if m.ready_to_lift_events and m.lift_now_events:
        first_ready = m.ready_to_lift_events[0][0]
        lift_time = m.lift_now_events[0][0]
        sync_latency = lift_time - first_ready

    victims_rescued = len(delivered_set)
    success_rate = victims_rescued / args.victims  # 0.0 .. 1.0
    outcome = ("COMPLETE" if mission_complete
               else f"TIMEOUT ({victims_rescued}/{args.victims} rescued)")

    print("\n" + "=" * 60)
    print(f"RUN SUMMARY — {outcome}")
    print("=" * 60)
    print(f"  Outcome:                 {outcome}")
    print(f"  Total time:              {total_time:.1f} s (limit: {args.timeout} s)")
    print(f"  Victims rescued:         {victims_rescued}/{args.victims} "
          f"({success_rate*100:.0f}%)")
    if m.first_sighting_t:
        print(f"  First sighting at:       {m.first_sighting_t:.1f} s")
    else:
        print(f"  First sighting at:       NONE")
    print(f"  Vision calls total:      {m.vision_calls_total}")
    print(f"  Vision calls skipped:    {m.vision_calls_skipped} "
          f"({100*m.vision_calls_skipped/max(1,m.vision_calls_total):.0f}%)")
    print(f"  Vision calls to LLM:     {m.vision_calls_llm}")
    if sync_latency is not None:
        print(f"  Heavy-lift sync latency: {sync_latency:.1f} s")
    print("=" * 60)

    # --- Append to CSV ---
    csv_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not csv_exists:
            w.writerow(["completed", "total_time_s", "victims_rescued",
                        "victims_total", "success_rate", "first_sighting_s",
                        "vision_total", "vision_skipped",
                        "vision_skipped_pct", "vision_llm_calls",
                        "sync_latency_s"])
        w.writerow([
            "yes" if mission_complete else "no",
            f"{total_time:.1f}",
            victims_rescued,
            args.victims,
            f"{success_rate:.2f}",
            f"{m.first_sighting_t:.1f}" if m.first_sighting_t else "",
            m.vision_calls_total,
            m.vision_calls_skipped,
            f"{100*m.vision_calls_skipped/max(1,m.vision_calls_total):.0f}",
            m.vision_calls_llm,
            f"{sync_latency:.1f}" if sync_latency is not None else ""
        ])
    print(f"\n[metrics] Results appended to {args.output}")


if __name__ == "__main__":
    main()
