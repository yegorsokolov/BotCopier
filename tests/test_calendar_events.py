import csv
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import bisect


def _load_calendar(file: Path):
    times = []
    impacts = []
    ids = []
    with open(file, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            ts, impact, eid = row
            times.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            impacts.append(float(impact))
            ids.append(int(eid))
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    impacts = [impacts[i] for i in order]
    ids = [ids[i] for i in order]
    return times, impacts, ids


def _load_calendar_manual(file: Path):
    times = []
    impacts = []
    ids = []
    with open(file, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            ts, impact, eid = row
            times.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            impacts.append(float(impact))
            ids.append(int(eid))
    for i in range(1, len(times)):
        t = times[i]
        imp = impacts[i]
        eid = ids[i]
        j = i - 1
        while j >= 0 and times[j] > t:
            times[j + 1] = times[j]
            impacts[j + 1] = impacts[j]
            ids[j + 1] = ids[j]
            j -= 1
        times[j + 1] = t
        impacts[j + 1] = imp
        ids[j + 1] = eid
    return times, impacts, ids


def _calendar_event_id_at(times, impacts, ids, ts, window_minutes):
    if not times:
        return -1
    # binary search for insertion point
    left, right = 0, len(times) - 1
    while left <= right:
        mid = (left + right) // 2
        if times[mid] < ts:
            left = mid + 1
        else:
            right = mid - 1
    best = -1
    max_imp = 0.0
    i = left - 1
    while i >= 0 and abs((ts - times[i]).total_seconds()) <= window_minutes * 60:
        if impacts[i] > max_imp:
            max_imp = impacts[i]
            best = ids[i]
        i -= 1
    i = left
    while i < len(times) and abs((ts - times[i]).total_seconds()) <= window_minutes * 60:
        if impacts[i] > max_imp:
            max_imp = impacts[i]
            best = ids[i]
        i += 1
    return best


def _calendar_event_id_at_bsearch(times, impacts, ids, ts, window_minutes):
    if not times:
        return -1
    idx = bisect.bisect_right(times, ts) - 1
    best = -1
    max_imp = 0.0
    i = idx
    while i >= 0 and abs((ts - times[i]).total_seconds()) <= window_minutes * 60:
        if impacts[i] > max_imp:
            max_imp = impacts[i]
            best = ids[i]
        i -= 1
    i = idx + 1
    while i < len(times) and abs((ts - times[i]).total_seconds()) <= window_minutes * 60:
        if impacts[i] > max_imp:
            max_imp = impacts[i]
            best = ids[i]
        i += 1
    return best


def test_event_id_lookup(tmp_path: Path):
    cal = tmp_path / "cal.csv"
    with open(cal, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "impact", "id"])
        writer.writerow(["2024-01-01 10:00:00", "0.5", "1"])
        writer.writerow(["2024-01-01 09:00:00", "1.0", "2"])
        writer.writerow(["2024-01-01 10:15:00", "0.8", "3"])
    times, impacts, ids = _load_calendar(cal)
    assert times == sorted(times)
    assert ids == [2, 1, 3]
    ts = datetime(2024, 1, 1, 10, 5)
    eid = _calendar_event_id_at(times, impacts, ids, ts, window_minutes=20)
    assert eid == 3
    ts2 = datetime(2024, 1, 1, 9, 5)
    eid2 = _calendar_event_id_at(times, impacts, ids, ts2, window_minutes=10)
    assert eid2 == 2
    ts3 = datetime(2024, 1, 1, 8, 0)
    assert _calendar_event_id_at(times, impacts, ids, ts3, window_minutes=59) == -1


def test_event_id_lookup_bsearch(tmp_path: Path):
    cal = tmp_path / "cal.csv"
    with open(cal, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "impact", "id"])
        writer.writerow(["2024-01-01 10:00:00", "0.5", "1"])
        writer.writerow(["2024-01-01 09:00:00", "1.0", "2"])
        writer.writerow(["2024-01-01 10:15:00", "0.8", "3"])
    times, impacts, ids = _load_calendar(cal)
    ts = datetime(2024, 1, 1, 10, 5)
    assert _calendar_event_id_at_bsearch(times, impacts, ids, ts, 20) == 3
    ts2 = datetime(2024, 1, 1, 9, 5)
    assert _calendar_event_id_at_bsearch(times, impacts, ids, ts2, 10) == 2
    ts3 = datetime(2024, 1, 1, 8, 0)
    assert _calendar_event_id_at_bsearch(times, impacts, ids, ts3, 59) == -1


def test_large_calendar_sort(tmp_path: Path):
    random.seed(0)
    cal = tmp_path / "calendar.csv"
    base = datetime(2024, 1, 1)
    with open(cal, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "impact", "id"])
        for i in range(2000):
            ts = base + timedelta(minutes=random.randint(0, 1000))
            writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), "1.0", str(i)])
    start = time.perf_counter()
    _load_calendar_manual(cal)
    manual_time = time.perf_counter() - start
    start = time.perf_counter()
    times, impacts, ids = _load_calendar(cal)
    sorted_time = time.perf_counter() - start
    assert times == sorted(times)
    groups = defaultdict(list)
    for t, eid in zip(times, ids):
        groups[t].append(eid)
    for eids in groups.values():
        if len(eids) > 1:
            assert eids == sorted(eids)
    assert sorted_time < manual_time
