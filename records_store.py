# -*- coding: utf-8 -*-

import os
import sqlite3
from datetime import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS enforcement_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    date TEXT NOT NULL,
    time_period TEXT NOT NULL,
    location TEXT NOT NULL,
    reason TEXT NOT NULL,
    plate_number TEXT NOT NULL,
    source_filename TEXT,
    image_path TEXT,
    mode TEXT NOT NULL,
    excel_file TEXT
);

CREATE INDEX IF NOT EXISTS idx_records_created_at
ON enforcement_records(created_at);

CREATE INDEX IF NOT EXISTS idx_records_plate_number
ON enforcement_records(plate_number);

CREATE INDEX IF NOT EXISTS idx_records_date
ON enforcement_records(date);
"""


def default_db_path(base_dir):
    return os.path.join(base_dir, "parking_records.sqlite3")


def init_db(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


def add_records(db_path, records):
    if not records:
        return 0

    init_db(db_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for record in records:
        rows.append((
            created_at,
            record.get("date", ""),
            record.get("time_period", ""),
            record.get("location", ""),
            record.get("reason", ""),
            record.get("plate_number", ""),
            record.get("source_filename", ""),
            record.get("image_path", ""),
            record.get("mode", ""),
            record.get("excel_file", ""),
        ))

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO enforcement_records (
                created_at, date, time_period, location, reason, plate_number,
                source_filename, image_path, mode, excel_file
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def fetch_recent_records(db_path, limit=100):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, date, time_period, location, reason,
                   plate_number, source_filename, image_path, mode, excel_file
            FROM enforcement_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_record_count(db_path):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM enforcement_records").fetchone()
    return int(row[0]) if row else 0


def fetch_daily_counts(db_path, limit=14):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT date, COUNT(*) AS count
            FROM enforcement_records
            GROUP BY date
            ORDER BY date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
