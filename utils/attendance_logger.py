import csv
import os
from datetime import datetime
from typing import Literal


class AttendanceLogger:
    """CSV-based attendance logger."""

    def __init__(self, csv_path: str = "attendance_log.csv") -> None:
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "employee_id", "employee_name", "mode"])

    def log(
        self,
        employee_id: str,
        employee_name: str,
        mode: Literal["checkin", "checkout"],
    ) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts, employee_id, employee_name, mode])
