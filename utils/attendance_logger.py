import csv
import os
from datetime import datetime


class AttendanceLogger:
    """CSV logger for employee attendance."""

    def __init__(self, csv_path: str = "attendance_log.csv") -> None:
        self.csv_path = csv_path
        # Ensure the CSV file has a header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "employee_id", "employee_name", "mode"])

    def log(self, employee_id: str, employee_name: str, mode: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, employee_id, employee_name, mode])
        print(
            f"[Attendance] {employee_name} ({employee_id}) {mode} at {timestamp}")
