
import json
import csv
from pathlib import Path

from core.report_generator import ReportGenerator


def main():
    reports_dir = Path(__file__).resolve().parent / "reports"

    sessions = sorted(
        reports_dir.glob("session_*/session.json")
    )

    if not sessions:
        print("No saved sessions found!")
        return

    session_file = sessions[-1]
    session_dir = session_file.parent

    with open(session_file, encoding="utf-8") as f:
        session_data = json.load(f)

    performance_file = session_dir / "performance.csv"
    fps_history = []

    if performance_file.exists():
        with open(
            performance_file,
            newline="",
            encoding="utf-8-sig"
        ) as f:
            fps_history = list(csv.DictReader(f))

    generator = ReportGenerator()

    pdf_path = generator.generate(
        session_data,
        fps_history,
        session_dir
    )

    print("PDF generated successfully!")
    print(pdf_path)


if __name__ == "__main__":
    main()
