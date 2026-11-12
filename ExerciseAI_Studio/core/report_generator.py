
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm


class ReportGenerator:
    """
    Generate a scientific session PDF report.

    Uses existing SessionManager data.
    The current assessment is depth-only,
    not a validated full-technique score.
    """

    def __init__(self):
        self.navy = colors.HexColor("#14243A")
        self.teal = colors.HexColor("#087F8C")
        self.gray = colors.HexColor("#526174")
        self.light = colors.HexColor("#EDF3F7")

        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(
                name="StudioTitle",
                parent=self.styles["Title"],
                fontSize=21,
                leading=27,
                textColor=self.navy,
                spaceAfter=10,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="StudioHeading",
                parent=self.styles["Heading2"],
                fontSize=12,
                leading=16,
                textColor=self.teal,
                spaceBefore=14,
                spaceAfter=8,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="StudioSmall",
                parent=self.styles["Normal"],
                fontSize=8,
                leading=12,
                textColor=self.gray,
            )
        )

    @staticmethod
    def _number(value, digits=2):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _table(self, rows, widths=None):
        table = Table(
            rows,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), self.navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, self.light]),
                ("GRID", (0, 0), (-1, -1), 0.25,
                 colors.HexColor("#D9E2E9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
        return table

    def _plot(self, values, title, ylabel, path, xvalues=None):
        fig, ax = plt.subplots(figsize=(8, 3.0))

        if xvalues is None:
            xvalues = list(range(1, len(values) + 1))

        ax.plot(
            xvalues,
            values,
            color="#087F8C",
            linewidth=1.8,
        )
        ax.set_title(title)
        ax.set_xlabel(
            "Repetition" if "Repetition" in title
            else "Elapsed time (s)"
        )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

        if "Repetition" in title and xvalues:
            ax.set_xticks(xvalues)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def generate(self, session_data, fps_history, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_dir / "report.pdf"
        repetitions = session_data.get("repetitions", [])

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=(210 * mm, 297 * mm),
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        )

        story = []

        def paragraph(text, style="Normal"):
            return Paragraph(
                escape(str(text)),
                self.styles[style],
            )

        def heading(text):
            story.append(
                paragraph(text, "StudioHeading")
            )

        story.append(
            paragraph("EXERCISE AI STUDIO", "StudioTitle")
        )
        story.append(
            paragraph(
                "Real-Time Exercise Monitoring | Session Report",
                "StudioSmall"
            )
        )
        story.append(Spacer(1, 8 * mm))

        heading("1. Session Information")

        fields = [
            ("Session ID", session_data.get("session_id")),
            ("Exercise", session_data.get("exercise")),
            ("Pose Model", session_data.get("model")),
            ("Started", session_data.get("started_at")),
            ("Ended", session_data.get("ended_at")),
            ("Duration (s)", self._number(
                session_data.get("duration_seconds")
            )),
            ("Total Repetitions", len(repetitions)),
            ("Depth-Pass Repetitions",
             session_data.get("depth_pass_repetitions")),
            ("Average FPS", self._number(
                session_data.get("average_fps")
            )),
        ]

        info_rows = [["Metric", "Value"]]
        info_rows.extend(
            [str(key), str(value) if value is not None else "N/A"]
            for key, value in fields
        )
        story.append(
            self._table(info_rows, [65 * mm, 109 * mm])
        )

        heading("2. Repetition Assessment")

        rep_rows = [[
            "Rep", "Duration (s)", "Min Knee (deg)",
            "Depth Score", "Status"
        ]]

        for rep in repetitions:
            rep_rows.append([
                str(rep.get("rep", "")),
                self._number(rep.get("duration")),
                self._number(rep.get("min_knee_angle"), 1),
                str(rep.get("score", "N/A")),
                str(rep.get("status", "N/A")),
            ])

        if not repetitions:
            rep_rows.append(["-", "-", "-", "-", "No reps"])

        story.append(
            self._table(
                rep_rows,
                [15 * mm, 31 * mm, 39 * mm, 34 * mm, 55 * mm]
            )
        )

        if repetitions:
            heading("3. Repetition Duration")

            duration_path = output_dir / "_duration_chart.png"

            self._plot(
                [float(r["duration"]) for r in repetitions],
                "Repetition Duration",
                "Duration (s)",
                duration_path,
                [r["rep"] for r in repetitions],
            )

            story.append(
                Image(str(duration_path), width=165 * mm,
                      height=62 * mm)
            )

            heading("4. Minimum Knee Angle")

            angle_path = output_dir / "_angle_chart.png"

            self._plot(
                [float(r["min_knee_angle"]) for r in repetitions],
                "Repetition Minimum Knee Angle",
                "Angle (degrees)",
                angle_path,
                [r["rep"] for r in repetitions],
            )

            story.append(
                Image(str(angle_path), width=165 * mm,
                      height=62 * mm)
            )

        if fps_history:
            heading("5. Real-Time Performance")

            fps_path = output_dir / "_fps_chart.png"

            self._plot(
                [float(item["fps"]) for item in fps_history],
                "Processing FPS",
                "Frames per second",
                fps_path,
                [float(item["elapsed_seconds"])
                 for item in fps_history],
            )

            story.append(
                Image(str(fps_path), width=165 * mm,
                      height=62 * mm)
            )

        heading("6. Scientific Interpretation")

        story.append(
            paragraph(
                "This report uses experimental, rule-based "
                "squat depth assessment. A depth-pass status "
                "does not establish overall exercise correctness, "
                "injury risk, or clinical safety."
            )
        )

        story.append(Spacer(1, 5 * mm))
        story.append(
            paragraph(
                "Generated: " +
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "StudioSmall"
            )
        )

        def page_footer(canvas, document):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(self.gray)
            canvas.drawString(
                18 * mm, 12 * mm,
                "ExerciseAI Studio | Experimental Research"
            )
            canvas.drawRightString(
                192 * mm, 12 * mm,
                f"Page {document.page}"
            )
            canvas.restoreState()

        doc.build(
            story,
            onFirstPage=page_footer,
            onLaterPages=page_footer,
        )

        return pdf_path
