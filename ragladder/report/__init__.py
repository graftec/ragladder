"""Reporting: terminal tables (rich) and machine-readable results.json."""

from ragladder.report.html import write_report_html
from ragladder.report.results import write_results_json
from ragladder.report.tables import (
    render_compare,
    render_diagnostics,
    render_inspect,
    render_run,
    render_stats,
    render_summary,
)

__all__ = [
    "render_compare",
    "render_diagnostics",
    "render_inspect",
    "render_run",
    "render_stats",
    "render_summary",
    "write_report_html",
    "write_results_json",
]
