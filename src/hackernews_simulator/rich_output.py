"""Rich-formatted CLI output for hackernews-simulator."""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

LABEL_COLORS = {
    "flop": "red",
    "low": "yellow",
    "moderate": "green",
    "hot": "blue",
    "viral": "magenta",
}

def label_color(label: str) -> str:
    return LABEL_COLORS.get(label, "white")

def print_prediction(title: str, result) -> None:
    """Print rich-formatted prediction output."""
    # Header panel
    console.print(Panel(f"[bold]{title}[/bold]", title="HN Reaction Simulator", border_style="blue"))
    console.print()

    # Metrics
    color = label_color(result.reception_label)
    console.print(f"  Predicted Score:    [bold]~{int(result.predicted_score)} points[/bold]")
    console.print(f"  Predicted Comments: [bold]~{int(result.predicted_comments)}[/bold]")
    console.print(f"  Reception: [bold {color}]{result.reception_label.upper()}[/bold {color}] ({int(result.confidence * 100)}% confidence)")
    console.print()

    # Percentile + expected score
    if result.percentile is not None:
        console.print(f"  Percentile: [bold]Top {result.percentile:.1f}%[/bold] of HN stories")
    if result.expected_score is not None:
        console.print(f"  Expected Score: [bold]~{int(result.expected_score)} points[/bold] (multiclass)")
    if result.percentile is not None or result.expected_score is not None:
        console.print()

    # Label distribution
    if result.label_distribution:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim", width=10)
        table.add_column("Bar", width=22)
        table.add_column("Pct", width=5)
        for label, prob in result.label_distribution.items():
            filled = round(prob * 20)
            bar = "█" * filled + "░" * (20 - filled)
            table.add_row(f"[{label_color(label)}]{label}[/]", bar, f"{int(prob*100)}%")
        console.print("  Label Distribution:")
        console.print(table)
        console.print()

    # SHAP
    if result.shap_features:
        console.print("  [bold]Why This Score:[/bold]")
        for feat in result.shap_features:
            arrow = "[green]↑[/green]" if feat["direction"] == "up" else "[red]↓[/red]"
            console.print(f"  {arrow} {feat['feature']} ({feat['importance']:+.2f})")
        console.print()

    # Time advice
    if result.time_recommendation:
        console.print(f"  [bold]Posting Advice:[/bold] {result.time_recommendation}")
        console.print()

    # Comments
    if result.simulated_comments:
        console.print("  [bold]Simulated Comments:[/bold]")
        console.print()
        for c in result.simulated_comments:
            username = c.get("username", "hn_user")
            tone = c.get("tone", "")
            text = c.get("comment", "")
            tone_str = f" [dim][{tone}][/dim]" if tone else ""
            console.print(f"  [bold]{username}[/bold]{tone_str}: {text}")
            console.print()

def print_comparison(results: list[dict]) -> None:
    """Print rich comparison table."""
    table = Table(title="Variant Comparison", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Title", max_width=60)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Reception", width=10)

    for i, r in enumerate(results):
        color = label_color(r.get("reception_label", ""))
        table.add_row(
            str(i + 1),
            r.get("title", ""),
            f"~{int(r.get('predicted_score', 0))}",
            f"[{color}]{r.get('reception_label', '').upper()}[/{color}]",
        )
    console.print(table)
