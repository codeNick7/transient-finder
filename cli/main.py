"""Command-line interface (CLI) for running the Transient-Finder NEO AI validation pipeline on laptop."""

import sys
from pathlib import Path

# Automatically ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import CONFIG
from core.models import CandidateRecord, OrbitalElements
from ingestion.jpl_scout import JPLScoutClient
from ingestion.mpc_neocp import MPCNEOCPClient
from ingestion.jpl_cad import JPLHistoricalCADClient
from ingestion.fixtures import get_ground_truth_fixtures, get_live_candidates_snapshot
from ai.validator import CandidateValidator

console = Console()
validator = CandidateValidator()


def cmd_live(args):
    """Fetch and validate live candidates from NASA JPL Scout and MPC NEOCP."""
    console.print(Panel(f"[bold cyan]{CONFIG.app_name}[/bold cyan] - Ingesting Live Observational Feeds", expand=False))
    
    scout_client = JPLScoutClient()
    neocp_client = MPCNEOCPClient()
    
    candidates = []
    with console.status("[bold green]Querying NASA JPL Scout API..."):
        candidates.extend(scout_client.fetch_current_candidates())
        
    with console.status("[bold green]Querying Minor Planet Center NEOCP..."):
        candidates.extend(neocp_client.fetch_current_candidates())
        
    if not candidates:
        console.print("[yellow]Live network feeds unreachable. Serving persistent telemetry cache...[/yellow]")
        candidates = get_live_candidates_snapshot()
        
    candidates = candidates[:args.limit]
    
    table = Table(title=f"AI Candidate Validation Results ({len(candidates)} Candidates Evaluated)")
    table.add_column("Designation", style="bold")
    table.add_column("Survey", style="dim")
    table.add_column("Motion (deg/d)", justify="right")
    table.add_column("RMS (arcsec)", justify="right")
    table.add_column("AI Confidence", justify="right")
    table.add_column("ML Real P", justify="right")
    table.add_column("Status", style="bold")
    table.add_column("PHA?", justify="center")

    for c in candidates:
        res = validator.validate(c)
        status_color = "green" if "VALIDATED" in res.status.value else ("red" if "BOGUS" in res.status.value or "REJECTED" in res.status.value else "yellow")
        pha_badge = "[bold red]YES[/bold red]" if res.is_potentially_hazardous else "[dim]NO[/dim]"
        table.add_row(
            c.candidate_id,
            c.source_survey,
            f"{c.motion_rate_deg_day:.2f}",
            f"{c.rms_residual_arcsec:.2f}",
            f"{res.composite_confidence_score:.1f}%",
            f"{res.ml_real_probability * 100:.1f}%",
            f"[{status_color}]{res.status.value}[/{status_color}]",
            pha_badge
        )
        
    console.print(table)


def cmd_historical(args):
    """Run 6-month longitudinal re-validation on candidates from May 2026."""
    console.print(Panel(f"[bold cyan]{CONFIG.app_name}[/bold cyan] - 6-Month Longitudinal Re-Validation Lab", expand=False))
    
    cad_client = JPLHistoricalCADClient()
    with console.status("[bold green]Fetching candidates from ~6 months ago (May 2026)..."):
        hist_cands = cad_client.fetch_close_approach_candidates(date_min="2026-05-01", date_max="2026-05-31")
        if not hist_cands:
            hist_cands = [f for f in get_ground_truth_fixtures() if "2026" in f.candidate_id]
            
    table = Table(title="Historical Candidate Re-Validation Audit (6 Months Old)")
    table.add_column("Candidate ID", style="bold")
    table.add_column("Discovery Date", style="dim")
    table.add_column("Age (Mo)", justify="right")
    table.add_column("Drift (arcsec)", justify="right")
    table.add_column("P(Recovery)", justify="right")
    table.add_column("Disposition", style="bold green")
    table.add_column("Notes", style="italic")

    for hc in hist_cands[:args.limit]:
        rep = validator.revalidate(hc)
        table.add_row(
            rep.candidate_id,
            rep.original_discovery_date.strftime("%Y-%m-%d"),
            f"{rep.age_months:.1f}",
            f"{rep.ephemeris_drift_arcsec:,.1f}",
            f"{rep.recovery_probability * 100:.1f}%",
            rep.final_disposition,
            rep.historical_notes[:65] + ("..." if len(rep.historical_notes) > 65 else "")
        )
    console.print(table)


def cmd_benchmark(args):
    """Run verification against benchmark testcases including false positives."""
    console.print(Panel("[bold cyan]Running Ground-Truth & False-Positive Diagnostic Benchmark Audit[/bold cyan]"))
    benchmarks = get_ground_truth_fixtures()
    
    table = Table(title="Benchmark Ground Truth & Diagnostic Suite")
    table.add_column("ID", style="bold")
    table.add_column("Survey Source", style="dim")
    table.add_column("Verdict", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Diagnostic Mode", style="magenta")
    table.add_column("Rejection Conf", justify="right")

    for b in benchmarks:
        res = validator.validate(b)
        diag = res.false_positive_diagnostic
        table.add_row(
            b.candidate_id,
            b.source_survey,
            res.status.value,
            f"{res.composite_confidence_score:.1f}%",
            diag.mode.value,
            f"{diag.rejection_confidence:.1f}%"
        )
    console.print(table)


def cmd_patch(args):
    """Query and analyze candidates & distant transients in an on-demand sky patch."""
    target_str = f" | Target: [yellow]{args.target}[/yellow]" if args.target else ""
    console.print(Panel(
        f"[bold cyan]{CONFIG.app_name}[/bold cyan] - On-Demand Sky Patch Analysis\n"
        f"[dim]Center: RA {args.ra:.3f}°, Dec {args.dec:.3f}° | FOV Radius: {args.radius:.2f}°{target_str}[/dim]",
        expand=False
    ))

    from ingestion.sky_patch import SkyPatchService
    patch_service = SkyPatchService()

    with console.status("[bold green]Querying survey streams, JPL Horizons, and transient catalogs..."):
        result = patch_service.query_sky_patch(
            ra_deg=args.ra,
            dec_deg=args.dec,
            radius_deg=args.radius,
            target_id=args.target,
            epoch=args.epoch
        )

    console.print(
        f"[bold]Detections Summary:[/bold] Total Found: [cyan]{result.total_detected}[/cyan] | "
        f"NEOs: [green]{result.neo_count}[/green] | "
        f"Distant Transients: [magenta]{result.distant_transient_count}[/magenta] | "
        f"Main Belt Asteroids: [blue]{result.asteroid_count}[/blue] | "
        f"Artifacts Vetoed: [red]{result.artifact_count}[/red]\n"
    )

    table = Table(title=f"Sky Patch Detections (RA: {args.ra:.2f}°, Dec: {args.dec:.2f}°, Radius: {args.radius:.2f}°)")
    table.add_column("Designation", style="bold")
    table.add_column("Sep (arcmin)", justify="right")
    table.add_column("Category", style="cyan")
    table.add_column("V Mag", justify="right")
    table.add_column("Motion (deg/d)", justify="right")
    table.add_column("AI Verdict", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Astrophysical Profile", style="dim")

    for obj in result.objects:
        cand = obj.candidate
        val = obj.validation
        status_color = "green" if "VALIDATED" in val.status.value or "IDENTIFIED" in val.status.value else (
            "magenta" if "DISTANT" in val.status.value else (
                "red" if "BOGUS" in val.status.value or "REJECTED" in val.status.value else "yellow"
            )
        )
        nature = (
            val.transient_properties.transient_type if val.transient_properties and val.transient_properties.transient_type
            else (val.orbit_classification if val.orbit_classification and val.orbit_classification != "UNKNOWN_SHORT_ARC" else obj.category.value)
        )
        table.add_row(
            cand.candidate_id,
            f"{obj.angular_separation_arcmin:.1f}'",
            obj.category.value,
            f"{cand.v_mag:.1f}" if cand.v_mag else "N/A",
            f"{cand.motion_rate_deg_day:.3f}",
            f"[{status_color}]{val.status.value}[/{status_color}]",
            f"{val.composite_confidence_score:.1f}%",
            str(nature)
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description=f"{CONFIG.app_name} CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # live command
    p_live = subparsers.add_parser("live", help="Validate live candidates from NASA Scout & MPC")
    p_live.add_argument("--limit", type=int, default=15, help="Max candidates to process")

    # historical command
    p_hist = subparsers.add_parser("historical", help="Re-validate candidates from 6 months ago")
    p_hist.add_argument("--limit", type=int, default=10, help="Max candidates to process")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suite on real & bogus candidates")

    # patch command
    p_patch = subparsers.add_parser("patch", help="Query and analyze sky patch (RA & Dec) for NEOs & distant transients")
    p_patch.add_argument("--ra", type=float, required=True, help="Central Right Ascension in degrees (0-360)")
    p_patch.add_argument("--dec", type=float, required=True, help="Central Declination in degrees (-90 to +90)")
    p_patch.add_argument("--radius", type=float, default=1.0, help="Search cone radius in degrees (default: 1.0)")
    p_patch.add_argument("--target", type=str, default=None, help="Optional target identifier (e.g. '1998 KY26')")
    p_patch.add_argument("--epoch", type=str, default=None, help="Observation date or epoch (e.g. '2024-06-01')")

    args = parser.parse_args()
    if args.command == "live":
        cmd_live(args)
    elif args.command == "historical":
        cmd_historical(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "patch":
        cmd_patch(args)


if __name__ == "__main__":
    main()
