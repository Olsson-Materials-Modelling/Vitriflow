from __future__ import annotations

import json
from pathlib import Path


def test_plot_scan_metric_all_stages(tmp_path: Path):
    from vitriflow.plotting import plot_scan_metric

    # consistent autotune json
    data = {
        "units": {"lammps_units": "metal", "time_unit_ps": 1.0},
        "tm_scan": {
            "outcomes": [
                {
                    "name": "t1000_r1",
                    "temperature_start": 1000.0,
                    "D": 0.05,
                    "density_mean": 2.20,
                    "pe_mean": -1000.0,
                    "msd_rms_last": 0.5,
                    "gr_peak_height": 2.5,
                    "gr_peak_fwhm": 0.20,
                },
                {
                    "name": "t1000_r2",
                    "temperature_start": 1000.0,
                    "D": 0.06,
                    "density_mean": 2.21,
                    "pe_mean": -999.0,
                    "msd_rms_last": 0.52,
                    "gr_peak_height": 2.45,
                    "gr_peak_fwhm": 0.22,
                },
                {
                    "name": "t1500_r1",
                    "temperature_start": 1500.0,
                    "D": 0.20,
                    "density_mean": 2.10,
                    "pe_mean": -980.0,
                    "msd_rms_last": 1.0,
                    "gr_peak_height": 2.20,
                    "gr_peak_fwhm": 0.30,
                },
            ]
        },
        "rate_scan": {
            "skipped": False,
            "rates": [
                {
                    "rate_K_per_time": 1.0,
                    "density_mean": 2.25,
                    "density_stderr": 0.01,
                    "metrics_mean": {"coord": 4.0},
                    "metrics_stderr": {"coord": 0.1},
                    "replicates": [
                        {"density": 2.24, "metrics": {"coord": 4.1}},
                        {"density": 2.26, "metrics": {"coord": 3.9}},
                    ],
                },
                {
                    "rate_K_per_time": 10.0,
                    "density_mean": 2.30,
                    "density_stderr": 0.02,
                    "metrics_mean": {"coord": 3.8},
                    "metrics_stderr": {"coord": 0.2},
                    "replicates": [
                        {"density": 2.29, "metrics": {"coord": 3.9}},
                        {"density": 2.31, "metrics": {"coord": 3.7}},
                    ],
                },
            ],
        },
        "size_scan": {
            "skipped": False,
            "sizes": [
                {
                    "n_atoms": 100,
                    "multiplier": 1,
                    "density_mean": 2.20,
                    "density_stderr": 0.01,
                    "metrics_mean": {"coord": 4.0},
                    "metrics_stderr": {"coord": 0.1},
                    "replicates": [
                        {"density": 2.19, "metrics": {"coord": 4.1}},
                        {"density": 2.21, "metrics": {"coord": 3.9}},
                    ],
                },
                {
                    "n_atoms": 800,
                    "multiplier": 8,
                    "density_mean": 2.22,
                    "density_stderr": 0.01,
                    "metrics_mean": {"coord": 4.02},
                    "metrics_stderr": {"coord": 0.1},
                    "replicates": [
                        {"density": 2.21, "metrics": {"coord": 4.1}},
                        {"density": 2.23, "metrics": {"coord": 3.9}},
                    ],
                },
            ],
        },
        "production": {
            "enabled": True,
            "boxes": [
                {"density": 2.20, "metrics": {"coord": 4.0}},
                {"density": 2.22, "metrics": {"coord": 4.1}},
                {"density": 2.21, "metrics": {"coord": 3.9}},
            ],
        },
    }

    jp = tmp_path / "autotune_results.json"
    jp.write_text(json.dumps(data))

    out_tm = tmp_path / "tm_density.pdf"
    plot_scan_metric(jp, out_tm, stage="tm_scan", metric="density", show_replicates=True)
    assert out_tm.exists() and out_tm.stat().st_size > 1_000

    out_rate = tmp_path / "rate_coord.pdf"
    plot_scan_metric(jp, out_rate, stage="rate_scan", metric="coord", show_replicates=True)
    assert out_rate.exists() and out_rate.stat().st_size > 1_000

    out_size = tmp_path / "size_density.pdf"
    plot_scan_metric(jp, out_size, stage="size_scan", metric="density", show_replicates=True)
    assert out_size.exists() and out_size.stat().st_size > 1_000

    out_prod = tmp_path / "prod_density.pdf"
    plot_scan_metric(jp, out_prod, stage="production", metric="density")
    assert out_prod.exists() and out_prod.stat().st_size > 1_000
