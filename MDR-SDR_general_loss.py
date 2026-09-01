import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from UNEC import (
    UNEC_MDR,
    UNEC_SDR,
    eval_MDR,
    eval_SDR,
    fit_mean_model,
    gen_data_Jin2023,
    gen_data_mix,
    loss_Jin2023,
    sheridan_score,
)


# ---------------------------------------------------------------------------
# Experimental settings
# ---------------------------------------------------------------------------
control_type = os.getenv("GENERAL_LOSS_CONTROL_TYPE", "SDR")  # "MDR" or "SDR"
q = 0.10
tau_list = [1, 2, 5, 10, 50, np.inf]
ncalib_list = [100, 500, 1000, 2000]
delta_list = [0, 0.5, 1]

ntrain_mu = 1000
ntrain_sigma = 1000
ntest = 1000
Nrep = 500
dim = 20

residual_type = "squared"  # "absolute" or "squared"


# The two Jin2023 cases differ only in their conditional sigma construction.
DGP_CONFIGS = {
    "Jin2023_sigma2": {
        "generator": gen_data_Jin2023,
        "kwargs": {"sigma_type": 2},
        "sig": 0.1,
        "c_val": 0,
    },
    "Jin2023_sigma3": {
        "generator": gen_data_Jin2023,
        "kwargs": {"sigma_type": 3},
        "sig": 0.1,
        "c_val": 0,
    },
    "mix_type2": {
        "generator": gen_data_mix,
        "kwargs": {"mix_type": 2},
        "sig": 1,
        "c_val": 1,
    },
}


def _generate(config, n, rng):
    """Generate one sample while keeping all DGP arguments in one place."""
    return config["generator"](
        n=n,
        sig=config["sig"],
        dim=dim,
        random_state=rng,
        **config["kwargs"],
    )


def main():
    if control_type not in {"MDR", "SDR"}:
        raise ValueError("control_type must be 'MDR' or 'SDR'.")
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1].")
    if residual_type not in {"absolute", "squared"}:
        raise ValueError("residual_type must be 'absolute' or 'squared'.")
    if not ncalib_list or any(n <= 0 for n in ncalib_list):
        raise ValueError("ncalib_list must contain positive integers.")

    select_method = UNEC_MDR if control_type == "MDR" else UNEC_SDR
    result_rows = []

    total_runs = len(DGP_CONFIGS) * Nrep
    progress = tqdm(total=total_runs, desc=f"UNEC-{control_type}")

    for dgp_name, config in DGP_CONFIGS.items():
        c_val = config["c_val"]

        for seed in range(Nrep):
            # A DGP-specific seed keeps results reproducible while avoiding
            # identical random streams across different DGPs.
            seed_sequence = np.random.SeedSequence(
                [seed, list(DGP_CONFIGS).index(dgp_name)]
            )
            rng = np.random.default_rng(seed_sequence)

            Xtrain_mu, _, _, Ytrain_mu = _generate(config, ntrain_mu, rng)
            Xtrain_sigma, _, _, Ytrain_sigma = _generate(config, ntrain_sigma, rng)
            Xcalib_pool, _, _, Ycalib_pool = _generate(config, max(ncalib_list), rng)
            Xtest, _, _, Ytest = _generate(config, ntest, rng)

            mean_model = fit_mean_model("rf", random_state=seed)
            mean_model.fit(Xtrain_mu, Ytrain_mu)

            residual = Ytrain_sigma - mean_model.predict(Xtrain_sigma)
            scale_target = np.abs(residual) if residual_type == "absolute" else residual ** 2
            squared_scale = residual_type == "squared"
            
            scale_model = RandomForestRegressor(random_state=seed, min_samples_leaf=10, n_jobs=-1)
            scale_model.fit(Xtrain_sigma, scale_target)

            Stest = sheridan_score(Xtest, mean_model, scale_model, c_val, squared_scale=squared_scale)
            # Binary reward is used only to describe useful selections; the controlled loss below remains the tau-dependent general loss.
            Rtest = (Ytest > c_val).astype(float)
            total_reward = np.sum(Rtest)

            for ncalib in ncalib_list:
                Xcalib = Xcalib_pool[:ncalib]
                Ycalib = Ycalib_pool[:ncalib]
                Scalib = sheridan_score(Xcalib, mean_model, scale_model, c_val, squared_scale=squared_scale)

                for tau in tau_list:
                    Lcalib = loss_Jin2023(Ycalib, tau=tau, c=c_val)
                    Ltest = loss_Jin2023(Ytest, tau=tau, c=c_val)

                    for delta in delta_list:
                        selected = select_method((Lcalib, Scalib), Stest, alpha=q, delta=delta)

                        if control_type == "MDR":
                            risk, reward = eval_MDR(Ltest, Rtest, selected)
                        else:
                            risk, _, reward = eval_SDR(Ltest, Rtest, selected)

                        nsel = len(selected)
                        result_rows.append(
                            {
                                "risk": risk,
                                "controlled": risk <= q,
                                "power": (
                                    reward / total_reward
                                    if total_reward > 0
                                    else 0
                                ),
                                "reward": reward,
                                "nsel": nsel,
                                "selection_ratio": nsel / ntest,
                                "control_type": control_type,
                                "dgp": dgp_name,
                                "tau": tau,
                                "ncalib": ncalib,
                                "delta": delta,
                                "q": q,
                                "seed": seed,
                                "residual_type": residual_type,
                            }
                        )

            progress.update(1)

    progress.close()
    all_results = pd.DataFrame(result_rows)

    summary = (
        all_results
        .groupby(
            ["control_type", "dgp", "tau", "ncalib", "delta", "q"],
            as_index=False,
        )
        .agg(
            mean_risk=("risk", "mean"),
            mean_power=("power", "mean"),
            mean_selection_ratio=("selection_ratio", "mean"),
            control_rate=("controlled", "mean"),
            risk_sd=("risk", "std"),
            power_sd=("power", "std"),
        )
    )

    results_dir = Path(__file__).resolve().parent / "results_general_loss"
    results_dir.mkdir(parents=True, exist_ok=True)
    detail_file = results_dir / f"UNEC_{control_type}_general_loss_detail.csv"
    summary_file = results_dir / f"UNEC_{control_type}_general_loss_summary.csv"
    all_results.to_csv(detail_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\nTarget {control_type}: q={q}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDetailed results saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
