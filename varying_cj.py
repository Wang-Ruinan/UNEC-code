import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from UNEC import (UNEC_MDR, UNEC_SDR, eval_MDR, eval_SDR, fit_mean_model, gen_data_Jin2023,
    gen_data_mix, loss_Jin2023)

# Experimental settings
ntrain_mu = 1000
ntrain_sigma = 1000
ncalib = 1000
ntest = 1000
Nrep = 100
dim = 20

q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
delta = 0
tau = 10
cj_half_width_list = [0, 0.3, 0.6, 1]
control_types = ["MDR", "SDR"]#

mean_model_type = "rf"  # "linear" or "rf"
residual_type = "squared"  # "absolute" or "squared"
min_scale = 0.01

# DGP2/3: Jin2023 sigma types 2/3; DGP4: mixture type 1.
DGP_CONFIGS = {
    "DGP2": {"generator": gen_data_Jin2023, "kwargs": {"sigma_type": 2}, "sig": 0.1, "cj_center": 0.0},
    "DGP3": {"generator": gen_data_Jin2023, "kwargs": {"sigma_type": 3}, "sig": 0.1, "cj_center": 0.0},
    "DGP4": {"generator": gen_data_mix, "kwargs": {"mix_type": 1}, "sig": 1.0, "cj_center": 1.0},
}


def _generate(config, n, rng):
    return config["generator"](
        n=n, sig=config["sig"], dim=dim, random_state=rng, **config["kwargs"]
    )


def main():
    if residual_type not in {"absolute", "squared"}:
        raise ValueError("residual_type must be 'absolute' or 'squared'.")
    if any(width < 0 for width in cj_half_width_list):
        raise ValueError("cj_half_width_list must contain non-negative values.")
    if not control_types or not set(control_types).issubset({"MDR", "SDR"}):
        raise ValueError("control_types may only contain 'MDR' and 'SDR'.")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive.")

    squared_scale = residual_type == "squared"
    result_rows = []
    progress = tqdm(total=len(DGP_CONFIGS) * Nrep, desc="UNEC varying-cj")

    for dgp_index, (dgp_name, config) in enumerate(DGP_CONFIGS.items()):
        for seed in range(Nrep):
            rng = np.random.default_rng(np.random.SeedSequence([seed, dgp_index]))
            Xtrain_mu, _, _, Ytrain_mu = _generate(config, ntrain_mu, rng)
            Xtrain_sigma, _, _, Ytrain_sigma = _generate(config, ntrain_sigma, rng)
            Xcalib, _, _, Ycalib = _generate(config, ncalib, rng)
            Xtest, _, _, Ytest = _generate(config, ntest, rng)

            mean_model = fit_mean_model(mean_model_type, random_state=seed)
            mean_model.fit(Xtrain_mu, Ytrain_mu)
            residual = Ytrain_sigma - mean_model.predict(Xtrain_sigma)
            scale_target = residual ** 2 if squared_scale else np.abs(residual)
            scale_model = RandomForestRegressor(
                random_state=seed, min_samples_leaf=10, n_jobs=-1
            )
            scale_model.fit(Xtrain_sigma, scale_target)

            pred_mu_calib = np.asarray(mean_model.predict(Xcalib), dtype=float)
            pred_mu_test = np.asarray(mean_model.predict(Xtest), dtype=float)
            pred_scale_calib = np.asarray(scale_model.predict(Xcalib), dtype=float)
            pred_scale_test = np.asarray(scale_model.predict(Xtest), dtype=float)
            if squared_scale:
                pred_scale_calib = np.sqrt(np.maximum(pred_scale_calib, min_scale ** 2))
                pred_scale_test = np.sqrt(np.maximum(pred_scale_test, min_scale ** 2))
            else:
                pred_scale_calib = np.maximum(pred_scale_calib, min_scale)
                pred_scale_test = np.maximum(pred_scale_test, min_scale)

            for cj_half_width in cj_half_width_list:
                cj_test = config["cj_center"] + rng.uniform(
                    -cj_half_width, cj_half_width, size=ntest
                )
                test_losses = loss_Jin2023(Ytest, tau=tau, c=cj_test)
                rewards = (Ytest > cj_test).astype(float)
                total_reward = float(np.sum(rewards))

                selected_by_control = {
                    kind: {q: np.zeros(ntest, dtype=bool) for q in q_list}
                    for kind in control_types
                }
                selection_methods = {"MDR": UNEC_MDR, "SDR": UNEC_SDR}

                if cj_half_width == 0:
                    # With a common c, calibrate once and select all test observations in one call, as in the fixed-c experiment.
                    cj = config["cj_center"]
                    Lcalib = loss_Jin2023(Ycalib, tau=tau, c=cj)
                    Scalib = (pred_mu_calib - cj) / pred_scale_calib
                    Stest = (pred_mu_test - cj) / pred_scale_test
                    for control_type in control_types:
                        select_method = selection_methods[control_type]
                        for q in q_list:
                            selected = select_method(
                                (Lcalib, Scalib),
                                Stest,
                                alpha=q,
                                delta=delta,
                            )
                            selected_by_control[control_type][q][selected] = True
                else:
                    # With varying c_j, recalibrate separately for each test observation using its own threshold.
                    for j, cj in enumerate(cj_test):
                        Lcalib_j = loss_Jin2023(Ycalib, tau=tau, c=cj)
                        Scalib_j = (pred_mu_calib - cj) / pred_scale_calib
                        Stest_j = np.asarray(
                            [(pred_mu_test[j] - cj) / pred_scale_test[j]]
                        )

                        for control_type in control_types:
                            select_method = selection_methods[control_type]
                            for q in q_list:
                                selected_j = select_method(
                                    (Lcalib_j, Scalib_j),
                                    Stest_j,
                                    alpha=q,
                                    delta=delta,
                                )
                                selected_by_control[control_type][q][j] = (
                                    len(selected_j) > 0
                                )

                for control_type in control_types:
                    selected_by_q = selected_by_control[control_type]
                    for q, selected_mask in selected_by_q.items():
                        selected = np.flatnonzero(selected_mask)
                        if control_type == "MDR":
                            risk, reward = eval_MDR(test_losses, rewards, selected)
                        else:
                            risk, _, reward = eval_SDR(test_losses, rewards, selected)
                        nsel = len(selected)
                        ave_reward = reward / nsel if nsel > 0 else 0
                        result_rows.append(
                            {
                                "risk": risk,
                                "controlled": risk <= q,
                                "power": reward / total_reward if total_reward > 0 else 0,
                                "reward": reward,
                                "ave_reward": ave_reward,
                                "nsel": nsel,
                                "selection_ratio": nsel / ntest,
                                "method": "UNEC",
                                "control_type": control_type,
                                "dgp": dgp_name,
                                "q": q,
                                "tau": tau,
                                "delta": delta,
                                "seed": seed,
                                "cj_center": config["cj_center"],
                                "cj_half_width": cj_half_width,
                                "cj_min": float(np.min(cj_test)),
                                "cj_max": float(np.max(cj_test)),
                                "cj_mean": float(np.mean(cj_test)),
                                "ntrain_mu": ntrain_mu,
                                "ntrain_sigma": ntrain_sigma,
                                "ncalib": ncalib,
                                "ntest": ntest,
                                "mean_model_type": mean_model_type,
                                "residual_type": residual_type,
                                "sigma": config["sig"],
                                "dim": dim,
                            }
                        )
            progress.update(1)

    progress.close()
    detail = pd.DataFrame(result_rows)
    summary = (
        detail.groupby(
            ["method", "control_type", "dgp", "cj_half_width", "tau", "delta", "q"],
            as_index=False,
        )
        .agg(
            mean_risk=("risk", "mean"),
            mean_power=("power", "mean"),
            mean_reward=("reward", "mean"),
            mean_ave_reward=("ave_reward", "mean"),
            mean_nsel=("nsel", "mean"),
            mean_selection_ratio=("selection_ratio", "mean"),
            control_rate=("controlled", "mean"),
            risk_sd=("risk", "std"),
            power_sd=("power", "std"),
        )
    )

    results_dir = Path(__file__).resolve().parent / "results_varying_cj"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_stem = "UNEC_MDR_SDR_DGP2-4_varying_cj"
    detail_file = results_dir / f"{file_stem}_detail.csv"
    summary_file = results_dir / f"{file_stem}_summary.csv"
    detail.to_csv(detail_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\nUNEC varying-c_j experiment finished, tau={tau}, delta={delta}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDetailed results saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
