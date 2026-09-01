import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from UNEC import (UNEC_MDR, UNEC_SDR, eval_MDR, eval_SDR, fit_mean_model, gen_data_Jin2023, loss_two_sided)

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
control_types = ["MDR", "SDR"]

# For test observation j:
# m_j = m0 + h * U_j, a_j = m_j - w0, b_j = m_j + w0,
# where U_j ~ Uniform(-0.5, 0.5).
m0 = -0.5
w0_list = [0.5, 1, 1.5]
h_list = [0, 1, 2]

mean_model_type = "rf"  # "linear" or "rf"
residual_type = "squared"  # "absolute" or "squared"
min_scale = 0.01

# Only DGP2 is used in this experiment.
dgp_name = "DGP2"
sig = 0.1
sigma_type = 2


def main():
    if residual_type not in {"absolute", "squared"}:
        raise ValueError("residual_type must be 'absolute' or 'squared'.")
    if not control_types or not set(control_types).issubset({"MDR", "SDR"}):
        raise ValueError("control_types may only contain 'MDR' and 'SDR'.")
    if any(w0 <= 0 for w0 in w0_list):
        raise ValueError("w0_list must contain positive half-widths.")
    if any(h < 0 for h in h_list):
        raise ValueError("h_list must contain non-negative values.")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive.")

    squared_scale = residual_type == "squared"
    selection_methods = {"MDR": UNEC_MDR, "SDR": UNEC_SDR}
    result_rows = []
    progress = tqdm(total=Nrep, desc="UNEC two-sided DGP2")

    for seed in range(Nrep):
        rng = np.random.default_rng(seed)
        generator_kwargs = {"sig": sig, "dim": dim, "random_state": rng, "sigma_type": sigma_type}
        Xtrain_mu, _, _, Ytrain_mu = gen_data_Jin2023(n=ntrain_mu, **generator_kwargs)
        Xtrain_sigma, _, _, Ytrain_sigma = gen_data_Jin2023(n=ntrain_sigma, **generator_kwargs)
        Xcalib, _, _, Ycalib = gen_data_Jin2023(n=ncalib, **generator_kwargs)
        Xtest, _, _, Ytest = gen_data_Jin2023(n=ntest, **generator_kwargs)

        mean_model = fit_mean_model(mean_model_type, random_state=seed)
        mean_model.fit(Xtrain_mu, Ytrain_mu)

        residual = Ytrain_sigma - mean_model.predict(Xtrain_sigma)
        scale_target = residual ** 2 if squared_scale else np.abs(residual)
        scale_model = RandomForestRegressor(random_state=seed, min_samples_leaf=10, n_jobs=-1)
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

        # Reuse the same U_j for every (w0, h) setting in this repetition.
        Utest = rng.uniform(-0.5, 0.5, size=ntest)

        for w0 in w0_list:
            for h in h_list:
                mj_test = m0 + h * Utest
                aj_test = mj_test - w0
                bj_test = mj_test + w0
                test_losses = loss_two_sided(Ytest, tau=tau, a=aj_test, b=bj_test)
                rewards = ((Ytest > aj_test) & (Ytest < bj_test)).astype(float)
                total_reward = float(np.sum(rewards))

                selected_by_control = {
                    kind: {q: np.zeros(ntest, dtype=bool) for q in q_list}
                    for kind in control_types
                }

                if h == 0:
                    # Fixed interval: calibrate once and select the complete test set in one call.
                    a_val = m0 - w0
                    b_val = m0 + w0
                    Lcalib = loss_two_sided(Ycalib, tau=tau, a=a_val, b=b_val)
                    Scalib = np.minimum(pred_mu_calib - a_val, b_val - pred_mu_calib) / pred_scale_calib
                    Stest = np.minimum(pred_mu_test - a_val, b_val - pred_mu_test) / pred_scale_test

                    for control_type in control_types:
                        select_method = selection_methods[control_type]
                        for q in q_list:
                            selected = select_method((Lcalib, Scalib), Stest, alpha=q, delta=delta)
                            selected_by_control[control_type][q][selected] = True
                else:
                    # Varying centers: use the j-th interval to recalibrate the original UNEC rule for the j-th test observation.
                    for j, (a_val, b_val) in enumerate(zip(aj_test, bj_test)):
                        Lcalib_j = loss_two_sided(Ycalib, tau=tau, a=a_val, b=b_val)
                        Scalib_j = np.minimum(pred_mu_calib - a_val, b_val - pred_mu_calib) / pred_scale_calib
                        Stest_j = np.asarray([min(pred_mu_test[j] - a_val, b_val - pred_mu_test[j]) / pred_scale_test[j]])

                        for control_type in control_types:
                            select_method = selection_methods[control_type]
                            for q in q_list:
                                selected_j = select_method((Lcalib_j, Scalib_j), Stest_j, alpha=q, delta=delta)
                                selected_by_control[control_type][q][j] = (
                                    len(selected_j) > 0
                                )

                for control_type in control_types:
                    for q, selected_mask in selected_by_control[control_type].items():
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
                                "power": (reward / total_reward if total_reward > 0 else 0),
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
                                "m0": m0,
                                "w0": w0,
                                "h": h,
                                "mean_mj": float(np.mean(mj_test)),
                                "min_aj": float(np.min(aj_test)),
                                "max_bj": float(np.max(bj_test)),
                                "ntrain_mu": ntrain_mu,
                                "ntrain_sigma": ntrain_sigma,
                                "ncalib": ncalib,
                                "ntest": ntest,
                                "mean_model_type": mean_model_type,
                                "residual_type": residual_type,
                                "sigma": sig,
                                "dim": dim,
                            }
                        )
        progress.update(1)

    progress.close()
    detail = pd.DataFrame(result_rows)
    summary = (
        detail.groupby(
            ["method", "control_type", "dgp", "m0", "w0", "h", "tau", "delta", "q"],
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

    results_dir = Path(__file__).resolve().parent / "results_two_sided"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_stem = "UNEC_MDR_SDR_DGP2_two_sided"
    detail_file = results_dir / f"{file_stem}_detail.csv"
    summary_file = results_dir / f"{file_stem}_summary.csv"
    detail.to_csv(detail_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\nUNEC two-sided DGP2 experiment finished, delta={delta}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDetailed results saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
