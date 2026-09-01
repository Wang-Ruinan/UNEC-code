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
    CS,
    NP,
    NP_score,
    UNEC_SDR,
    eval_SDR,
    fit_mean_model,
    gen_data_Jin2023,
    gen_data_mix,
    loss_Jin2023,
    sheridan_score,
)


# Experimental settings
ntrain_mu = 1000
ntrain_sigma = 1000
ncalib = 1000
ntest = 1000

tau = np.inf
q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
delta = float(os.getenv("FDR_DELTA", "0"))
Nrep = 500

dim = 20
mean_model_type = "rf"  # "linear" or "rf"
method = os.getenv("FDR_METHOD", "UNEC")  # "CS", "NP", or "UNEC"
setting = int(os.getenv("FDR_SETTING", "1"))  # 1: Jin2023, 2: mixture
residual_type = "squared"  # "absolute" or "squared" (used by UNEC)
jin_sigma_type = int(os.getenv("FDR_JIN_SIGMA_TYPE", "1"))  # DGP1-3
mix_type = int(os.getenv("FDR_MIX_TYPE", "1"))  # DGP4-5

DATA_GENERATORS = {
    1: gen_data_Jin2023,
    2: gen_data_mix,
}

C_VAL_BY_SETTING = {
    1: 0,
    2: 1,
}

SIG_BY_SETTING = {
    1: 0.1,
    2: 1,
}


def main():
    result_rows = []

    if setting not in DATA_GENERATORS:
        raise ValueError("setting must be 1 or 2.")
    if method not in {"CS", "NP", "UNEC"}:
        raise ValueError("method must be 'CS', 'NP', or 'UNEC'.")
    if residual_type not in {"absolute", "squared"}:
        raise ValueError("residual_type must be 'absolute' or 'squared'.")
    if jin_sigma_type not in {1, 2, 3}:
        raise ValueError("jin_sigma_type must be 1, 2, or 3.")
    if mix_type not in {1, 2}:
        raise ValueError("mix_type must be 1 or 2.")
    method_label = (
        f"{method}_delta{delta:g}" if method in {"UNEC", "NP"} else method
    )
    generate_data = DATA_GENERATORS[setting]
    if setting == 1:
        generator_kwargs = {"sigma_type": jin_sigma_type}
        dgp_name = f"DGP{jin_sigma_type}"
    else:  # setting == 2
        generator_kwargs = {"mix_type": mix_type}
        dgp_name = f"DGP{mix_type + 3}"
    c_val = C_VAL_BY_SETTING[setting]
    sig = SIG_BY_SETTING[setting]

    for seed in tqdm(range(Nrep)):
        rng = np.random.default_rng(seed)

        Xtrain_mu_part, _, _, Ytrain_mu_part = generate_data(n=ntrain_mu, sig=sig, dim=dim, random_state=rng, **generator_kwargs)
        Xtrain_sigma, _, _, Ytrain_sigma = generate_data(n=ntrain_sigma, sig=sig, dim=dim, random_state=rng, **generator_kwargs)
        Xcalib, _, _, Ycalib = generate_data(n=ncalib, sig=sig, dim=dim, random_state=rng, **generator_kwargs)
        Xtest, _, _, Ytest = generate_data(n=ntest, sig=sig, dim=dim, random_state=rng, **generator_kwargs)

        if method == "UNEC":
            # UNEC uses sample splitting: one part for mu and one for sigma.
            Xtrain_mu = Xtrain_mu_part
            Ytrain_mu = Ytrain_mu_part
        else:
            # CS and NP do not fit a sigma model, so give their mean model the same total training-data budget available to UNEC.
            Xtrain_mu = np.concatenate((Xtrain_mu_part, Xtrain_sigma), axis=0)
            Ytrain_mu = np.concatenate((Ytrain_mu_part, Ytrain_sigma), axis=0)

        mean_model = fit_mean_model(mean_model_type, random_state=seed)
        mean_model.fit(Xtrain_mu, Ytrain_mu)

        if method == "UNEC":
            # Fit the scale model using the selected residual transformation.
            Ytrain_sigma_pred = mean_model.predict(Xtrain_sigma)
            residual = Ytrain_sigma - Ytrain_sigma_pred
            scale_target = np.abs(residual) if residual_type == "absolute" else residual ** 2
            squared_scale = residual_type == "squared"
            #sigma_model = LinearRegression()
            #sigma_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
            sigma_model = RandomForestRegressor(random_state=seed, min_samples_leaf=10, n_jobs=-1)
            #sigma_model = HistGradientBoostingRegressor(learning_rate=0.08,max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,early_stopping=True,random_state=seed)
            sigma_model.fit(Xtrain_sigma, scale_target)
            Stest = sheridan_score(Xtest, mean_model, sigma_model, c_val, squared_scale=squared_scale)

        elif method == "NP":
            # Match summary(lm)$sigma in the R experiment for a linear model.
            train_resid = Ytrain_mu - mean_model.predict(Xtrain_mu)
            if mean_model_type == "linear":
                residual_df = max(len(Ytrain_mu) - Xtrain_mu.shape[1] - 1, 1)
                residual_scale = np.sqrt(np.sum(train_resid ** 2) / residual_df)
            else:
                residual_scale = np.sqrt(np.mean(train_resid ** 2))
            residual_scale = max(float(residual_scale), 1e-8)

            Stest = NP_score(Xtest, mean_model, c_val, residual_scale)

        else:  # method == "CS"
            Stest = -mean_model.predict(Xtest)

        # Reward is 1 only when the true test outcome is above c_val.
        Rtest = (Ytest > c_val).astype(float)
        total_reward = np.sum(Rtest)

        if method == "UNEC":
            Scalib = sheridan_score(Xcalib, mean_model, sigma_model, c_val, squared_scale=squared_scale)
        elif method == "NP":
            Scalib = NP_score(Xcalib, mean_model, c_val, residual_scale)
        else:  # method == "CS"
            Scalib = -mean_model.predict(Xcalib)

        Lcalib = loss_Jin2023(Ycalib, tau=tau, c=c_val)
        Ltest = loss_Jin2023(Ytest, tau=tau, c=c_val)

        for q in q_list:
            if method == "UNEC":
                selected = UNEC_SDR((Lcalib, Scalib), Stest, alpha=q, delta=delta)
            elif method == "NP":
                selected = NP((Lcalib, Scalib), Stest, alpha=q, delta=delta)
            else:
                selected = CS((Lcalib, Scalib), Stest, alpha=q, mult_test=True)

            sdr, _, reward = eval_SDR(Ltest, Rtest, selected)
            power = reward / total_reward if total_reward > 0 else 0
            ave_reward = reward / len(selected) if len(selected) > 0 else 0
            nsel = len(selected)
            selection_ratio = nsel / ntest

            result_rows.append({
                "sdr": sdr,
                "power": power,
                "reward": reward,
                "ave_reward": ave_reward,
                "nsel": nsel,
                "selection_ratio": selection_ratio,
                "method": method,
                "dgp": dgp_name,
                "setting": setting,
                "mean_model_type": mean_model_type,
                "residual_type": residual_type if method == "UNEC" else None,
                "jin_sigma_type": jin_sigma_type if setting == 1 else None,
                "mix_type": mix_type if setting == 2 else None,
                "q": q,
                "tau": tau,
                "ncalib": ncalib,
                "ntest": ntest,
                "ntrain_mu": len(Ytrain_mu),
                "ntrain_sigma": ntrain_sigma if method == "UNEC" else 0,
                "ntrain_total": ntrain_mu + ntrain_sigma,
                "c_val": c_val,
                "delta": delta,
                "sigma": sig,
                "dim": dim,
                "seed": seed,
            })

    all_res = pd.DataFrame(result_rows)
    all_res["controlled"] = all_res["sdr"] <= all_res["q"]

    summary = (
        all_res
        .groupby(
            ["method", "dgp", "mean_model_type", "residual_type", "ncalib", "delta", "q"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_sdr=("sdr", "mean"),
            mean_power=("power", "mean"),
            mean_reward=("reward", "mean"),
            mean_ave_reward=("ave_reward", "mean"),
            mean_nsel=("nsel", "mean"),
            mean_selection_ratio=("selection_ratio", "mean"),
            control_rate=("controlled", "mean"),
            sdr_sd=("sdr", "std"),
            power_sd=("power", "std"),
        )
    )

    results_dir = Path(__file__).resolve().parent / "results_FDR"
    results_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"{method}_{dgp_name}_FDR"
    detail_file = results_dir / f"{file_stem}_detail.csv"
    summary_file = results_dir / f"{file_stem}_summary.csv"
    all_res.to_csv(detail_file, index=False)
    summary.to_csv(summary_file, index=False)

    print(f"\nAverage SDR, power, average reward, and selection ratio over {Nrep} repetitions:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDetailed results saved to: {detail_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
