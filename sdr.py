import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
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
    gen_data_Jin2023,
    gen_data_group,
    gen_data_mix,
    gen_data_s1,
    loss_Jin2023,
    sheridan_score,
)


# Experimental settings
ntrain_mu = 1250 #2000
ntrain_sigma = 1250 #2000
ncalib_list = [1250] #[500, 1000, 2000]
ntest = 1250 #2000

tau_list = [np.inf] #[1, 5, 10, 30, np.inf]
q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
delta = 1 #1
Nrep = 100

dim = 20
mean_model_type = "rf"  # "linear" or "rf"
method = "UNEC"  # "CS", "NP", or "UNEC"
setting = 1  # 1: Jin2023, 2: group DGP, 3: S3 mixture DGP, 4: S1 DGP

DATA_GENERATORS = {
    1: gen_data_Jin2023,
    2: gen_data_group,
    3: gen_data_mix,
    4: gen_data_s1,
}

C_VAL_BY_SETTING = {
    1: 0,
    2: 0,
    3: 1,
    4: -2,
}

SIG_BY_SETTING = {
    1: 0.1,
    2: 1,
    3: 1,
    4: 1,
}


def main():
    result_rows = []

    if setting not in DATA_GENERATORS:
        raise ValueError("setting must be 1, 2, 3, or 4.")
    if method not in {"CS", "NP", "UNEC"}:
        raise ValueError("method must be 'CS', 'NP', or 'UNEC'.")
    if method in {"CS", "NP"} and any(tau != np.inf for tau in tau_list):
        raise ValueError("The CS and NP methods require binary loss, so tau_list must contain only np.inf.")
    generate_data = DATA_GENERATORS[setting]
    c_val = C_VAL_BY_SETTING[setting]
    sig = SIG_BY_SETTING[setting]

    for seed in tqdm(range(Nrep)):
        rng = np.random.default_rng(seed)

        Xtrain_mu, _, _, Ytrain_mu = generate_data(n=ntrain_mu, sig=sig, dim=dim, random_state=rng)
        Xtrain_sigma, _, _, Ytrain_sigma = generate_data(n=ntrain_sigma, sig=sig, dim=dim, random_state=rng)
        Xcalib_pool, _, _, Ycalib_pool = generate_data(n=max(ncalib_list), sig=sig, dim=dim, random_state=rng)
        Xtest, _, _, Ytest = generate_data(n=ntest, sig=sig, dim=dim, random_state=rng)

        if mean_model_type == "linear":
            mean_model = LinearRegression()
            #mean_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
        elif mean_model_type == "rf":
            mean_model = RandomForestRegressor(random_state=seed, min_samples_leaf=10, n_jobs=-1)
        else:
            raise ValueError("mean_model_type must be 'linear' or 'rf'.")

        mean_model.fit(Xtrain_mu, Ytrain_mu)

        if method == "UNEC":
            # Fit the variance model using squared residuals from the selected mean model.
            Ytrain_sigma_pred = mean_model.predict(Xtrain_sigma)
            residual = Ytrain_sigma - Ytrain_sigma_pred
            #abs_resid = np.abs(residual)
            squared_resid = residual ** 2
            #sigma_model = LinearRegression()
            #sigma_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
            sigma_model = RandomForestRegressor(random_state=seed, min_samples_leaf=10, n_jobs=-1)
            #sigma_model = HistGradientBoostingRegressor(learning_rate=0.08,max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=1.0,early_stopping=True,random_state=seed)
            #sigma_model.fit(Xtrain_sigma, abs_resid)
            sigma_model.fit(Xtrain_sigma, squared_resid)
            Stest = sheridan_score(Xtest, mean_model, sigma_model, c_val, squared_scale=True)

        elif method == "NP":
            # Match summary(lm)$sigma in the R experiment for a linear model.
            train_resid = Ytrain_mu - mean_model.predict(Xtrain_mu)
            if mean_model_type == "linear":
                residual_df = max(ntrain_mu - Xtrain_mu.shape[1] - 1, 1)
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

        for ncalib in ncalib_list:
            Xcalib = Xcalib_pool[:ncalib]
            Ycalib = Ycalib_pool[:ncalib]

            if method == "UNEC":
                Scalib = sheridan_score(Xcalib, mean_model, sigma_model, c_val, squared_scale=True)
            elif method == "NP":
                Scalib = NP_score(Xcalib, mean_model, c_val, residual_scale)
            else:  # method == "CS"
                Scalib = -mean_model.predict(Xcalib)

            for tau in tau_list:
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
                        "setting": setting,
                        "mean_model_type": mean_model_type,
                        "q": q,
                        "tau": tau,
                        "ncalib": ncalib,
                        "ntest": ntest,
                        "ntrain_mu": ntrain_mu,
                        "ntrain_sigma": ntrain_sigma,
                        "c_val": c_val,
                        "delta": delta,
                        "sigma": sig,
                        "dim": dim,
                        "seed": seed,
                    })

    all_res = pd.DataFrame(result_rows)
    #results_dir = Path(__file__).resolve().parent / "results"
    #results_dir.mkdir(parents=True, exist_ok=True)
    #output_file = results_dir / (
    #    f"UNEC_SDR_S3, mean_model={mean_model_type}, ntest={ntest}, Nrep={Nrep}, "
    #    f"c={c_val}, delta={delta}, sigma={sig}, dim={dim}.csv"
    #)
    #all_res.to_csv(output_file, index=False)
    #print(f"Results saved to: {output_file}")

    mean_results = (
        all_res
        .groupby(["method", "setting", "mean_model_type", "ncalib", "delta", "tau", "q"], as_index=False)
        [["sdr", "power", "ave_reward", "selection_ratio"]]
        .mean()
    )
    print(f"\nAverage SDR, power, average reward, and selection ratio over {Nrep} repetitions:")
    print(mean_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
