import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "data/Survey.csv"

race_order = [
    "Black or African\nAmerican",
    "Hispanic\nor Latino",
    "Asian",
    "White",
]

education_order = [
    "High school or less",
    "Some college or\nassociate degree",
    "Bachelor’s degree",
    "Master's degree\nor higher",
]

gender_order = ["Male", "Female"]
immigration_order = ["Non-immigrant", "Immigrant"]

BASE_RACE = "White"
BASE_GENDER = "Male"
BASE_EDU = "High school or less"

TREAT_AGE_ZERO_AS_MISSING = True

# US population targets (your provided proportions)
US_CENSUS_RACE_PROP = {
    "White": 0.575,                          # White alone, non-Hispanic
    "Hispanic\nor Latino": 0.200,            # Hispanic or Latino
    "Black or African\nAmerican": 0.137,     # Black or African American alone
    "Asian": 0.067,                          # Asian alone
}


def remap_race(v):
    if pd.isna(v):
        return np.nan
    v = int(v)
    if v == 2:
        return "Asian"
    if v == 3:
        return "Black or African\nAmerican"
    if v == 6:
        return "Hispanic\nor Latino"
    return "White"


def remap_education(v):
    if pd.isna(v):
        return np.nan
    v = float(v)
    if v <= 6:
        return "High school or less"
    if 7 <= v <= 10:
        return "Some college or\nassociate degree"
    if v == 11:
        return "Bachelor’s degree"
    return "Master's degree\nor higher"


def remap_gender(v):
    if pd.isna(v):
        return np.nan
    try:
        vv = float(v)
        if vv == 1:
            return "Male"
        if vv == 2:
            return "Female"
    except Exception:
        pass

    s = str(v).strip().lower()
    if s in ["m", "male", "man", "men", "boy"]:
        return "Male"
    if s in ["f", "female", "woman", "women", "girl"]:
        return "Female"
    return np.nan


def remap_immigration(v):
    if pd.isna(v):
        return np.nan
    try:
        v = int(v)
    except Exception:
        return np.nan
    if v == 1:
        return "Non-immigrant"
    if v == 2:
        return "Immigrant"
    return np.nan


def zscore(s):
    s = s.astype(float)
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return np.nan * s
    return (s - s.mean()) / sd


def debug_counts(df, msg):
    print(f"{msg}: n={len(df)}")
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = debug_counts(df, "Loaded")

    # rename to patsy safe
    rename_map = {}
    if "Self-Objective" in df.columns:
        rename_map["Self-Objective"] = "Self_Objective"
    if "Self-Subjective" in df.columns:
        rename_map["Self-Subjective"] = "Self_Subjective"
    df = df.rename(columns=rename_map)

    required_cols = [
        "Self_Objective",
        "Self_Subjective",
        "Race",
        "Gender",
        "Education",
        "Age",
        "Immigration_status",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Available={list(df.columns)}"
        )

    # numeric
    df["Self_Objective"] = pd.to_numeric(df["Self_Objective"], errors="coerce")
    df["Self_Subjective"] = pd.to_numeric(df["Self_Subjective"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if TREAT_AGE_ZERO_AS_MISSING:
        df.loc[df["Age"] == 0, "Age"] = np.nan

    # remap
    df["Race"] = df["Race"].apply(remap_race)
    df["Gender"] = df["Gender"].apply(remap_gender)
    df["Education"] = df["Education"].apply(remap_education)
    df["Immigration_status"] = df["Immigration_status"].apply(remap_immigration)

    df = debug_counts(df, "After remap")

    # filter to target categories
    df = df[df["Race"].isin(race_order)].copy()
    df = debug_counts(df, "After Race filter")

    df = df[df["Gender"].isin(gender_order)].copy()
    df = debug_counts(df, "After Gender filter")

    df = df[df["Education"].isin(education_order)].copy()
    df = debug_counts(df, "After Education filter")

    df = df[df["Immigration_status"].isin(immigration_order)].copy()
    df = debug_counts(df, "After Immigration_status filter")

    # listwise deletion only on core vars
    df = df.dropna(subset=["Self_Objective", "Self_Subjective", "Age"])
    df = debug_counts(df, "After dropping NA on Self vars and Age")

    if len(df) == 0:
        raise ValueError("No data left after filtering, check mappings or missingness.")

    # categorical encoding
    df["Race"] = pd.Categorical(df["Race"], categories=race_order)
    df["Gender"] = pd.Categorical(df["Gender"], categories=gender_order)
    df["Education"] = pd.Categorical(df["Education"], categories=education_order)
    df["Immigration_status"] = pd.Categorical(
        df["Immigration_status"], categories=immigration_order
    )

    # standardize for correlation interpretation
    df["Self_Objective"] = zscore(df["Self_Objective"])
    df["Self_Subjective"] = zscore(df["Self_Subjective"])
    df["Age_z"] = zscore(df["Age"])

    df = df.dropna(subset=["Self_Objective", "Self_Subjective", "Age_z"])
    df = debug_counts(df, "Final analytic sample (pre-weight)")

    # ===== Race post-stratification weights based on your provided proportions =====
    # sample proportions in analytic sample
    sample_race_prop = df["Race"].value_counts(normalize=True).to_dict()

    print("\nSample race proportions (analytic sample):")
    for k in race_order:
        if k in sample_race_prop:
            print(f"{k}: {sample_race_prop[k]:.4f}")

    # compute weights: population share / sample share
    # guard against missing groups
    missing_groups = [r for r in race_order if r not in sample_race_prop]
    if missing_groups:
        raise ValueError(
            f"Some race groups are missing in analytic sample, cannot compute weights: {missing_groups}"
        )

    df["Race_weight"] = df["Race"].map(
        lambda r: US_CENSUS_RACE_PROP[r] / sample_race_prop[r]
    ).astype(float)
    # import pdb; pdb.set_trace()
    print(df.groupby("Race")["Race_weight"].max().reindex(race_order))
    # sanity checks
    if df["Race_weight"].isna().any():
        raise ValueError("Race_weight has NA. Check race mapping and US_CENSUS_RACE_PROP keys.")

    if (df["Race_weight"] <= 0).any():
        raise ValueError("Race_weight has non-positive values. Check construction.")

    print("\nRace_weight summary:")
    print(df["Race_weight"].describe())

    # ===================== MODELS =====================

    # ---------- Model 1: Objective only (+ age, immigration) ----------
    formula_m1 = """
        Self_Subjective ~
        Self_Objective
    """

    model_1 = smf.wls(
        formula_m1,
        data=df,
        weights=df["Race_weight"]
    ).fit(cov_type="HC1")

    print("\n================ MODEL 1: OBJECTIVE ONLY =================\n")
    print(model_1.summary())


    # ---------- Model 2: Main effects only (NO interactions) ----------
    formula_m2 = f"""
        Self_Subjective ~
        Self_Objective
        + Age_z
        + C(Immigration_status, Treatment(reference='Non-immigrant'))
        + C(Race, Treatment(reference='{BASE_RACE}'))
        + C(Gender, Treatment(reference='{BASE_GENDER}'))
        + C(Education, Treatment(reference="{BASE_EDU}"))
    """

    model_2 = smf.wls(
        formula_m2,
        data=df,
        weights=df["Race_weight"]
    ).fit(cov_type="HC1")

    print("\n================ MODEL 2: MAIN EFFECTS ONLY =================\n")
    print(model_2.summary())


    # ---------- Model 3: Full model (with interactions) ----------
    formula_m3 = f"""
        Self_Subjective ~
        Self_Objective
        + Age_z
        + C(Immigration_status, Treatment(reference='Non-immigrant'))
        + C(Race, Treatment(reference='{BASE_RACE}'))
        + C(Gender, Treatment(reference='{BASE_GENDER}'))
        + C(Education, Treatment(reference="{BASE_EDU}"))

        + Self_Objective:C(Race, Treatment(reference='{BASE_RACE}'))
        + Self_Objective:C(Gender, Treatment(reference='{BASE_GENDER}'))
        + Self_Objective:C(Education, Treatment(reference="{BASE_EDU}"))
        + Self_Objective:Age_z
    """

    model_3 = smf.wls(
        formula_m3,
        data=df,
        weights=df["Race_weight"]
    ).fit(cov_type="HC1")

    print("\n================ MODEL 3: FULL INTERACTION MODEL =================\n")
    print(model_3.summary())
    # ---------- Model 4: Unadjusted (only Obj and Obj×Race; no controls) ----------
    formula_m4 = f"""
        Self_Subjective ~
        Self_Objective
        + Self_Objective:C(Race, Treatment(reference='{BASE_RACE}'))
    """

    model_4 = smf.wls(
        formula_m4,
        data=df,
        weights=df["Race_weight"]
    ).fit(cov_type="HC1")

    print("\n================ MODEL 4: UNADJUSTED (Obj × Race only) =================\n")
    print(model_4.summary())


    # ---------- Model 5: Adjusted (controls included, but ONLY Obj×Race interaction) ----------
    formula_m5 = f"""
        Self_Subjective ~
        Self_Objective
        + Age_z
        + C(Immigration_status, Treatment(reference='Non-immigrant'))
        + C(Race, Treatment(reference='{BASE_RACE}'))
        + C(Gender, Treatment(reference='{BASE_GENDER}'))
        + C(Education, Treatment(reference="{BASE_EDU}"))
        + Self_Objective:C(Race, Treatment(reference='{BASE_RACE}'))
    """

    model_5 = smf.wls(
        formula_m5,
        data=df,
        weights=df["Race_weight"]
    ).fit(cov_type="HC1")

    print("\n================ MODEL 5: ADJUSTED (controls + Obj × Race only) =================\n")
    print(model_5.summary())


    # Order on x-axis
    race_order_plot = [
        "Black or African\nAmerican",
        "Hispanic\nor Latino",
        "Asian",
        "White",
    ]


    # ===================== F TESTS: Joint significance of Obj × Race interactions =====================
    def joint_f_test_obj_x_race(model, race_order_plot):
        """
        Wald F test (using the model's robust cov) for:
        H0: all Self_Objective × Race[T.*] interaction terms == 0
        """
        prefix = f"Self_Objective:C(Race, Treatment(reference='{BASE_RACE}'))"

        terms = []
        for g in race_order_plot:
            if g == BASE_RACE:
                continue
            t = f"{prefix}[T.{g}]"
            if t in model.params.index:
                terms.append(t)
            else:
                raise ValueError(
                    f"Could not find term '{t}' in model params.\n"
                    f"Available terms include:\n{list(model.params.index)}"
                )

        hyp = " = 0, ".join(terms) + " = 0"
        res = model.f_test(hyp)

        print(hyp)
        print(f"F = {float(res.fvalue):.4f}, df_num = {int(res.df_num)}, df_denom = {int(res.df_denom)}, p = {float(res.pvalue):.6g}")
        return res

    print("\n================ F TEST: Obj × Race interactions (Model 4, Unadjusted) =================\n")
    ft_m4 = joint_f_test_obj_x_race(model_4, race_order_plot)

    print("\n================ F TEST: Obj × Race interactions (Model 5, Adjusted) =================\n")
    ft_m5 = joint_f_test_obj_x_race(model_5, race_order_plot)


    import matplotlib.pyplot as plt

    Z = 1.96  # 95% CI

    def beta_se_sum(model, base_term, interaction_term=None):
        b = model.params
        V = model.cov_params()

        beta = float(b[base_term])
        var = float(V.loc[base_term, base_term])

        if interaction_term is not None:
            if interaction_term not in b.index:
                raise ValueError(f"Missing term in model params: {interaction_term}")
            beta += float(b[interaction_term])
            var += float(V.loc[interaction_term, interaction_term]) + 2.0 * float(V.loc[base_term, interaction_term])

        return beta, np.sqrt(var)

    def joint_f_test_terms(model, terms):
        hyp = " = 0, ".join(terms) + " = 0"
        res = model.f_test(hyp)
        print(hyp)
        print(f"F = {float(res.fvalue):.4f}, df_num = {int(res.df_num)}, df_denom = {int(res.df_denom)}, p = {float(res.pvalue):.6g}")
        return res

    def plot_two_series(labels, unadj_beta, unadj_se, adj_beta, adj_se, title, out_path):
        x = np.arange(len(labels))
        offset = 0.12

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x - offset, unadj_beta, yerr=Z * unadj_se, fmt="o", capsize=3, label="Unadjusted")
        ax.errorbar(x + offset, adj_beta, yerr=Z * adj_se, fmt="^", capsize=3, label="Adjusted")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
        ax.axhline(0, linewidth=1)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.show()
        print(f"\nSaved figure to: {out_path}\n")

    # ===================== Plot race (Model 4 vs Model 5) =====================

    def build_race_rows(model, race_order_plot, base_term="Self_Objective"):
        """
        Group-specific slope (= corr since z-scored) for each race:
        beta(White) = base_term
        beta(group) = base_term + interaction(group)
        """
        prefix = f"Self_Objective:C(Race, Treatment(reference='{BASE_RACE}'))"
        rows = []
        for g in race_order_plot:
            if g == BASE_RACE:  # White (reference)
                beta, se = beta_se_sum(model, base_term, None)
            else:
                term = f"{prefix}[T.{g}]"
                beta, se = beta_se_sum(model, base_term, term)
            rows.append((g, beta, se))
        return rows

    # Extract (beta, se) for each race
    rows_unadj_r = build_race_rows(model_4, race_order_plot)
    rows_adj_r   = build_race_rows(model_5, race_order_plot)

    labels_r = [r[0] for r in rows_unadj_r]
    unadj_beta_r = np.array([r[1] for r in rows_unadj_r])
    unadj_se_r   = np.array([r[2] for r in rows_unadj_r])

    adj_beta_r = np.array([r[1] for r in rows_adj_r])
    adj_se_r   = np.array([r[2] for r in rows_adj_r])

    plot_two_series(
        labels_r, unadj_beta_r, unadj_se_r, adj_beta_r, adj_se_r,
        title="By race/ethnicity",
        out_path="fig_subj_obj_corr_race_unadjusted_vs_adjusted.png"
    )
    # ===================== Gender models =====================

    # Model 6: Unadjusted (Obj + Obj×Gender only)
    formula_m6 = f"""
        Self_Subjective ~
        Self_Objective
        + Self_Objective:C(Gender, Treatment(reference='{BASE_GENDER}'))
    """
    model_6 = smf.wls(formula_m6, data=df, weights=df["Race_weight"]).fit(cov_type="HC1")
    print("\n================ MODEL 6: UNADJUSTED (Obj × Gender only) =================\n")
    print(model_6.summary())

    # Model 7: Adjusted (controls + Obj×Gender only)
    formula_m7 = f"""
        Self_Subjective ~
        Self_Objective
        + Age_z
        + C(Immigration_status, Treatment(reference='Non-immigrant'))
        + C(Race, Treatment(reference='{BASE_RACE}'))
        + C(Gender, Treatment(reference='{BASE_GENDER}'))
        + C(Education, Treatment(reference="{BASE_EDU}"))
        + Self_Objective:C(Gender, Treatment(reference='{BASE_GENDER}'))
    """
    model_7 = smf.wls(formula_m7, data=df, weights=df["Race_weight"]).fit(cov_type="HC1")
    print("\n================ MODEL 7: ADJUSTED (controls + Obj × Gender only) =================\n")
    print(model_7.summary())

    # F test for Obj×Gender (Female term)
    prefix_g = f"Self_Objective:C(Gender, Treatment(reference='{BASE_GENDER}'))"
    term_g = f"{prefix_g}[T.Female]"
    print("\n================ F TEST: Obj × Gender (Model 6) =================\n")
    _ = joint_f_test_terms(model_6, [term_g])
    print("\n================ F TEST: Obj × Gender (Model 7) =================\n")
    _ = joint_f_test_terms(model_7, [term_g])

    # Plot gender
    rows_unadj_g = [("Male",) + beta_se_sum(model_6, "Self_Objective", None),
                    ("Female",) + beta_se_sum(model_6, "Self_Objective", term_g)]
    rows_adj_g   = [("Male",) + beta_se_sum(model_7, "Self_Objective", None),
                    ("Female",) + beta_se_sum(model_7, "Self_Objective", term_g)]

    labels_g = [r[0] for r in rows_unadj_g]
    unadj_beta_g = np.array([r[1] for r in rows_unadj_g])
    unadj_se_g   = np.array([r[2] for r in rows_unadj_g])
    adj_beta_g   = np.array([r[1] for r in rows_adj_g])
    adj_se_g     = np.array([r[2] for r in rows_adj_g])

    plot_two_series(
        labels_g, unadj_beta_g, unadj_se_g, adj_beta_g, adj_se_g,
        title="By gender",
        out_path="fig_subj_obj_corr_gender_unadjusted_vs_adjusted.png"
    )


    # ===================== Education models =====================

    # Model 8: Unadjusted (Obj + Obj×Education only)
    formula_m8 = f"""
        Self_Subjective ~
        Self_Objective
        + Self_Objective:C(Education, Treatment(reference="{BASE_EDU}"))
    """
    model_8 = smf.wls(formula_m8, data=df, weights=df["Race_weight"]).fit(cov_type="HC1")
    print("\n================ MODEL 8: UNADJUSTED (Obj × Education only) =================\n")
    print(model_8.summary())

    # Model 9: Adjusted (controls + Obj×Education only)
    formula_m9 = f"""
        Self_Subjective ~
        Self_Objective
        + Age_z
        + C(Immigration_status, Treatment(reference='Non-immigrant'))
        + C(Race, Treatment(reference='{BASE_RACE}'))
        + C(Gender, Treatment(reference='{BASE_GENDER}'))
        + C(Education, Treatment(reference="{BASE_EDU}"))
        + Self_Objective:C(Education, Treatment(reference="{BASE_EDU}"))
    """
    model_9 = smf.wls(formula_m9, data=df, weights=df["Race_weight"]).fit(cov_type="HC1")
    print("\n================ MODEL 9: ADJUSTED (controls + Obj × Education only) =================\n")
    print(model_9.summary())

    # F test for Obj×Education (3 terms)
    edu_order_plot = [
        "High school or less",
        "Some college or\nassociate degree",
        "Bachelor’s degree",
        "Master's degree\nor higher",
    ]
    prefix_e = f"Self_Objective:C(Education, Treatment(reference=\"{BASE_EDU}\"))"
    edu_terms = [f"{prefix_e}[T.{g}]" for g in edu_order_plot if g != BASE_EDU]

    print("\n================ F TEST: Obj × Education (Model 8) =================\n")
    _ = joint_f_test_terms(model_8, edu_terms)
    print("\n================ F TEST: Obj × Education (Model 9) =================\n")
    _ = joint_f_test_terms(model_9, edu_terms)

    # Plot education
    rows_unadj_e = []
    rows_adj_e = []
    for g in edu_order_plot:
        if g == BASE_EDU:
            rows_unadj_e.append((g,) + beta_se_sum(model_8, "Self_Objective", None))
            rows_adj_e.append((g,) + beta_se_sum(model_9, "Self_Objective", None))
        else:
            t = f"{prefix_e}[T.{g}]"
            rows_unadj_e.append((g,) + beta_se_sum(model_8, "Self_Objective", t))
            rows_adj_e.append((g,) + beta_se_sum(model_9, "Self_Objective", t))

    labels_e = [r[0] for r in rows_unadj_e]
    unadj_beta_e = np.array([r[1] for r in rows_unadj_e])
    unadj_se_e   = np.array([r[2] for r in rows_unadj_e])
    adj_beta_e   = np.array([r[1] for r in rows_adj_e])
    adj_se_e     = np.array([r[2] for r in rows_adj_e])

    plot_two_series(
        labels_e, unadj_beta_e, unadj_se_e, adj_beta_e, adj_se_e,
        title="By education",
        out_path="fig_subj_obj_corr_education_unadjusted_vs_adjusted.png"
    )
if __name__ == "__main__":
    main()

