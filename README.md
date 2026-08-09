# Detecting Unlicensed Commercial Short-Term Rental Operators in Las Vegas

A multimodal **Positive-Unlabeled (PU) learning** system that ranks unlicensed Airbnb listings by
behavioral similarity to a commercial hospitality operation, so a county enforcement office can
prioritise limited audit capacity.

**[→ Open the notebook](str_operator_detection.ipynb)** · runs top-to-bottom on Google Colab free GPU.

---

> ### ⚠️ What this model claims
>
> Every score in this project measures **behavioral similarity to a commercial operation**. None of
> them measures compliance with the law. A high-priority listing is a listing worth a human look —
> it is not an allegation, a finding, or evidence of one. The notebook's §9 sets out the deployment
> controls (human review, register reconciliation, disparate-impact auditing) that any real use
> would require.
---

## The problem

Clark County requires short-term rentals to hold a license. In this Inside Airbnb snapshot of
20,296 Las Vegas listings:

| `license` field | Listings | Interpretation |
|---|---:|---|
| Real county license (`G##-#####`) | 684 | Confirmed permitted operators |
| `"Exempt"` | 6,009 | **Ambiguous** — genuine exemption *or* evasion |
| Null / blank | 13,209 | No license information at all |
| Other free-text (`STR-…`, state business IDs) | 384 | Licensed under a *different* municipal scheme |

Meanwhile **34.7% of listings belong to hosts running 20+ properties** — unregistered hospitality
businesses at scale.

There is no verified negative anywhere in the data:no listing is stamped "confirmed unlicensed."
That rules out ordinary binary classification and makes this a textbook PU problem.

## The finding that reframed the project

The obvious approach is to train on the 684 confirmed licensees and rank unlicensed listings by
similarity. Working through the EDA first showed why that fails:

| Host portfolio | Listings | Holds a real license | Rate |
|---|---:|---:|---:|
| 1 property | 4,466 | 204 | **4.6%** |
| 2–5 | 4,848 | 265 | 5.5% |
| 6–19 | 3,884 | 135 | 3.5% |
| 20+ | 7,036 | 80 | **1.1%** |

**The licensing rate falls as portfolio size rises.** Large operators don't register — 47% of their
listings claim `"Exempt"`, against 14% for single-property hosts. So "looks like a licensed listing"
is close to the *opposite* of the enforcement target, and a model that stopped there would have
shipped an audit queue pointed at spare-bedroom hosts while 537-property operators sat at the bottom.

The notebook builds the specified model, demonstrates the failure explicitly with a face-validity
table, and then composes a deployable score from two visible factors:

```
priority = percentile(commercial evidence index) × (1 − P(residential-permit-like))
```

The first factor is a published, hand-weighted rubric over behavioral features — deliberately
transparent, because "the model said so" is not something you can argue in front of a hearing
officer. The second is where the learned nnPU model contributes.

## What's in the notebook

| § | Contents |
|---|---|
| 1 | EDA — price/stay-length distributions, host concentration, license × host size, spatial clustering, missingness |
| 2 | Cleaning — conservative, every imputation paired with an indicator flag |
| 3 | PU label construction; why `"Exempt"` can be neither positive nor negative |
| 4 | Feature engineering — behavioral, spatial (haversine to the Strip), and title regex features |
| 5 | **Multimodal nnPU network** (PyTorch) — frozen MiniLM text branch + entity embeddings + numerics |
| 6 | Optuna hyperparameter search, 30 trials |
| 7 | Baselines — LightGBM (tabular-only) and a deep autoencoder, same 30-trial budget |
| 8 | Evaluation — AUC, precision@k, calibration, ablation, leakage robustness, face validity |
| 9 | The deployable enforcement score, limitations, ethics |

### Technical highlights

- **nnPU loss implemented from the paper** (Kiryo et al., NeurIPS 2017), with the derivation in
  markdown and **four unit tests before it touches real data** — including a closed-form check
  (the risk must equal exactly 0.5 at zero logits for *any* prior) and a synthetic experiment with
  known hidden labels showing uPU driving empirical risk negative while nnPU recovers the positives.
- **An independent validation set that fell out of the data.** The 384 listings holding a
  *different-format* real license were left in the unlabeled pool and never used as positives —
  giving a genuinely held-out set of licensed listings to check the model against.
- **Leakage measured, not assumed.** The largest host owns 537 listings, so a random split lets a
  model score a test listing by having memorised its siblings. Both models are re-fitted on a
  host-grouped split and the gap is reported.
- **Class prior treated as an assumption.** π is not identifiable from PU data; the notebook sweeps
  π ∈ {0.2, 0.35, 0.5} and reports the rank correlation between the resulting queues.

## Running it

**Colab** — upload `str_operator_detection.ipynb`, choose a GPU runtime, Run All. The first cell
installs any missing dependencies; the notebook prompts for the CSV upload if it can't find one.

**Locally**

```bash
git clone https://github.com/zhabibi-z/str-operator-detection.git
cd str-operator-detection
pip install numpy pandas scikit-learn scipy matplotlib torch sentence-transformers optuna lightgbm
jupyter lab str_operator_detection.ipynb
```

All seeds are fixed (`SEED = 42`) and cuDNN is set deterministic, so a fresh top-to-bottom run
reproduces the reported numbers.

> **macOS note:** the notebook caps `OMP_NUM_THREADS` to 1 on Darwin before importing torch or
> LightGBM. Both ship their own OpenMP runtime and the two thread pools segfault the interpreter
> when LightGBM trains after torch has been used. Linux and Colab are unaffected.

## Data

`airbnb_listings.csv` — [Inside Airbnb](https://insideairbnb.com/) snapshot, Las Vegas / Clark
County. 20,296 listings × 19 columns. Inside Airbnb publishes under CC BY 4.0.

## Repository layout

```
str_operator_detection.ipynb   the deliverable, with outputs
airbnb_listings.csv            the source snapshot
artifacts/                     generated at runtime (embedding cache, audit queue CSV)
```

## License

Code is MIT. The dataset retains Inside Airbnb's CC BY 4.0 terms.
