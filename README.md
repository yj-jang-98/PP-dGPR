# Privacy-preserving distributed Gaussian process regression

Experimental code for secure average consensus, distributed GP hyperparameter
optimization, and product-of-experts (PoE) prediction on SARCOS and Diabetes.

## Run the experiments

Install the Python dependencies in your environment:

```sh
python -m pip install -r requirements.txt
jupyter lab
```

Open a notebook with the project folder as the working directory, restart its
kernel, and run its cells in order. Keep the source folders and `data/` beside the
notebooks. The manuscript reports Python 3.12.13, GPyTorch 1.15.2, and an NVIDIA T4
on Google Colab. Runtime depends on the hardware, library versions, and worker
counts; `requirements.txt` lists dependencies rather than a frozen environment.

| Notebook | Experiment | Output |
| --- | --- | --- |
| `figs2-3.ipynb` | Hyperparameter convergence and consensus error versus scale and iteration | `img/hyp.pdf`, `img/accuracy.pdf`, supplementary timing summary |
| `table1.ipynb` | Runtime and PoE agreement across network configurations | Runtime, accuracy, graph, and training-time tables displayed in the notebook |

Create `img/` if it is absent. Saved notebook outputs are cleared so that tables
and figures are generated from the current settings.

## Settings and data

Both notebooks use 4,000 SARCOS training observations and the first 200 observations
of `data/sarcos_inv_test.mat`. Each SARCOS observation has 21 inputs and seven
outputs; one GP is fitted per output. Diabetes is loaded from scikit-learn and
split with `random_state=42` into 353 training and 89 test observations. Each
notebook seeds its own training-sample selection and shuffling; their sampling
sequences differ.

| Setting | Figures 2–3 | Table I |
| --- | --- | --- |
| Network `(agents, neighbors)` | `(20, 4)` | SARCOS: `(20, 4)`, `(20, 19)`, `(40, 4)`; Diabetes: `(10, 4)`, `(20, 4)`, `(20, 19)` |
| Hyperparameter iterations | 30 | 30 per dataset and topology |
| Hyperparameter consensus scale | `2**-20` | `2**-20` |
| Prediction consensus iterations | 20, retaining iterations 0–20 | 100, retaining the final state |
| Prediction consensus scales | `1/10`, `1/100`, `1/10000` | `1/10000` |
| Modulus and weight scale | `q=2**40`, `Lw=1/40` | Same |
| Timing repeats | 20 | 20 |
| Modeled delay per communication phase | 0.2 ms | 0.2 ms |

`Lz_values` stores the **reciprocals** of prediction scales. Hyperparameters use
step size `0.1 * 0.99**t`. Lengthscale and signal variance are initialized within
`[5, 15]`; noise variance is initialized within `[5, 15]` for SARCOS and `[15, 25]`
for Diabetes.

PoE-agreement RMSE measures deviation from plaintext PoE, averaged over agents
for the proposed method. SARCOS sums squared errors over its seven outputs before
averaging over test points and taking the square root. Target-prediction RMSE is
also displayed in `table1.ipynb` as a separate diagnostic.

Online totals cover the configured test batch and exclude hyperparameter training
and data preprocessing. Local prediction and aggregation times are added per
repeat before calculating the mean and population standard deviation. The
proposed protocol models two communication phases per consensus iteration.

## Comparison methods and Lattigo

`table1.ipynb` includes two adaptations, labeled explicitly in its output:

- **Nawaz-inspired HE-PoE:** reuse exact local GP predictions and aggregate
  encrypted precision-weighted means and precisions with Lattigo multiparty
  CKKS. This implementation does not use sparse GP summaries. Its total includes
  local prediction, the backend call, final division, and six communication
  phases.
- **Luo-inspired SS-PoE:** evaluate local GPs on simulated secret shares, then
  aggregate predictions without opening the local values. The simulation uses
  floating-point shares and bounded masks with fixed-point wrapping. Its total
  includes computation, serialization, and modeled latency and bandwidth.

Both comparisons run once per dataset and agent count, using the local parameters
and PoE reference from the first corresponding topology (four neighbors).
The experiments run logical parties within local Python/Go processes.

The HE backend requires Go 1.24 or newer and Lattigo v6.2.0, pinned in
`nawaz_sdsgp/lattigo_backend/go.mod`. The first cell of `table1.ipynb` builds the
backend on Linux/Colab. On Windows, build it before running the notebook:

```powershell
cd nawaz_sdsgp/lattigo_backend
go build -o nawaz_lattigo.exe .
```

Return to the project folder before starting the notebook. A custom binary path
can be supplied through `NAWAZ_LATTIGO_BACKEND`. In Colab, upload the complete
project to Drive; the setup cell finds it automatically when there is one copy.
If needed, set `PPDGPR_PROJECT_ROOT` to its absolute folder path before setup.
First-time dependency installation and Go builds require internet access.

## Source files

| File or folder | Purpose |
| --- | --- |
| `hypOpt.py` | Exact local GP models and consensus/gradient hyperparameter updates |
| `share.py` | Graph caches, vectorized additive masks, and secure average consensus |
| `plot.py` | Figure 2 and Figure 3 plotting functions |
| `final_experiments.py` | Table I graph construction, prediction, aggregation, and timing helpers |
| `luo_ppgpr/` | Secret-sharing simulation for the SS-PoE comparison |
| `nawaz_sdsgp/` | Python interface and Go source for the HE-PoE comparison |
| `data/sarcos_inv.mat` | SARCOS training source |
| `data/sarcos_inv_test.mat` | SARCOS test source |

Include both notebooks, these source files/folders, the two `.mat` files,
`requirements.txt`, and this README with the submission. Existing `data/*.pt`
checkpoints and `img/*.pdf` files are previous run artifacts; neither notebook
loads them. Python caches and prebuilt executables can be omitted from a source
submission. Build the HE executable for the machine on which it will run.
