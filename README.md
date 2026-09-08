# Deep Learning Project repository (forked from the official **LATINO-PRO** repo)

> **LAtent consisTency INverse sOlver with PRompt Optimization** – <https://arxiv.org/abs/2503.12615>.

This is the fork with our implementations for the Deep Learning Project, in addition to the base
code from the original repo.

---

## 📦 Installation

```bash
# 1. Clone the repo and enter it
git clone https://github.com/Bar-Ben-Haim/LATINO-PRO.git
cd LATINO-PRO

# 2. (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# Windows-PowerShell:  .venv\Scripts\Activate.ps1

# 3. Install all Python dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

No custom CUDA extensions are required; GPU acceleration is handled automatically by PyTorch if a compatible device is present.

---

## 🚀 Quick start

The repository contains two ready‑to‑run scripts. **All hyper‑parameters are controlled by the YAML files inside the **configs** directory**, so the basic usage is simply:

```bash
# Baseline LATINO model
python main_LATINO.py            # uses configs/LATINO.yaml by default

# Prompt‑optimized LATINO‑PRO model
python main_LATINO_PRO.py        # uses configs/LATINO_PRO.yaml by default
```

```configs/problem``` offers different inverse problem operators to choose from.

```configs/image``` includes two examples taken from the FFHQ and AFHQ datasets.

---

# 🔬 Fork additions

> **Everything in this section is our work for the Deep Learning Project.**
> We implemented a batch runner, two extra blur operators, and two experiments on the
> data-fidelity weight.

## New files

| File | What it is |
| --- | --- |
| `data_prep.py` | Fetches datasets for the experiments. |
| `run_batch.py` | Runs LATINO / LATINO-PRO on a batch of images and scores them. |
| `run_experiments.py` | Calls the batch runner for the different experiment configurations. |
| `experiment_consts.py` | Experiment constants. |
| `delta_sweep.py` | Sweeps the data-fidelity weight δ. |
| `morozov.py` | Sets the prox weight using the Morozov principle, instead of the hard-coded δ table. |
| `vae_domain_check.py` | Measures the SDXL VAE round-trip `D(E(x))` per dataset. |
| `data/sngfaces_*.json` | Metadata used to fetch the SNGFaces dataset. |
| `morozov_results/rho_ladder.png` | Result figure from the Morozov experiments. |

## How to run

**1. Get the data**

```bash
python data_prep.py prepare --only ffhq1024 ood_sngfaces ood_medical
```

**2. Batch run on dataset**

```bash
python run_batch.py \
  --images-dir data/ffhq1024 \
  --problem deblurring_motion \
  --prompt "a sharp photo of a face" \
  --out-dir results/motion \
  --override problem.sigma_y=0.01 \
  --limit 10
```

Add `--pro` for LATINO-PRO.

**3. The experiment tables**

```bash
python run_experiments.py --arm latino     # FFHQ + SNGFaces + chest X-ray
python run_experiments.py --arm morozov    # table δ vs. Morozov δ, 6 operators
```

**4. δ sweep**

```bash
python delta_sweep.py --dataset ood_sngfaces --problem deblurring_motion -n 25
```

**5. VAE domain check**

```bash
python vae_domain_check.py
```

---

## 📓 Interactive notebooks

| Notebook           | Purpose                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| LATINO.ipynb       | Hands‑on introduction to the baseline solver: load a sample, apply degradation, reconstruct, inspect metrics. |
| LATINO_PRO.ipynb   | Full prompt‑optimization workflow adapted to the LoRA-LCM model to work on small GPUs (eg. Colab T4 GPU).     |

---

## 🗂️ Repository layout

```
LATINO-PRO/
├── configs/              # YAML config files controlling every experiment
├── samples/              # example images for tests
├── data/                 # (fork) datasets fetched by data_prep.py
├── morozov_results/      # (fork) Morozov figures
├── LATINO.ipynb          # baseline interactive notebook
├── LATINO_PRO.ipynb      # full prompt‑optimization notebook
├── LICENSE               # License
├── inverse_problems.py   # deepinverse operators are defined here
├── main_LATINO.py        # base LATINO restoration
├── main_LATINO_PRO.py    # prompt‑optimized restoration
├── motionblur.py         # helper code for motion‑blur degradations
├── noise_schemes.py      # definition of various inverse solvers
├── utils.py              # miscellaneous utilities
├── data_prep.py          # (fork) dataset download
├── experiment_consts.py  # (fork) experiment constants
├── run_batch.py          # (fork) batch runner
├── run_experiments.py    # (fork) the experiment table
├── delta_sweep.py        # (fork) data-fidelity weight sweep
├── morozov.py            # (fork) Morozov principle experiments
├── vae_domain_check.py   # (fork) VAE domain check
├── requirements.txt      # Python dependencies
└── README.md             # this file
```

---

## 📄 Citation

If you use LATINO‑PRO in academic work, please cite:

```bibtex
@misc{spagnoletti2025latinoprolatentconsistencyinverse,
      title={LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization}, 
      author={Alessio Spagnoletti and Jean Prost and Andrés Almansa and Nicolas Papadakis and Marcelo Pereyra},
      year={2025},
      eprint={2503.12615},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12615}, 
}
```

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
