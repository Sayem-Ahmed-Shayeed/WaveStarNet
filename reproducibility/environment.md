# Execution environments

Exact software and accelerator records captured by each run are stored beside the corresponding results in `environment.json`, `benchmark_config.json`, and notebook outputs. These run-specific files are authoritative. Install the recorded GPU runtime with `pip install -r requirements.txt`.

The experiments used Python, PyTorch, timm, NumPy, pandas, scikit-learn, Pillow, and the libraries imported by the included notebooks. CUDA runs used the GPU and mixed-precision settings recorded by each experiment. The local CPU measurements were performed on an Intel Core i5-1135G7 processor. The model-only GPU efficiency benchmark was performed on an NVIDIA T4 and records its timing configuration in `plantwild/efficiency/results/plantwild_t4_efficiency_v02/benchmark_config.json`.

Install package versions matching the `environment.json` records before reproducing a run. Dataset locations are configurable in the notebooks and must point to the downloaded image data and the fixed split files included here.
