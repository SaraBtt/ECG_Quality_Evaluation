# ECG Quality Evaluation

This package was developed from the paper:

Sara Battiston, Roberto Sassi, Massimo W. Rivolta, "Evaluating the Quality of CycleGAN Generated ECG Data for Myocardial Infarction Classification," Computing in Cardiology 2024.

Direct PDF: https://www.cinc.org/archives/2024/pdf/CinC2024-457.pdf

DOI: https://doi.org/10.22489/CinC.2024.457

## What is included

- `utils.py` provides `make_data_class`, which builds paired ECG samples from real and generated inputs. Each pair stores the real sample and the synthetic sample together, with their labels, so the rest of the package can process them consistently.
- `plot_ecg_bands.py` plots real and synthetic ECGs lead by lead. It can show individual traces or percentile bands, which makes it easy to compare the spread of generated signals against the real ones.
- `plot_umap_maps.py` reduces ECG samples to 2D with UMAP and plots the real distribution as a density map with synthetic samples overlaid as scatter points.
- `GAN_train_test.py` trains and evaluates the GAN-train and GAN-test classifiers and saves their scores and loss curves.
- `GAN_train_test_classifier.py` defines the ECG classifier used by the training and testing workflow.
- `ST_segments_evaluation.py` compares real and synthetic ECG ST segments with radar-style plots and confidence bounds.

## Try it out

There is a notebook in the repository for a guided run, and test data is included in `TestData/` so you can try the plots and evaluation scripts without preparing your own dataset first.

## Files to look at first

- `TEST_scripts.ipynb` for a worked example
- `TestData/` for sample ECG arrays
