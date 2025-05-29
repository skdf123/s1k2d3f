# SkelDiff: A Skeleton-Guided Diffusion Model for Oracle Bone Script Decipherment

This repository is part of an anonymous submission to NeurIPS 2025. Code and pre-trained models will be released upon acceptance.

---

## Overview

This repository contains configuration files, model definitions, and minimal demo data for **SkelDiff**, a skeleton-guided diffusion model for structure-aware glyph reconstruction under weak supervision.

For full details, please refer to the accompanying anonymous manuscript.

---

## Repository Structure

```text
main/
├── configs.yml             # Experiment configuration file
├── dataset.py              # Minimal data loader
├── main.py                 # Entry point 
├── models/
│   ├── extract_skeleton/   # Skeleton extractor components
│   └── unet/               # UNet backbone
├── spi_loss.py             # SPI loss implementation
├── train.py                # Training script
├── demo_data/
│   └── train/
│       ├── input/          # Sample training inputs
│       └── target/         # Corresponding ground truths
```

---

## Note

This repository is a **review-phase stub** designed to demonstrate structure and intent.  
The full implementation will be made public upon paper acceptance.

---

## Contact

For reproducibility or artifact review purposes, please reach out via the OpenReview discussion thread.

