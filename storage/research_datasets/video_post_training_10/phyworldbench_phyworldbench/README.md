---
license: mit
task_categories:
- text-to-video
tags:
- physics
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: category
    dtype: string
  - name: subcategory
    dtype: string
  - name: Prompt
    dtype: string
  - name: Physics
    dtype: string
  - name: Detailed
    dtype: string
  - name: Detailed_long
    dtype: string
  - name: Prompt_index
    dtype: string
  - name: Physics_index
    dtype: string
  - name: Detailed_index
    dtype: string
  splits:
  - name: train
    num_bytes: 457059
    num_examples: 350
  download_size: 249627
  dataset_size: 457059
---

# PhyWorldBench

This repository hosts the core assets of **PhyWorldBench**, the 1,050 JSON prompt files, the evaluation standards, and the physics categories and subcategories, used to assess physical realism in text-to-video models. 
We have also attached over 10k of the generated videos that we used for experimentation. Since the videos are **not** part of our benchmark, we did not include all the generated videos.