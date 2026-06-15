---
language:
- en
license: cc-by-4.0
size_categories:
- 100K<n<1M
task_categories:
- text-to-video
tags:
- time-lapse
- video-generation
- text-to-video
- metamorphic
configs:
- config_name: default
  data_files:
  - split: test
    path: Captions/ChronoMagic-Pro.csv
---

# ChronoMagic Dataset

This dataset contains time-lapse video-text pairs curated for metamorphic video generation. It was presented in the paper [ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation](https://huggingface.co/papers/2406.18522).

Project page: https://pku-yuangroup.github.io/ChronoMagic-Bench

# Usage

```
cat ChronoMagic-Pro.zip.part-* > ChronoMagic-Pro.zip 
unzip ChronoMagic-Pro.zip
```

<div align=center>
<img src="https://github.com/PKU-YuanGroup/ChronoMagic-Bench/blob/ProjectPage/static/images/logo_bench.jpg?raw=true" width="450px">
</div>
<h2 align="center"> <a href="https://pku-yuangroup.github.io/ChronoMagic-Bench/">[NeurIPS D&B 2024 Spotlight] ChronoMagic-Bench: A Benchmark for Metamorphic Evaluation of Text-to-Time-lapse Video Generation </a></h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h5>


## 💡 Description
- **Repository:** [Code](https://github.com/PKU-YuanGroup/ChronoMagic-Bench), [Page](https://pku-yuangroup.github.io/ChronoMagic-Bench/), [Data](https://huggingface.co/collections/BestWishYsh/chronomagic-bench-667bea7abfe251ebedd5b8dd)
- **Paper:** [https://huggingface.co/papers/2406.18522](https://huggingface.co/papers/2406.18522)
- **Point of Contact:** [Shenghai Yuan](shyuan-cs@hotmail.com)
- **License:** CC-BY-4.0

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star and citation.

```BibTeX
@article{yuan2024chronomagic,
  title={Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation},
  author={Yuan, Shenghai and Huang, Jinfa and Xu, Yongqi and Liu, Yaoyang and Zhang, Shaofeng and Shi, Yujun and Zhu, Rui-Jie and Cheng, Xinhua and Luo, Jiebo and Yuan, Li},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={21236--21270},
  year={2024}
}
```