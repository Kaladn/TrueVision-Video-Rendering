---
license: cc-by-4.0
---

# Data for PAPER [How Far is Video Generation from World Model: A Physical Law Perspective](https://huggingface.co/papers/2411.02385)

Project page: https://phyworld.github.io/

## Download Data

| Data Type            | Train Data (30K/300K/3M)                                                                                                                  | Eval Data                                                                                                                | Description                                                                                                 |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Uniform Motion**   | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/uniform_motion_eval.hdf5)                    | Eval data includes both in-distribution and out-of-distribution data                                        |
| **Parabola**         | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/parabola_eval.hdf5)                          | -          |
| **Collision**        | [30K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_30K.hdf5), [300K](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_300K.hdf5), [3M](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_3M.hdf5) | [Eval](https://huggingface.co/datasets/magicr/phyworld/blob/main/id_ood_data/collision_eval.hdf5)                          | -                                                                            | -                                                                                                                        | -                                                                                                           |
| **Combinatorial Data** | [In-template 6M templates00:59](https://huggingface.co/datasets/magicr/phyworld/tree/main/combinatorial_data)                                                                                                               | [Out-of-template](https://huggingface.co/datasets/magicr/phyworld/blob/main/combinatorial_data/combinatorial_out_of_template_eval_1K.hdf5) | In-template-6M includes train data (0:990 videos in each train template) and in-template eval data (990:1000 videos in each train template). Out-template refers to eval data from reserved 10 templates (templates60:69). |

## Citation

```
@article{kang2024how,
  title={How Far is Video Generation from World Model? -- A Physical Law Perspective},
  author={Kang, Bingyi and Yue, Yang and Lu, Rui and Lin, Zhijie and Zhao, Yang, and Wang, Kaixin and Gao, Huang and Feng Jiashi},
  journal={arXiv preprint arXiv:2406.16860},
  year={2024}
}
```