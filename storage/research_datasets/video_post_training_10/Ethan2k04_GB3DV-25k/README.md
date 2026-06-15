---
license: cc
task_categories:
- text-to-video
language:
- en
pretty_name: GB3DV-25k
size_categories:
- 10K<n<100K
---
# Good Bad Geometry 3D Video Dataset (GB3DV-25k)

<table>
  <tr>
    <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/68383a9465209bfdd67267cd/ney_agz1yLzNm5NH5K6Ml.gif"/></td>
    <td align="center"><img src="https://cdn-uploads.huggingface.co/production/uploads/68383a9465209bfdd67267cd/GvlL-ibs6CSA4UHsExAMS.gif"/></td>
  </tr>
  <tr>
    <td align="center"><b>Bad Geometry</b></td>
    <td align="center"><b>Good Geometry</b></td>
  </tr>
</table>

This is the official dataset of paper VIGOR: VIdeo Geometry-Oriented Reward for Temporal Generative Alignment.
The dataset contains 4 scene types: static_indoor, static_outdoor, dynamic_indoor, and dynamic_outdoor.
Each scene type constitutes a folder containing various prompt subfolders (10 seeds each prompt).
The resulting dataset has 25,600 unique videos with diverse degrees of geometric consistency.