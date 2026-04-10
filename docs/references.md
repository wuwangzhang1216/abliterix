[← back to README](../README.md)

# References

Papers Abliterix builds on, plus the BibTeX entries you'll need for citation.

Abliterix builds on the following research:

- **Abliteration**: Arditi, A., et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). *NeurIPS 2024*.
- **Projected Abliteration**: grimjim (2025). [Projected Abliteration](https://huggingface.co/blog/grimjim/projected-abliteration). Norm-preserving biprojection for refusal removal.
- **COSMIC**: Siu, V., et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations](https://arxiv.org/abs/2506.00085). *ACL 2025 Findings*.
- **Angular Steering**: Vu, H. M. & Nguyen, T. M. (2025). [Angular Steering: Behavior Control via Rotation in Activation Space](https://arxiv.org/abs/2510.26243). *NeurIPS 2025 Spotlight*.
- **Selective Steering**: (2026). [Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection](https://arxiv.org/abs/2601.19375).
- **Surgical Refusal Ablation**: Cristofano, A. (2026). [Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning](https://arxiv.org/abs/2601.08489).
- **Spherical Steering**: (2026). [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://arxiv.org/abs/2602.08169).
- **Steering Vector Fields**: (2026). [Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models](https://arxiv.org/abs/2602.01654).
- **Optimal Transport**: (2026). [Efficient Refusal Ablation in LLM through Optimal Transport](https://arxiv.org/abs/2603.04355).
- **Multi-Direction Refusal**: (2026). [There Is More to Refusal in Large Language Models than a Single Direction](https://arxiv.org/abs/2602.02132).

<details>
<summary>Classic references</summary>

- **Abliteration (original)**: Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). *NeurIPS 2024*.
- **Representation Engineering**: Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405). *arXiv:2310.01405*.
- **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022*.
- **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902). *KDD 2019*.
- **TPE**: Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011). [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization). *NeurIPS 2011*.
- **PaCMAP**: Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). [Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization](https://jmlr.org/papers/v22/20-1061.html). *JMLR*, 22, 1–73.

</details>

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11717}
}

@article{zou2023representation,
  title   = {Representation Engineering: A Top-Down Approach to AI Transparency},
  author  = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and Dombrowski, Ann-Kathrin and Goel, Shashwat and Li, Nathaniel and Byun, Michael J. and Wang, Zifan and Mallen, Alex and Basart, Steven and Koyejo, Sanmi and Song, Dawn and Fredrikson, Matt and Kolter, J. Zico and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2310.01405},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.01405}
}

@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2106.09685}
}

@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages     = {2623--2631},
  year      = {2019},
  url       = {https://arxiv.org/abs/1907.10902}
}

@inproceedings{bergstra2011algorithms,
  title     = {Algorithms for Hyper-Parameter Optimization},
  author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {2546--2554},
  year      = {2011},
  url       = {https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization}
}

@article{cristofano2026sra,
  title   = {Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning},
  author  = {Cristofano, Andrea},
  journal = {arXiv preprint arXiv:2601.08489},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.08489}
}

@article{spherical2026,
  title   = {Spherical Steering: Geometry-Aware Activation Rotation for Language Models},
  journal = {arXiv preprint arXiv:2602.08169},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.08169}
}

@article{svf2026,
  title   = {Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models},
  journal = {arXiv preprint arXiv:2602.01654},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.01654}
}

@article{selective2026,
  title   = {Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection},
  journal = {arXiv preprint arXiv:2601.19375},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.19375}
}

@inproceedings{siu2025cosmic,
  title     = {{COSMIC}: Generalized Refusal Direction Identification in {LLM} Activations},
  author    = {Siu, Vincent and others},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2506.00085}
}

@inproceedings{vu2025angular,
  title     = {Angular Steering: Behavior Control via Rotation in Activation Space},
  author    = {Vu, Hieu M. and Nguyen, Tan M.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Spotlight},
  url       = {https://arxiv.org/abs/2510.26243}
}

@article{wang2021pacmap,
  title   = {Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  author  = {Wang, Yingfan and Huang, Haiyang and Rudin, Cynthia and Shaposhnik, Yaron},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  pages   = {1--73},
  year    = {2021},
  url     = {https://jmlr.org/papers/v22/20-1061.html}
}
```

</details>
