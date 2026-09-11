# Banking77 provenance and use

Intent augmentation uses the official [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset, not an MTEB mirror. The loader uses the source URLs specified in that dataset's official loading script and caches the training CSV as `data/banking77_train.csv`.

Banking77 contains banking queries, so it is used only as bounded auxiliary supervision for transferable support concepts: delivery/status, missing/pending outcomes, refunds, account/security, and payments. It never supplies reply text, historical resolutions, golden-set labels, or Amazon-specific policy. `src/model_trainer.py` documents the label map and caps each target intent at 350 examples to keep the Twitter corpus dominant.

Citation: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, ACL 2020. Dataset license: CC BY 4.0.
