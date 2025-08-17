# Hardware-aware Training

The training utilities automatically inspect the host machine and adapt the
model to available resources.  :mod:`scripts.train_target_clone` samples RAM,
CPU frequency, free disk space and GPU memory before any feature extraction
occurs.  Based on these metrics a **mode** is selected and recorded in
``model.json`` together with a ``feature_flags`` section.

Modes roughly correspond to the following configurations:

| Mode     | Description |
|----------|-------------|
| ``lite`` | Minimal feature set without order‑book inputs.  Designed for low
memory VPS hosts. |
| ``standard``/``heavy`` | Enables technical indicators and order‑book
features when sufficient CPU/GPU resources are present. |
| ``deep``/``rl`` | Allows transformer or reinforcement learning refinements on
powerful hardware. |

Downstream components read these flags to stay in sync:

* :mod:`scripts.generate_mql4_from_model.py` skips embedding order‑book
  functions unless ``feature_flags.order_book`` is ``true``.
* :mod:`scripts.online_trainer.py` filters unsupported features and passes the
  appropriate ``--lite-mode`` flag when regenerating Expert Advisors.

This metadata is persisted in ``model.json`` so that models trained on one
machine can be safely deployed on another with different capabilities.

