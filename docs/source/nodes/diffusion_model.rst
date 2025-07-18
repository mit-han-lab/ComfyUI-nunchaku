Diffusion Model Nodes
=====================

.. _nunchaku-flux-dit-loader:

Nunchaku FLUX DiT Loader
------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/nodes/NunchakuFluxDiTLoader.png
    :alt: NunchakuFluxDiTLoader

A node for loading Nunchaku FLUX models. It manages model loading, device selection, attention implementation, CPU offload, and caching for efficient inference.

**Inputs:**

- **model_path**: The path to the Nunchaku FLUX model folder (legacy format) or ``.safetensors`` file. You can download the model from `HuggingFace <hf_nunchaku_>`_ or `ModelScope <ms_nunchaku_>`_.
- **cache_threshold**: Adjusts the first-block caching tolerance like ``residual_diff_threshold`` in WaveSpeed. Increasing the value enhances speed at the cost of quality. A typical setting is 0.12. Setting it to 0 disables the effect. See :ref:`nunchaku:usage-fbcache` for more details.
- **attention**: Attention implementation. Options include ``flash-attention2`` and ``nunchaku-fp16``. The ``nunchaku-fp16`` uses FP16 attention, offering ~1.2× speedup. Note that 20-series GPUs can only use ``nunchaku-fp16``.
- **cpu_offload**: Whether to enable CPU offload for the transformer model. Options include:

  - ``auto``: Will enable it if the GPU memory is less than 14GiB
  - ``enable``: Force enable CPU offload
  - ``disable``: Disable CPU offload
- **device_id**: The GPU device ID to use for the model.
- **data_type**: Specifies the model's data type. Default is ``bfloat16``. For 20-series GPUs, which do not support ``bfloat16``, use ``float16`` instead.
- **i2f_mode**: For Turing (20-series) GPUs, controls the GEMM implementation mode. Options are `enabled` and `always`. This option is ignored on other GPU architectures.

**Outputs:**

- **model**: The loaded diffusion model.

.. seealso::

    See API reference: :class:`~comfyui_nunchaku.nodes.models.flux.NunchakuFluxDiTLoader`.
