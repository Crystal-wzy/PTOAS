# RMSNorm FP32 7168 Persistent Fragment Case

The PTODSL source is in `kernel.py`.

This PTODSL source is the persistent-fragment variant of the TileLang
RMSNorm kernel for `batch=4096` and `d=7168`.

- The weight buffer is loaded from GM to UB once.
- A SIMT init section copies the 28 valid `f32` weight elements into the
  persistent fragment.
- Each token SIMT section resumes and re-keeps those 28 elements.
- The remaining four elements of the 32-element physical fragment are layout
  padding and do not consume persistent slots.

The source uses the same UB offsets, `256` SIMT threads, and `64` token loop
as the corresponding TileLang-generated PTO kernel.
