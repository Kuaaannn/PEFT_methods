"""LoRA-OFT: controlled comparison of LoRA, DoRA and original OFT in LLM post-training.

Design notes live in ../notes. The two entry points are:
    loraoft.train     one training run, any method, any task
    sweeps/           materialise and run a grid of cells
"""

__version__ = "0.1.0"
