# Third-party notices

Our implementations build on the following Hugging Face PEFT codebases.

- [PEFT](https://github.com/huggingface/peft) provides the adapter implementations used in the language and image experiments.
- [PEFT MetaMathQA](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) provides the basis for the LLM mathematics benchmark, including data processing and evaluation conventions.
- [PEFT image-gen](https://github.com/huggingface/peft/tree/main/method_comparison/image-gen) provides the basis for the image training and evaluation code.

We credit the upstream authors and contributors for this work. The [PEFT citation](https://github.com/huggingface/peft#citing--peft) and both benchmark references are recorded in [CITATION.cff](CITATION.cff).

The language and image PEFT snapshots retain their upstream copyright notices and Apache-2.0 license files. The image benchmark sources also retain their Hugging Face Apache-2.0 headers. Existing source-specific licensing remains in force. This package does not assign a new blanket license to original project code, models or datasets.

EvalPlus is downloaded from its pinned public repository when requested by the preparation command. Its license remains with that repository. Models and datasets are subject to their own licenses and access terms.

The retained FineWiki evaluation sample is attributed to HuggingFaceFW/FineWiki and its underlying Wikipedia contributors. Consult the [dataset documentation](https://huggingface.co/datasets/HuggingFaceFW/finewiki) for the source terms.

## README icons

`assets/arxiv-icon.png` is the official 32px arXiv favicon, retained unchanged from https://arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png (retrieved 8 October 2026).

`assets/project-icon.svg` is this project's existing three-bar favicon, extracted without changing its shapes or colors. The same SVG is used by both local project-page copies.
