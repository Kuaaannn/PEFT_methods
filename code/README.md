# Scientific implementations

Start with the [release guide](../README.md) and [experiment commands](../docs/experiments.md).

This directory contains the original trainers, evaluators, matrix operators and analysis routines. The portable launchers in `../scripts/` prepare their inputs and run the paper settings. The language and image PEFT runtimes are kept separately because the experiments used different builds.

The shared operator libraries retain internal utilities required by the original modules. The release launchers select the experiments used in the current manuscript.
