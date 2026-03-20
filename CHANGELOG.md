# Changelog

## [1.0.0] - 2026-03-21

### Initial Release

- Core OSA-Adapt framework: FiLM adapter, severity conditioner, severity-aware N1 loss
- Two-phase progressive adaptation protocol (BN adaptation + FiLM fine-tuning)
- Two-pass inference for resolving AHI circular dependency
- AHI estimator from sleep staging predictions
- Patient-level stratified cross-validation
- Severity-stratified few-shot sampling
- 8 baseline adaptation methods (Full FT, Last Layer, LoRA, Standard FiLM, BN Only, CORAL, MMD, No Adaptation)
- Chambon2018 and TinySleepNet model architectures with optional PhysioEx integration
- Medical metrics (sensitivity, specificity, AUC with bootstrap CI)
- Statistical significance testing (Wilcoxon, McNemar, Bonferroni correction, Cohen's d)
- Experiment scripts for main experiments, ablation study, data efficiency, severity analysis
- Default configuration file
- Comprehensive test suite
