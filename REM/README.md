# Robust Unlearnable Examples: Protecting Data Against Adversarial Learning
You can generate the poisoned dataset with a demo script below:
```shell
python rem_poisons_generate.py --dataset <Dataset> --eps <Epsilon of perturbation>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`, `c100`, `svhn`, `imagenet100`
- --eps `<Epsilon of perturbation>`: `8/255`, `16/255`, ...

### Acknowledgement
- Code adapted from the official implementation of REM:
  [[Code]](https://openreview.net/pdf?id=baUQQPwQiAg)
  [[Paper]](https://arxiv.org/pdf/2205.12141.pdf).
