# Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training
You can generate the poisoned dataset with a demo script below:
```shell
python hypo_poisons_generate.py --dataset <Dataset> --eps <Epsilon of perturbation> --step_size <Step>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`, `c100`, `svhn`, `imagenet100`
- --eps `<Epsilon of perturbation>`: `8/255`, `16/255`, ...
- --step_size `<Step>`: `0.8/255`, `1.6/255`, ...

### Acknowledgement
- Code adapted from the official implementation of HYPO:
  [[Code]](https://github.com/TLMichael/Delusive-Adversary)
  [[Paper]](https://arxiv.org/pdf/2102.04716.pdf).
