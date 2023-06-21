# Unlearnable Examples: Making Personal Data Unexploitable
You can generate the poisoned dataset with a demo script below:
```shell
python em_poisons_generate.py --dataset <Dataset> --eps <Epsilon of perturbation>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`, `c100`, `svhn`, `imagenet100`
- --eps `<Epsilon of perturbation>`: `8/255`, `16/255`, ...

### Acknowledgement
- Code adapted from the official implementation of EM:
  [[Code]](https://github.com/HanxunH/Unlearnable-Examples)
  [[Paper]](https://openreview.net/pdf?id=iAmZUo0DxC0).