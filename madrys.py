import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MadrysLoss(nn.Module):
    def __init__(
        self,
        step_size=2/255,
        epsilon=8/255,
        perturb_steps=10,
        distance="L_inf",
    ):
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        if self.distance == "L_inf":
            x_adv = x_natural.clone() + self.step_size * torch.randn(
                x_natural.shape
            ).to(device)
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(
                    torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon
                )
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        elif self.distance == "L_2":
            l = len(x_natural.shape) - 1
            rp = torch.randn_like(x_natural)
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1] * l))
            x_adv = x_natural.clone() + self.step_size * rp / (rp_norm + 1e-10).to(
                device
            )
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]

                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(
                    -1, *([1] * l)
                )
                scaled_g = grad / (g_norm + 1e-10)
                x_adv = x_adv.detach() + self.step_size * (scaled_g).detach()
                diff = x_adv - x_natural
                diff = diff.renorm(p=2, dim=0, maxnorm=self.epsilon)
                x_adv = x_natural.clone() + diff
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = x_natural.clone() + self.epsilon * torch.randn(x_natural.shape).to(
                device
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        outputs = model(x_adv)
        loss = self.cross_entropy(outputs, y)

        return outputs, loss
