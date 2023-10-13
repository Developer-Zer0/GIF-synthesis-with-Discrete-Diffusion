import hydra
import torch
from torchmetrics import Metric
from src.models.metrics.loss_func import get_loss_function

class ComputeLosses(Metric):
    def __init__(self,
                 loss_dict = {},
                 loss_opts = {},
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        losses = loss_dict
        losses["total"] = 1.0
        for loss in losses:
            self.add_state(loss, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses
        self.loss_opts = loss_opts

        # Instantiate loss functions
        self._losses_func = {loss: get_loss_function(loss)
                             for loss in losses.keys() if loss != "total"}
        # Save the lambda parameters
        #self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}
        # save the lambda weighting for each loss
        self._params = loss_dict

    def update(self, output):
        total: float = 0.0

        for loss in self.losses.keys():
            if loss == 'total':
                continue

            total += self._update_loss(loss, output)
        self.total += total.detach()
        self.count += 1
        return total

    def compute(self):
        count = getattr(self, "count")
        return {loss: getattr(self, loss)/count for loss in self.losses}

    def _update_loss(self, loss: str, outputs):
        # Update the loss
        val = self._losses_func[loss](outputs, self.loss_opts)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

    @property
    def is_differentiable(self) -> bool:
        return True
