from mmcv.runner.hooks import HOOKS, Hook
from spikingjelly.activation_based import functional


@HOOKS.register_module()
class NetResetHook(Hook):

    def before_train_iter(self, runner):
        """Check whether the training dataset is compatible with head.
        Args:
            runner (obj: `IterBasedRunner`): Iter based Runner.
        """
        functional.reset_net(runner.model)



    def before_val_iter(self, runner):
        """Check whether the eval dataset is compatible with head.
        Args:
            runner (obj:`IterBasedRunner`): Iter based Runner.
        """
        functional.reset_net(runner.model)
