import math


class LRScheduler(object):
    """
    Learning Rate Scheduler
        Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
        Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
        Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, restart_step=20, warmup_epochs=0):
        self.mode = mode
        print('> Using {} LR Scheduler!'.format(self.mode))

        self.lr = base_lr

        if mode == 'step':
            assert lr_step

        self.lr_step = lr_step
        self.restart_step = restart_step
        self.iters_per_epoch = iters_per_epoch

        self.N = num_epochs * iters_per_epoch
        self.restart_period = restart_step * iters_per_epoch

        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i

        if self.mode == 'cos':
            batch_idx = float(T)
            while batch_idx / self.restart_period > 1.0:
                batch_idx = batch_idx - self.restart_period

            lr = self.lr * 0.5 * (1.0 + math.cos(1.0 * (batch_idx / self.restart_period) * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters

        if epoch > self.epoch:
            # print('\n=>Epoches %i, learning rate = %.4f, \
            #     previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch

        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


def cosine_annealing_lr(optimizer, init_lr, period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    # \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
    # \cos(\frac{T_{cur}}{T_{max}}\pi))

    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx / restart_period > 1.:
        batch_idx = batch_idx - restart_period
        # restart_period = restart_period * 2.

    radians = math.pi * (batch_idx / restart_period)
    lr = init_lr * 0.5 * (1.0 + math.cos(radians))

    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0:
            param_group['lr'] = lr

    return optimizer, lr
