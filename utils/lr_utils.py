class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value


class MyLRScheduler:
    def __init__(self, optimizer, last_epoch, niter, niter_decay, verbose):
        self.last_epoch = max(0, last_epoch)
        self.optimizer = optimizer
        self.niter = niter
        self.niter_decay = niter_decay
        self.verbose = verbose
        self.__update_lr(optimizer)
        self.__calculate_extra_info()

    def __update_lr(self, optimizer):
        self.lr = [group['lr'] for group in optimizer.param_groups]

    def __calculate_extra_info(self):
        if self.last_epoch <= self.niter:
            epoch_decay = self.niter_decay
        else:
            epoch_decay = self.niter + self.niter_decay - self.last_epoch

        if epoch_decay > 0:
            self.lrd = [i / epoch_decay for i in self.lr]
        else:
            self.lrd = [0 for _ in self.lr]

    def step(self):
        current_epoch = self.last_epoch + 1
        self.__calculate_new_lr(current_epoch)
        self.last_epoch = current_epoch
        self.__update_lr(self.optimizer) 

    def __calculate_new_lr(self, current_epoch):
        if current_epoch <= self.niter:
            return

        for lr, lrd, group in zip(self.lr, self.lrd, self.optimizer.param_groups):
            group['lr'] = lr - lrd
            if self.verbose:
                print('Update learning rate: %f -> %f' % (lr, group['lr']))


if __name__ == "__main__":
    opt = AttributeDict({
        'verbose': True,
        'niter': 2,
        'niter_decay': 2
    })
    optimizer = AttributeDict({
        'param_groups': [{ 'lr': 5e-3 }]
    })
    epoch_resume = 0
    lr = MyLRScheduler(optimizer, epoch_resume, opt.niter, opt.niter_decay, opt.verbose)
    for i in range(epoch_resume + 1, opt.niter + opt.niter_decay + 1):
        lr.step()

