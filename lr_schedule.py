def inv_lr_scheduler(optimizer, iter_num, gamma=0.01, power=0.75, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    # [FOR SEED, SEED-IV]
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        # [FOR SEED, SEED-IV]
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer
