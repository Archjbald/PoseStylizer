import matplotlib.pyplot as plt


def print_loss(log_path, mode='train'):
    with open(log_path) as f_log:
        loss_logs = f_log.read().split('\n')

    loss_logs = [l.split() for l in loss_logs if "===" not in l and
                 (('Val' not in l and not mode == 'val') or ('Val' in l and mode == 'val'))]

    losses = {}
    lbls = []

    for iter in loss_logs:
        if not iter:
            continue
        epoch = int(iter[1][:-1])
        if epoch < 2:
            continue
        loss = []
        lbl = []
        for k in iter[6 + (1 if mode == 'val' else 0):]:
            if ':' in k:
                lbl.append(k[:-1])
            else:
                loss.append(float(k))
        if not lbls:
            lbls = lbl

        losses[epoch] = losses.setdefault(epoch, []) + [loss, ]

    epochs = list(losses.keys())
    epochs.sort()
    for epoch in epochs:
        losses[epoch] = [sum([it[l] for it in losses[epoch]]) / len(losses[epoch]) for l in range(len(lbls))]

    fig, ax = plt.subplots()
    for i, lbl in enumerate(lbls):
        if lbl in ['pair_L1loss', 'origin_L1', 'perceptual']:
            # if lbl in ['pair_L1loss', 'D_PP', 'D_PB', 'pair_GANloss']:
            continue
        ax.plot(epochs, [losses[e][i] for e in epochs], label=lbl)

    ax.legend()
    plt.show()

    return


LOG_PATH = 'logs/draiver_market/loss_log.txt'
print_loss(LOG_PATH, mode='val')
