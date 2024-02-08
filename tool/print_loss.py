import matplotlib.pyplot as plt

WEIGHTS = {
    'pair_L1loss': 1,
    'D_PP': 1,
    'D_PB': 1,
    'pair_GANloss': 1,
    'origin_L1': 1,
    'perceptual': 1,
    'loss_NCE': 1,
    'loss_NCE_both': 1,
    'loss_NCE_Y': 1,
}

colors = {
    'pair_L1loss': 'tab:red',
    'D_PP': 'tab:blue',
    'D_PB': 'tab:orange',
    'pair_GANloss': 'tab:green',
    'origin_L1': 'tab:brown',
    'perceptual': 'tab:pink',
    'loss_NCE': 'tab:brown',
    'loss_NCE_both': 'tab:red',
    'loss_NCE_Y': 'tab:pink',
}

equiv = {
    'A': 'origin_L1',
    'B': 'pair'
}
colors = {}


def print_loss(log_path, mode='train', weights={}):
    with open(log_path) as f_log:
        loss_logs = f_log.read().split('\n')

    loss_logs = [l.split() for l in loss_logs if "===" not in l and
                 (('Val' not in l and not mode == 'val' and 'count' in l) or ('Val' in l and mode == 'val'))]

    losses = {}
    lbls = []

    for iter in loss_logs:
        if not iter:
            continue
        epoch = int(iter[1][:-1])
        if epoch < 0:
            continue
        loss = []
        lbl = []
        for k in iter[7 if mode == 'val' else 8:]:
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
    # ax.set_ylim([0, 7])

    losses = {lbl: [losses[e][i] for e in epochs] for i, lbl in enumerate(lbls)}
    ban_list = ['D_BP', 'IS', 'IS std']
    len_lbl = len([lbl for lbl in lbls if lbl not in ban_list])
    losses['total'] = [sum([v[e] for k, v in losses.items() if k not in ban_list]) / len_lbl for e in
                       range(len(epochs))]
    lbls.append('total')

    for lbl in lbls:
        # if lbl in ['pair_L1loss', 'origin_L1', 'perceptual']:
        if False and lbl in ['pair_L1loss', 'D_PP', 'D_PB', 'IS_val', 'IS_std']:  # 'pair_GANloss'
            # if lbl in ['loss_NCE_both', 'D', 'D_PP', 'D_PB', 'adv', ]:
            # if 'IS' not in lbl:
            # if lbl not in "total":
            continue

        w = weights.setdefault(lbl, 1)
        # ax.plot(epochs, losses[lbl], label=f"{lbl} ({w})",
        ax.plot(epochs, losses[lbl], label=f"{lbl}",
                color=colors.setdefault(lbl, None))

    ax.legend()
    plt.show()

    return


def get_weights(options):
    options = options.split('\n')
    options = {l.split(': ')[0]: l.split(': ')[1] for l in options if l}
    equiv = {
        'identity': 'idt',
        'adversarial': 'adv',
    }
    lambdas = {}
    for opt, val in options.items():
        if opt[:7] == 'lambda_':
            key = opt[7:]
            key = equiv[key] if key in equiv else key
            lambdas[key] = float(val)

    return lambdas


FOLDER = r"D:\Networks\PoseStylizer\logs\synthe\synthe_mark_3_bis"
LOG_PATH = f"{FOLDER}/loss_log.txt"
OPT_PATH = f"{FOLDER}/options.txt"
with open(OPT_PATH) as f:
    options = f.read()
    weights = get_weights(options)

print_loss(LOG_PATH, mode='train', weights={})
print(weights)
