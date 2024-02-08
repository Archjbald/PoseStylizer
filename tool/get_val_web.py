def get_val_web(html_path):
    with open(html_path) as f:
        html_data = f.read().split('\n')

    val_data = []
    val_imgs = {}
    img = []
    id = None
    keep = True
    header = []
    for l in html_data:
        if '<h3>' in l:
            if not header:
                header = val_data.copy()
            if "val" in l:
                keep = True
                id = int(l.split(' ')[-1].split('<')[0])
            else:
                keep = False
        if keep:
            val_data.append(l)
            if id is not None:
                img.append(l)

        if '</table>' in l:
            keep = False
            if id is not None:
                val_imgs[id] = val_imgs.setdefault(id, []) + ['\n'.join(img)]
            img = []
            id = None

    with open(html_path.replace('.html', '_val.html'), 'w') as f:
        f.write('\n'.join(header))
        for im in val_imgs.values():
            f.write('\n'.join(im))


if __name__ == '__main__':
    get_val_web(r"D:\Networks\PoseStylizer\logs\multi\synthe_multi_small_w\web\index.html")
