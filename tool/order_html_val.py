import os

web_path = r"C:\Users\Romain Guesdon\Desktop\fashion_cycle_PATN\web"

with open(os.path.join(web_path, "index.html")) as f:
    html = f.read()

html = html.split("<h3>")
imgs = dict([i.split('</h3>\n') for i in html[1:]])
vals = [{k: v for k, v in imgs.items() if f" val {i}" in k} for i in range(5)]

out_html = html[0]
for i, d in enumerate(vals):
    out_html += f"<h2>Val {i}</h2>\n"
    out_html += "\n".join([f"<h3>{k}</h3>\n" + v for k, v in d.items()])

out_html += "  </body>\n<html>"
with open(os.path.join(web_path, "index_vals.html"), 'w') as f:
    f.write(out_html)

print("")
