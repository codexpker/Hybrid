"""Plot sealed source-only FIT and SELECT curves with both fixed selections."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

root=Path('docs/experiments/native_supervised_mlp_20261123')
fit=pd.DataFrame(json.loads((root/'source/FIT_CURVE.json').read_text()))
select=pd.DataFrame(json.loads((root/'source/SELECT_CURVE.json').read_text()))
seal=json.loads((root/'source/ALL_SELECTION.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for lr,color in [(1e-4,'#2166ac'),(1e-3,'#b2182b')]:
    a=fit[fit.lr==lr];b=select[select.lr==lr]
    axes[0].plot(a.step,a.loss,color=color,alpha=.18,linewidth=.6)
    axes[0].plot(a.step,a.loss.rolling(40,min_periods=1).mean(),color=color,label=f'lr {lr:g}, 40-step mean')
    axes[1].plot(b.step,b.risk,color=color,label=f'lr {lr:g}',linewidth=1.4)
for ax in axes:
    ax.axvline(240,color='gray',linestyle=':',linewidth=1)
    ax.set(xlabel='Source update',ylabel='Class-balanced BCE')
    ax.legend(frameon=False)
axes[0].set_title('FIT loss on the repeated fixed exposure plan')
axes[1].set_title('SOURCE SELECT loss on fixed event strata')
for name,marker in [('formal','o'),('aligned240','x')]:
    r=seal[name]
    axes[1].scatter([r['step']],[r['risk']],marker=marker,s=65,color='black',zorder=5)
fig.savefig(root/'source_curves.pdf')
fig.savefig(root/'source_curves.png',dpi=180)
fit.to_csv(root/'source_fit.csv',index=False)
select.to_csv(root/'source_select.csv',index=False)
