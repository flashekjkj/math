"""Render actual Trainer logs; no smoothing, interpolation or invented validation points."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/assets';OUT.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#334155','text.color':'#0f172a','axes.edgecolor':'#cbd5e1','savefig.facecolor':'white','axes.unicode_minus':False})
runs=['local-sft-v1','analysis-sft-v1','multiturn-sft-v1','analysis-dpo-v1'];records={}
frozen=json.loads((OUT/'training-history.json').read_text(encoding='utf-8')) if (OUT/'training-history.json').exists() else {}
for name in runs:
    src=ROOT/f'runs/{name}/trainer_state.json'
    if not src.exists():
        records[name]=frozen[name]
        continue
    state=json.loads(src.read_text(encoding='utf-8'));history=state['log_history'];ev=ROOT/f'runs/{name}/eval_results.json'
    records[name]={'source':f'runs/{name}/trainer_state.json','sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'history':history,'final_eval':json.loads(ev.read_text(encoding='utf-8')) if ev.exists() else None,'eval_source':str(ev.relative_to(ROOT)) if ev.exists() else None,'eval_sha256':hashlib.sha256(ev.read_bytes()).hexdigest() if ev.exists() else None}
(OUT/'training-history.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
plot_runs=['local-sft-v1','analysis-sft-v1','analysis-dpo-v1']
fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
labels=['单轮 SFT','教师分析 SFT','DPO']
for ax,name,label in zip(axes.flat,plot_runs,labels):
    h=records[name]['history'];tr=[r for r in h if 'loss' in r];ev=[r for r in h if 'eval_loss' in r]
    ax.plot([r['step'] for r in tr],[r['loss'] for r in tr],color='#2563eb',lw=2,marker='o',markersize=3.5,label=label+' 训练 loss')
    if name=='multiturn-sft-v1':ev=[{'step':64,'eval_loss':records[name]['final_eval']['eval_loss']}]
    if ev:ax.plot([r['step'] for r in ev],[r['eval_loss'] for r in ev],color='#ea580c',lw=1.5,marker='D',markersize=5,label=label+' 验证 loss'+('（最终点）' if len(ev)==1 else ''))
    ax.set(xlabel='优化器更新步数',ylabel='Loss');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left' if name=='multiturn-sft-v1' else 'upper right',fontsize=9)
    if name=='analysis-sft-v1':ax.text(.02,.05,'该轮未记录匹配的验证 loss',ha='left',transform=ax.transAxes,color='#64748b',fontsize=9)
fig.savefig(OUT/'training-loss.png',dpi=160);plt.close(fig)
h=records['analysis-dpo-v1']['history'];tr=[r for r in h if 'rewards/margins' in r];ev=[r for r in h if 'eval_rewards/margins' in r]
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,key,ekey,title,scale in [(axes[0],'rewards/margins','eval_rewards/margins','偏好隐式奖励间隔',1),(axes[1],'rewards/accuracies','eval_rewards/accuracies','偏好隐式奖励排序命中率',100)]:
    ax.plot([r['step'] for r in tr],[r[key]*scale for r in tr],color='#0f766e',lw=2,marker='o',label='训练日志区间')
    ax.scatter([r['step'] for r in ev],[r[ekey]*scale for r in ev],color='#ea580c',s=70,marker='D',label='最终偏好验证（30 对）',zorder=3)
    ax.set(xlabel='优化器更新步数',ylabel='隐式奖励间隔' if scale==1 else '排序命中率（%）');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    if scale==100:ax.set_ylim(0,105)
fig.savefig(OUT/'dpo-diagnostics.png',dpi=160);plt.close(fig)
print('Saved training-loss.png, dpo-diagnostics.png and actual source data')
