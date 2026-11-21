"""Figures for Note 05 -- Optimizers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title


def _style(ax):
    ax.grid(alpha=0.18, color=LINE)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)


@figure("optimizer_role")
def optimizer_role():
    fig, ax = plt.subplots(figsize=(12, 3.8))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 4))
    for spine in ax.spines.values(): spine.set_visible(False)
    boxes = [(1.45, "Loss", "how wrong?", MIST, INK), (4.1, "Backprop", "gradients: ∂L/∂w", SKY, PAPER),
             (6.95, "Optimizer", "how to update?", BLUE, PAPER), (10.0, "New weights", "lower loss", MIST, INK)]
    for x, title, sub, face, text in boxes:
        ax.add_patch(FancyBboxPatch((x-1.05, 1.4), 2.1, 1.2, boxstyle="round,pad=0.05,rounding_size=0.12", facecolor=face, edgecolor=NAVY, linewidth=1.7))
        ax.text(x, 2.2, title, ha="center", fontsize=11, fontweight="bold", color=text)
        ax.text(x, 1.75, sub, ha="center", fontsize=8.3, color=text)
    for a,b,label in [(2.55,3.0,"differentiate"),(5.15,5.85,"choose step"),(8.05,8.95,"apply")]:
        ax.add_patch(FancyArrowPatch((a,2),(b,2),arrowstyle="-|>",mutation_scale=16,color=GOOD,linewidth=1.8))
        ax.text((a+b)/2,2.42,label,ha="center",fontsize=8,color=GOOD,fontweight="bold")
    ax.text(6,0.55,"Loss tells us how wrong. Backpropagation supplies the direction. The optimizer chooses the move.",ha="center",fontsize=9.5,color=NAVY,fontweight="bold")
    fig.tight_layout(); return fig


@figure("batch_gradient_variants")
def batch_gradient_variants():
    fig, axes=plt.subplots(1,3,figsize=(12.8,4.0))
    rng=np.random.default_rng(8)
    for ax,title,groups,subtitle in [
        (axes[0],"Batch Gradient Descent",[10],"all 10 samples → 1 update"),
        (axes[1],"Stochastic GD",[1]*10,"1 sample each → 10 updates"),
        (axes[2],"Mini-Batch GD",[3,3,3,1],"small groups → 4 updates"),
    ]:
        x=0
        for j,n in enumerate(groups):
            pts=np.arange(x,x+n)
            ax.scatter(pts,np.zeros(n),s=95,color=[MIST,SKY,BLUE,GOOD][j%4],edgecolor=NAVY,linewidth=.8,zorder=3)
            if n>1: ax.plot([pts[0],pts[-1]],[0,0],color=[MIST,SKY,BLUE,GOOD][j%4],linewidth=7,alpha=.55,zorder=2)
            ax.annotate("update",xy=(x+n-.5,-.1),xytext=(x+n-.5,-.65),ha="center",fontsize=7.5,color=WARN,fontweight="bold",arrowprops=dict(arrowstyle="->",color=WARN))
            x+=n
        ax.set_xlim(-.8,10); ax.set_ylim(-1.1,.8); ax.set_xticks([]); ax.set_yticks([]); _style(ax)
        panel_title(ax,title,subtitle)
    fig.tight_layout(); return fig


@figure("momentum_valley")
def momentum_valley():
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    x=np.linspace(-3,3,250); y=np.linspace(-2,2,250); X,Y=np.meshgrid(x,y); Z=X**2+18*Y**2
    for ax,title,momentum,color in [(axes[0],"Plain SGD — zigzags",False,WARN),(axes[1],"Momentum — builds speed",True,GOOD)]:
        ax.contour(X,Y,Z,levels=[1,4,9,16,25,36],colors=[LINE],linewidths=1)
        p=np.array([-2.4,1.25]); vel=np.zeros(2); pts=[p.copy()]
        for _ in range(11):
            g=np.array([2*p[0],36*p[1]])
            if momentum: vel=.82*vel-.035*g; p=p+vel
            else: p=p-.035*g
            pts.append(p.copy())
        pts=np.array(pts)
        ax.plot(pts[:,0],pts[:,1],"o-",color=color,linewidth=1.8,markersize=5,zorder=4)
        ax.scatter([0],[0],s=80,color=BLUE,edgecolor=PAPER,linewidth=1.3,zorder=5)
        ax.text(0,-.35,"minimum",ha="center",fontsize=8,color=NAVY,fontweight="bold")
        ax.set_xlim(-3,3);ax.set_ylim(-2,2);ax.set_xticks([]);ax.set_yticks([]);_style(ax)
        panel_title(ax,title,"same start, same learning rate")
    fig.tight_layout(); return fig


@figure("adagrad_vs_rmsprop_memory")
def adagrad_vs_rmsprop_memory():
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    t=np.arange(1,101); g=np.where(t<40,4.0,0.7)
    ada=np.cumsum(g**2); rms=np.zeros_like(g,dtype=float)
    for i,v in enumerate(g): rms[i]=(.9*(rms[i-1] if i else 0)+.1*v*v)
    for ax,series,title,sub,color in [(axes[0],ada,"AdaGrad remembers FOREVER","accumulator only grows",WARN),(axes[1],rms,"RMSprop forgets gradually","moving average follows recent gradients",GOOD)]:
        ax.plot(t,series,color=color,linewidth=2.6)
        ax.axvline(40,color=GREY,linestyle=":",linewidth=1.2)
        ax.text(42,max(series)*.82,"gradient scale\nchanges here",fontsize=8,color=GREY)
        ax.set_xlabel("update",fontsize=9,color=INK);ax.set_ylabel("stored squared-gradient scale",fontsize=9,color=INK);_style(ax);panel_title(ax,title,sub)
    fig.tight_layout();return fig


@figure("adam_two_moments")
def adam_two_moments():
    fig,ax=plt.subplots(figsize=(12,4.6));blank_axes(ax,xlim=(0,12),ylim=(0,5.1))
    for spine in ax.spines.values(): spine.set_visible(False)
    boxes=[(2,3.4,"gradient g","current slope",MIST,INK),(5.0,3.4,"first moment m","smoothed direction",SKY,PAPER),(5.0,1.35,"second moment v","smoothed magnitude",SKY,PAPER),(9,2.4,"Adam update","direction / size",BLUE,PAPER)]
    for x,y,title,sub,face,text in boxes:
        ax.add_patch(FancyBboxPatch((x-1.15,y-.52),2.3,1.04,boxstyle="round,pad=.05,rounding_size=.12",facecolor=face,edgecolor=NAVY,linewidth=1.6))
        ax.text(x,y+.12,title,ha="center",fontsize=10,fontweight="bold",color=text);ax.text(x,y-.23,sub,ha="center",fontsize=8,color=text)
    for a,b in [((3.2,3.4),(3.8,3.4)),((3.2,3.25),(3.8,1.55)),((6.2,3.25),(7.8,2.65)),((6.2,1.55),(7.8,2.15))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,color=GOOD,linewidth=1.7))
    ax.text(5,4.42,"Momentum",ha="center",fontsize=9,color=GOOD,fontweight="bold");ax.text(5,.5,"RMSprop-like adaptation",ha="center",fontsize=9,color=GOOD,fontweight="bold")
    ax.text(9,1.2,"Adam = direction memory\n+ per-parameter step sizing",ha="center",fontsize=9,color=NAVY,fontweight="bold")
    ax.set_title("Adam combines two memories",fontsize=12.5,fontweight="bold",color=NAVY,pad=10);fig.tight_layout();return fig


@figure("adamw_weight_decay")
def adamw_weight_decay():
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,title,decoupled in [(axes[0],"Adam + L2 penalty",False),(axes[1],"AdamW — decoupled decay",True)]:
        blank_axes(ax,xlim=(0,10),ylim=(0,6));
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(1.2,4.65,"weight w",fontsize=11,fontweight="bold",color=NAVY)
        if not decoupled:
            ax.text(4.5,4.65,"task gradient\n+ L2 penalty",ha="center",fontsize=9,color=INK)
            ax.add_patch(FancyArrowPatch((2.6,4.5),(6.0,4.5),arrowstyle="-|>",mutation_scale=16,color=WARN,linewidth=1.9))
            ax.text(7.8,4.65,"Adam's\nadaptive scaling",ha="center",fontsize=9,color=INK)
            ax.add_patch(FancyArrowPatch((6.8,4.5),(9.2,4.5),arrowstyle="-|>",mutation_scale=16,color=WARN,linewidth=1.9))
            ax.text(5,1.65,"Penalty gets rescaled\ndifferently per parameter",ha="center",fontsize=10,color=WARN,fontweight="bold")
        else:
            ax.text(4.7,4.65,"task gradient",ha="center",fontsize=9,color=INK);ax.text(4.7,2.3,"weight shrinkage\n−ηλw",ha="center",fontsize=9,color=INK)
            ax.add_patch(FancyArrowPatch((2.6,4.5),(7.4,4.5),arrowstyle="-|>",mutation_scale=16,color=GOOD,linewidth=1.9))
            ax.add_patch(FancyArrowPatch((2.6,2.25),(7.4,2.25),arrowstyle="-|>",mutation_scale=16,color=GOOD,linewidth=1.9))
            ax.text(8.5,3.45,"new weight",ha="center",fontsize=10,color=NAVY,fontweight="bold")
            ax.text(5,1.0,"Learn the task and control\ncomplexity as separate jobs",ha="center",fontsize=10,color=GOOD,fontweight="bold")
        panel_title(ax,title,"weight decay behaviour")
    fig.tight_layout();return fig
