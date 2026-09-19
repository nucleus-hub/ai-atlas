"""Figures for Note 06 -- Regularization in Neural Networks."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title


def _style(ax):
    ax.grid(alpha=.18,color=LINE)
    for s in ax.spines.values(): s.set_edgecolor(LINE)

@figure("overfitting_train_validation")
def overfitting_train_validation():
    fig,ax=plt.subplots(figsize=(10.5,4.8)); e=np.arange(1,51)
    train=.68*np.exp(-e/14)+.06
    val=.68*np.exp(-e/13)+.13+np.maximum(0,e-24)*.010
    ax.plot(e,train,color=BLUE,lw=2.7,label="training loss")
    ax.plot(e,val,color=WARN,lw=2.7,label="validation loss")
    ax.axvline(24,color=GOOD,ls="--",lw=1.5);ax.scatter([24],[val[23]],s=80,color=GOOD,zorder=5,edgecolor=PAPER,lw=1.5)
    ax.annotate("best validation point\nstop here",xy=(24,val[23]),xytext=(31,.57),fontsize=9,color=GOOD,fontweight="bold",arrowprops=dict(arrowstyle="->",color=GOOD))
    ax.text(39,.35,"training keeps improving\nwhile validation worsens",ha="center",fontsize=9,color=WARN,fontweight="bold")
    ax.set_xlabel("epoch",color=INK);ax.set_ylabel("loss",color=INK);ax.legend();_style(ax);ax.set_title("Overfitting: training loss falls while validation loss turns upward",color=NAVY,fontweight="bold",fontsize=12.5,pad=10);fig.tight_layout();return fig

@figure("dropout_training_inference")
def dropout_training_inference():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,title,drop in [(axes[0],"TRAINING — Dropout(0.3)",True),(axes[1],"INFERENCE — no dropout",False)]:
        blank_axes(ax,xlim=(0,10),ylim=(0,6));
        for s in ax.spines.values():s.set_visible(False)
        left=[(2,1.2+i*1.2) for i in range(4)]; right=[(7,1.2+i*1.2) for i in range(4)]
        for p in left:
            for q in right: ax.plot([p[0],q[0]],[p[1],q[1]],color=LINE,lw=.8,zorder=1)
        dead={1,3} if drop else set()
        for i,p in enumerate(left+right):
            is_dead=(i%4 in dead and i>=4)
            face=GREY if is_dead else (MIST if i<4 else SKY)
            ax.add_patch(Circle(p,.29,facecolor=face,edgecolor=NAVY,lw=1.1,zorder=3))
            if is_dead: ax.text(*p,"×",ha="center",va="center",fontsize=16,color=PAPER,fontweight="bold",zorder=4)
        ax.text(5,.42,"random neurons removed\nfor this update" if drop else "all neurons active\nfor prediction",ha="center",fontsize=9,color=WARN if drop else GOOD,fontweight="bold")
        panel_title(ax,title,"different view on every training update" if drop else "full network available")
    fig.tight_layout();return fig

@figure("batch_normalization")
def batch_normalization():
    fig,axes=plt.subplots(1,2,figsize=(12,4.4));rng=np.random.default_rng(4)
    before=rng.normal(5,2.0,600);after=(before-before.mean())/(before.std())
    for ax,data,title,colour,mean,std in [(axes[0],before,"Before Batch Normalization",WARN,before.mean(),before.std()),(axes[1],after,"After Batch Normalization",GOOD,after.mean(),after.std())]:
        ax.hist(data,bins=30,color=colour,alpha=.75,edgecolor=PAPER)
        ax.axvline(mean,color=NAVY,lw=2,ls="--",label=f"mean = {mean:.2f}")
        ax.text(.5,.08,f"standard deviation = {std:.2f}",transform=ax.transAxes,ha="center",fontsize=9,color=INK,fontweight="bold")
        ax.set_xlabel("activation value",color=INK);ax.set_ylabel("count",color=INK);ax.legend();_style(ax);panel_title(ax,title,"stabilise what the next layer receives")
    fig.tight_layout();return fig

@figure("l2_weight_decay")
def l2_weight_decay():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));w=np.linspace(-3,3,300)
    axes[0].plot(w,w*w,color=BLUE,lw=2.8);axes[0].fill_between(w,0,w*w,color=MIST,alpha=.35);axes[0].set_xlabel("weight w",color=INK);axes[0].set_ylabel("L2 penalty: w²",color=INK);_style(axes[0]);panel_title(axes[0],"L2 penalises large weights","small weights cost little; large weights cost a lot")
    ax=axes[1];x=np.linspace(-2,2,200);ax.plot(x,np.sin(2*x),color=GREY,lw=2,label="underlying pattern");ax.plot(x,np.sin(2*x)+.55*np.sin(12*x),color=WARN,lw=2,label="large weights: wiggly fit");ax.plot(x,.88*np.sin(2*x),color=GOOD,lw=2.5,label="decayed weights: smoother fit");ax.legend(fontsize=8);ax.set_xlabel("input",color=INK);ax.set_ylabel("prediction",color=INK);_style(ax);panel_title(ax,"Weight decay favours simpler behaviour","less sensitivity to noisy quirks")
    fig.tight_layout();return fig
