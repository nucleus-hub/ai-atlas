"""Figures for Note 08 -- RNNs and LSTMs."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from .core import BLUE,GOOD,GREY,INK,LINE,MIST,NAVY,PAPER,SKY,WARN,blank_axes,figure,panel_title

@figure("rnn_unrolled_time")
def rnn_unrolled_time():
 fig,ax=plt.subplots(figsize=(12.5,4));blank_axes(ax,xlim=(0,13),ylim=(0,5));
 for s in ax.spines.values():s.set_visible(False)
 xs=[1.5,4.2,6.9,9.6,12.3]
 for i,x in enumerate(xs):
  ax.add_patch(FancyBboxPatch((x-.62,2.1),1.24,1.05,boxstyle="round,pad=.04,rounding_size=.1",facecolor=SKY,edgecolor=NAVY,lw=1.4));ax.text(x,2.63,"RNN\ncell",ha="center",va="center",fontsize=9,fontweight="bold",color=PAPER)
  ax.text(x,1.45,f"x{i+1}",ha="center",fontsize=10,color=INK,fontweight="bold");ax.add_patch(FancyArrowPatch((x,1.75),(x,2.05),arrowstyle="-|>",mutation_scale=13,color=GOOD,lw=1.5));ax.text(x,3.65,f"h{i+1}",ha="center",fontsize=10,color=BLUE,fontweight="bold")
  ax.add_patch(FancyArrowPatch((x,3.15),(x,3.45),arrowstyle="-|>",mutation_scale=13,color=BLUE,lw=1.5))
  if i: ax.add_patch(FancyArrowPatch((xs[i-1]+.67,2.63),(x-.67,2.63),arrowstyle="-|>",mutation_scale=14,color=WARN,lw=1.8))
 ax.text(6.9,.55,"Same cell and SAME weights reused at every time step",ha="center",fontsize=10,color=NAVY,fontweight="bold");ax.set_title("An RNN unrolled through time",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig

@figure("rnn_vanishing_gradient")
def rnn_vanishing_gradient():
 fig,ax=plt.subplots(figsize=(11,4.5));steps=np.arange(0,11);grad=.6**steps
 ax.semilogy(steps,grad,"o-",color=WARN,lw=2.6,markersize=7);ax.axhline(.01,color=GREY,ls="--",lw=1.2);ax.text(6.3,.013,"almost no learning signal",fontsize=9,color=GREY,fontweight="bold")
 ax.annotate("gradient multiplied\nat every time step",xy=(2,grad[2]),xytext=(3.3,.55),fontsize=9,color=NAVY,fontweight="bold",arrowprops=dict(arrowstyle="->",color=NAVY))
 ax.annotate(f"0.6^10 = {grad[-1]:.4f}",xy=(10,grad[-1]),xytext=(7.4,.004),fontsize=9,color=WARN,fontweight="bold",arrowprops=dict(arrowstyle="->",color=WARN))
 ax.set_xlabel("steps backward through time",color=INK);ax.set_ylabel("gradient magnitude (log scale)",color=INK);ax.grid(alpha=.2,color=LINE,which="both");[s.set_edgecolor(LINE) for s in ax.spines.values()];ax.set_title("Vanishing gradients: earlier timesteps stop learning",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig

@figure("lstm_cell_gates")
def lstm_cell_gates():
 fig,ax=plt.subplots(figsize=(12.5,4.6));blank_axes(ax,xlim=(0,13),ylim=(0,6));
 for s in ax.spines.values():s.set_visible(False)
 ax.add_patch(FancyArrowPatch((.8,4.6),(12.2,4.6),arrowstyle="-|>",mutation_scale=17,color=BLUE,lw=3));ax.text(6.5,5.12,"cell state c — long-term memory conveyor belt",ha="center",fontsize=10,color=BLUE,fontweight="bold")
 gates=[(3,"Forget gate","what to erase",WARN),(6.5,"Input gate","what to add",GOOD),(10,"Output gate","what to expose",SKY)]
 for x,t,sub,col in gates:
  ax.add_patch(FancyBboxPatch((x-1.05,1.8),2.1,1.45,boxstyle="round,pad=.05,rounding_size=.12",facecolor=col,edgecolor=NAVY,lw=1.5));ax.text(x,2.72,t,ha="center",fontsize=9.5,fontweight="bold",color=PAPER if col in (WARN,GOOD) else NAVY);ax.text(x,2.23,sub,ha="center",fontsize=8,color=PAPER if col in (WARN,GOOD) else INK)
  ax.add_patch(FancyArrowPatch((x,3.3),(x,4.45),arrowstyle="-|>",mutation_scale=14,color=col,lw=1.8))
 ax.text(6.5,.72,"At each time step: forget irrelevant information → add useful information → reveal needed output",ha="center",fontsize=9.5,color=NAVY,fontweight="bold");ax.set_title("Inside an LSTM cell",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig
