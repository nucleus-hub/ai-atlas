"""Figures for Note 09 -- architecture choice."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from .core import BLUE,GOOD,GREY,INK,LINE,MIST,NAVY,PAPER,SKY,WARN,blank_axes,figure

@figure("architecture_selection_flow")
def architecture_selection_flow():
 fig,ax=plt.subplots(figsize=(12.5,6.0));blank_axes(ax,xlim=(0,13),ylim=(0,8));
 for s in ax.spines.values():s.set_visible(False)
 boxes=[(6.5,7,"What structure\ndoes the data have?",MIST),(2,4.6,"Structured / tabular\ncustomer rows, numbers",SKY),(6.5,4.6,"Spatial grid\nimages, maps",SKY),(11,4.6,"Sequence / time\ntext, audio, time series",SKY),(2,1.8,"ANN\nDense layers",BLUE),(6.5,1.8,"CNN\nFilters + pooling",BLUE),(11,1.8,"RNN / LSTM\nMemory through time",BLUE)]
 for x,y,t,c in boxes:
  ax.add_patch(FancyBboxPatch((x-1.35,y-.58),2.7,1.16,boxstyle="round,pad=.05,rounding_size=.12",facecolor=c,edgecolor=NAVY,lw=1.6));ax.text(x,y,t,ha="center",va="center",fontsize=9.5,fontweight="bold",color=PAPER if c==BLUE else NAVY)
 for a,b in [((5.5,6.4),(2.5,5.2)),((6.5,6.4),(6.5,5.25)),((7.5,6.4),(10.5,5.2)),((2,4),(2,2.45)),((6.5,4),(6.5,2.45)),((11,4),(11,2.45))]: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,color=GOOD,lw=1.7))
 ax.text(6.5,.45,"Choose from the DATA'S structure first — not from whatever model is fashionable this week.",ha="center",fontsize=9.5,color=NAVY,fontweight="bold");ax.set_title("Choosing a neural-network architecture",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig

@figure("pretrained_model_workflow")
def pretrained_model_workflow():
 fig,ax=plt.subplots(figsize=(12.8,3.8));blank_axes(ax,xlim=(0,14),ylim=(0,4));
 for s in ax.spines.values():s.set_visible(False)
 stages=[(1.4,"Large public\ndataset",MIST),(4.2,"Pretrained\nmodel",SKY),(7,"Replace / add\nsmall output head",MIST),(9.8,"Fine-tune on\nyour data",SKY),(12.5,"Deploy\npredictions",BLUE)]
 for i,(x,t,c) in enumerate(stages):
  ax.add_patch(FancyBboxPatch((x-.95,1.45),1.9,1.2,boxstyle="round,pad=.05,rounding_size=.12",facecolor=c,edgecolor=NAVY,lw=1.5));ax.text(x,2.05,t,ha="center",va="center",fontsize=9,fontweight="bold",color=PAPER if c==BLUE else NAVY)
  if i:ax.add_patch(FancyArrowPatch((stages[i-1][0]+1,2.05),(x-1,2.05),arrowstyle="-|>",mutation_scale=14,color=GOOD,lw=1.6))
 ax.text(7,.6,"Reuse broadly learned features; train only what your task needs.",ha="center",fontsize=10,color=NAVY,fontweight="bold");ax.set_title("Pretrained model workflow",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig
