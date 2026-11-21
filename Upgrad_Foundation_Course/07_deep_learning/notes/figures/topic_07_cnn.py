"""Figures for Note 07 -- Convolutional Neural Networks."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title

@figure("cnn_convolution_operation")
def cnn_convolution_operation():
    image=np.array([[1,2,0,1,3],[0,1,2,3,1],[2,0,1,0,2],[1,3,0,2,1],[0,1,2,1,0]])
    kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    out=np.array([[(image[i:i+3,j:j+3]*kernel).sum() for j in range(3)] for i in range(3)])
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.3))
    for ax,arr,title,cmap in [(axes[0],image,"Input image  (5×5)","Blues"),(axes[1],kernel,"Filter / kernel  (3×3)","RdBu"),(axes[2],out,"Feature map  (3×3)","Purples")]:
        ax.imshow(arr,cmap=cmap)
        for (i,j),v in np.ndenumerate(arr):ax.text(j,i,str(v),ha="center",va="center",fontsize=10,fontweight="bold",color="white" if abs(v)>1 else INK)
        ax.set_xticks([]);ax.set_yticks([]);panel_title(ax,title,"stride = 1, padding = 0")
    axes[0].add_patch(Rectangle((-.5,-.5),3,3,fill=False,ec=WARN,lw=2.5));axes[0].text(1,4.85,"first 3×3 patch",ha="center",fontsize=8.5,color=WARN,fontweight="bold")
    fig.suptitle("Convolution: slide a filter, multiply matching cells, then add",fontsize=11.5,fontweight="bold",color=NAVY,y=.02);fig.tight_layout(rect=(0,.05,1,1));return fig

@figure("cnn_stride_padding")
def cnn_stride_padding():
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.9))
    for ax,title,n,k,p,s,colour in [(axes[0],"No padding",5,3,0,1,WARN),(axes[1],"Padding = 1",5,3,1,1,GOOD),(axes[2],"Stride = 2",5,3,0,2,BLUE)]:
        size=n+2*p; ax.set_xlim(0,size);ax.set_ylim(0,size);ax.invert_yaxis();ax.set_aspect("equal")
        for i in range(size):
            for j in range(size):
                face=MIST if p and (i in (0,size-1) or j in (0,size-1)) else PAPER
                ax.add_patch(Rectangle((j,i),1,1,facecolor=face,edgecolor=LINE))
        ax.add_patch(Rectangle((p,p),k,k,fill=False,ec=colour,lw=2.5))
        out=(n-k+2*p)//s+1
        ax.set_xticks([]);ax.set_yticks([]);panel_title(ax,title,f"output: {out}×{out}")
    fig.suptitle("Output size = floor((N − K + 2P) / S) + 1",fontsize=11.5,fontweight="bold",color=NAVY,y=.02);fig.tight_layout(rect=(0,.06,1,1));return fig

@figure("cnn_pooling")
def cnn_pooling():
    x=np.array([[1,3,2,1],[4,6,5,2],[0,2,8,3],[1,4,7,2]]);out=np.array([[6,5],[4,8]])
    fig,axes=plt.subplots(1,2,figsize=(8.5,4))
    for ax,arr,title in [(axes[0],x,"Feature map  (4×4)"),(axes[1],out,"Max-pooled output  (2×2)")]:
        ax.imshow(arr,cmap="Blues")
        for (i,j),v in np.ndenumerate(arr):ax.text(j,i,str(v),ha="center",va="center",fontweight="bold",color=PAPER if v>4 else INK)
        ax.set_xticks([]);ax.set_yticks([]);panel_title(ax,title,"2×2 max pooling" if arr.shape==(4,4) else "one maximum per patch")
    for i in (0,2):
        for j in (0,2):axes[0].add_patch(Rectangle((j-.5,i-.5),2,2,fill=False,ec=WARN,lw=2))
    fig.suptitle("Pooling shrinks maps while retaining the strongest signal",fontsize=11,color=NAVY,fontweight="bold",y=.02);fig.tight_layout(rect=(0,.06,1,1));return fig

@figure("cnn_architecture_stack")
def cnn_architecture_stack():
    fig,ax=plt.subplots(figsize=(13,4.1));blank_axes(ax,xlim=(0,14),ylim=(0,5));
    for s in ax.spines.values():s.set_visible(False)
    stages=[(1.1,"Image","28×28×1",MIST),(3.3,"Conv + ReLU","26×26×32",SKY),(5.7,"MaxPool","13×13×32",MIST),(8.1,"Conv + ReLU","11×11×64",SKY),(10.4,"MaxPool","5×5×64",MIST),(12.7,"Dense +\nSoftmax","10 classes",BLUE)]
    for i,(x,title,shape,col) in enumerate(stages):
        ax.add_patch(Rectangle((x-.72,1.55),1.44,2.0,facecolor=col,edgecolor=NAVY,lw=1.5))
        ax.text(x,2.8,title,ha="center",va="center",fontsize=9,fontweight="bold",color=PAPER if col==BLUE else NAVY)
        ax.text(x,1.95,shape,ha="center",fontsize=7.8,color=PAPER if col==BLUE else INK)
        if i:ax.add_patch(FancyArrowPatch((stages[i-1][0]+.76,2.55),(x-.76,2.55),arrowstyle="-|>",mutation_scale=14,color=GOOD,lw=1.6))
    ax.text(7,.68,"early layers: edges  →  middle: textures/shapes  →  deep layers: objects  →  output: class probabilities",ha="center",fontsize=9.3,color=NAVY,fontweight="bold")
    ax.set_title("The CNN architecture stack",fontsize=12.5,color=NAVY,fontweight="bold",pad=10);fig.tight_layout();return fig
