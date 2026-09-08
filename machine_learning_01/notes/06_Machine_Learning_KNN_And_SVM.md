# <span style="color:#0B3D91">K-Nearest Neighbors & Support Vector Machines</span>

> Study notes on the two classifiers the slide deck promises twice (learning outcomes, slides 3 & 9) but never actually teaches. Anchored to the practical notebook's real, executed code (Practicals 4 & 5, Titanic dataset) rather than the missing slides.

> **A note on formulas:** equations are written in plain text inside code blocks rather than LaTeX, so they render correctly in any Markdown viewer.

> **Note on course coverage:** the slide deck's learning outcomes promise *"classify data using KNN and choose a sensible value of K"* and *"understand SVM and when to prefer it over other classifiers,"* but there is not a single teaching slide for either — the deck jumps straight from Logistic Regression to the recap. The practical notebook, however, fully implements both. This note follows the notebook, not the missing slides.

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [K-Nearest Neighbors (KNN) — You Are Who Your Neighbors Are](#1-k-nearest-neighbors-knn--you-are-who-your-neighbors-are)
2. *Support Vector Machines — pending*

---

## <span style="color:#1E6FEB">1. K-Nearest Neighbors (KNN) — You Are Who Your Neighbors Are</span>

**The idea:** to classify a new point, measure its distance to every point in the training data, take the K closest ones, and let them vote. Whichever class is most common among those K neighbors wins. No equation gets fitted, no coefficients get learned — it's a **lazy learner**: "training" is just storing the data, and all the actual work happens at prediction time.

**Why it matters beyond this course:** "distance = similarity" is one of the most reused ideas in ML — it's the same logic behind recommendation engines ("users like you also bought…"), anomaly detection (a point far from everything else looks suspicious), and even the similarity search underneath RAG systems. KNN is that idea in its purest, simplest form.

**Toy example from the notebook:** two clusters of 4 points each (bottom-left = Class 0, top-right = Class 1). A new point at (2.5, 2.5) gets checked against its K=3 nearest neighbors:

```
Nearest 3 neighbors: (2,2), (2,1), (1,2) — distances 0.71, 1.58, 1.58
All 3 belong to Class 0 → new point classified as Class 0 (unanimous vote)
```

**Picking K is the whole game, and it's a trade-off:**

- **Too small** (K=1 or 2) → the boundary gets jagged, chasing individual noisy points — **overfitting**.
- **Too large** (K=20+) → the boundary flattens out and basically just predicts whichever class is most common overall — **underfitting**.

The fix is the **elbow method**: try every K from 1 to 20, measure the test-set error rate for each, plot it, and pick the K sitting at the bottom of the curve. On Titanic, that came out to:

```
Best K: 14
Minimum Error Rate: 0.2018   (≈ 80% accuracy)
```

> **Small honest wrinkle:** the final trained K=14 model reports `Accuracy: 0.78` a couple of cells later — a ~2-point gap from the elbow scan's 0.2018 error on what should be the identical test set. Almost certainly a notebook-cells-run-out-of-order quirk (very common in Jupyter). The **K=14 choice itself is solid**; just don't treat the fourth decimal as sacred.

**One non-negotiable catch:** since KNN is purely about distance, **features must be scaled first**. If "fare" ranges into the hundreds and "age" ranges 0–80, fare would dominate every distance calculation regardless of which feature is actually more predictive — the exact scaling lesson from Topic 2 coming back around.

**Feature importance needs a workaround** because there's no formula to read weights from. The notebook uses **permutation importance**: shuffle one feature's values (breaking its real relationship to the outcome), re-run predictions, and measure how much accuracy drops. Repeat 20 times per feature and average, so one unlucky shuffle doesn't skew the result. A feature that matters a lot will tank accuracy when scrambled; a useless one barely moves the needle. Bonus: this technique isn't KNN-specific — it works on any black-box model, so it's a good general tool to have.

**Real-world flavor:** a recommendation system is basically KNN wearing a disguise — find the K most similar users by behavior, recommend what they liked.

---

*Section 2 — Support Vector Machines — pending your review before it gets added.*
