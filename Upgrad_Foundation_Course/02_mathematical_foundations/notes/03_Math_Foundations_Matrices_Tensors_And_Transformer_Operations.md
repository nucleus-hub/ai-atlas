# <span style="color:#0B3D91">Mathematics Foundations: Matrices, Tensors &amp; Transformer Operations</span>

> Study notes on how machines store and transform data in bulk — and how those simple grid
> operations build all the way up to how a transformer "thinks":
> **Matrices** (grids of numbers) → **Shapes** (rows × columns, and why they must line up) →
> **Addition** → **Scaling** → **Multiplication** → **Transpose** → **Identity** → **Inverse** →
> **Determinant** → **Softmax** → **Attention**.
> An intuition-first foundation for reading model cards, debugging shape errors, sizing GPU
> memory, and understanding what a transformer's "attention" actually computes.
>
> **Note on formulas:** these notes use plain-text formulas (real symbols in code blocks)
> instead of LaTeX, so every equation renders correctly in any Markdown viewer.
>
> **Status:** complete — all planned topics covered (Matrices → Attention), plus a bonus
> **Softmax** section (the function attention depends on).

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [Matrices](#1-matrices)
2. [Shapes](#2-shapes)
3. [Addition](#3-addition)
4. [Scaling](#4-scaling-scalar-multiplication)
5. [Multiplication](#5-multiplication-matrix-multiplication)
6. [Transpose](#6-transpose)
7. [Identity](#7-identity-the-identity-matrix)
8. [Inverse](#8-inverse-the-inverse-matrix)
9. [Determinant](#9-determinant)
10. [Softmax](#10-softmax-bonus) *(bonus — the function attention uses)*
11. [Attention](#11-attention)

---

## <span style="color:#1E6FEB">1. Matrices</span>

### 1.1 Theory
A **matrix** is just a **grid of numbers** — rows and columns, like a spreadsheet.

Think of the whole family of "number containers," each one adding a dimension:

| Rank | Name | Looks like | Example |
|------|------|-----------|---------|
| 0-D | Scalar | a single number | `7` |
| 1-D | Vector | a list | `[7, 2, 9]` |
| 2-D | **Matrix** | a grid | `[[7, 2], [9, 4]]` |
| 3-D+ | Tensor | a stack of grids | a batch of images |

Each step just adds **one more axis** — nothing new is introduced, the container just gets one dimension deeper. (Pro tip: in practice "tensor" often just means "the array object my framework uses," whatever its rank — a PyTorch scalar is still called a tensor.)

**Why it's useful:** a matrix lets you hold *lots* of data in one tidy object and transform all of it at once. In AI, a matrix might be a **batch of embeddings** (each row = one thing's meaning) or a **weight layer** in a neural network. Instead of looping over numbers one-by-one, you do one clean operation on the whole grid. That's why GPUs and ML frameworks live and breathe matrices.

### 1.2 Mathematical Formula (Notation)
A matrix is written as a grid of rows and columns:

```
      col1   col2
row1 [ a11    a12 ]
row2 [ a21    a22 ]
```

- **a11** → entry in **row 1, column 1**
- **a12** → row 1, column 2
- General entry: **aij** = the value at **row i, column j** (row first, then column — always)

A matrix's **shape** is written **rows × columns** (e.g. a `2 × 3` matrix has 2 rows and 3 columns).

### 1.3 Mathematical Example
Take this matrix:

```
A = [ 7   2 ]
    [ 9   4 ]
```

Step-by-step reading:
- **Step 1 — Shape?** 2 rows, 2 columns → it's a **2 × 2** matrix.
- **Step 2 — a11** = row 1, col 1 = **7**
- **Step 3 — a12** = row 1, col 2 = **2**
- **Step 4 — a21** = row 2, col 1 = **9**
- **Step 5 — a22** = row 2, col 2 = **4**

That's it — you can now point to any number by its address.

### 1.4 Real-World Example
Imagine a **customer database** where every customer is described by `[age, spend]`:

```
[ 25   200 ]   <- customer 1
[ 40   850 ]   <- customer 2
[ 33   500 ]   <- customer 3
   ^     ^
  age   spend
```

This is a **3 × 2 matrix**: 3 customers (rows) × 2 features (columns). In a recommendation system, each row is one customer's "profile vector," and stacking them into a matrix lets the model process all customers in a single sweep. In a transformer, this exact idea shows up as a **batch of word embeddings** fed through the network together.

### 1.5 Key Takeaway
> **A matrix is a grid of numbers (rows × columns), and you locate any value by its row-then-column address (aij).**
> It's the workhorse container of AI — one object holding a whole batch of data you can transform all at once.

---

## <span style="color:#1E6FEB">2. Shapes</span>

### 2.1 Theory
A **shape** answers one simple question about a matrix: **"how many rows by how many columns?"** — written as **rows × columns**.

Why care so much? Because in AI, **almost every practical error is a shape mismatch.** When you combine matrices (especially with multiplication), their dimensions have to *line up* — otherwise the math is undefined and your code explodes. Knowing shapes lets you read model cards, size GPU memory, and debug those cryptic `shapes cannot be multiplied` errors.

The golden rules for **matrix multiplication** shapes:
- **Inner must match** — for `(m × n) × (n × p)`, the two **n**'s must be identical. Each output entry is a sum over `n` paired terms; if the counts differ, there's nothing to pair the leftovers with.
- **Outer survive** — the result is `(m × p)`: as many rows as the first matrix, as many columns as the second. The `m` and `p` don't have to agree with anything.
- **Order matters** — `A × B` is usually *not* the same as `B × A`, and often `B × A` isn't even legal. Matrix multiplication is **not commutative**.

### 2.2 Mathematical Formula (The shape rule)

```
(m × n)  ×  (n × p)   =   (m × p)
     ^        ^
     └── these must match ──┘   (INNER)

  m ─────────────────── p       (OUTER: survive into result)
```

- **m** → rows of the first matrix
- **n** → columns of the first = rows of the second (the shared "inner" dimension)
- **p** → columns of the second matrix

### 2.3 Mathematical Example
Can we multiply these, and what's the output shape?

```
A shape = 2 × 3        B shape = 3 × 4
```

Step-by-step:
- **Step 1** — Write them side by side: `(2 × 3) × (3 × 4)`.
- **Step 2 — Check inner:** 3 and 3 → they **match**. Legal!
- **Step 3 — Take outer:** first's rows = **2**, second's cols = **4**.
- **Step 4 — Result shape = 2 × 4.**

Now flip the order — `B × A` = `(3 × 4) × (2 × 3)`:
- **Check inner:** 4 and 2 → they **don't match**. Illegal! (Order matters.)

### 2.4 Real-World Example
You feed a **batch of 32 word embeddings**, each of size **768**, into a weight layer:

```
Input:   32 × 768        Weights:  768 × 512
         (32 tokens,               (maps 768 features
          768 features each)        down to 512)

(32 × 768) × (768 × 512)  ->  32 × 512
```

The inner 768's match, so it works — and each token now has 512 features. If instead you saw the error:

```
shapes cannot be multiplied (32×768 and 512×768)
```

...the inner numbers are **768** and **512** — they don't match. Fix: **transpose** one of them so the 768's line up. This is *the* most common bug in deep learning.

### 2.5 Key Takeaway
> **A shape is just rows × columns. To multiply, the inner dimensions must match; the outer dimensions become the result's shape.**
> When code breaks in ML, look at the shapes first — it's almost always a mismatch.

---

## <span style="color:#1E6FEB">3. Addition</span>

### 3.1 Theory
Matrix **addition** is the most relaxing operation you'll ever learn: you just add **element by element, position by position.** Top-left plus top-left, top-right plus top-right, and so on.

There's exactly **one rule** — both matrices must have the **same shape**. Same rows, same columns. If they don't match, addition is simply undefined (there'd be leftover entries with no partner to add to). No clever tricks, no exceptions. Subtraction works identically — just subtract instead of add.

**Why it's useful:** adding matrices lets you *combine* two grids of information in one clean sweep — merge two batches of adjustments, accumulate changes, or blend two signals. In deep learning it shows up constantly (more on that in the real-world bit).

### 3.2 Mathematical Formula
For two matrices **A** and **B** of the **same shape**, the sum **C = A + B** is:

```
C[i][j] = A[i][j] + B[i][j]
```

- **A[i][j]** → the entry in row *i*, column *j* of A
- **B[i][j]** → the entry in the *same* position of B
- **C[i][j]** → their sum, sitting in that same position

In words: **every output cell = the two input cells at that exact address, added together.** The shape never changes — a `2 × 2` plus a `2 × 2` gives a `2 × 2`.

### 3.3 Mathematical Example
Add these two `2 × 2` matrices:

```
[ 1   2 ]   +   [ 5   6 ]
[ 3   4 ]       [ 7   8 ]
```

Go position by position:
- **Step 1 — Top-left:** `1 + 5 = 6`
- **Step 2 — Top-right:** `2 + 6 = 8`
- **Step 3 — Bottom-left:** `3 + 7 = 10`
- **Step 4 — Bottom-right:** `4 + 8 = 12`

Result:

```
[ 6    8 ]
[ 10   12 ]
```

That's the whole operation — no carrying, no ordering worries. (And `A + B = B + A` here, unlike multiplication — addition *is* commutative.)

### 3.4 Real-World Example
**Residual connections in a transformer.** Every modern deep network (GPT, BERT, etc.) uses a trick where a layer's **output is added back to its input**:

```
output = Layer(input) + input     <- a matrix addition
```

Both `Layer(input)` and `input` are matrices of the **same shape** (e.g. `32 × 768`), so they add element-by-element perfectly. This tiny addition is a huge deal: it gives the signal a "shortcut" path straight through the network, which is *the* reason very deep networks (dozens of layers) can train at all instead of collapsing. So the humble `+` you just did by hand is quietly holding up modern AI.

### 3.5 Key Takeaway
> **Matrix addition is element-by-element: add the numbers at each matching position. Both matrices must have the exact same shape, and the result keeps that shape.**
> It's how you combine two grids of data in one step — and it's the "residual connection" that makes deep networks trainable.

---

## <span style="color:#1E6FEB">4. Scaling (Scalar Multiplication)</span>

### 4.1 Theory
**Scaling** means multiplying an *entire* matrix by a **single number** (a "scalar"). That one number multiplies **every single entry** in the grid.

The key insight: **the shape never changes — only the *size* of the values does.** A `2 × 2` matrix stays a `2 × 2` matrix; its numbers just get bigger (scalar > 1), smaller (scalar between 0 and 1), or flipped in sign (negative scalar). Think of it like a volume knob for the whole matrix at once.

**Why it's useful:** scaling by one number is *everywhere* in AI. Any time you want to dial a whole grid of values up or down uniformly — control a step size, normalize magnitudes, soften or sharpen a signal — you multiply by a scalar. It's the simplest possible way to adjust an entire matrix in one move.

### 4.2 Mathematical Formula
For a scalar **k** and a matrix **A**, the scaled matrix **C = k · A** is:

```
C[i][j] = k * A[i][j]
```

- **k** → the scalar (one single number)
- **A[i][j]** → the entry in row *i*, column *j* of A
- **C[i][j]** → that entry, multiplied by *k*, in the same position

In words: **hit every cell with the same multiplier.** Shape in = shape out.

### 4.3 Mathematical Example
Scale this matrix by `0.5` (i.e. halve everything):

```
0.5 × [ 4    6 ]
      [ 8   10 ]
```

Multiply each entry by 0.5:
- **Step 1 — Top-left:** `0.5 × 4 = 2`
- **Step 2 — Top-right:** `0.5 × 6 = 3`
- **Step 3 — Bottom-left:** `0.5 × 8 = 4`
- **Step 4 — Bottom-right:** `0.5 × 10 = 5`

Result:

```
[ 2   3 ]
[ 4   5 ]
```

Same shape (`2 × 2`), just half the size. If we'd used `k = 2`, everything would double; if `k = -1`, every sign would flip.

### 4.4 Real-World Example
Scalar multiplication is quietly running the whole show in model training:

- **Learning rate** — a single scalar multiplies an entire **gradient matrix** to control how big a step the model takes when learning: `update = learning_rate × gradient`. Too big → training explodes; too small → it crawls. One number, whole matrix.
- **Temperature** — a scalar that *divides* a whole vector of logits before softmax, making a model's output more random (high temp) or more confident (low temp).
- **Attention scaling** — attention divides its scores by `√d` (a scalar) before the softmax to keep the numbers stable.

Same simple operation you just did by hand — one number, applied to every entry — tuning behaviour all over modern AI.

### 4.5 Key Takeaway
> **Scaling multiplies every entry of a matrix by one number (a scalar). The shape stays the same; only the magnitude of the values changes.**
> It's the "volume knob" of linear algebra — behind learning rates, temperature, and attention scaling.

---

## <span style="color:#1E6FEB">5. Multiplication (Matrix Multiplication)</span>

### 5.1 Theory
Matrix multiplication is **not** element-by-element (that's addition's job). Instead, it's a **"rows meet columns" dance**:

> Each entry of the result = one *row* of the first matrix combined with one *column* of the second, by multiplying paired numbers and adding them up.

That "multiply the pairs, then add" move is called a **dot product** (you met it in the Similarity notes!). So matrix multiplication is really just "do a dot product for every row-column pairing."

Remember the shape rules from Topic 2 — they matter here:
- **Inner must match:** `(m × n) × (n × p)` — the two `n`'s must be equal (each row and column need the same length to pair up).
- **Outer survive:** the result is `(m × p)`.
- **Order matters:** `A × B ≠ B × A` in general. Not commutative.

**Why it's useful:** matrix multiplication is the core operation of AI. Every neural network layer, every transformation of embeddings, and the entire attention mechanism are matrix multiplications. When people say "GPUs do trillions of operations," they mostly mean this. Master this one and the rest of deep learning stops looking like magic.

### 5.2 Mathematical Formula
For **A** of shape `(m × n)` and **B** of shape `(n × p)`, the product **C = A × B** has shape `(m × p)`, where:

```
C[i][j] = A[i][1]*B[1][j] + A[i][2]*B[2][j] + ... + A[i][n]*B[n][j]
```

- **C[i][j]** → the result entry at row *i*, column *j*
- **row *i* of A** → `[A[i][1], A[i][2], ..., A[i][n]]`
- **column *j* of B** → `[B[1][j], B[2][j], ..., B[n][j]]`
- The formula = **dot product** of row *i* (from A) with column *j* (from B): multiply matching entries, add them all up.

The recipe in plain words: **to fill cell (i, j), walk across row i of A and down column j of B, multiply each pair, and sum.**

### 5.3 Mathematical Example
Multiply these two `2 × 2` matrices:

```
A = [ 1   2 ]        B = [ 5   6 ]
    [ 3   4 ]            [ 7   8 ]
```

Shapes: `(2×2) × (2×2)` → inner 2's match → result is `2 × 2`. Now fill each cell with a row·column dot product:
- **Step 1 — C[1][1]** = (row 1 of A) · (col 1 of B) = `(1×5) + (2×7) = 5 + 14 = 19`
- **Step 2 — C[1][2]** = (row 1 of A) · (col 2 of B) = `(1×6) + (2×8) = 6 + 16 = 22`
- **Step 3 — C[2][1]** = (row 2 of A) · (col 1 of B) = `(3×5) + (4×7) = 15 + 28 = 43`
- **Step 4 — C[2][2]** = (row 2 of A) · (col 2 of B) = `(3×6) + (4×8) = 18 + 32 = 50`

Result:

```
C = [ 19   22 ]
    [ 43   50 ]
```

Quick sanity check that order matters: if you compute `B × A` instead, you get `[[23, 34], [31, 46]]` — totally different numbers. **A × B ≠ B × A.**

### 5.4 Real-World Example
**A neural network layer.** Say you have a batch of 32 word embeddings, each of size 768, and a weight matrix that maps 768 features down to 512:

```
Input:   32 × 768        Weights:  768 × 512

(32 × 768) × (768 × 512)  ->  32 × 512
```

The inner 768's match, so it's legal — and in one multiplication, all 32 tokens get transformed into their new 512-dimensional representations simultaneously. Every dot product = one neuron combining all 768 input features into one output number.

Stack a few of these multiplications with some non-linearities between them, and you've got a deep network. Attention itself is just a specific sequence of these matrix multiplications (Query × Key, then × Value). This single operation, run billions of times, is how models "think."

### 5.5 Key Takeaway
> **Matrix multiplication fills each result cell with a dot product of one row (from the first matrix) and one column (from the second): multiply the pairs, add them up.**
> The inner dimensions must match, the outer ones become the result's shape, and order matters (A×B ≠ B×A). It's the single most important operation in all of deep learning.

---

## <span style="color:#1E6FEB">6. Transpose</span>

### 6.1 Theory
**Transpose** is delightfully simple: you **flip a matrix over its diagonal**, swapping rows and columns. Row 1 becomes column 1, row 2 becomes column 2, and so on. That's the *entire* operation.

The immediate consequence: an **`m × n` matrix becomes `n × m`**. A "wide" matrix turns "tall," and vice versa. The numbers themselves don't change — they just move to new addresses (the entry at row *i*, col *j* lands at row *j*, col *i*).

**Why it's useful:** transpose is the go-to fix for **shape mismatches**. Remember from the Shapes topic that multiplication needs the inner dimensions to match? When they *don't*, transposing one matrix often flips its dimensions so they line up. It's also how you reinterpret data — turning "rows are people, columns are features" into "rows are features, columns are people" without touching a single value.

### 6.2 Mathematical Formula
The transpose of **A** is written **Aᵀ** (or `A^T`), and:

```
A^T[i][j] = A[j][i]
```

- **A[j][i]** → the entry at row *j*, column *i* in the **original** matrix
- **A^T[i][j]** → that same value, now sitting at row *i*, column *j* in the transpose
- **Shape rule:** if A is `m × n`, then Aᵀ is `n × m`

In plain words: **the row index and column index simply swap places.**

### 6.3 Mathematical Example
Transpose this `2 × 3` matrix:

```
A = [ 1   2   3 ]     shape 2 × 3
    [ 4   5   6 ]
```

Turn each **row** into a **column**:
- **Step 1 — Row 1 `[1, 2, 3]`** becomes **column 1** (goes down the left)
- **Step 2 — Row 2 `[4, 5, 6]`** becomes **column 2** (goes down the right)

Result:

```
A^T = [ 1   4 ]        shape 3 × 2
      [ 2   5 ]
      [ 3   6 ]
```

Notice the shape flipped from `2 × 3` to `3 × 2`, and every number just relocated — `A[2][3] = 6` is now `Aᵀ[3][2] = 6`.

### 6.4 Real-World Example
**Attention's `Q × Kᵀ`.** In a transformer, the **Query (Q)** and **Key (K)** matrices are *both* shaped `(tokens × dimensions)` — say `(32 × 768)`. You can't multiply them directly: `(32 × 768) × (32 × 768)` has inner dims `768` and `32`, which **don't match**.

The fix is to **transpose K**:

```
Q × K^T  =  (32 × 768) × (768 × 32)  ->  32 × 32
```

Now the inner 768's line up, and the result is a `32 × 32` grid — **every token dotted with every other token**, i.e. "how much should each word pay attention to each other word." That single transpose is what makes the whole attention mechanism computable. Not bad for "flip the rows and columns."

### 6.5 Key Takeaway
> **Transpose flips a matrix over its diagonal: rows become columns, so an `m × n` matrix becomes `n × m`. The values don't change — their addresses swap (Aᵀ[i][j] = A[j][i]).**
> It's the standard trick for lining up shapes so matrices can multiply — including the `Q × Kᵀ` at the heart of attention.

---

## <span style="color:#1E6FEB">7. Identity (The Identity Matrix)</span>

### 7.1 Theory
The **identity matrix** is matrix multiplication's version of the **number 1**. Just as `x × 1 = x` for ordinary numbers, `A × I = A` for matrices — multiply anything by the identity and **nothing changes.**

What does it look like? **Ones down the leading diagonal (top-left to bottom-right), zeros everywhere else.** It's always **square** (same number of rows and columns), and it's written **I** (or `Iₙ` for an `n × n` one).

**Why it's useful:** the identity is how we write down **"no transformation" / "do nothing"** in the language of matrices. That sounds boring, but it's foundational:
- It's the **definition of an inverse** — a matrix's inverse is whatever you multiply it by to get back to `I`.
- It's the "neutral starting point" for building up transformations.
- A **residual connection** (from the Addition topic) is essentially what a layer approximates when it's learned to *do nothing* — behaving like the identity.

### 7.2 Mathematical Formula
The `n × n` identity matrix has entries:

```
I[i][j] = 1   if i = j   (on the diagonal)
I[i][j] = 0   if i ≠ j   (everywhere else)
```

For example, the `3 × 3` identity:

```
I = [ 1   0   0 ]
    [ 0   1   0 ]
    [ 0   0   1 ]
```

The defining property (the whole reason it exists):

```
A × I = A       and       I × A = A
```

- **I** → the identity matrix (ones on the diagonal, zeros elsewhere)
- Multiplying by I leaves **A completely unchanged**, from either side

### 7.3 Mathematical Example
Multiply a matrix by the `2 × 2` identity and watch nothing happen:

```
[ 3   7 ]   ×   [ 1   0 ]
[ 2   5 ]       [ 0   1 ]
```

Fill each cell with the usual row·column dot product:
- **Step 1 — C[1][1]** = `(3×1) + (7×0) = 3 + 0 = 3`
- **Step 2 — C[1][2]** = `(3×0) + (7×1) = 0 + 7 = 7`  ← the row comes through untouched
- **Step 3 — C[2][1]** = `(2×1) + (5×0) = 2 + 0 = 2`
- **Step 4 — C[2][2]** = `(2×0) + (5×1) = 0 + 5 = 5`

Result:

```
[ 3   7 ]     <- exactly the original, unchanged
[ 2   5 ]
```

See how the `1`s pick out each value and the `0`s zero out everything else? That's the whole trick — the identity "selects" each entry back into place.

### 7.4 Real-World Example
The identity is the quiet backbone behind a few big ideas:
- **Defining the inverse** — "undoing" a transformation means finding the matrix `A⁻¹` such that `A × A⁻¹ = I`. Without the identity as the "back to start" target, the inverse couldn't even be defined.
- **Residual connections** — when a transformer layer decides the best thing to do is *nothing* (leave the input alone), it's effectively acting like the identity: `output = input`. This is part of *why* residuals help deep networks — a layer can safely "opt out" without harming the signal.
- **Initialization & no-op transforms** — starting a transformation from the identity means "begin from no change, then learn adjustments from there."

### 7.5 Key Takeaway
> **The identity matrix (`I`) is matrix multiplication's "1": ones on the diagonal, zeros elsewhere, always square. Multiplying by it changes nothing — `A × I = A`.**
> It's how "do nothing" is written in matrix form, and it's the anchor that defines the inverse and underpins residual connections.

---

## <span style="color:#1E6FEB">8. Inverse (The Inverse Matrix)</span>

### 8.1 Theory
The **inverse** of a matrix is its **"undo" button.**

Think of a matrix `A` as a machine that *does something* to your data — stretches it, rotates it, reshuffles it. The inverse, written **`A⁻¹`**, is the machine that **puts everything back exactly how it was.** Do `A`, then do `A⁻¹`, and it's like nothing happened.

It's the same idea as everyday "undo" pairs:
- Multiply by 5 → undo by dividing by 5.
- Put your shoes on → undo by taking them off.
- Apply `A` → undo by applying `A⁻¹`.

"Back to how it was" has a name from the last topic: the **identity** (`I`, the do-nothing matrix). So the inverse is simply *the thing that gets you back to `I`.*

Two simple facts worth knowing:
- **Some matrices have no undo button.** If a matrix throws information away (imagine squashing a 3-D object into a flat shadow — you can't rebuild the object from the shadow), there's no way to reverse it. Mathematicians call such a matrix **singular** (just means "non-reversible").
- **Only square matrices** (same number of rows and columns) can have an inverse.

**Why it's useful:** the inverse is the classic way to **solve for an unknown** — the matrix version of "divide both sides to get x by itself." It's the engine behind classical methods like linear regression.

### 8.2 Mathematical Formula
The one line that defines it:

```
A × A⁻¹ = I
```

- **A** → your original matrix (the "do something" machine)
- **A⁻¹** → its inverse (the "undo it" machine)
- **I** → the identity matrix (everything back to the start)

For a small **`2 × 2`** matrix, there's a plug-in recipe:

```
A = [ a   b ]        A⁻¹ =  1/(ad − bc) × [  d   −b ]
    [ c   d ]                              [ −c    a ]
```

The steps hiding in that formula are easy: **(1)** swap the two diagonal numbers, **(2)** flip the sign of the other two, **(3)** divide everything by `ad − bc`.

That `ad − bc` number is the key gatekeeper: **if it's 0, you're dividing by zero → no inverse exists** (the matrix is singular). We'll meet `ad − bc` properly in the next topic — it's the **determinant**.

### 8.3 Mathematical Example
Find the inverse of:

```
A = [ 4   7 ]
    [ 2   6 ]
```

- **Step 1 — Compute `ad − bc`:** `(4×6) − (7×2) = 24 − 14 = 10`. Not zero → an undo button exists.
- **Step 2 — Swap the diagonal** (the 4 and the 6): → `[[6, 7], [2, 4]]`
- **Step 3 — Flip the sign of the other two** (the 7 and the 2): → `[[6, −7], [−2, 4]]`
- **Step 4 — Divide every entry by 10:**

```
A⁻¹ = [ 0.6   −0.7 ]
      [ −0.2   0.4 ]
```

**Quick check — does `A × A⁻¹` really give the do-nothing matrix `I`?**
- Top-left: `(4×0.6) + (7×−0.2) = 2.4 − 1.4 = 1`
- Top-right: `(4×−0.7) + (7×0.4) = −2.8 + 2.8 = 0`
- Bottom-left: `(2×0.6) + (6×−0.2) = 1.2 − 1.2 = 0`
- Bottom-right: `(2×−0.7) + (6×0.4) = −1.4 + 2.4 = 1`

That's `[[1, 0], [0, 1]] = I`. The undo button works perfectly.

### 8.4 Real-World Example
**Solving for an unknown — the basis of regression.** Loads of problems look like `A x = b`: you know `A` and `b`, and you want to find `x`. Just like you'd divide to isolate `x` with normal numbers, here you multiply by the inverse:

```
A x = b   ->   x = A⁻¹ b
```

Fitting the best line through data (**linear regression**) is solved with exactly this move.

**The plot twist for AI:** modern neural networks *deliberately avoid* computing inverses, for two down-to-earth reasons:
1. **It's slow** — inverting a big matrix takes a huge amount of computation.
2. **It's fragile** — if a matrix is *nearly* non-reversible, its inverse becomes gigantic and jumpy, so a tiny change in the input causes a wild swing in the output. Not something you want.

So instead of one exact-but-risky inverse, deep learning **creeps toward the answer step-by-step** using **gradient descent** — take a small step, check, adjust, repeat. That's why your course spends its next part on **gradients** rather than on matrix inversion.

### 8.5 Key Takeaway
> **The inverse `A⁻¹` is a matrix's undo button — apply `A` then `A⁻¹` and you're back to the start (`A × A⁻¹ = I`). Matrices that throw away information (singular) have no inverse.**
> It's the classic tool for solving `A x = b` (like linear regression), but deep learning skips it — it's slow and unstable — and uses step-by-step gradient descent instead.

---

## <span style="color:#1E6FEB">9. Determinant</span>

### 9.1 Theory
The **determinant** is just **one number** that answers a single question about a matrix:

> **"Does this matrix squash things flat, or not?"**

Here's the friendly picture. Think of a matrix as something that *reshapes* a square of dough — it can stretch it bigger, shrink it smaller, or (in the worst case) flatten it into a thin line with no thickness at all. The determinant is a number that measures **how the size of that dough changed.**

- **Big number** → the matrix made things **bigger** (stretched the dough).
- **Small number** → the matrix made things **smaller** (shrank the dough).
- **Zero** → uh oh. The matrix **flattened the dough into a line** — squashed it to nothing.

That **zero** is the one case you really need to remember. When the determinant is `0`, information got crushed away, and — just like you can't un-flatten a pancake back into a ball — you **can't undo the matrix.** In other words:

> **Determinant = 0 → the matrix is "broken" (singular) → it has no inverse (no undo button).**
>
> **Determinant ≠ 0 → the matrix is fine → it *does* have an inverse.**

That's the whole point of the determinant: it's a quick **"is this matrix reversible or not?"** check, packed into a single number.

### 9.2 Mathematical Formula
For a small **`2 × 2`** matrix, the recipe is just *multiply the diagonals and subtract*:

```
A = [ a   b ]        determinant = (a × d) − (b × c)
    [ c   d ]
```

- **a × d** → multiply the two numbers on the **main diagonal** (top-left and bottom-right)
- **b × c** → multiply the two numbers on the **other diagonal** (top-right and bottom-left)
- **subtract** → main-diagonal product minus the other one

That's it. Two multiplications and one subtraction.

### 9.3 Mathematical Example
Find the determinant of:

```
A = [ 4   7 ]
    [ 2   6 ]
```

- **Step 1 — Multiply the main diagonal:** `4 × 6 = 24`
- **Step 2 — Multiply the other diagonal:** `7 × 2 = 14`
- **Step 3 — Subtract:** `24 − 14 = 10`

```
determinant = 10
```

It's **not zero**, so this matrix is reversible (it has an undo button). No surprise — it's the same matrix we found the inverse of earlier!

**Now a "broken" one:**

```
B = [ 2   4 ]        determinant = (2 × 8) − (4 × 4) = 16 − 16 = 0
    [ 4   8 ]
```

Determinant is `0` → this matrix squashes things flat → **no inverse.** (Notice row 2 is just row 1 doubled — it's secretly repeating itself, which is what causes the flattening.)

### 9.4 Real-World Example
The determinant is your **quick "can I undo this?" gut-check** before doing heavier math.

Say you're trying to solve for some unknowns — like fitting a best-fit line to data (regression). Before you trust the answer, you glance at the determinant:
- **Not zero?** Great, there's a clean, reversible solution.
- **Zero?** Stop — the data was redundant (some info was repeated or missing), so there's no unique answer to be found.

It's like checking if a door is actually locked before you keep yanking the handle. One number saves you a lot of wasted effort.

### 9.5 Key Takeaway
> **The determinant is one number that tells you if a matrix squashes space flat. For a 2×2 it's `(a×d) − (b×c)`.**
> The only rule you must remember: **zero = flattened = no undo button (no inverse); not zero = reversible.**

---

## <span style="color:#1E6FEB">10. Softmax (bonus)</span>

> Not one of the original ten topics, but Attention leans on it — so here's a quick, friendly detour.

### 10.1 Theory
**Softmax** is a little function that turns a list of **random-looking numbers into percentages that add up to 100%.**

Imagine you've got some raw scores — say `[2, 1, 0]` — and you want to answer: *"out of these, how much weight should each one get?"* Softmax converts them into something like `[66%, 24%, 10%]`. Now they're **easy to compare**, they're all **positive**, and they **sum to 1** (100%). It's like turning exam marks into "share of the total."

Two nice things softmax does on purpose:
- **Bigger scores get a bigger slice** — the highest number always wins the largest percentage.
- **It exaggerates the winner** — because of a squaring-like effect (it uses `e^x`), a score that's a bit higher gets a *lot* more of the pie. So softmax gently says "this one's the favourite" rather than treating everything as roughly equal.

**Why it's useful:** any time a model has to **choose** or **focus** — pick the next word, classify an image, or decide which words to pay attention to — softmax is how it converts raw scores into clean, comparable "how confident am I in each option" percentages.

### 10.2 Mathematical Formula
For a list of numbers `x1, x2, ..., xn`:

```
softmax(xi) = e^(xi) / ( e^(x1) + e^(x2) + ... + e^(xn) )
```

- **e^(xi)** → take each number and raise `e` (≈ 2.718) to that power. This makes everything **positive** and makes bigger numbers stand out more.
- **the bottom (denominator)** → add up all those `e^(...)` values — this is the "total pie."
- **divide** → each number's share of the total → a percentage.

In plain words: **"exponentiate each number, then divide by the sum of all of them."** The results always land between 0 and 1 and add up to 1.

### 10.3 Mathematical Example
Turn the scores `[2, 1, 0]` into percentages.

- **Step 1 — Exponentiate each** (using `e^x`):
  - `e^2 ≈ 7.39`
  - `e^1 ≈ 2.72`
  - `e^0 = 1.00`
- **Step 2 — Add them up (the total pie):** `7.39 + 2.72 + 1.00 = 11.11`
- **Step 3 — Divide each by the total:**
  - `7.39 / 11.11 ≈ 0.665` → **66.5%**
  - `2.72 / 11.11 ≈ 0.245` → **24.5%**
  - `1.00 / 11.11 ≈ 0.090` → **9.0%**

Result:

```
softmax([2, 1, 0]) ≈ [0.665, 0.245, 0.090]
```

They add up to 1 (100%), they're all positive, and the biggest input (`2`) grabbed the biggest slice (66.5%). Notice it wasn't just "2 is twice as big as 1" — softmax **amplified** the lead. That's the whole idea.

### 10.4 Real-World Example
**Predicting the next word.** When ChatGPT is about to write a word, it produces a raw score for *every* word in its vocabulary — thousands of numbers. Softmax turns those into **probabilities**: maybe "dog" → 60%, "cat" → 30%, "banana" → 0.1%, and so on. The model then picks from that distribution. Same trick powers image classifiers ("90% cat, 8% dog, 2% fox") — softmax is the universal "turn scores into confident percentages" step.

### 10.5 Key Takeaway
> **Softmax turns any list of numbers into positive percentages that add up to 100% — with the biggest number getting the biggest (and exaggerated) share.**
> It's how models convert raw scores into "how much should I focus on / believe in each option."

---

## <span style="color:#1E6FEB">11. Attention</span>

### 11.1 Theory
**Attention** is how an AI model figures out **which words in a sentence matter to which other words.**

Everyday intuition: read *"The cat sat on the mat because **it** was tired."* Does *"it"* mean the cat or the mat? You instantly know — **the cat**. Your brain paid *attention* to the right word. Attention gives a model that same ability: for every word, it decides **how much to focus on each other word** to understand the meaning.

The lovely part: **attention is just the matrix stuff you've already learned, glued together** — multiply, transpose, scale, and (now!) softmax. No new magic.

Every word gets three roles, with easy nicknames:
- **Query (Q)** → "what am I looking for?"
- **Key (K)** → "what do I have to offer?"
- **Value (V)** → "the actual information I'll pass along"

The idea: each word's **Query** is matched against every word's **Key** to score *who should focus on whom*; **softmax** turns those scores into focus-percentages; and those percentages **blend the Values** into a new, context-aware meaning for each word.

### 11.2 Mathematical Formula
The whole thing in one line — and every piece is something you know:

```
Attention = softmax( (Q × Kᵀ) / √d ) × V
```

- **Q × Kᵀ** → multiply Queries by the **transpose** of Keys → a grid of "every word vs every word" scores *(Transpose topic!)*
- **÷ √d** → **scale** the numbers down so they stay stable *(Scaling topic!)*
- **softmax(...)** → turn each row of scores into **focus percentages that add up to 100%** *(the function above!)*
- **× V** → use those percentages to **blend the Values** into each word's new meaning *(Multiplication topic!)*

So attention = *a couple of matrix multiplications + one softmax.* That's the entire thing.

### 11.3 Mathematical Example
Let's walk one word through it with 3 tiny words — **"cat", "sat", "mat"**.

Suppose the score step (`Q × Kᵀ`) already gave us the row for **"cat"**: how much "cat" relates to each word:

```
cat's raw scores:  [ 2,  1,  0 ]   (vs cat, vs sat, vs mat)
```

- **Step 1 — Softmax those scores** (we literally just did this one!):

```
softmax([2, 1, 0]) ≈ [0.665, 0.245, 0.090]
```

So "cat" decides to focus **66.5% on itself, 24.5% on "sat", 9% on "mat".**

- **Step 2 — Blend the Values** using those percentages. If the words' Value vectors are `V_cat`, `V_sat`, `V_mat`, then cat's new meaning is:

```
new_cat = 0.665 × V_cat  +  0.245 × V_sat  +  0.090 × V_mat
```

That's it — "cat"'s updated meaning is a **weighted mix**, leaning mostly on itself but flavoured by the words it cares about. Do this for every word and you've run attention.

> **The `Q × Kᵀ` score step in full** (from the Transpose topic): with `Q = K = [[1,0],[0,1],[1,1]]` for "cat", "sat", "mat", transposing K and multiplying gives the score grid `[[1,0,1],[0,1,1],[1,1,2]]` — one "how related?" number per word-pair. Those rows are exactly what softmax then turns into focus-percentages.

### 11.4 Real-World Example
**Understanding "it".** When a model reads *"The cat sat on the mat because it was tired,"* the attention percentages for *"it"* come out high on *"cat"* — so the model correctly links them. No grammar rules written by hand; just score → softmax → blend, learned from data.

And it's everywhere: **every transformer layer** runs this "compare every word to every other word" step (often several times in parallel — "multi-head"), which is how models track long-range context and connect a pronoun to a noun 40 words earlier. Stacked ~90 times, this single operation is the backbone of GPT, Claude, BERT, and friends.

### 11.5 Key Takeaway
> **Attention lets a model decide which words matter to which: it scores every word against every other word, uses softmax to turn those scores into focus-percentages, and blends the words' information accordingly.**
> Under the hood it's nothing new — `softmax(Q × Kᵀ / √d) × V` — just the transpose, scaling, matrix-multiply, and softmax you already know. It's what powers every modern transformer.

---

*End of notes — all topics covered: Matrices → Shapes → Addition → Scaling → Multiplication → Transpose → Identity → Inverse → Determinant → Softmax → Attention. From "a matrix is a grid of numbers" all the way to "attention is how a transformer thinks."*
