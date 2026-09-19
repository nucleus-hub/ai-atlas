# <span style="color:#0B3D91">Data Engineering for AI: Text Preprocessing &amp; OCR</span>

> Study notes covering how raw language and scanned documents become model-ready — text preprocessing (tokenization, normalization, stopword removal, stemming vs lemmatization) and OCR workflow awareness (turning pixels into machine-readable text, and where it breaks down).
> The text-and-documents branch of the data pipeline, made practical.

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [Text Preprocessing](#1-text-preprocessing)
2. [OCR Workflow Awareness](#2-ocr-workflow-awareness)

---

## <span style="color:#1E6FEB">1. Text Preprocessing</span>

### 1.1 Overview / What is it?
**Text preprocessing** standardizes raw, messy language into a clean, consistent form *before* it ever reaches a model. It is the cleaning pipeline's equivalent for words instead of table rows.

### 1.2 Why does it matter for AI?
Raw text is messy. The same word can appear as *"Running"*, *"running"*, or *"RUN"* across a dataset — and without preprocessing, a model may treat these as **entirely unrelated tokens**. Preprocessing collapses that surface-level variation so identical meaning maps to identical text.

> **Raw Text → Model-Ready Text:** `"Running, RUNNING, and running!!"` all become the *same* clean token after preprocessing.

### 1.3 Key Concepts — The Preprocessing Steps

**1. Tokenization** — split raw text into individual units (tokens), the building blocks for every later NLP step:

- **Word tokenization** — split on spaces and punctuation.
- **Sub-word tokenization** — split rare words into meaningful fragments.
- **Sentence tokenization** — split a document into individual sentences.
- *Example:* `"AI systems need clean data."` → `["AI", "systems", "need", "clean", "data", "."]`

**2. Normalization & Case Folding** — standardize formatting so identical meanings are represented identically:

- **Case folding** — convert all text to lowercase.
- Remove punctuation, extra whitespace, and special characters.
- Handle accents and unicode variations consistently.
- *Example:* `"Café RESET!!  Password"` → `"cafe reset password"`

**3. Stopword Removal** — filter out common words that carry little meaning on their own (`"the"`, `"is"`, `"at"`, `"and"`):

- Common in classic NLP pipelines: search, keyword extraction, topic modeling.
- **Modern transformer models often skip this** — they learn context directly.
- Removing stopwords too aggressively can **strip meaningful negation** (e.g., `"not"`).
- *Example:* `"This is the best product I have used"` → `"best product used"`

**4. Stemming vs Lemmatization** — two ways to reduce words to a common base form:

| | **Stemming** | **Lemmatization** |
|---|---|---|
| **Method** | Chops words down using simple rules | Uses vocabulary & grammar to find the dictionary form |
| **Speed / output** | Fast, but can produce non-words | Slower, but always a real word |
| **"Running", "Runs"** | → "Run" | → "Run" |
| **"Better"** | → "Better" (unchanged) | → "Good" |

### 1.4 Simple Example
`"Running, RUNNING, and running!!"` → after case folding + punctuation removal + lemmatization → a single token **`run`**. Three surface forms, one meaning, one token.

### 1.5 How it works — A Full Pipeline

**Raw text:**
> "The Support Team was AMAZING!! I've NEVER had such quick replies... Will definitely be recommending this to friends."
> *(mixed casing, punctuation, contractions, and stopwords throughout)*

**Preprocessed tokens** (lowercased, punctuation removed, stopwords dropped, lemmatized):

```python
['support', 'team', 'amazing',
 'never', 'quick', 'reply',
 'definitely', 'recommend', 'friend']
```

Note `"replies"` → `"reply"` and `"recommending"` → `"recommend"` (lemmatization), while `"NEVER"` survives — a good pipeline keeps meaningful words even as it drops filler.

### 1.6 Practical Example / Use Case
A team analyzing thousands of customer-feedback messages for **topic modeling** runs every message through tokenize → normalize → remove stopwords → lemmatize. Suddenly *"loved it"*, *"Loving it!"*, and *"LOVE IT"* all collapse to the same signal, so themes like `refund`, `delivery`, `support` emerge cleanly instead of being scattered across dozens of surface variants.

### 1.7 Key Takeaways
> - Text preprocessing = **standardize messy language into consistent, model-ready text**.
> - Core steps: **tokenize → normalize/case-fold → (optionally) remove stopwords → stem or lemmatize**.
> - **Stemming** is fast but crude (may yield non-words); **lemmatization** is slower but returns real dictionary words (`better → good`).
> - **Modern transformers often skip stopword removal**; and beware stripping negations like `"not"`.
> - The payoff: the same meaning maps to the same token, so downstream NLP/AI sees clean signal.

---

## <span style="color:#1E6FEB">2. OCR Workflow Awareness</span>

### 2.1 Overview / What is it?
**OCR (Optical Character Recognition)** is the process that turns **scanned documents and images into machine-readable text**. It is how a photo of a page or a scanned PDF — which contains *pixels*, not text — becomes searchable, extractable words.

### 2.2 Why does it matter for AI?
Unstructured documents are everywhere, and many arrive as **scans with no text layer** (scanned PDFs, photos of forms, images of receipts). Without OCR, an AI pipeline literally cannot read them. But OCR is *lossy and error-prone*, so knowing when it is needed — and where it breaks — protects your downstream data quality.

### 2.3 Key Concepts — How OCR Works
OCR is a small pipeline of its own, turning pixels into text:

**Scanned Image → Preprocessing → OCR Engine → Postprocessing → Extracted Text**

| Stage | What happens |
|---|---|
| **Scanned Image** | A photo or scan of a page |
| **Preprocessing** | Deskew, denoise, binarize (clean up the image) |
| **OCR Engine** | Detects & reads characters |
| **Postprocessing** | Spellcheck & layout cleanup |
| **Extracted Text** | Machine-readable, searchable output |

### 2.4 Common Pitfalls — Where OCR Breaks Down
Accuracy drops sharply outside of clean, high-quality scans:

| Pitfall | Effect |
|---|---|
| **Skewed scans** | Tilted pages confuse character detection and reading order |
| **Poor contrast** | Faded ink or low-resolution scans produce garbled output |
| **Handwriting** | Standard OCR engines struggle *badly* with handwritten text |
| **Layout loss** | Tables and multi-column layouts often get flattened incorrectly |

### 2.5 OCR vs Native Text Extraction — Always Check First
Before running OCR, check whether it is even necessary:

| | **Native Text Extraction** | **OCR** |
|---|---|---|
| **When** | Digitally-created PDFs/DOCX with embedded text | Scanned documents/images with **no** embedded text |
| **Speed & accuracy** | Fast, accurate, no image processing | Slower; accuracy depends on scan quality |
| **Fidelity** | Preserves exact original characters | Introduces potential recognition errors |
| **Rule** | **Always try this first** | Only when there is no text layer to extract |

### 2.6 Common OCR Tools & Libraries
Once OCR is actually needed, you will encounter:

| Tool | Notes |
|---|---|
| **Tesseract OCR** | Free, open-source; the most widely used starting point |
| **Google Cloud Vision** | Managed cloud OCR API with strong layout detection |
| **AWS Textract** | Managed OCR with built-in table & form extraction |
| **Azure AI Vision** | Managed OCR API integrated with the Azure ecosystem |
| **EasyOCR / PaddleOCR** | Open-source Python libraries, strong multi-language support |

### 2.7 Simple Example
A **digitally-created PDF invoice** has a text layer → use *native extraction*: fast and character-perfect. A **photographed paper invoice** is just pixels → it needs *OCR*, which might misread `"O"` as `"0"` or scramble a two-column layout. Same document type, two totally different (and differently reliable) paths.

### 2.8 Practical Example / Use Case
An insurance team ingests claim documents: some are clean digital PDFs, others are phone photos of handwritten forms. The pipeline first attempts **native extraction**; if no text layer exists, it falls back to **OCR** (Tesseract for typed pages, a cloud API like Textract for tables). Handwritten sections are flagged for **human review** because OCR cannot be trusted there — preventing garbled data from silently poisoning downstream models.

### 2.9 Key Takeaways
> - **OCR turns pixels into machine-readable text** via: image → preprocess → OCR engine → postprocess → text.
> - **Always try native text extraction first** — it is faster, exact, and error-free; only use OCR when there is no text layer.
> - OCR accuracy collapses on **skew, poor contrast, handwriting, and complex layouts** (tables/multi-column).
> - Start with **Tesseract**; reach for managed APIs (**Google Vision, AWS Textract, Azure AI Vision**) for tables/forms and tough scans.
> - Treat OCR output as *potentially noisy* — validate it, and flag hard cases (like handwriting) for human review.

---

*End of file 03 — Text Preprocessing & OCR complete.*
