# <span style="color:#0B3D91">Data Engineering for AI: Foundations &amp; File Formats</span>

> Study notes covering the groundwork of data engineering for AI systems — why data-centric thinking matters, the three tiers of data organization (structured, semi-structured, unstructured), and the everyday file formats (CSV, JSON, PDF, DOCX, HTML) that AI pipelines must ingest.
> A practical, intuition-first foundation for building trustworthy AI systems — because every model is only as good as the data behind it.

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [Why Data-Centric Thinking Matters for AI](#1-why-data-centric-thinking-matters-for-ai)
2. [Structured, Semi-Structured &amp; Unstructured Data](#2-structured-semi-structured--unstructured-data)
3. [File Formats: CSV, JSON, PDF, DOCX &amp; HTML](#3-file-formats-csv-json-pdf-docx--html)

---

## <span style="color:#1E6FEB">1. Why Data-Centric Thinking Matters for AI</span>

### 1.1 Overview / What is it?
**Data engineering** is the discipline that makes AI systems *trustworthy* — it ensures the **right data, in the right shape, reaches the model**. This module deliberately moves *beyond model architecture* to focus on the quality, structure, and pipelines behind the data, because that is what actually determines whether an AI system can be relied on.

### 1.2 Why does it matter for AI?
> **"Garbage In, Garbage Out."**
> A model trained or prompted with flawed data will *confidently* produce flawed answers — no algorithm can fix that downstream.

Every AI and GenAI system is **only as good as the data behind it**. Messy, incomplete, or poorly structured data leads directly to:

- **Wrong predictions**
- **Biased outcomes**
- **Hallucinated answers**

### 1.3 Key Concepts — Real AI Failures Caused by Bad Data
These are not hypothetical; poor data practices sit behind many well-known AI failures.

| Failure | Root data cause |
|---|---|
| **Hallucinating chatbot** | Sparse, outdated context data → the support bot *invented* policies that never existed |
| **Biased hiring model** | Historical hiring data skewed toward one group → model learned to penalize other candidates |
| **Broken pipeline** | An unversioned schema change *silently* broke a downstream training pipeline for weeks |
| **Faulty financial model** | Missing transaction values silently became zeros → understated risk in a credit model |

### 1.4 Simple Example
The words *"Running"*, *"running"*, and *"RUN"* all mean the same thing — but to a model, without any data preparation, they can look like three unrelated things. Data-centric thinking says: *fix that in the data first*, before blaming the model.

### 1.5 How it works — The Data Engineering Lifecycle for AI
Every stage in this module maps to one pipeline. Each stage is covered hands-on:

**Raw Data → Clean → Annotate → Version → Query → Govern**

| Stage | What happens | Covered in |
|---|---|---|
| **Raw Data** | CSV, JSON, PDFs… collected as-is | Foundations & file formats |
| **Clean** | ETL & cleaning pipelines | ETL, cleaning |
| **Annotate** | Label data for supervised learning | Annotation |
| **Version** | Track dataset changes over time | Dataset versioning |
| **Query** | SQL for structured access | SQL basics |
| **Govern** | Privacy & compliance | Privacy, governance |

### 1.6 Practical Example / Use Case
A customer-support bot confidently tells a user about a refund policy that was **deprecated 8 months ago**. Nothing was wrong with the language model — its *retrieved context document was never updated*. The fix lives entirely in the data layer (versioning + governance + fresh grounding), not in the model. This is data-centric thinking in action.

### 1.7 Key Takeaways
> - AI performance is decided by **data quality and structure**, not just model architecture.
> - **Garbage In → Garbage Out**: no clever algorithm rescues bad data.
> - Bad data produces three signature failures: **wrong predictions, biased outcomes, hallucinations**.
> - The lifecycle — **Raw → Clean → Annotate → Version → Query → Govern** — is the roadmap for the whole module.

---

## <span style="color:#1E6FEB">2. Structured, Semi-Structured &amp; Unstructured Data</span>

### 2.1 Overview / What is it?
AI systems must consume many different **data shapes**, and each shape needs a different processing approach. There are three tiers of data organization:

- **Structured** — organized in rows & columns; easy for machines to query directly.
- **Semi-structured** — has organizational markers (tags, keys, labels) but *no fixed schema*.
- **Unstructured** — no fixed format; needs extraction (chunking & embedding) before AI can use it.

### 2.2 Why does it matter for AI?
The tier decides **how much preprocessing you need before the data is usable**. Structured data is nearly query-ready; unstructured data (the majority of real-world content — documents, images, audio) must be *extracted and transformed* before a model can touch it. Misjudging the tier means underestimating the pipeline work.

### 2.3 Key Concepts

**Structured Data** — organized in rows & columns, easy to query directly:

- **CSV / Tables** → rows of records: sales logs, user tables, transaction history.
- **Relational DB** → structured storage queried directly using **SQL**.

**Semi-Structured Data — the middle ground** — organizational markers but no rigid table schema, allowing flexible, *nested* structure:

- Examples: **JSON, XML, YAML, log files, NoSQL documents**.
- Common sources: APIs, config files, application logs, NoSQL databases.
- AI systems need **nested parsing logic** — fields can be optional or deeply nested.

**Unstructured Data** — no fixed format; needs extraction before any AI use:

- **JSON** (semi-structured API/config data), **PDFs/Documents** (reports, manuals, policy docs), **Images** (scans, photos, diagrams), **Audio** (calls, voice notes, transcripts).

### 2.4 Simple Example
A semi-structured JSON record — note the *nesting* a flat table can't express:

```json
{
  "customer": "A. Rao",
  "orders": [
    {"item": "Laptop", "amount": 52000},
    {"item": "Mouse",  "amount": 800}
  ]
}
```

One customer, a *list* of orders inside — that nesting is exactly what "semi-structured" means.

### 2.5 How it works — Choosing the Tier for the Task

| Tier | Formats | Best for | Key handling need |
|---|---|---|---|
| **Structured** | CSV, SQL tables | Analytics, dashboards, direct querying | Easiest to validate and clean |
| **Semi-Structured** | JSON, XML, logs | APIs, configs, NoSQL pipelines | Needs schema-aware / nested parsing |
| **Unstructured** | PDF, DOCX, HTML, images, audio | Documents, RAG, OCR pipelines | Needs extraction before any AI use |

### 2.6 Practical Example / Use Case
Building a **RAG chatbot** over company policy PDFs: those PDFs are *unstructured*, so before the model can answer anything, you must extract the text, chunk it, and embed it. Meanwhile the customer table it cross-references is *structured* and can be queried directly with SQL. One product, two tiers, two very different pipelines.

### 2.7 Key Takeaways
> - Data comes in **three tiers**: structured (rows/columns), semi-structured (nested tags/keys, no fixed schema), unstructured (no format at all).
> - The tier dictates **how much preprocessing** is required — structured is query-ready, unstructured needs extraction first.
> - **JSON/XML/logs** are the classic semi-structured middle ground; **PDFs, images, audio** are unstructured.
> - Match your pipeline to the tier: validate structured data, parse semi-structured nesting, extract unstructured content.

---

## <span style="color:#1E6FEB">3. File Formats: CSV, JSON, PDF, DOCX &amp; HTML</span>

### 3.1 Overview / What is it?
The **file format** is the container your raw data arrives in. Each format has its own structure, its own parsing quirks, and its own sweet-spot use case. Knowing the format tells you *how much work* it takes to turn that file into model-ready data.

### 3.2 Why does it matter for AI?
> Format choice shapes how much preprocessing is needed.

A CSV can be loaded and queried almost immediately; a scanned PDF might need a whole OCR pipeline before it yields a single usable word. Picking (or recognizing) the right format up front saves enormous downstream effort — and prevents subtle bugs like scrambled reading order.

### 3.3 Key Concepts — The Five Formats

**CSV — Comma-Separated Values** (simplest *structured* format):

- Stores tabular data as plain text; each line = a row, commas separate columns.
- Extremely **portable** — opens in Excel, pandas, or any text editor.
- **No native support for nested data or data types** — everything is text.
- Common use: exported reports, transaction logs, tabular datasets.

```
id,name,amount,date
1,A. Rao,1200,2026-01-04
2,B. Khan,980,2026-01-05
```

**JSON — JavaScript Object Notation** (flexible, *nested* key-value format):

- Represents data as nested key-value pairs and lists — the standard for **APIs and config files**.
- Supports nested objects and arrays, unlike flat CSV; human-readable, widely supported.
- Common use: API responses, config files, NoSQL document stores.

```json
{ "name": "A. Rao", "city": "Mumbai", "active": true }
```

**PDF — Portable Document Format** (fixed-layout, built for *humans* not machines):

- Preserves exact visual layout across devices — which is *precisely* what makes it hard to parse reliably.
- Text, tables, and images can all be embedded in one file.
- Extraction libraries can **scramble reading order** in multi-column layouts.
- **Scanned PDFs contain no text at all** — they need OCR (covered later).
- *Parsing challenge:* a two-column report may extract as jumbled, interleaved text unless a **layout-aware parser** is used.

**DOCX — Word Documents** (structured, XML-based):

- Actually a **zipped bundle of XML files** — giving programmatic access to text, styles, tables, and comments.
- Structure (headings, styles, tables) is preserved and extractable.
- Libraries like **`python-docx`** can read and edit content directly.
- Common use: contracts, letters, policy drafts, reports.
- *Parsing note:* unlike PDF, DOCX retains a structured XML tree, making heading levels and tables **far easier to extract accurately**.

**HTML — HyperText Markup Language** (the markup behind every web page):

- Structures content using **nested tags**: `<p>`, `<table>`, `<div>`, `<span>`, and more.
- Scraping requires **filtering out** navigation, ads, and scripts to reach meaningful text.
- Common use: web scraping, documentation ingestion, email bodies.

```html
<article>
  <h1>Billing FAQ</h1>
  <p>How do I reset my password?</p>
</article>
```

### 3.4 Simple Example
The *same* customer record in three formats:

- **CSV:** `1,A. Rao,Mumbai` — flat, compact, no nesting.
- **JSON:** `{"id":1,"name":"A. Rao","city":"Mumbai"}` — labeled keys, extensible.
- **HTML:** `<td>A. Rao</td><td>Mumbai</td>` — wrapped in tags meant for display, not analysis.

Same information, wildly different parsing effort.

### 3.5 How it works — Matching Format to Workflow

| Tier | Formats | Best for | Preprocessing burden |
|---|---|---|---|
| **Structured** | CSV, SQL tables | Analytics, dashboards, direct querying | Lowest — easy to validate & clean |
| **Semi-structured** | JSON, XML, logs | APIs, configs, NoSQL pipelines | Medium — schema-aware parsing |
| **Unstructured** | PDF, DOCX, HTML, images, audio | Documents, RAG, OCR pipelines | Highest — extraction before any AI use |

### 3.6 Practical Example / Use Case
An ingestion pipeline for a document-QA system receives a mix of files. A **DOCX** contract extracts cleanly via its XML tree; a **digitally-created PDF** extracts with a layout-aware parser; a **scanned PDF** falls through to OCR; and **HTML** help-center pages get scraped with nav/ads stripped out. The format of each file decides which branch of the pipeline it takes.

### 3.7 Key Takeaways
> - **CSV** = simple flat tables, everything is text, no nesting.
> - **JSON** = nested key-value pairs, the standard for APIs & configs.
> - **PDF** = fixed visual layout, hard to parse; multi-column scrambles, scanned = OCR needed.
> - **DOCX** = zipped XML, structure preserved, easiest of the "documents" to extract (`python-docx`).
> - **HTML** = nested tags; scraping means filtering out the noise (nav, ads, scripts).
> - Rule of thumb: **structure ↑ = preprocessing effort ↓**. Always match format to the task before building the pipeline.

---

*End of file 01 — Foundations & File Formats complete.*
