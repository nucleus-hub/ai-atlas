# <span style="color:#0B3D91">Data Engineering for AI: Versioning, SQL, Privacy &amp; Governance</span>

> Study notes covering the responsible-data-practice layer — dataset versioning (reproducibility & rollback), SQL basics (querying structured data), and data privacy & governance (PII, anonymization, ownership, access control, compliance).
> The practices that keep AI systems reproducible, queryable, and responsible.

---

## <span style="color:#1E6FEB">Table of Contents</span>

1. [Dataset Versioning](#1-dataset-versioning)
2. [SQL Basics](#2-sql-basics)
3. [Data Privacy &amp; Governance](#3-data-privacy--governance)

---

## <span style="color:#1E6FEB">1. Dataset Versioning</span>

### 1.1 Overview / What is it?
**Dataset versioning** is version control *for data* — tracking how a dataset changes over time (new rows, corrected labels, removed records), so you always know exactly which data produced a given model.

> **"Which data trained this model?" should always have a clear answer.**

### 1.2 Why does it matter for AI?
Datasets change constantly. Without versioning, it becomes **impossible to reproduce or debug** a model's behavior. Versioning gives you three superpowers:

- **Reproducibility** — recreate a model's exact training conditions.
- **Rollback** — undo a bad data update.
- **Auditability** — prove what data a model was trained on.

*Example:* a model's accuracy suddenly drops after retraining — with dataset versioning, the team traces it to a corrupted label update in **v1.4** and rolls back.

### 1.3 Key Concepts — Basic Practices
Simple habits that make datasets traceable and trustworthy:

| Practice | What it does |
|---|---|
| **Naming Conventions** | Clear, dated version IDs, e.g. `dataset_v1.2_2026-01` |
| **DVC / Git-LFS** | Version control built for large data files (Git can't handle big binaries alone) |
| **Checksums / Hashing** | Detect *unintended* changes to a file |
| **Changelogs** | Document what changed, and why |
| **Dataset Registries** | A central catalog of dataset versions |

### 1.4 Simple Example
Two folders: `dataset_v1.3_2026-01` and `dataset_v1.4_2026-02`. A checksum reveals the label file changed between them; the changelog says *"relabeled 400 ambiguous reviews."* When v1.4's model underperforms, you know *precisely* what differed and can revert to v1.3.

### 1.5 How it works — The Workflow
1. **Snapshot** the dataset with a clear, dated version name.
2. **Track large files** with DVC or Git-LFS (code in Git, data pointers alongside).
3. **Hash** each version so unintended changes are detectable.
4. **Record a changelog** entry: what changed, why, and by whom.
5. **Register** the version in a central catalog so teams can find and reference it.
6. **Link model → dataset version** so every trained model points back to its exact data.

### 1.6 Practical Example / Use Case
An ML team retrains their recommender weekly. Each run pins an exact dataset version (`recs_v2.7_2026-03`) via DVC, hashed and changelogged. When week 8's model tanks, they diff v2.7 vs v2.6, spot a corrupted import that duplicated 5% of rows, roll back to v2.6, and redeploy — all in an afternoon, because the data history was there.

### 1.7 Key Takeaways
> - Dataset versioning = **version control for data**, so "which data trained this model?" always has an answer.
> - Enables **reproducibility, rollback, and auditability**.
> - Core practices: **dated naming, DVC / Git-LFS, checksums/hashing, changelogs, dataset registries**.
> - Always **link a model to the exact dataset version** it trained on — future-you will be grateful.

---

## <span style="color:#1E6FEB">2. SQL Basics</span>

### 2.1 Overview / What is it?
**SQL (Structured Query Language)** is the universal language for querying structured, relational data. It turns raw tables into direct, decision-ready answers — no manual spreadsheet work needed.

### 2.2 Why does it matter for AI? (SQL, Even in an AI World)
GenAI has not replaced SQL — **it depends on it**:

- Most enterprise data still lives in **structured, relational databases**.
- AI pipelines often combine **SQL filtering** with other retrieval methods.
- **Feature stores** for ML models are commonly queried using SQL.
- SQL is a **universal, portable skill** across almost every data platform.

### 2.3 Key Concepts — Anatomy of a Query
Every query is built from the same clauses, **always in this order**:

```sql
SELECT   category, SUM(amount)   -- choose columns / calculations
FROM     sales                   -- choose the source table
WHERE    quarter = 'Q4'          -- filter individual rows
GROUP BY category                -- aggregate rows into groups
ORDER BY SUM(amount) DESC        -- sort the final results
LIMIT    5;                      -- restrict the number of rows
```

**SELECT** — choose which columns to retrieve:

- `SELECT name, age FROM customers;` → only those two columns.
- `SELECT * FROM customers;` → every column (use sparingly in production).
- Rename on the fly with `AS`: `SELECT name AS full_name`.

**WHERE** — keep only rows matching a condition (runs *before* grouping):

- Comparisons: `=, !=, >, <, >=, <=`
- Logic: `AND, OR, NOT`
- Pattern matching: `LIKE '%mumbai%'`

**JOIN** — merge rows from two related tables on a shared key:

```sql
SELECT c.name, o.amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;
```

**JOIN types** (which rows survive the merge):

| Join | Returns |
|---|---|
| **INNER JOIN** | Only rows with a match in *both* tables |
| **LEFT JOIN** | All rows from the left table + matches from the right (NULLs if none) |
| **RIGHT JOIN** | All rows from the right table + matches from the left |
| **FULL (OUTER) JOIN** | All rows from *both* tables, matched where possible |

**GROUP BY & aggregation** — summarize many rows into totals:

- `COUNT()` — number of rows; `SUM()` / `AVG()`; `MIN()` / `MAX()`.
- Rule: every non-aggregated column in `SELECT` must also appear in `GROUP BY`.

**HAVING vs WHERE** — a classic beginner confusion (they filter at *different stages*):

| **WHERE** | **HAVING** |
|---|---|
| Filters individual rows | Filters grouped results |
| Runs **before** grouping | Runs **after** GROUP BY |
| Can't reference aggregates | Can reference aggregates |
| `WHERE amount > 100` | `HAVING SUM(amount) > 10000` |

**ORDER BY & LIMIT** — sort and cap results (almost always the *last* clauses):

- `ORDER BY amount DESC` (highest first) / `ASC` (default, alphabetical).
- `LIMIT 5` — keep only the first N rows after sorting.

**Subqueries** — a query nested inside another (inner runs first):

```sql
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE amount > 1000
);
-- finds all customers who placed at least one order over 1000
```

### 2.4 Common Beginner Mistakes
| Mistake | Why it bites |
|---|---|
| **No WHERE on UPDATE/DELETE** | Updates or deletes *every* row in the table |
| **`=` instead of `IS NULL`** | NULL can't be compared with `=`; use `IS NULL` / `IS NOT NULL` |
| **Confusing HAVING & WHERE** | Filtering rows with HAVING (or aggregates with WHERE) errors out |
| **Overusing `SELECT *`** | Wastes performance and hides intent in production code |

### 2.5 Simple Example
> **Business question:** Which product category generated the most revenue last quarter?

```sql
SELECT category, SUM(amount) AS revenue
FROM sales
WHERE quarter = 'Q4'
GROUP BY category
ORDER BY revenue DESC;
```

**Result:**

| category | revenue |
|---|---|
| Electronics | ₹ 18,42,300 |
| Home & Kitchen | ₹ 12,05,750 |
| Fashion | ₹ 9,88,420 |

### 2.6 How it works — Reading That Query Clause by Clause
1. **SELECT** — choose `category` plus a computed revenue total.
2. **FROM** — read from the `sales` table.
3. **WHERE** — keep only Q4 rows *before* any aggregation.
4. **GROUP BY** — collapse rows into one per category.
5. **ORDER BY** — sort categories from highest to lowest revenue.

### 2.7 Practical Example / Use Case — SQL + AI, Together
Modern AI systems combine SQL for structured metadata with other retrieval for unstructured meaning — each covering what the other can't:

- **RAG-style systems** use SQL to filter documents by metadata *before* deeper retrieval.
- **Feature stores** are queried with SQL for training data.
- **Text-to-SQL** is itself a growing GenAI application category.

*Example:* a support-bot pipeline first uses SQL to filter documents to the customer's product line, then runs deeper retrieval only within that filtered set.

**Practice challenge:** find the top 3 customers by total amount spent.

```sql
SELECT customer_id, SUM(amount) AS total_spent
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 3;
```

### 2.8 Key Takeaways
> - **SQL is alive and essential in the AI era** — enterprise data, feature stores, and RAG metadata filtering all run on it.
> - Clause order is fixed: **SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT**.
> - **WHERE filters rows (before grouping); HAVING filters groups (after)** — the #1 confusion.
> - **JOINs** combine tables on a shared key; know INNER vs LEFT/RIGHT/FULL.
> - Avoid the classics: no `WHERE` on UPDATE/DELETE, `= NULL`, and `SELECT *` in production.

---

## <span style="color:#1E6FEB">3. Data Privacy &amp; Governance</span>

### 3.1 Overview / What is it?
This topic covers the **responsible-data** layer of AI. Two intertwined ideas:

- **Data Privacy** — protecting personal information (PII) throughout the pipeline.
- **Data Governance** — the rules for *who owns data, who can touch it, and how it is kept compliant*.

### 3.2 Why does it matter for AI?
AI pipelines constantly handle sensitive data. Mishandling it risks **legal penalties, broken trust, and real harm to individuals**. Privacy and governance are what keep an AI system not just accurate, but *responsible* and *lawful*.

### 3.3 Key Concepts

**What is PII (Personally Identifiable Information)?** — any data that can identify a specific individual, alone or combined with other data:

- **Direct identifiers:** name, email, phone number, government ID.
- **Indirect identifiers:** date of birth + zip code can *together* identify someone.
- AI pipelines must treat PII with **extra caution at every stage**: storage, processing, and output.
- *Example:* a support-ticket dataset with customer emails, phone numbers, and billing addresses is full of PII — it cannot be used or shared carelessly.

**Anonymization techniques** — reduce identifiability while keeping data useful:

| Field | Before | After (anonymized) |
|---|---|---|
| name | Aditi Rao | `A***  R**` |
| email | aditi.rao@email.com | `a****@email.com` |
| phone | 98765 43210 | `98765 XXXXX` |
| city | Mumbai | Mumbai *(kept — not identifying alone)* |

Masked fields **retain usable structure without exposing identity**.

**Governance — Ownership & Access Control** — defines clear ownership and restricts access to only roles that genuinely need it:

- **Data owner:** accountable for a dataset's quality and appropriate use.
- **Access control:** role-based permissions (read, write, admin) per dataset.
- **Every access should be logged** — supporting audits and incident response.
- *Example:* a customer PII dataset is owned by the Data Privacy team; only specific approved ML pipelines get read access, and every access is logged.

**Compliance basics** — concepts you will meet in industry:

| Concept | What it means |
|---|---|
| **GDPR / Data Laws** | Regional regulations on personal data |
| **Retention Policies** | How long data can legally be kept |
| **Audit Trails** | Records of who accessed what, when |
| **Consent Management** | Tracking user permission for data use |

### 3.4 Simple Example
Before feeding support tickets into a model, a script masks every name, email, and phone (`aditi.rao@email.com` → `a****@email.com`) while keeping non-identifying fields like `city`. The model still learns useful patterns; no individual is exposed.

### 3.5 How it works — The Workflow
1. **Identify PII** in the dataset (direct + indirect identifiers).
2. **Anonymize / mask** sensitive fields while preserving analytical usefulness.
3. **Assign an owner** accountable for the dataset.
4. **Apply role-based access control** — least privilege, per dataset.
5. **Log every access** for audit trails.
6. **Enforce compliance** — respect retention policies, consent, and data laws (GDPR, etc.).

### 3.6 Practical Example / Use Case
A team building a support chatbot must use real ticket data. Governance kicks in: the **Data Privacy team owns** the dataset, grants **read-only** access to one approved training pipeline, and **logs** every access. Before training, PII is **masked**. Old tickets past the **retention window** are purged, and only tickets from users who **consented** to data use are included. The result: a useful model that is also legally and ethically sound.

### 3.7 Key Takeaways
> - **PII = any data that can identify a person**, directly (name, email, ID) or indirectly (DOB + zip).
> - **Anonymize/mask** sensitive fields to reduce identifiability while keeping data useful.
> - **Governance** = clear **ownership**, **role-based access control** (least privilege), and **logged access**.
> - Know the compliance pillars: **GDPR/data laws, retention policies, audit trails, consent management**.
> - Privacy + governance make AI systems **responsible and lawful**, not just accurate.

---

## <span style="color:#1E6FEB">Module Wrap-Up: Overarching Key Takeaways</span>

> - **Data-centric thinking** means treating data quality and structure as core engineering work.
> - Structured, semi-structured, and unstructured data — and formats like CSV, JSON, PDF, DOCX, HTML — each need different handling.
> - **ETL and repeatable cleaning pipelines** turn messy raw data into trustworthy input.
> - **Text preprocessing, OCR awareness, and annotation quality** directly shape model performance.
> - **Dataset versioning, SQL, privacy, and governance** keep AI systems reproducible, queryable, and responsible.

> **The one-line summary:** An AI system is only as trustworthy as the data behind it — quality, structure, and pipelines decide everything downstream.

---

*End of file 05 — Versioning, SQL, Privacy & Governance complete.*
*End of Module 4 — Data Engineering for AI Applications. All 12 topics covered.*
