# Module 4 — Data Engineering for AI Applications

Companion materials for **Session 4: Data Engineering for AI Applications**.

```
data_engineering/
├── Demo/          # the runnable notebook + generated data
└── resources/     # static content (deck PDF, agenda PDF, original zip)
```

The runnable program lives in `Demo/` — see [`Demo/README.md`](Demo/README.md) for a
full description of the four demos (ETL, Data Cleaning Pipeline, Text Preprocessing,
SQL Basics).

---

## How to run it — the commands

> All commands are run from inside the `Demo/` folder, and follow the setup in
> [`Demo/README.md`](Demo/README.md).

### 1. Create a virtual environment (recommended)
```bash
cd Demo
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```
Then open `Data Engineering for AI Applications_Demos.ipynb` and run all cells
(`Cell > Run All`, or `Kernel > Restart & Run All`).

No internet connection is required — every demo runs fully offline.

### Running the SQL demo standalone (optional)
The notebook creates `training.db` for you, but you can also run the SQL directly:
```bash
sqlite3 training.db < sql_commands.sql          # run all queries in one pass
# or work interactively:
sqlite3 training.db
sqlite> .tables
sqlite> SELECT * FROM customers LIMIT 5;
sqlite> .quit
```
Install the SQLite CLI first if needed — macOS: `brew install sqlite`,
Windows: `winget install SQLite.SQLite`, Debian/Ubuntu: `sudo apt-get install sqlite3`.

> **Note:** files like `raw_orders.csv`, `clean_orders.csv`, `training.db`, and
> `cleaning_change_log.csv` are **generated** when the notebook/SQL runs — they're
> safe to delete and will be recreated on the next run.
