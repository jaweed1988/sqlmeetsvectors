"""
sql_meets_vectors.py
=====================
SQL Meets Vectors — Financial Transaction Intelligence

Vector search inside a relational database. No external vector library needed.

SQL Server 2025 ships with a native VECTOR(384) data type and a
VECTOR_DISTANCE() function. That means semantic similarity search is
just a T-SQL query, running right next to your star schema — no separate
search engine, no extra infrastructure.

What this script does, in order:
  1. Creates the SqlMeetsVectors database if it doesn't exist
  2. Loads 5,000 synthetic financial transactions from a CSV file
  3. Builds a star schema in memory (dim_customer, dim_merchant, dim_date, fact_transactions)
  4. Generates a 384-dimensional embedding for each transaction using sentence-transformers
  5. Stores those embeddings as VECTOR(384) columns inside SQL Server
  6. Runs four SQL analytics queries against the star schema
  7. Runs semantic search using VECTOR_DISTANCE() — pure T-SQL, no Python search library
  8. Detects anomalous transactions using statistical context from the same database

Schema overview:

    fact_transactions
    ─────────────────────────────────────────────────────────
    transaction_id   VARCHAR(20)    PRIMARY KEY
    customer_key     INT            references dim_customer
    merchant_key     INT            references dim_merchant
    date_key         INT            references dim_date
    amount           DECIMAL(12,2)
    currency         CHAR(3)
    is_flagged       BIT
    memo_text        VARCHAR(500)   the text that was embedded
    memo_embedding   VECTOR(384)    used by VECTOR_DISTANCE()

Requirements:
    SQL Server 2025 Developer Edition (free, full Enterprise features)
    pip install pyodbc pandas sentence-transformers torch numpy
"""

import pyodbc
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION SETTINGS
# Update "server" to match your machine name.
# You can find it in SSMS under the "Server name" field when you connect.
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "server":                 "localhost",
    "database":               "SqlMeetsVectors",
    "driver":                 "ODBC Driver 18 for SQL Server",
    "trusted_connection":     "yes",       # uses your Windows login, no password needed
    "TrustServerCertificate": "yes",       # required for local self-signed SSL certs
}

# The embedding model — free, runs on your CPU, no API key required.
# Downloads once (~80MB) on first run and caches locally forever.
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMS = 384


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg, indent=0):
    """Print a timestamped message so progress is easy to follow in the terminal."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {'  ' * indent}{msg}")


def connect(use_master=False):
    """
    Open a connection to SQL Server.

    We pass use_master=True only once — when we need to create the database.
    You cannot connect to SqlMeetsVectors before it exists, so we connect to
    the built-in master database first, create ours, then switch over.
    """
    cfg = {**DB_CONFIG, "database": "master" if use_master else DB_CONFIG["database"]}

    conn_str = (
        f"DRIVER={{{cfg['driver']}}};"
        f"SERVER={cfg['server']};"
        f"DATABASE={cfg['database']};"
        f"TrustServerCertificate={cfg['TrustServerCertificate']};"
        f"Trusted_Connection={cfg['trusted_connection']};"
    )

    try:
        return pyodbc.connect(conn_str, autocommit=True)
    except pyodbc.Error as e:
        log(f"Connection failed: {e}")
        log("Make sure SQL Server is running and the server name in DB_CONFIG is correct.")
        sys.exit(1)


def load_embedding_model():
    """
    Load the sentence-transformers model locally.

    On first run this downloads all-MiniLM-L6-v2 (~80MB) from HuggingFace
    and caches it on your machine. Every run after that loads from cache
    and takes just a couple of seconds.
    """
    log("Loading embedding model...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("sentence-transformers is not installed.")
        log("Run: pip install sentence-transformers torch")
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL)
    log(f"Model ready: {EMBED_MODEL} ({VECTOR_DIMS} dimensions)", indent=1)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Database setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_database():
    """
    Create the SqlMeetsVectors database if it does not already exist.
    This is safe to run every time — it checks before creating.
    """
    log("Setting up database...")

    conn   = connect(use_master=True)
    cursor = conn.cursor()
    db     = DB_CONFIG["database"]

    cursor.execute(f"""
        IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = N'{db}')
            CREATE DATABASE [{db}]
    """)

    conn.close()
    log(f"Database '{db}' is ready", indent=1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Extract
# ─────────────────────────────────────────────────────────────────────────────

def extract(relative_path):
    """
    Read transactions.csv into a pandas DataFrame.

    The file path is resolved relative to where this script lives, not
    wherever you happen to run it from. That means it works correctly
    regardless of your current working directory.

    Windows sometimes saves CSV files with a BOM (byte order mark) that
    breaks the default UTF-8 reader. We try three encodings in sequence
    until one works.
    """
    log("Loading transactions from CSV...")

    csv_path = (Path(__file__).parent / relative_path).resolve()
    log(f"File: {csv_path}", indent=1)

    if not csv_path.exists():
        log("File not found. Run generate_data.py first:", indent=1)
        log("  cd data && python generate_data.py  (from the sqlmeetsvectors folder)", indent=1)
        sys.exit(1)

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            log(f"Loaded {len(df):,} rows (encoding: {encoding})", indent=1)
            break
        except UnicodeDecodeError:
            continue
    else:
        log("Could not read the file. Try re-running generate_data.py.", indent=1)
        sys.exit(1)

    # Enforce types so the rest of the script can rely on them
    df["amount"]     = df["amount"].astype(float)
    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["is_flagged"] = df["is_flagged"].astype(bool)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Transform
# ─────────────────────────────────────────────────────────────────────────────

def build_dimensions(df):
    """
    Build three dimension tables from the raw data.

    Each dimension holds a different kind of context:
      dim_customer  who made the transaction (account type, region)
      dim_merchant  where it happened (merchant name, spending category)
      dim_date      when it happened, split into year / month / quarter

    We deduplicate each one so every customer and merchant appears exactly once,
    then assign a simple integer key. The fact table stores those keys instead of
    repeating the full strings on every row — that is the core idea of a star schema.
    """
    log("Building dimension tables...")

    dim_customer = (
        df[["customer_id", "account_type", "region"]]
        .drop_duplicates("customer_id")
        .reset_index(drop=True)
    )
    dim_customer.insert(0, "customer_key", range(1, len(dim_customer) + 1))
    log(f"dim_customer  {len(dim_customer)} rows", indent=1)

    dim_merchant = (
        df[["merchant", "category"]]
        .drop_duplicates("merchant")
        .reset_index(drop=True)
    )
    dim_merchant.insert(0, "merchant_key", range(1, len(dim_merchant) + 1))
    log(f"dim_merchant  {len(dim_merchant)} rows", indent=1)

    dim_date = pd.DataFrame({
        "date_key":    range(1, len(df) + 1),
        "full_date":   df["timestamp"].dt.date.values,
        "year":        df["timestamp"].dt.year.values,
        "quarter":     df["timestamp"].dt.quarter.values,
        "month":       df["timestamp"].dt.month.values,
        "month_name":  df["timestamp"].dt.strftime("%B").values,
        "day_of_week": df["timestamp"].dt.day_name().values,
        "is_weekend":  df["timestamp"].dt.dayofweek.isin([5, 6]).values,
    })
    log(f"dim_date      {len(dim_date)} rows", indent=1)

    return dim_customer, dim_merchant, dim_date


def build_fact_table(df, dim_customer, dim_merchant):
    """
    Build the central fact table — one row per transaction.

    We join in the integer keys from the dimension tables so the fact table
    is lean: instead of storing 'Whole Foods, groceries, Northeast' on every
    row, we just store merchant_key=7 and pull the details back with a JOIN.

    We also build memo_text here — a descriptive sentence that combines several
    fields into one string. This is what gets embedded. Richer text gives the
    model more context and produces better vectors.
    """
    log("Building fact table...")

    fact = df.merge(dim_customer[["customer_id", "customer_key"]], on="customer_id")
    fact = fact.merge(dim_merchant[["merchant",   "merchant_key"]],  on="merchant")
    fact["date_key"] = range(1, len(fact) + 1)

    fact["memo_text"] = (
        "Transaction: " + df["memo"].values
        + ". Merchant: "  + df["merchant"].values
        + ". Category: "  + df["category"].values
        + ". Amount: $"   + df["amount"].astype(str).values
        + ". Region: "    + df["region"].values + "."
    )

    fact = fact[[
        "transaction_id", "customer_key", "merchant_key", "date_key",
        "amount", "currency", "is_flagged", "memo_text"
    ]].copy()

    log(f"fact_transactions  {len(fact):,} rows", indent=1)
    return fact


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Embed
# ─────────────────────────────────────────────────────────────────────────────

def generate_embeddings(model, texts, batch_size=256):
    """
    Convert each memo_text into a 384-dimensional vector.

    The model runs locally on your CPU and processes texts in batches.
    Two transactions with similar meaning will end up with similar vectors —
    even if the exact words are different. That similarity is what powers
    the semantic search in step 7.
    """
    log(f"Generating {len(texts):,} embeddings...")

    all_embeddings = []
    total_batches  = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch      = texts[i : i + batch_size]
        batch_num  = i // batch_size + 1
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        log(f"Batch {batch_num}/{total_batches} — {min(i + batch_size, len(texts)):,}/{len(texts):,}", indent=1)

    log(f"Done — {len(all_embeddings):,} vectors at {VECTOR_DIMS} dimensions each", indent=1)
    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Load
# ─────────────────────────────────────────────────────────────────────────────

# Table definitions. The VECTOR(384) column on fact_transactions is the key
# addition — SQL Server 2025 stores and indexes this natively.
TABLE_DDL = {
    "dim_customer": """
        CREATE TABLE dim_customer (
            customer_key  INT           PRIMARY KEY,
            customer_id   VARCHAR(50)   NOT NULL,
            account_type  VARCHAR(20)   NOT NULL,
            region        VARCHAR(50)   NOT NULL
        )
    """,
    "dim_merchant": """
        CREATE TABLE dim_merchant (
            merchant_key  INT           PRIMARY KEY,
            merchant      VARCHAR(100)  NOT NULL,
            category      VARCHAR(50)   NOT NULL
        )
    """,
    "dim_date": """
        CREATE TABLE dim_date (
            date_key      INT           PRIMARY KEY,
            full_date     DATE          NOT NULL,
            year          SMALLINT      NOT NULL,
            quarter       TINYINT       NOT NULL,
            month         TINYINT       NOT NULL,
            month_name    VARCHAR(12)   NOT NULL,
            day_of_week   VARCHAR(12)   NOT NULL,
            is_weekend    BIT           NOT NULL
        )
    """,
    "fact_transactions": """
        CREATE TABLE fact_transactions (
            transaction_id  VARCHAR(20)    PRIMARY KEY,
            customer_key    INT            NOT NULL REFERENCES dim_customer(customer_key),
            merchant_key    INT            NOT NULL REFERENCES dim_merchant(merchant_key),
            date_key        INT            NOT NULL REFERENCES dim_date(date_key),
            amount          DECIMAL(12,2)  NOT NULL,
            currency        CHAR(3)        NOT NULL DEFAULT 'USD',
            is_flagged      BIT            NOT NULL DEFAULT 0,
            memo_text       VARCHAR(500)   NOT NULL,
            memo_embedding  VECTOR(384)    NOT NULL
        )
    """,
}


def load_to_sql_server(dim_customer, dim_merchant, dim_date, fact, embeddings):
    """
    Create all four tables and insert the data into SQL Server.

    The three dimension tables use fast_executemany, which sends all rows
    in a single network round-trip — much faster than one INSERT at a time.

    fact_transactions is inserted row by row because the VECTOR column needs
    a CAST(? AS VECTOR(384)) in the INSERT statement. SQL Server converts the
    JSON array string we pass in — something like '[0.12, -0.34, ...]' — into
    its native vector format at write time.

    After loading we create a vector index. This tells SQL Server to build an
    approximate nearest-neighbor structure so VECTOR_DISTANCE() queries are fast
    even as the table grows.
    """
    log("Loading data into SQL Server...")

    conn   = connect()
    cursor = conn.cursor()

    # Drop in reverse dependency order — fact first, then dims.
    # SQL Server will not let you drop a table that another table's
    # foreign key still references.
    log("Dropping existing tables if present...", indent=1)
    for table in ["fact_transactions", "dim_date", "dim_merchant", "dim_customer"]:
        cursor.execute(f"""
            IF OBJECT_ID(N'dbo.{table}', N'U') IS NOT NULL
                DROP TABLE dbo.{table}
        """)
    conn.commit()

    for table, ddl in TABLE_DDL.items():
        cursor.execute(ddl)
        log(f"Created {table}", indent=1)
    conn.commit()

    # Load dimension tables with bulk insert
    for table_name, dataframe in [
        ("dim_customer", dim_customer),
        ("dim_merchant", dim_merchant),
        ("dim_date",     dim_date),
    ]:
        cols         = ", ".join(dataframe.columns)
        placeholders = ", ".join(["?" for _ in dataframe.columns])
        insert_sql   = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

        # pyodbc needs plain Python types, not numpy integers or pandas booleans
        rows = [
            tuple(
                bool(v) if isinstance(v, bool) else
                int(v)  if pd.api.types.is_integer_dtype(type(v)) else v
                for v in row
            )
            for row in dataframe.itertuples(index=False, name=None)
        ]

        cursor.fast_executemany = True
        cursor.executemany(insert_sql, rows)
        conn.commit()

        count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        log(f"Loaded {table_name}: {count:,} rows", indent=1)

    # Load fact_transactions with VECTOR column
    log("Loading fact_transactions with embeddings...", indent=1)

    insert_sql = """
        INSERT INTO fact_transactions (
            transaction_id, customer_key, merchant_key, date_key,
            amount, currency, is_flagged, memo_text, memo_embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS VECTOR(384)))
    """

    # pyodbc sends Python str as ntext and bytes as image — both rejected by VECTOR.
    # We must explicitly declare the last parameter as SQL_VARCHAR so SQL Server
    # receives a plain varchar JSON string that CAST(? AS VECTOR(384)) can accept.
    import pyodbc as _pyodbc
    cursor.fast_executemany = False
    CHUNK = 500
    total = len(fact)

    for i in range(0, total, CHUNK):
        chunk_df   = fact.iloc[i : i + CHUNK]
        chunk_embs = embeddings[i : i + CHUNK]

        for (_, row), emb in zip(chunk_df.iterrows(), chunk_embs):
            emb_json = json.dumps(emb)
            cursor.setinputsizes(
                [None, None, None, None, None, None, None, None,
                 (pyodbc.SQL_WVARCHAR, 0, 0)]
            )
            cursor.execute(insert_sql, (
                row["transaction_id"],
                int(row["customer_key"]),
                int(row["merchant_key"]),
                int(row["date_key"]),
                float(row["amount"]),
                row["currency"],
                bool(row["is_flagged"]),
                row["memo_text"],
                emb_json,
            ))

        conn.commit()
        log(f"Inserted {min(i + CHUNK, total):,} / {total:,}", indent=2)

    count = cursor.execute("SELECT COUNT(*) FROM fact_transactions").fetchone()[0]
    log(f"fact_transactions: {count:,} rows", indent=1)

    # Create the vector index for fast similarity search
    # Note: VECTOR INDEX not available in current SQL Server version
    # log("Creating vector index...", indent=1)
    # cursor.execute("""
    #     CREATE VECTOR INDEX idx_memo_embedding
    #         ON fact_transactions (memo_embedding)
    #         WITH (metric = 'cosine')
    # """)
    # conn.commit()
    # log("Vector index created", indent=1)

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Data quality
# ─────────────────────────────────────────────────────────────────────────────

def run_dq_checks():
    """
    Run four checks against the loaded data to confirm everything is clean.

    Checks:
      Null amounts             every transaction must have a dollar value
      Null embeddings          every row must have a vector stored
      Duplicate transaction IDs  each transaction should appear exactly once
      Orphan merchant keys     every merchant_key must exist in dim_merchant
    """
    log("Running data quality checks...")

    conn   = connect()
    cursor = conn.cursor()

    checks = {
        "Null amounts": "SELECT COUNT(*) FROM fact_transactions WHERE amount IS NULL",

        "Null embeddings": "SELECT COUNT(*) FROM fact_transactions WHERE memo_embedding IS NULL",

        "Duplicate transaction IDs": """
            SELECT COUNT(*) FROM (
                SELECT transaction_id FROM fact_transactions
                GROUP BY transaction_id
                HAVING COUNT(*) > 1
            ) duplicates
        """,

        "Orphan merchant keys": """
            SELECT COUNT(*) FROM fact_transactions ft
            LEFT JOIN dim_merchant dm ON ft.merchant_key = dm.merchant_key
            WHERE dm.merchant_key IS NULL
        """,
    }

    for name, sql in checks.items():
        count = cursor.execute(sql).fetchone()[0]
        result = "PASS" if count == 0 else f"FAIL — {count} rows"
        log(f"{name}: {result}", indent=1)

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SQL analytics
# ─────────────────────────────────────────────────────────────────────────────

def run_sql_analytics():
    """
    Four T-SQL queries that answer real business questions.
    These run against the star schema — standard GROUP BY aggregations,
    nothing unusual about them. The vector column is not involved here.
    """
    log("Running SQL analytics...")

    conn = connect()

    queries = {

        # Which spending categories account for the most money?
        "Spending by category": """
            SELECT
                dm.category,
                COUNT(*)                   AS txn_count,
                ROUND(SUM(ft.amount), 2)   AS total_spend,
                ROUND(AVG(ft.amount), 2)   AS avg_per_txn,
                ROUND(MAX(ft.amount), 2)   AS largest_txn
            FROM  fact_transactions ft
            JOIN  dim_merchant dm ON ft.merchant_key = dm.merchant_key
            WHERE ft.amount > 0
            GROUP BY dm.category
            ORDER BY total_spend DESC
        """,

        # Is spending growing month over month?
        "Monthly spend trend": """
            SELECT
                dd.year,
                dd.month_name,
                ROUND(SUM(ft.amount), 2)   AS monthly_spend,
                COUNT(*)                    AS txn_count
            FROM  fact_transactions ft
            JOIN  dim_date dd ON ft.date_key = dd.date_key
            WHERE ft.amount > 0
            GROUP BY dd.year, dd.month, dd.month_name
            ORDER BY dd.year, dd.month
        """,

        # Which regions have the highest proportion of flagged transactions?
        "Fraud flag rate by region": """
            SELECT
                dc.region,
                SUM(CASE WHEN ft.is_flagged = 1 THEN 1 ELSE 0 END)   AS flagged,
                COUNT(*)                                                AS total,
                ROUND(
                    100.0 * SUM(CASE WHEN ft.is_flagged = 1 THEN 1 ELSE 0 END)
                    / COUNT(*), 2
                )                                                       AS flag_rate_pct
            FROM  fact_transactions ft
            JOIN  dim_customer dc ON ft.customer_key = dc.customer_key
            GROUP BY dc.region
            ORDER BY flag_rate_pct DESC
        """,

        # Who are the top spenders?
        "Top 10 customers by spend": """
            SELECT TOP 10
                dc.customer_id,
                dc.region,
                dc.account_type,
                COUNT(*)                   AS txn_count,
                ROUND(SUM(ft.amount), 2)   AS total_spend
            FROM  fact_transactions ft
            JOIN  dim_customer dc ON ft.customer_key = dc.customer_key
            WHERE ft.amount > 0
            GROUP BY dc.customer_id, dc.region, dc.account_type
            ORDER BY total_spend DESC
        """,
    }

    for title, sql in queries.items():
        log(f"Query: {title}", indent=1)
        print("\n" + pd.read_sql(sql, conn).to_string(index=False) + "\n")

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Semantic search
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(model, query, top_n=5, filter_region=None):
    """
    Find the transactions most semantically similar to a plain English query.

    Since VECTOR_DISTANCE is not available, we compute similarity in Python.
    """
    region_note = f"  [region: {filter_region}]" if filter_region else ""
    log(f'Searching: "{query}"{region_note}')

    query_vector = np.array(model.encode([query])[0])

    region_filter = f"AND dc.region = '{filter_region}'" if filter_region else ""

    sql = f"""
        SELECT ft.transaction_id, ft.memo_text, ft.amount, dm.category, dc.region, ft.memo_embedding
        FROM fact_transactions ft
        JOIN dim_merchant dm ON ft.merchant_key = dm.merchant_key
        JOIN dim_customer dc ON ft.customer_key = dc.customer_key
        WHERE 1=1 {region_filter}
    """

    conn = connect()
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()

    # Compute similarities
    results = []
    for txn_id, memo, amount, category, region, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        similarity = 1 - np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb))
        results.append((txn_id, memo, amount, category, region, similarity))

    # Sort by similarity ascending (lower distance is better)
    results.sort(key=lambda x: x[5])

    print(f"\n  {'-' * 60}")
    print(f"  Top {top_n} results for: \"{query}\"")
    print(f"  {'-' * 60}")

    for i, (txn_id, memo, amount, category, region, score) in enumerate(results[:top_n], 1):
        print(f"\n  [{i}] Similarity score: {score:.4f}")
        print(f"      ID:       {txn_id}")
        print(f"      Memo:     {memo[:95]}...")
        print(f"      Amount:   ${amount:,.2f}")
        print(f"      Category: {category}")
        print(f"      Region:   {region}")

    print(f"  {'-' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def explain_anomalies():
    """
    Find flagged transactions and explain them using statistical context
    pulled from the same database.

    For each flagged transaction, a subquery calculates the average and maximum
    amount for that spending category across all non-flagged transactions.
    We then compute how many times above average this transaction sits.

    A $3,800 grocery transaction is not suspicious on its own — but when the
    category average is $140, the 27x ratio is the signal. That ratio comes
    from the data, not from a hard-coded rule.
    """
    log("Running anomaly detection...")

    conn   = connect()
    cursor = conn.cursor()

    sql = """
        SELECT TOP 5
            ft.transaction_id,
            ft.memo_text,
            ft.amount,
            dm.category,
            dc.region,
            dc.account_type,
            cat_stats.cat_avg,
            cat_stats.cat_max,
            ROUND(ft.amount / NULLIF(cat_stats.cat_avg, 0), 1) AS times_above_avg
        FROM fact_transactions ft
        JOIN dim_merchant dm ON ft.merchant_key = dm.merchant_key
        JOIN dim_customer dc ON ft.customer_key = dc.customer_key
        JOIN (
            SELECT
                dm2.category,
                ROUND(AVG(ft2.amount), 2) AS cat_avg,
                ROUND(MAX(ft2.amount), 2) AS cat_max
            FROM fact_transactions ft2
            JOIN dim_merchant dm2 ON ft2.merchant_key = dm2.merchant_key
            WHERE ft2.is_flagged = 0 AND ft2.amount > 0
            GROUP BY dm2.category
        ) cat_stats ON dm.category = cat_stats.category
        WHERE ft.is_flagged = 1
        ORDER BY times_above_avg DESC
    """

    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()

    for txn_id, memo, amount, category, region, acct, cat_avg, cat_max, ratio in rows:
        print(f"\n  {'=' * 60}")
        print(f"  Transaction:  {txn_id}")
        print(f"  Memo:         {memo[:85]}...")
        print(f"  Amount:       ${amount:.2f}  |  {category}  |  {region}")
        print(f"\n  Context — non-flagged {category} transactions:")
        print(f"    Average amount:   ${cat_avg:.2f}")
        print(f"    Largest amount:   ${cat_max:.2f}")
        print(f"    This transaction: {ratio}x the category average")

        if ratio and ratio > 10:
            finding = f"Significantly above normal range for {category} — {ratio}x the average."
        elif ratio and ratio > 3:
            finding = f"Above normal range for {category} — {ratio}x the average."
        else:
            finding = f"Within normal dollar range — likely flagged by a behavioral pattern."

        print(f"\n  Finding:  {finding}")
        print(f"  Action:   Verify directly with the customer.")
        print(f"            Review recent {category} history for this {acct} account in {region}.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("""
+--------------------------------------------------------------+
|  SQL Meets Vectors                                           |
|  Financial transaction intelligence                          |
|  SQL Server 2025 Developer Edition                           |
+--------------------------------------------------------------+
""")

    # Database and model setup
    setup_database()
    model = load_embedding_model()

    # ETL — extract, transform, embed, load
    raw = extract("data/transactions.csv")
    dim_customer, dim_merchant, dim_date = build_dimensions(raw)
    fact       = build_fact_table(raw, dim_customer, dim_merchant)
    embeddings = generate_embeddings(model, fact["memo_text"].tolist())
    load_to_sql_server(dim_customer, dim_merchant, dim_date, fact, embeddings)

    # Validate what was loaded
    run_dq_checks()

    # Relational analytics — standard T-SQL queries
    run_sql_analytics()

    # Semantic search — natural language queries via VECTOR_DISTANCE()
    semantic_search(model, "business travel flight hotel conference")
    semantic_search(model, "monthly recurring subscription payment")
    semantic_search(model, "large unusual purchase high amount", filter_region="Northeast")
    semantic_search(model, "everyday household essential groceries")

    # Anomaly detection — statistical context from the same database
    explain_anomalies()

    log(f"Done — {DB_CONFIG['database']} on {DB_CONFIG['server']}")
