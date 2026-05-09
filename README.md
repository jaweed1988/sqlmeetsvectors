# SQL Meets Vectors

Vector search inside a relational database. No external vector library needed.

SQL Server 2025 Developer Edition ships with a native `VECTOR(384)` data type and a `VECTOR_DISTANCE()` function. That means you can run semantic similarity search as a T-SQL query, right next to your relational star schema, without Pinecone, ChromaDB, Weaviate, or any other dedicated vector database.

This project demonstrates that end to end using 5,000 synthetic financial transactions.


## What it does

The pipeline runs eight steps in sequence:

1. Creates the `SqlMeetsVectors` database if it does not already exist
2. Loads 5,000 synthetic financial transactions from a CSV file
3. Builds a star schema in memory: `dim_customer`, `dim_merchant`, `dim_date`, `fact_transactions`
4. Generates a 384-dimensional embedding for each transaction using `sentence-transformers` locally (no API key, no internet required after first run)
5. Stores those embeddings as `VECTOR(384)` columns inside SQL Server
6. Runs four SQL analytics queries against the star schema
7. Runs semantic search using `VECTOR_DISTANCE()` in pure T-SQL
8. Detects anomalous transactions using statistical context from the same database


## Database schema

```
fact_transactions
----------------------------------------------------------
transaction_id    VARCHAR(20)    PRIMARY KEY
customer_key      INT            references dim_customer
merchant_key      INT            references dim_merchant
date_key          INT            references dim_date
amount            DECIMAL(12,2)
currency          CHAR(3)
is_flagged        BIT
memo_text         VARCHAR(500)   the text that was embedded
memo_embedding    VECTOR(384)    used by VECTOR_DISTANCE()
```

The three dimension tables hold descriptive context:

| Table | Contents |
|---|---|
| `dim_customer` | customer ID, account type, region |
| `dim_merchant` | merchant name, spending category |
| `dim_date` | full date, year, quarter, month, day of week |


## Project structure

```
sql-meets-vectors/
    assets/
        sqlmeetsvectorsarchitecturediagram.png   architecture overview diagram
    data/
        generate_data.py        generates transactions.csv (5,000 rows)
        transactions.csv        5,000 synthetic financial transactions
    sql_meets_vectors.py        the main pipeline
    requirements.txt
    README.md
```


## Requirements

SQL Server 2025 Developer Edition (free). Download from https://www.microsoft.com/en-us/sql-server/sql-server-downloads and choose Developer. It installs as a default instance so you connect with `localhost`.

Python 3.11 or later.

ODBC Driver 18 for SQL Server. On Windows this is bundled with SQL Server. On Linux or macOS see https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/jaweed1988/sqlmeetsvectors.git
cd sqlmeetsvectors
```

### 2. Install Python dependencies

CPU-only PyTorch is recommended unless you have a GPU. It is around 250MB vs 2GB for the full build.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Update the server name

Open `sql_meets_vectors.py` and update `DB_CONFIG` with your machine name. You can find it in SSMS under the Server name field when you connect.

```python
DB_CONFIG = {
    "server": "YOUR-MACHINE-NAME",   # change this
    ...
}
```

### 4. Generate the data

```bash
cd data
python generate_data.py
cd ..
```

This creates `data/transactions.csv` with 5,000 synthetic financial transactions.

### 5. Run the pipeline

```bash
python sql_meets_vectors.py
```

On first run, `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~80MB) and caches it locally. Every run after that skips the download and takes under a minute for the embedding step.


## How semantic search works

When you call `semantic_search(model, "business travel flight hotel conference")`:

1. The query string is embedded into a 384-dimensional vector using the same model that embedded the transactions
2. SQL Server's `VECTOR_DISTANCE('cosine', ...)` computes the distance between the query vector and every `memo_embedding` in the table
3. `ORDER BY distance ASC` returns the closest matches first

The result is a ranked list of transactions ordered by semantic similarity, not keyword match. A transaction with the memo "Flight to Chicago for annual conference" will score highly for "business travel" even though the word "business" never appears in the memo.

Because the vectors live in the same table as the relational columns, you can combine vector similarity with standard SQL filters in a single query:

```sql
SELECT TOP 5
    ft.transaction_id,
    ft.memo_text,
    ROUND(
        1 - VECTOR_DISTANCE('cosine', ft.memo_embedding, CAST(? AS VECTOR(384))),
        4
    ) AS similarity_score
FROM fact_transactions ft
JOIN dim_customer dc ON ft.customer_key = dc.customer_key
WHERE dc.region = 'Northeast'
ORDER BY VECTOR_DISTANCE('cosine', ft.memo_embedding, CAST(? AS VECTOR(384))) ASC
```

And in SSMS, since `memo_embedding` is natively `VECTOR(384)`, no extra cast is needed on the column side:

```sql
DECLARE @query_vec VECTOR(384) = CAST('[...]' AS VECTOR(384));

SELECT TOP 5
    ft.transaction_id,
    ft.memo_text,
    ft.amount,
    dm.category,
    dc.region,
    VECTOR_DISTANCE('cosine', ft.memo_embedding, @query_vec) AS distance
FROM fact_transactions ft
JOIN dim_merchant dm ON ft.merchant_key = dm.merchant_key
JOIN dim_customer dc ON ft.customer_key = dc.customer_key
ORDER BY distance ASC;
```


## Sample output

### SQL analytics

```
Spending by category

 category     txn_count  total_spend  avg_per_txn  largest_txn
  shopping          498   621,432.10      1,248.86     4,499.12
    travel          501   599,213.44      1,196.03     4,490.55
healthcare          499   578,887.22      1,160.10     4,487.00
    dining          502   561,104.78      1,118.14     4,481.33
```

### Semantic search

```
Searching: "business travel flight hotel conference"

Top 5 results for: "business travel flight hotel conference"

[1] Similarity score: 0.9241
    ID:       TXN002341
    Memo:     Transaction: Flight to Chicago via Delta Airlines for annual conference...
    Amount:   $834.00  |  travel  |  Midwest

[2] Similarity score: 0.9108
    ID:       TXN000892
    Memo:     Transaction: Hotel stay at Marriott Hotels during product summit...
    Amount:   $412.50  |  travel  |  West
```

### Anomaly detection

```
Transaction:  TXN001847
Memo:         Transaction: Quick stop at Trader Joe's for produce and dairy...
Amount:       $3,891.22  |  groceries  |  Northeast

Context -- non-flagged groceries transactions:
  Average amount:   $143.22
  Largest amount:   $498.77
  This transaction: 27.2x the category average

Finding:  Significantly above normal range for groceries -- 27.2x the average.
Action:   Verify directly with the customer.
          Review recent groceries history for this credit account in Northeast.
```


## Tech stack

| | |
|---|---|
| Language | Python 3.11+ |
| Database | SQL Server 2025 Developer Edition |
| Connectivity | pyodbc + ODBC Driver 18 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) |
| Vector search | SQL Server VECTOR_DISTANCE() in pure T-SQL |
| Analytics | T-SQL star schema queries |
| Data | Synthetic, generated by generate_data.py |


## Extending the project

A few ideas for taking it further:

Add a `VECTOR INDEX` and measure the query speed difference with and without it. Swap `all-MiniLM-L6-v2` for a larger model like `all-mpnet-base-v2` (768 dimensions) and update `VECTOR(384)` to `VECTOR(768)`. Build a Streamlit front end that lets you type natural language queries and see results in real time. Add real transaction data from a public dataset such as the Kaggle Credit Card Fraud dataset. Explore SQL Server 2025's `VECTOR_SEARCH` stored procedure as an alternative to the manual `ORDER BY VECTOR_DISTANCE()` pattern.


## Acknowledgements

This project was built with the help of [Claude Code](https://claude.ai/code), Anthropic's AI coding assistant. Claude Code helped with writing and debugging the pipeline, generating the T-SQL queries, and structuring the README.


## License

MIT. Use freely, learn well.
