import pandas as pd

url = "https://example.com/some_page_with_table"
# Read all HTML tables on the page
tables = pd.read_html(url)

print(f"Found {len(tables)} tables")

# Pick the first table
df = tables[0]
print(df.head())

# Save to CSV
df.to_csv("scraped_table.csv", index=False)
