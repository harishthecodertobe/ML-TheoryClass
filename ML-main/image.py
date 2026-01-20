import pandas as pd
def image_to_dataframe(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = f.read()
    return pd.DataFrame({"path": [path], "bytes": [data]})
img_path = r"<location>/health.jpeg"
csv_output = r"<location>/health.jpeg"
df = image_to_dataframe(img_path)
df.to_csv(csv_output, index=False)
print("CSV saved at:", csv_output)
