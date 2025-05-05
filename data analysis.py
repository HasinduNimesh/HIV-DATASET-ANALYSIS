import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import tempfile
import zipfile
from urllib.request import urlopen
from io import BytesIO

# Create the figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Load cleaned data from CSV
df = pd.read_csv("data_clean/orphans_clean.csv")

# Drop unused columns
df = df.drop(columns=["type","indicator","value","lower","upper","unicef_region"], errors='ignore')
df = df.rename(columns={"country_region":"country"})

# Global trend plot
plt.figure(figsize=(8,4))
sns.lineplot(data=df.groupby("year")["orphans_0_17"].sum().reset_index(),
             x="year", y="orphans_0_17")
plt.title("Global Total AIDS Orphans (2000–2023)")
plt.ylabel("Number of orphans")
plt.tight_layout()
plt.savefig("figures/global_trend.png")
plt.close()

# Top 10 countries plot
top10 = df[df.year==2023].nlargest(10, "orphans_0_17")
plt.figure(figsize=(6,4))
sns.barplot(data=top10, y="country", x="orphans_0_17")
plt.title("Top 10 Countries by AIDS Orphans (2023)")
plt.tight_layout()
plt.savefig("figures/top10_2023.png")
plt.close()

# Distribution histogram
plt.figure(figsize=(6,4))
sns.histplot(df.orphans_0_17, log_scale=(False,True), bins=50)
plt.title("Distribution of Country-Year Orphan Counts")
plt.xlabel("Orphans (0–17)")
plt.tight_layout()
plt.savefig("figures/hist_orphans.png")
plt.close()

# Population scatterplot (if column exists)
if "population" in df.columns:
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="population", y="orphans_0_17", alpha=0.5)
    plt.xscale("log"); plt.yscale("log")
    plt.title("Orphans vs Population (2000–2023)")
    plt.tight_layout()
    plt.savefig("figures/scatter_pop.png")
    plt.close()
else:
    print("⚠️ Warning: 'population' column not found, skipping population scatterplot")
    
    # Alternative visualization: Top 5 countries trend
    plt.figure(figsize=(8,5))
    top5_countries = df.groupby('country')['orphans_0_17'].sum().nlargest(5).index
    top5_data = df[df['country'].isin(top5_countries)]
    sns.lineplot(data=top5_data, x="year", y="orphans_0_17", hue="country")
    plt.title("Orphans Trend in Top 5 Countries (2000–2023)")
    plt.tight_layout()
    plt.savefig("figures/top5_countries_trend.png")
    plt.close()

# World map plot using Natural Earth data 
try:
    print("Downloading Natural Earth data...")
    
    # A more reliable URL from the official NACIS CDN
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    
    from urllib.request import Request
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    print("Trying NACIS CDN URL...")
    req = Request(url, headers=headers)
    
    with urlopen(req) as response:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(BytesIO(response.read())) as z:
                z.extractall(tmpdir)
                # Find the shapefile
                shp_file = [f for f in os.listdir(tmpdir) if f.endswith('.shp')][0]
                world = gpd.read_file(os.path.join(tmpdir, shp_file))
    
    # Make sure we have the ISO code
    if 'ISO_A3' in world.columns:
        world = world.rename(columns={'ISO_A3': 'iso_a3'})
    
    # Create the map without using mapclassify
    map_df = world.merge(
        df[df.year==2023],
        left_on="iso_a3", right_on="iso3",
        how="left"
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Use simple choropleth without quantiles scheme
    map_df.plot(column="orphans_0_17", cmap="OrRd", legend=True, ax=ax,
                missing_kwds={"color":"lightgrey"})
    
    ax.set_title("AIDS Orphans by Country (2023)")
    ax.axis("off")
    fig.savefig("figures/map_2023.png", bbox_inches="tight")
    plt.close()
    
    print("✅ World map created successfully")
    
except Exception as e:
    print(f"⚠️ Error creating world map: {e}")
    print("Consider installing additional packages:")
    print("pip install mapclassify")
    print("\nOr download the Natural Earth data manually from:")
    print("https://www.naturalearthdata.com/downloads/110m-cultural-vectors/")
    
    # Create a simple alternative visualization instead with fixed seaborn syntax
    plt.figure(figsize=(10,6))
    
    # Updated barplot syntax to avoid FutureWarning
    top20 = df[df.year==2023].nlargest(20, "orphans_0_17")
    sns.barplot(
        data=top20,
        y="country", 
        x="orphans_0_17",
        hue="country",  # Assign to hue instead of using palette directly
        palette="OrRd_r",
        legend=False  # Hide the legend since it would be redundant
    )
    plt.title("Top 20 Countries by AIDS Orphans (2023)")
    plt.tight_layout()
    plt.savefig("figures/top20_2023.png")
    plt.close()
    print("✅ Created alternative Top 20 visualization instead")

print("✅ Analysis complete! Check the 'figures' directory for outputs.")