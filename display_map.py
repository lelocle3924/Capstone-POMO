import pandas as pd
import folium

# Read CSV file
csv_path = './clean_data/cleaned_Canlubang_TransportOrder.csv'  # Change 'your_file.csv' to your actual filename
df = pd.read_csv(csv_path)

# Assume columns are named 'latitude' and 'longitude'
lat_col = 'latitude'
lon_col = 'longitude'

# Center map on the first coordinate
center = [df[lat_col].iloc[0], df[lon_col].iloc[0]]
m = folium.Map(location=center, zoom_start=12)

# Add points to map
for _, row in df.iterrows():
    folium.Marker([row[lat_col], row[lon_col]]).add_to(m)

# Save map to HTML file
m.save('coordinates_map.html')
print("Map saved as coordinates_map.html")