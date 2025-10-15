import pandas as pd
import numpy as np
import glob
import os
import folium

# --- Load depot coordinates ---
depot_ids = ['2501', '2513', '25P2', '2522']
depot_master = pd.read_csv('./clean_data/DepotMaster.csv', dtype={'DepotID': str})
depot_coords = depot_master[depot_master['DepotID'].isin(depot_ids)][['DepotID', 'DepotName', 'Latitude', 'Longitude']].set_index('DepotID')
depot_centers = depot_coords[['Latitude', 'Longitude']].values

# --- Find all customer CSV files ---
customer_files = glob.glob('./clean_data/cleaned_*_TransportOrder.csv')

def assign_depot(lat, lon, depot_centers):
    # Compute distances to each depot
    dists = np.linalg.norm(depot_centers - np.array([lat, lon]), axis=1)
    return depot_ids[np.argmin(dists)]

# --- Assign clusters and collect all customers ---
all_customers = []
for file in customer_files:
    df = pd.read_csv(file)
    # Assign nearest depot
    df['AssignedDepotID'] = df.apply(lambda row: assign_depot(row['Latitude'], row['Longitude'], depot_centers), axis=1)
    all_customers.append(df)
    # Save result
    out_path = os.path.splitext(file)[0] + '_clustered.csv'
    df.to_csv(out_path, index=False)
    print(f"Clustered file saved: {out_path}")

all_customers_df = pd.concat(all_customers, ignore_index=True)
print(all_customers_df.head())

def display_cluster_counts(customers_df):
    counts = customers_df['AssignedDepotID'].value_counts()
    print("Total number of customers in each cluster:")
    for depot_id, count in counts.items():
        print(f"Depot {depot_id}: {count}")

display_cluster_counts(all_customers_df)

# --- Display clusters on map ---
colors = {'2501': 'red', '2513': 'blue', '25P2': 'green', '2522': 'purple'}
center_lat = depot_coords['Latitude'].mean()
center_lon = depot_coords['Longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Plot depots
for depot_id, row in depot_coords.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=f"Depot {depot_id}: {row['DepotName']}",
        icon=folium.Icon(color='black', icon='home')
    ).add_to(m)

# Plot customers
for depot_id in depot_ids:
    cluster = all_customers_df[all_customers_df['AssignedDepotID'] == depot_id]
    for _, row in cluster.iterrows():
        folium.CircleMarker(
            [row['Latitude'], row['Longitude']],
            radius=4,
            color=colors[depot_id],
            fill=True,
            fill_color=colors[depot_id],
            fill_opacity=0.7,
            popup=f"Customer {row['ShipToID']}<br>Depot: {depot_id}"
        ).add_to(m)

m.save('customer_clusters_map.html')
print("Map saved as customer_clusters_map.html")