import pandas as pd
import numpy as np
import glob
import os
import folium
import requests
import datetime

# --- Parameters ---
depot_ids = ['2501', '2522', '25P2']
max_customers_per_depot = 300  # Ceiling for customers per depot
osrm_url = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=false"

# --- Load depot coordinates ---
depot_master = pd.read_csv('./clean_data/DepotMaster.csv', dtype={'DepotID': str})
depot_coords = depot_master[depot_master['DepotID'].isin(depot_ids)][['DepotID', 'DepotName', 'Latitude', 'Longitude']].set_index('DepotID')
depot_centers = depot_coords[['Latitude', 'Longitude']].values

# --- Find all customer CSV files ---
customer_files = glob.glob('./clean_data/cleaned_*_TransportOrder.csv')

def osrm_distance(lat1, lon1, lat2, lon2):
    try:
        url = osrm_url.format(lon1, lat1, lon2, lat2)
        response = requests.get(url)
        data = response.json()
        if 'routes' in data and len(data['routes']) > 0:
            return data['routes'][0]['distance'] / 1000  # distance in km
        else:
            return np.inf
    except Exception:
        return np.inf

def assign_depot_osrm(lat, lon, depot_coords, depot_ids, cluster_sizes, max_customers_per_depot):
    # Compute OSRM distances to each depot
    dists = []
    for depot_id in depot_ids:
        depot_lat = depot_coords.loc[depot_id, 'Latitude']
        depot_lon = depot_coords.loc[depot_id, 'Longitude']
        dist = osrm_distance(lat, lon, depot_lat, depot_lon)
        dists.append(dist)
    sorted_depot_idx = np.argsort(dists)
    # Assign to nearest depot that is not full
    for depot_idx in sorted_depot_idx:
        depot_id = depot_ids[depot_idx]
        if cluster_sizes[depot_id] < max_customers_per_depot:
            cluster_sizes[depot_id] += 1
            return depot_id
    # If all depots are full, assign to depot with smallest size
    min_depot = min(cluster_sizes, key=cluster_sizes.get)
    cluster_sizes[min_depot] += 1
    return min_depot

# --- Process files and assign clusters ---
all_customers = []
cluster_sizes = {depot_id: 0 for depot_id in depot_ids}
for file in customer_files:
    df = pd.read_csv(file)
    df['AssignedDepotID'] = df.apply(
        lambda row: assign_depot_osrm(row['Latitude'], row['Longitude'], depot_coords, depot_ids, cluster_sizes, max_customers_per_depot),
        axis=1
    )
    all_customers.append(df)
    out_path = os.path.splitext(file)[0] + '_clustered.csv'
    df.to_csv(out_path, index=False)
    print(f"Clustered file saved: {out_path}")

all_customers_df = pd.concat(all_customers, ignore_index=True)

def display_cluster_counts(customers_df):
    counts = customers_df['AssignedDepotID'].value_counts()
    print("Total number of customers in each cluster:")
    for depot_id, count in counts.items():
        print(f"Depot {depot_id}: {count}")

display_cluster_counts(all_customers_df)

# --- Display clusters on map ---
colors = {'2501': 'red', '25P2': 'blue', '2522': 'purple'}
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

# Save map with date and time in filename
dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
map_filename = f'customer_clusters_map_{dt_str}.html'
m.save(map_filename)
print(f"Map saved as {map_filename}")