"""Create an interactive HTML map of job stations and extraction boxes."""

import folium
from job_definitions_readin import JOB_DEFINITIONS


# Bounds are [west_lon, east_lon, south_lat, north_lat].
BOX_DEFINITIONS = {
    "RCA_1": ([-130.026663, -129.972718, 45.915716, 45.955691], "RCA"),
    "RCA_2": ([-129.766136, -129.738212, 45.807694, 45.842728], "RCA"),
    "RCA_3": ([-125.410172, -125.370190, 44.500876, 44.538380], "RCA"),
    "RCA_4": ([-125.148693, -125.144887, 44.567351, 44.572292], "RCA"),
    "RCA_5": ([-124.966230, -124.947191, 44.358269, 44.378032], "RCA"),
    "RCA_6": ([-124.309678, -124.303650, 44.632928, 44.638093], "RCA"),
    "CEA_1": ([-130.026512, -129.972880, 45.915716, 45.955691], "CEA"),
    "CEA_2": ([-129.766292, -129.738365, 45.807694, 45.842728], "CEA"),
    "CEA_3": ([-125.410141, -125.370155, 44.500876, 44.538380], "CEA"),
    "CEA_4": ([-125.148652, -125.144843, 44.567351, 44.572292], "CEA"),
    "CEA_5": ([-124.966180, -124.944600, 44.358269, 44.378930], "CEA"),
    "CEA_6": ([-124.309600, -124.301983, 44.632928, 44.638093], "CEA"),
    "glider_CE247": ([-125.972423, -124.099105, 44.294520, 45.018573], "glider"),
    "glider_CE311": ([-128.002100, -124.090214, 43.374255, 47.136371], "glider"),
    "glider_CE312": ([-127.994481, -124.150946, 43.345876, 47.135052], "glider"),
    "glider_CE319": ([-128.003233, -124.094811, 43.317032, 47.157970], "glider"),
    "glider_CE320": ([-128.004720, -124.094682, 43.296557, 47.132319], "glider"),
    "glider_CE326": ([-127.985857, -124.090363, 43.368176, 47.257975], "glider"),
    "glider_CE327": ([-127.994524, -124.088601, 43.430351, 48.277302], "glider"),
    "glider_CE381": ([-128.003097, -124.259125, 44.584162, 48.318096], "glider"),
    "glider_CE382": ([-128.005282, -124.103957, 43.382465, 48.071842], "glider"),
    "glider_CE383": ([-127.999521, -124.085420, 43.047500, 47.128229], "glider"),
    "glider_CE384": ([-128.008930, -124.166900, 44.304104, 47.081660], "glider"),
    "glider_CE386": ([-128.018376, -124.101999, 43.536917, 47.671559], "glider"),
    "glider_CE871": ([-127.996127, -124.300001, 44.381385, 44.908733], "glider"),
    "glider_CE917": ([-126.072364, -124.125683, 43.426230, 44.686111], "glider"),
    "glider_dfo_eva035": ([-128.702767, -128.077391, 51.342925, 51.727880], "glider"),
    "glider_osu033": ([-125.187129, -124.070020, 44.267890, 45.066013], "glider"),
}


def add_box_layers(station_map):
    colors = {"RCA": "red", "CEA": "purple", "glider": "blue"}
    groups = {}

    for box_name, (bounds, box_type) in BOX_DEFINITIONS.items():
        groups.setdefault(
            box_type,
            folium.FeatureGroup(name=f"{box_type} extraction boxes", show=True),
        )
        west, east, south, north = bounds
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=colors[box_type],
            weight=2,
            fill=True,
            fill_color=colors[box_type],
            fill_opacity=0.12,
            tooltip=box_name,
            popup=(
                f"<b>{box_name}</b><br>"
                f"West: {west}<br>East: {east}<br>"
                f"South: {south}<br>North: {north}"
            ),
        ).add_to(groups[box_type])

    for group in groups.values():
        group.add_to(station_map)


def create_map(output_file="job_stations_and_boxes_map.html"):
    station_map = folium.Map(
        location=[47.5, -123.5],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    colors = {
        "OCNMS_jobs": "blue",
        "ORCA_jobs": "green",
        "RCA_jobs": "red",
        "CEA_jobs": "purple",
        "ANeMoNe_jobs": "orange",
        "Hatchery_jobs": "darkred",
    }

    for job_name, stations in JOB_DEFINITIONS.items():
        group = folium.FeatureGroup(name=job_name, show=True)
        color = colors.get(job_name, "gray")

        for station_name, (longitude, latitude) in stations.items():
            folium.Marker(
                location=[latitude, longitude],
                tooltip=station_name,
                popup=(
                    f"<b>{station_name}</b><br>"
                    f"Job: {job_name}<br>"
                    f"Longitude: {longitude}<br>Latitude: {latitude}"
                ),
                icon=folium.Icon(color=color, icon="map-marker"),
            ).add_to(group)

        group.add_to(station_map)

    add_box_layers(station_map)
    folium.LayerControl(collapsed=False).add_to(station_map)
    station_map.save(output_file)
    print(f"Saved map to {output_file}")


if __name__ == "__main__":
    create_map()
