#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:55:42 2026

Creating maps of the global, regional, and subregional location of Puget Sound 
for presentation and such.

@author: lara
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Approximate bounding box for Puget Sound
PS_LON_MIN, PS_LON_MAX = -123.2, -122.0
PS_LAT_MIN, PS_LAT_MAX = 47.0, 48.6


def add_base(ax, *, land_color="#e8e8e8", ocean_color="#cfe8f3", borders=False):
    ax.set_facecolor(ocean_color)
    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=-1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#333333", zorder=2)
    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#666666", zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.35, edgecolor="#888888", zorder=2)


def add_box(ax, linewidth=2.0, color="crimson"):
    box = mpatches.Rectangle(
        (PS_LON_MIN, PS_LAT_MIN),
        PS_LON_MAX - PS_LON_MIN,
        PS_LAT_MAX - PS_LAT_MIN,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.add_patch(box)


def add_gridlines(ax, labels=False, fontsize=8, major=False):
    gl = ax.gridlines(
        draw_labels=labels,
        linewidth=0.5 if major else 0.35,
        color="gray",
        alpha=0.65 if major else 0.55,
        linestyle="-" if major else "--",
    )
    if labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": fontsize}
        gl.ylabel_style = {"size": fontsize}


# 1. Global locator map
fig1 = plt.figure(figsize=(7, 5.4), constrained_layout=True)
ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax1.set_global()
add_base(ax1)
add_gridlines(ax1, major=True)
add_box(ax1, linewidth=2.2)
ax1.text(
    -155,
    45.8,
    "Puget Sound",
    fontsize=12,
    color="black",
    ha="center",
    transform=ccrs.PlateCarree(),
    zorder=6,
)
fig1.savefig("puget_sound_global.png", dpi=300, bbox_inches="tight")

# 2. Regional context map
fig2 = plt.figure(figsize=(7, 5.4), constrained_layout=True)
ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax2.set_extent([-145, -105, 30, 65], crs=ccrs.PlateCarree())
add_base(ax2, borders=True)
add_box(ax2, linewidth=2.2)
add_gridlines(ax2, labels=True)
ax2.text(
    -128.4,
    47.56,
    "Puget Sound",
    fontsize=16,
    color="black",
    ha="center",
    transform=ccrs.PlateCarree(),
    zorder=6,
)
fig2.savefig("puget_sound_regional.png", dpi=300, bbox_inches="tight")

# 3. Puget Sound zoom
fig3 = plt.figure(figsize=(7, 5.4), constrained_layout=True)
ax3 = fig3.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax3.set_extent([-124.0, -121.35, 46.55, 49.15], crs=ccrs.PlateCarree())
add_base(ax3, borders=True)
ax3.add_feature(
    cfeature.LAKES,
    facecolor="#cfe8f3",
    edgecolor="#4f8ca3",
    linewidth=0.45,
)
ax3.add_feature(cfeature.RIVERS, edgecolor="#4f8ca3", linewidth=0.35)
add_gridlines(ax3, labels=True)

# Outline the Puget Sound extent; mark and label Seattle.
add_box(ax3, linewidth=2.0, color="crimson")
ax3.plot(
    -122.3321,
    47.6062,
    marker="o",
    markersize=4.5,
    color="black",
    transform=ccrs.PlateCarree(),
    zorder=6,
)
ax3.text(
    -122.3,
    47.555,
    "Seattle",
    fontsize=14,
    color="black",
    transform=ccrs.PlateCarree(),
    zorder=6,
)
ax3.text(
    -123,
    48.65,
    "Puget Sound",
    fontsize=18,
    color="black",
    transform=ccrs.PlateCarree(),
    zorder=6,
)
fig3.savefig("puget_sound_zoom.png", dpi=300, bbox_inches="tight")

plt.show()
