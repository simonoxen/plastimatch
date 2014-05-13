#!/usr/bin/env python

"""
This script analyzes the seg_dice.csv mabs file and compute statistic on it.
Author: Paolo Zaffino  (p.zaffino@unicz.it)
Rev 1
"""

import argparse
import numpy as np
import sys

## Parser settings
parser = argparse.ArgumentParser(description='MABS seg_dice.csv analyzer')
parser.add_argument('--input', help='Mabs seg_dice.csv file', type=str, required=True)
parser.add_argument('--structures', help='Subset of structures to analyze. String as "stru1 stru2"', type=str, required=False)
parser.add_argument('--thresholds', help='Subset of thresholds to analyze. String as "thr1 thr2"', type=str, required=False)
args = parser.parse_args()

# Read data from file
seg_dice = open(args.input)
raw_lines = [l.strip().split(",") for l in seg_dice.readlines()]
seg_dice.close()

# Structure of the list (variable, index):
#
# atlas 0, parms 1, struct 2, rho 3, sigma 4, 5th_percsim 5, thresh 6, dice 7, tp 8
# tn 9, fp 10, fn 11, hd 12, 95hd 13, ahd 14, bhd 15, 95bhd 16, abhd 17

# Remove variable name from data, keep just the values
data_as_str = []
for line in raw_lines:
    data_as_str.append([s.split("=")[-1] for s in line])

# Leave string type as string, and convert numbers in float 
data = []
for d in data_as_str:
    item = []
    for ind, itm in enumerate(d):
        if ind < 3: item.append(itm)
        else: item.append(float(itm))
    data.append(item)

# Seek which structures and thresholds are present inside the file
thresholds = set([i[6] for i in data])
if args.thresholds != None:
    selected_thresholds = thresholds.intersection([float(i) for i in args.thresholds.split(" ")])
else:
    selected_thresholds = thresholds

structures = set([i[2] for i in data])
if args.structures != None:
    selected_structures = structures.intersection(args.structures.split(" "))
else:
    selected_structures = structures

# Create data structures
# eg:
# stats_left_parotid (dict) -
#                            | --> dice_04 (list)
#                            | --> dice_05 (list)
#                            | --> b_avg_dist_04 (list)
#                            | --> b_avg_dist_05 (list)
#                            | --> b_95_dist_04 (list)
#                            | --> b_95_dist_05 (list)
for structure in structures:
    vars()["stats_%s" % structure] = {}
    for threshold in thresholds:
        threshold_str = str(threshold).replace(".", "")
        vars()["stats_%s" % structure]["dice_%s" % threshold_str] = []
        vars()["stats_%s" % structure]["b_avg_dist_%s" % threshold_str] = []
        vars()["stats_%s" % structure]["b_95_dist_%s" % threshold_str] = []

# Insert data inside the structure
for d in data:
    threshold_str = str(d[6]).replace(".", "")
    vars()["stats_%s" % d[2]]["dice_%s" % threshold_str].append(d[7])
    vars()["stats_%s" % d[2]]["b_avg_dist_%s" % threshold_str].append(d[17])
    vars()["stats_%s" % d[2]]["b_95_dist_%s" % threshold_str].append(d[16])

# Medians and percentiles
for d in data:
    threshold_str = str(d[6]).replace(".", "")
    vars()["median_dice_%s_%s" % (d[2], threshold_str)] = np.median(vars()["stats_%s" % d[2]]["dice_%s" % threshold_str])
    vars()["95th_perc_dice_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["dice_%s" % threshold_str], 95)
    vars()["5th_perc_dice_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["dice_%s" % threshold_str], 5)

    vars()["median_b_avg_dist_%s_%s" % (d[2], threshold_str)] = np.median(vars()["stats_%s" % d[2]]["b_avg_dist_%s" % threshold_str])
    vars()["95th_perc_b_avg_dist_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["b_avg_dist_%s" % threshold_str], 95)
    vars()["5th_perc_b_avg_dist_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["b_avg_dist_%s" % threshold_str], 5)

    vars()["median_b_95_dist_%s_%s" % (d[2], threshold_str)] = np.median(vars()["stats_%s" % d[2]]["b_95_dist_%s" % threshold_str])
    vars()["95th_perc_b_95_dist_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["b_95_dist_%s" % threshold_str], 95)
    vars()["5th_perc_b_95_dist_%s_%s" % (d[2], threshold_str)] = np.percentile(vars()["stats_%s" % d[2]]["b_95_dist_%s" % threshold_str], 5)

# Print results
for structure in selected_structures:
    for threshold in selected_thresholds:
        threshold_str = str(threshold).replace(".", "")
        print("Structure = %s" % structure)
        print("  threshold = %s" % threshold)
        print("    dice = median %f  5th_perc %f  95th_perc %f" % (vars()["median_dice_%s_%s" % (structure, threshold_str)],
            vars()["5th_perc_dice_%s_%s" % (structure, threshold_str)],
            vars()["95th_perc_dice_%s_%s" % (structure, threshold_str)]))
        print("    average boundary distance = median %f  5th_perc %f  95th_perc %f" % (vars()["median_b_avg_dist_%s_%s" % (structure, threshold_str)],
            vars()["5th_perc_b_avg_dist_%s_%s" % (structure, threshold_str)],
            vars()["95th_perc_b_avg_dist_%s_%s" % (structure, threshold_str)]))
        print("    95th percentile boundary distance = median %f  5th_perc %f  95th_perc %f" % (vars()["median_b_95_dist_%s_%s" % (structure, threshold_str)],
            vars()["5th_perc_b_95_dist_%s_%s" % (structure, threshold_str)],
            vars()["95th_perc_b_95_dist_%s_%s" % (structure, threshold_str)]))

