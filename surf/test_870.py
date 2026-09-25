import os
import numpy as np
import pandas as pd
import astropy.units as u

from datetime import datetime, timedelta
from astropy.time import Time

import surf
import surf_analysis as surfA
import surf_inputs as surfIN
import surf_insitu as surfIS


# ============================================================
# Settings
# ============================================================

RMIN = 21.5 * u.solRad
RMAX = 230 * u.solRad

TARGET_SPEED = 870.0


# ============================================================
# Helper: convert dates from the CME catalogue
# ============================================================

def convert_to_datetime(date_str):

    date_formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H%M',
        '%Y/%m/%d %H%M(%S)',
        '%Y/%m/%d %H%M(S)',
        '%Y/%m/%d %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M(S)',
        '%d/%m/%Y %H:%M'
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue

    return None


# ============================================================
# Load CME catalogue
# ============================================================

project_dirs = surf._setup_dirs_()

crpath = os.path.join(
    project_dirs['input'],
    '(I)CMEs.csv'
)

crlist = pd.read_csv(crpath)


# Convert dates
for column in ['Time_21.5', 'Disturbance_Time']:

    crlist[column] = crlist[column].apply(
        lambda x: convert_to_datetime(x)
        if isinstance(x, str)
        else None
    )


# ============================================================
# Select the 870 km/s CME
# ============================================================

matches = np.isclose(
    crlist['CME_V'].astype(float),
    TARGET_SPEED
)

if not np.any(matches):
    raise ValueError(
        f"No CME with speed {TARGET_SPEED} km/s found."
    )

cme_row = crlist.loc[matches].iloc[0]


print("\n")
print("=" * 70)
print("870 km/s CME DIAGNOSTIC")
print("=" * 70)

print(f"CME speed       : {cme_row['CME_V']} km/s")
print(f"Longitude       : {cme_row['lon']} deg")
print(f"Latitude        : {cme_row['lat']} deg")
print(f"Angular halfwidth: {cme_row['Ang_rad']} deg")
print(f"Full width      : {2*cme_row['Ang_rad']} deg")
print(f"Time at 21.5 Rs : {cme_row['Time_21.5']}")
print(f"Observed arrival: {cme_row['Disturbance_Time']}")


# ============================================================
# Create the SAME background model as main_fcst.py
# ============================================================

ftime = cme_row['Time_21.5']

print("\nCreating OMNI background...")

model = surfIS.omniSURF_forecast(
    ftime,
    rmin=RMIN,
    rmax=RMAX,
    dt_scale=4,
    run_2d=False,
    solver='huxt'
)

print("Background model created.")


# ============================================================
# Create the SAME spheroidal CME
# ============================================================

cme = surf.ConeCME(
    t_launch=0.0 * u.day,
    longitude=cme_row['lon'] * u.deg,
    latitude=cme_row['lat'] * u.deg,
    initial_height=RMIN,
    width=2.0 * cme_row['Ang_rad'] * u.deg,
    v=cme_row['CME_V'] * (u.km / u.s),
    thickness=0.0 * u.solRad,
    cme_expansion=False,
    cme_fixed_duration=False
)


# ============================================================
# Run model
# ============================================================

print("\nRunning CME simulation...")

model.solve([cme])

cme_out = model.cmes[0]

print("Simulation complete.")


# ============================================================
# First: use the normal SURF arrival calculation
# ============================================================

stats = cme_out.compute_arrival_at_body('ACE')


print("\n")
print("=" * 70)
print("NORMAL SURF ARRIVAL RESULT")
print("=" * 70)

print(f"Hit          : {stats['hit']}")
print(f"Arrival time : {stats['t_arrive']}")
print(f"Transit time : {stats['t_transit']}")
print(f"Arrival radius: {stats['r']}")
print(f"Arrival speed : {stats['v']}")
print(f"Hit longitude: {stats['lon']}")


# ============================================================
# Reproduce compute_arrival_at_body() diagnostics
# ============================================================

print("\n")
print("=" * 70)
print("RECONSTRUCTING t_front / r_front")
print("=" * 70)


# Get ACE positions at the model output times
ace = model.get_observer('ACE')

# Earth/ACE radial distance and longitude
arrive_rad = ace.r
arrive_lon = ace.lon


# CME longitude offset used by SURF
# This follows the logic in compute_arrival_at_body()
arrive_lon_test = arrive_lon.copy()

id_low = arrive_lon_test < 0 * u.deg
id_high = arrive_lon_test > 180 * u.deg

if np.any(id_low):
    arrive_lon_test[id_low] += 360 * u.deg
elif np.any(id_high):
    arrive_lon_test[id_high] -= 360 * u.deg


t_front = []
r_front = []
v_front = []

hit_index = None


# ============================================================
# Walk through every CME timestep
# ============================================================

for i, coord in cme_out.coords.items():

    if len(coord['r']) == 0:
        continue

    r_cme = coord['r']
    v_cme = coord['v']
    lon_cme = coord['lon']

    front_id = coord['front_id'] == 1.0

    r_cme = r_cme[front_id]
    v_cme = v_cme[front_id]
    lon_cme = lon_cme[front_id] - cme_out.longitude


    if not np.any(front_id):
        continue


    # --------------------------------------------------------
    # Single longitude case
    # --------------------------------------------------------

    if len(lon_cme) == 1:

        longitude_difference = (
            arrive_lon_test[i] - lon_cme[0]
        ).to(u.deg).value

        # Wrap difference to [-180, 180]
        longitude_difference = (
            longitude_difference + 180
        ) % 360 - 180

        longitude_hit = abs(longitude_difference) <= 1.5

        if longitude_hit:

            t_front.append(coord['time'].jd)
            r_front.append(r_cme[0])
            v_front.append(v_cme[0].value)

            # Print the FIRST longitude hit
            if len(t_front) == 1:

                print("\n*** FIRST LONGITUDE HIT ***")

                print(
                    "Model time      :",
                    Time(coord['time'].jd, format='jd').isot
                )

                print(
                    "CME longitude   :",
                    lon_cme[0].to(u.deg)
                )

                print(
                    "ACE longitude   :",
                    arrive_lon_test[i].to(u.deg)
                )

                print(
                    "Longitude diff  :",
                    longitude_difference,
                    "deg"
                )

                print(
                    "CME front radius:",
                    r_cme[0].to(u.solRad)
                )

                print(
                    "ACE radius      :",
                    arrive_rad[i].to(u.solRad)
                )

                print(
                    "CME front speed :",
                    v_cme[0]
                )

                print(
                    "t_front[0]      :",
                    Time(t_front[0], format='jd').isot
                )


    # --------------------------------------------------------
    # Check radial crossing
    # --------------------------------------------------------

    if len(r_front) > 0:

        if r_front[-1] > arrive_rad[i]:

            hit_index = i

            # Interpolate exactly as SURF does
            t_arrive_jd = np.interp(
                arrive_rad[i].value,
                np.asarray(r_front) * r_front[-1].unit.to(
                    u.solRad
                ) if hasattr(r_front[-1], 'unit') else
                np.asarray([
                    x.to(u.solRad).value
                    if hasattr(x, 'unit') else x
                    for x in r_front
                ]),
                np.asarray(t_front)
            )

            t_arrive = Time(
                t_arrive_jd,
                format='jd'
            )

            t_front_0 = Time(
                t_front[0],
                format='jd'
            )

            t_transit = (
                t_arrive_jd - t_front[0]
            )

            print("\n*** RADIAL CROSSING DETECTED ***")

            print(
                "Crossing model time:",
                t_arrive.isot
            )

            print(
                "t_front[0]        :",
                t_front_0.isot
            )

            print(
                "t_arrive           :",
                t_arrive.isot
            )

            print(
                "t_transit          :",
                t_transit,
                "days"
            )

            print(
                "CME radius at crossing:",
                arrive_rad[i].to(u.solRad)
            )

            print(
                "Number of longitude-matched points:",
                len(t_front)
            )

            # Show the first few and last few points
            print("\nFirst longitude-matched points:")

            for j in range(min(5, len(t_front))):

                print(
                    f"  {j}: "
                    f"{Time(t_front[j], format='jd').isot}   "
                    f"r = "
                    f"{r_front[j].to(u.solRad) if hasattr(r_front[j], 'to') else r_front[j]}"
                )

            print("\nLast longitude-matched points:")

            start = max(0, len(t_front) - 5)

            for j in range(start, len(t_front)):

                print(
                    f"  {j}: "
                    f"{Time(t_front[j], format='jd').isot}   "
                    f"r = "
                    f"{r_front[j].to(u.solRad) if hasattr(r_front[j], 'to') else r_front[j]}"
                )

            break


# ============================================================
# Final diagnostic
# ============================================================

print("\n")
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if len(t_front) == 0:

    print("NO longitude-matched CME points were found.")

elif hit_index is None:

    print("Longitude-matched CME points were found,")
    print("but the CME did not cross the ACE radius.")

else:

    first_radius = (
        r_front[0].to(u.solRad)
        if hasattr(r_front[0], 'to')
        else r_front[0] * u.solRad
    )

    ace_radius = arrive_rad[hit_index].to(u.solRad)

    print(
        "t_front[0] =",
        Time(t_front[0], format='jd').isot
    )

    print(
        "r_front[0] =",
        first_radius
    )

    print(
        "t_arrive   =",
        t_arrive.isot
    )

    print(
        "ACE radius =",
        ace_radius
    )

    print()

    if first_radius > ace_radius:

        print(
            "!!! SUSPICIOUS CASE CONFIRMED !!!"
        )

        print(
            "The FIRST longitude-matched CME point is "
            "already beyond ACE."
        )

        print(
            "Therefore t_front[0] occurs after the "
            "CME has already crossed 1 AU."
        )

        print(
            "This can explain the near-zero transit time."
        )

    else:

        print(
            "The first longitude-matched point is "
            "inside ACE's orbit."
        )

        print(
            "So the zero/near-zero transit time must "
            "have another cause."
        )

print("=" * 70)
