import surf as surf
import surf_analysis as surfA
import surf_inputs as surfIN
import surf_insitu as surfIS

import numpy as np
import os
import astropy.units as u
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
import re
from sunpy.coordinates import sun
import helio_coords as hcoords

#===============================================================================
# <codecell> Import ICME-CME data

# Function to convert a date string to a datetime object
def convert_to_datetime(date_str):
    # List of possible date formats
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
    # If no format matches, return None
    return None
    
# Function to convert Time_Error from string to timedelta
def convert_to_timedelta(time_str):
    try:
        return pd.to_timedelta(time_str)
    except ValueError:
        return None

# Function to get earth's latitude
def get_earth_lat(dt):
    """
    A function to return Earth latitude for a given date, in radians
 
    Parameters
    ----------
    dt : datetime
 
    Returns
    -------
    E_lat: Earth latitude, with astropy units of radians
 
    """
    cr, cr_lon_init = surfIN.datetime2surfinputs(dt)
    
    # Use the HUXt ephemeris data to get Earth lat over the CR
    # ========================================================
    dummymodel = surf.SURF(v_boundary=np.ones(128)*400*(u.km/u.s), simtime=0.1*u.day,
                         cr_num=cr,cr_lon_init=cr_lon_init, lon_out=0.0*u.deg)
    # retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    # get average Earth lat
    E_lat = np.nanmean(earth.lat_c)
    E_lat = E_lat.to(u.deg)  # Convert to degrees
    return E_lat
    

# --- Load data ----------------------------------------------------------------
# Read in Met Office Cone CME files
project_dirs = surf._setup_dirs_()
crpath = os.path.join(project_dirs['input'],'(I)CMEs.csv')

# Load the CSV file into a DataFrame
crlist = pd.read_csv(crpath,nrows=2)

# Convert date columns to datetime objects using datetime library
date_columns = ['Time_21.5', 'Disturbance_Time']
for column in date_columns:
    crlist[column] = crlist[column].apply(lambda x: convert_to_datetime(x) if isinstance(x, str) else None)

# compute 21.5-215 transit time
for irow in range(0, len(crlist)):
    crlist.loc[irow,'tt_21'] = crlist.loc[irow,'Disturbance_Time'] - crlist.loc[irow,'Time_21.5']
# Convert  from timedelta to days
crlist['tt_21'] = crlist['tt_21'].apply(lambda x: x.days + x.seconds / 86400 if isinstance(x, timedelta) else None)

# Add a new column with carrington rotation number of CME time
crlist['cr_num'] = crlist['Time_21.5'].apply(lambda dt: surfIN.datetime2surfinputs(dt)[0])
crlist['cr_lon_init'] = crlist['Time_21.5'].apply(lambda dt: surfIN.datetime2surfinputs(dt)[1])
crlist['earth_lat'] = crlist['Time_21.5'].apply(lambda dt: get_earth_lat(dt))

#===============================================================================
# <codecell> Create OMNI inner boundary conditions

# --- Load ICME  ---------------------------------------------------------------

#HUXt run parameters
dt_scale = 4
rmin = 21.5*u.solRad
rmax = 230*u.solRad #outer boundary for HUXt runs

#===============================================================================
# <codecell> Functions for spheroidal and fixed duration ConeCMEs
def spheroidal(onecme,model):
    '''Solve HUXt using a spheroidal cone CME'''
    
    cme = surf.ConeCME(t_launch=0.0 * u.day,
                    longitude=onecme['lon'] * u.deg,
                    latitude=onecme['lat'] * u.deg,
                    initial_height=rmin,
                    width=2.0 * onecme['Ang_rad'] * u.deg,
                    v=onecme['CME_V']* (u.km / u.s),
                    thickness=0.0 * u.solRad,
                    cme_expansion=False,
                    cme_fixed_duration=False)

    model.solve([cme])
    cme_out = model.cmes[0]

    # Compute the transit time
    stats = cme_out.compute_arrival_at_body('ACE')
    tt = stats['t_transit'].value
    arrival_time = stats['t_arrive']
    
    # Find the arrival speed within 1 day of arrival
    ace_series = surfA.get_observer_timeseries(model, observer='ACE')
    mask = (Time(ace_series['time']) >= arrival_time) & (
            Time(ace_series['time']) <= arrival_time + 1 * u.day)
    v_1au = ace_series.loc[mask, 'vsw'].max()
    
    return tt, v_1au
    
def fixed_duration(onecme,model,duration,rmin=rmin):
    '''Solve HUXt using a fixed pulse duration cone CME'''
    
    cme = surf.ConeCME(t_launch=0.0 * u.day,
                    longitude=onecme['lon'] * u.deg,
                    latitude=onecme['lat'] * u.deg,
                    initial_height=rmin,
                    width=2.0 * onecme['Ang_rad'] * u.deg,
                    v=onecme['CME_V'] * (u.km / u.s),
                    thickness=0.0 * u.solRad,
                    cme_expansion=False,
                    cme_fixed_duration=True,
                    fixed_duration=duration * 60 * 60 * u.s)

    model.solve([cme])
    cme_out = model.cmes[0]

    # Compute the transit time
    stats = cme_out.compute_arrival_at_body('ACE')
    tt = stats['t_transit'].value
    arrival_time = stats['t_arrive']
    
    # Find the arrival speed within 1 day of arrival
    ace_series = surfA.get_observer_timeseries(model, observer='ACE')
    mask = (Time(ace_series['time']) >= arrival_time) & (
            Time(ace_series['time']) <= arrival_time + 1 * u.day)
    v_1au = ace_series.loc[mask, 'vsw'].max()
    
    return tt, v_1au
    
#===============================================================================
# <codecell> Run model

# Initialize Data Storage
transit_time = []
arrival_speed = []

# First three rows
transit_time.append(['Angular_Width'] + list(crlist['Ang_rad']))
arrival_speed.append(['Angular_Width'] + list(crlist['Ang_rad']))
transit_time.append(['Velocity'] + list(crlist['CME_V']))
arrival_speed.append(['Velocity'] + list(crlist['CME_V']))
transit_time.append(['Observed'] + list(crlist['tt_21']))
arrival_speed.append(['Observed'] + list(crlist['V_max']))

durations = np.arange (10.0,11.0,0.5)#np.arange(1.0, 30.5, 0.5)  # CME durations in hours

# Pre-initialize rows
sph_tt_row = ['Spheroidal']
sph_av_row = ['Spheroidal']
tt_rows = [[f"tt_{d:.1f}h"] for d in durations]
av_rows = [[f"tt_{d:.1f}h"] for d in durations]
    
# Iterate over all CMEs in crlist
for _, onecme in crlist.iterrows():
    
    #========================================================
    # Obtain OMNI boundary conditions at 21.5 rS

    ftime = onecme['Time_21.5']
    
    model = surfIS.omniSURF_reconstruction(ftime, ftime +timedelta(days=27),
                                           rmin=rmin, rmax=rmax,
                                           dt_scale=4, dt=1*u.day,
                                           run_2d=False,
                                           solver='huxt')
    
    #========================================================
    # Spheroidal cone cme
    sph_tt_val = np.nan
    sph_av_val = np.nan
    sph_tt_val, sph_av_val = spheroidal(onecme, model)
    sph_tt_row.append(sph_tt_val)
    sph_av_row.append(sph_av_val)
    
    #========================================================
    # Fixed duration cone cmes
    for i, duration in enumerate(durations):
        tt_val = np.nan
        av_val = np.nan
        tt_val, av_val = fixed_duration(onecme, model, duration)
        tt_rows[i].append(tt_val)
        av_rows[i].append(av_val)
    
transit_time.append(sph_tt_row)
arrival_speed.append(sph_av_row)
transit_time.extend(tt_rows)
arrival_speed.extend(av_rows)

#==============================================================
# Save results

pd.DataFrame(transit_time).to_csv(os.path.join(project_dirs['output'],"cme_transit_time.csv"), index=False, header=False)
pd.DataFrame(arrival_speed).to_csv(os.path.join(project_dirs['output'],"cme_arrival_speed.csv"), index=False, header=False)
