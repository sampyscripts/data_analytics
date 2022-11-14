import numpy as np #Numpy library import


def haversine_py(pickup_lat, pickup_long, dropoff_lat, dropoff_long, unit="mile"):
    '''
    Function to convert coordinates into measurable distances using Haversine method.
    '''
    try: #trying different measuring units
        if unit=="km":
            R = 6378.14
        elif unit=="m":
            R = 6371000
        elif unit=="ft":
            R = 20930000
        else:
            R = 3961.19
    except:
        print(f"Unit can only be in 'mile', 'm' or 'km'")
        
    rad_pickup_lat = np.radians(pickup_lat) #returns pick_up latitude in radians
    rad_dropoff_lat = np.radians(dropoff_lat) #returns drop_off latitude in radians
    rad_lat_change = np.radians(dropoff_lat -pickup_lat) #returns the difference between drop_off and pick_up latitudes in radians
    rad_long_change = np.radians(dropoff_long - pickup_long) #returns the difference between drop_off and pick_up longitudes in radians
    
    a = np.square(np.sin(rad_lat_change/2)) + (np.cos(rad_pickup_lat)*np.cos(rad_dropoff_lat)*
                                                          np.square(np.sin(rad_long_change/2)))
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    if __name__ == "__main__":
        return f"The distance between these coordinates is {R*c}{unit}"
    else:
        return R*c

