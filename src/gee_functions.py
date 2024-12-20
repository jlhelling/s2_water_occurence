import ee
import rasterio
from shapely.geometry import box

#------------------ HELPER FUNCTIONS ----------------------#

def apply_cloud_filtering(COL, CS = 0.65):
        """
        Apply cloud filtering to an Earth Engine ImageCollection using the Cloud Score Plus Collection.
        Args:
            COL (ee.ImageCollection): The input ImageCollection to be filtered.
            CS (float, optional): The cloud score threshold for filtering. Default is 0.65.
        Returns:
            ee.ImageCollection: The cloud-filtered ImageCollection.
        """

        # Load Cloud Score Plus Collection for cloud filtering, select only cs band
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').select('cs')

        # Function to apply the cloudscore threshold mask
        def apply_cloud_mask(img):
            return img.updateMask(img.select('cs').gte(CS))

        # link COLLECTION and CS+ to remove unclear pixels below cs-threshold
        COL_clean = COL.linkCollection(csPlus, ['cs']).map(apply_cloud_mask)

        return COL_clean


def filter_by_year_and_month(COL, year, month):
    """
    Filters an Earth Engine ImageCollection by a specific year and month.
    Args:
        COL (ee.ImageCollection): The ImageCollection to filter.
        year (int): The year to filter by.
        month (int): The month to filter by (1-12).
    Returns:
        ee.ImageCollection: The filtered ImageCollection containing images from the specified year and month.
    """
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    return COL.filterDate(start_date, end_date)

def combine_bands(img):
    """
    Combines the 'water' and 'NDWI' bands of an image, averages them, and adds the result as a new band named 'water_prob'.

    Args:
        img (ee.Image): The input image containing 'water' and 'NDWI' bands.
    Returns:
        ee.Image: The input image with an additional 'water_prob' band.
    """
    # Select 'water' and 'NDWI' bands, add them, and rename the result to 'water_prob'
    combined_band = img.select('water').add(img.select('NDWI')).divide(2).rename('water_prob')
    # Add the new 'water_prob' band to the original image
    return img.addBands(combined_band)


def calc_ndwi(img):
    """
    Calculate the Normalized Difference Water Index (NDWI) for an image.
    The resulting NDWI values are adjusted to a range of [0, 1] by adding 1 and dividing by 2.

    Args:
        img (ee.Image): The input image containing the necessary bands (B3 and B8).
    Returns:
        ee.Image: The input image with an additional band named 'NDWI' containing the
                  calculated NDWI values, and only the 'water' and 'NDWI' bands selected.
    """
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    adjusted_ndwi = ndwi.add(1).divide(2).rename('NDWI')
    return img.addBands(adjusted_ndwi).select(['water', 'NDWI'])

# create monthly and yearly water rasters from S2 collection with NDWI and DW
def get_monthly_water_occurence_yr(YEAR, AOI, WATER_THRESHOLD = 0.5, FILTER_CLOUDS = True, CLOUD_THRESHOLD = 0.70):
    """
    Extracts monthly water occurence frequency at the pixel-scale for a region of interest over a defined year. Water occurence is calculated for individual months. 
    Water is detected via a combined index from the S2-collection NDWI and DW water probability.
    A pixel is defined as water if the combined water probability value is greater than or equal to the threshold. Cloud filtering can be applied via the Cloudscore+ collection.
    The returned image contains the following bands:
    - 'freq_{1-12}': Water occurrence ratio for each month (e.g., 'freq_1' for January, 'freq_2' for February, etc.).
    - 'freq_year': Total yearly water occurrence ratio.
    
    Args:
        YEAR (str): The year to extract combined water probability values.
        AOI (ee.FeatureCollection): The region of interest.
        WATER_THRESHOLD (float, optional): The threshold value to define water pixels. Default is 0.5.
        FILTER_CLOUDS (bool, optional): Whether to apply cloud filtering. Default is False.
        CLOUD_THRESHOLD (float, optional): The cloud score threshold for filtering. Default is 0.65.
    Returns:
        ee.Image: A composite image containing monthly water occurrence frequency bands ('{0-11}_freq_month') and the yearly water occurrence band ('freq_year').
    """

    # Function to apply the cloudscore threshold mask
    def apply_water_mask(img):
        return img.updateMask(img.select('water_prob').gte(WATER_THRESHOLD))

    # Load S-2 Surface Reflectance Collection
    S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Load Dynamic World Collection
    DW = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')

    # Date filter
    date = YEAR + '-01-01'
    START = ee.Date(date) # oldest possible date: 2015-06-27
    END = START.advance(1,'year')

    # Spatial filter 
    ROI = AOI.geometry()

    # combine filters
    colFilter = ee.Filter.And(
        ee.Filter.bounds(ROI),
        ee.Filter.date(START, END))

    # filter collections
    dw_filtered = DW.filter(colFilter).select('water')
    s2_filtered = S2.filter(colFilter)

    col_filtered = dw_filtered.linkCollection(s2_filtered, ['B3','B8'])

    if col_filtered.size().getInfo() == 0:
        raise ValueError(f"No images found for the year {YEAR} in the specified region.")

    # apply cloud filtering with CloudScore+ collection
    if FILTER_CLOUDS: 
        col = apply_cloud_filtering(col_filtered, CS=CLOUD_THRESHOLD).select(['water', 'B3', 'B8'])
    else:
        col = col_filtered

    # Apply NDWI calculation directly on the collection
    col = col.map(calc_ndwi)

    # Apply the combine_bands function to each image in the collection
    col_combined = col.map(combine_bands)

    months = range(1, 13)  # Months from January to December    
    
    # create monthly occurence collection: 
    monthly_freqs_list = []

    # loop for each month over all years and add band to collection
    for month in months:
        # Convert YEAR string to integer
        year_int = int(YEAR) 
        
        monthly_col = filter_by_year_and_month(col_combined, year_int, month)
        
        # get valid and total values for selected month
        count_total = monthly_col.select('water_prob').count().rename(['count_total'])
        count_valid = monthly_col.select('water_prob').map(apply_water_mask).count().rename(['count_valid'])
            
        # calculate monthly frequency
        monthly_freq = count_valid.divide(count_total).rename(f'freq_month')

        # Mask frequencies without observations
        monthly_freq = monthly_freq.updateMask(count_total.gt(0))  

        # add monthly frequency to list
        monthly_freqs_list.append(monthly_freq)
            
    # Create ImageCollection of monthly frequencies
    monthly_freq_collection = ee.ImageCollection(monthly_freqs_list)
    
    # Calculate sum and count of observed months
    sum_monthly_freq = monthly_freq_collection.sum().rename('sum_monthly_freq')  # Pixel-wise sum of monthly frequencies
    count_observed_months = monthly_freq_collection.count().rename('count_observed_months')
    
    # Calculate month-specific frequency by dividing the sum of frequencies by the count of instances
    yearly_freq = sum_monthly_freq.divide(count_observed_months).rename('freq_year')

    # join bands and convert all bands to float
    composite = monthly_freq_collection.toBands().addBands(yearly_freq).clip(ROI).toFloat()

    return composite