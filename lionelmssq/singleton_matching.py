import polars as pl
import tqdm as tqdm
from lionelmssq.masses import ELEMENT_MASSES, PHOSPHATE_LINK_MASS, MASSES
from clr_loader import get_mono
rt = get_mono()

PPM_TOLERANCE = 10

#Calculate ion mass from monoisotopic mass of each nucleoside in the reference table
UNIQUE_MASSES = (
    (MASSES.group_by("monoisotopic_mass", maintain_order=True)
     .agg(pl.col("nucleoside").unique())
    )
    .with_columns(pl.col("monoisotopic_mass")
                  .add(PHOSPHATE_LINK_MASS - ELEMENT_MASSES["H+"]) #Add phosphate adduct and subtract one proton
                  .alias("theoretical_mz")
                  )
                ).sort("theoretical_mz")

#Compute for max and min theoretical m/z with 2*PPM_TOLERANCE excess
THEO_MZ_MAX = UNIQUE_MASSES["theoretical_mz"].max() + (UNIQUE_MASSES["theoretical_mz"].max()*2*PPM_TOLERANCE)/(1e6)
THEO_MZ_MIN = UNIQUE_MASSES["theoretical_mz"].min() - (UNIQUE_MASSES["theoretical_mz"].min()*2*PPM_TOLERANCE)/(1e6)

def generate_mz_df(bunch):
    scan_id = int(bunch.scan_id.split("scan=")[-1])

    b_peak_index = []
    b_intensity = []
    b_mz = []
    
    if len(bunch.peaks)>0:
        for i in range(len(bunch.peaks)):
            peak_mz = bunch.peaks[i].mz
            #Limit peak m/z search by only considering ion masses within the theoretical bounds
            if THEO_MZ_MIN <= peak_mz < THEO_MZ_MAX:
                b_peak_index.append(i)
                b_intensity.append(bunch.peaks[i].intensity)
                b_mz.append(peak_mz)
        mz_df = pl.DataFrame(data = {"scan_id": scan_id,
                                        "peak_index": b_peak_index,
                                        "intensity": b_intensity,
                                        "mz": b_mz})
        return mz_df
    else:
        return None

def match_singletons(df_mz):
    """
    Match observed m/z from .RAW file to theoretical m/z from reference mass table.

    Parameters
    ----------
    df_mz : polars DataFrame
        DataFrame containing observed/experimental m/z.

    Returns
    -------
    df_matches : polars dataframe
        Dataframe containing list of candidate nucleosides obtained by matching singletons
    """

    #Group observed m/z together on the cluster index, aggregate by taking the mean of the m/z
    df_mz_agg = df_mz.group_by("ppm_group").agg(observed_mz = pl.col("mz").mean(),
                                                count = pl.len()).sort("observed_mz")
    
    #Match mean observed m/z to theoretical m/z from the reference table 
    df_mz_agg = (df_mz_agg.sort("observed_mz")
                          .join_asof(UNIQUE_MASSES.sort("theoretical_mz"),
                                     left_on = "observed_mz",
                                     right_on = "theoretical_mz",
                                     strategy = "nearest"))
    
    #Compute mass error between observed and theoretical m/z
    df_mz_agg = df_mz_agg.sort("observed_mz").with_columns((1e6*abs(pl.col("observed_mz")
                                                                    -pl.col("theoretical_mz"))
                                                                    /pl.col("observed_mz"))
                                            .fill_null(0).fill_nan(0)
                                            .lt(PPM_TOLERANCE).alias("is_match"))
    
    #Take mass error that is less than PPM_TOLERANCE and flatten list of candidate singletons
    df_matches = (
        df_mz_agg.filter(pl.col("is_match"))
                 .select(["nucleoside", "count"])
                 .sort("count", descending = True)
                 .explode("nucleoside"))
    
    return df_matches

def match_singletons_only(raw_file_read):
    """
    Full pipeline to match observed m/z from .RAW file to theoretical m/z from reference mass table without deisotoping first.

    Parameters
    ----------
    raw_file_read : ms_deisotope.data_source.thermo_raw_net.ThermoRawLoader
        Thermo RAW iterator.

    Returns
    -------
    df_matches : polars dataframe
        Dataframe containing list of candidate nucleosides obtained by matching singletons
    """

    #Initialize the raw_file_read iterator while ungrouping the MS1 from the MS2 scans
    raw_file_read.make_iterator(grouped=False)
    #Initialize the first scan from the iterator
    bunch = next(raw_file_read)

    mz_list = []
    for _ in tqdm.tqdm(range(len(raw_file_read)-1), desc = "Deisotoping MS2 scans"):
        if bunch.ms_level == 2: #Only consider MS2 scans

            #Centroid the scans
            bunch.pick_peaks()
        
            mz_df = generate_mz_df(bunch)
            if mz_df is not None:
                mz_list.append(mz_df)

        #Move on to the next scan
        bunch = next(raw_file_read)

    #Post-processing for df_mz
    #Generate cluster indices when m/z is within PPM_TOLERANCE of each other
    df_mz = pl.concat(mz_list)
    df_mz = df_mz.sort("mz").with_columns((1e6*abs(pl.col("mz").shift(1)
                                                    -pl.col("mz"))
                                                    /pl.col("mz").shift(1))
                                        .fill_null(0).fill_nan(0)
                                        .gt(PPM_TOLERANCE)
                                        .cum_sum().alias("ppm_group"))
    
    df_matches = match_singletons(df_mz)
    return df_matches