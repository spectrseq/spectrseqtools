import polars as pl
import tqdm as tqdm
from lionelmssq.masses import ELEMENT_MASSES, PHOSPHATE_LINK_MASS, MASSES
from clr_loader import get_mono
from dbscan1d.core import DBSCAN1D
from sklearn.metrics import silhouette_score

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
    scan_time = bunch.scan_time

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
                                     "scan_time": scan_time,
                                     "peak_index": b_peak_index,
                                     "intensity": b_intensity,
                                     "mz": b_mz})
        return mz_df
    else:
        return None

def cluster_score(scantimes):
    """
    Provide a score on how clustered each of the scan peaks by scan time, using DBSCAN and Silhouette scores

    Parameters
    ----------
    scantimes : polars Series
        Series containing scan times

    Returns
    -------
    score : float
        Silhouette score for the DBSCAN cluster of scan times
    """

    scan_arr = scantimes.sort().to_numpy()

    #Cluster scan times using 1D DBSCAN
    dbs = DBSCAN1D(eps = 0.5, min_samples = 10)
    clusters = dbs.fit_predict(scan_arr)

    X = scan_arr.reshape(-1, 1)
    if len(set(clusters)) == 1:
        #If there are only noise clusters (cluster = -1), set score to -1.0
        if list(set(clusters))[0] == -1:
            score = -1.0
        #If there is only one cluster, set score to 0.0
        else:
            score = 0.0
    else:
        #Compute silhouette score otherwise
        score = silhouette_score(X, clusters)

    return score

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

    #Match m/z to theoretical m/z from the reference table 
    df_mz = (
        df_mz.sort("mz")
        .join_asof(UNIQUE_MASSES.sort("theoretical_mz"),
                   left_on = "mz",
                   right_on = "theoretical_mz",
                   strategy = "nearest"
                   )
            )
    
    #Compute mass error between observed and theoretical m/z
    df_mz = (
        df_mz.sort("mz")
        .with_columns((1e6*abs(pl.col("mz")
                               -pl.col("theoretical_mz"))
                               /pl.col("mz")
                        ).fill_null(0).fill_nan(0)
                        .lt(PPM_TOLERANCE).alias("is_match")
                    )
        .filter(pl.col("is_match"))
        .sort(["nucleoside", "scan_time"])
            )
    
    df_mz_agg = (
        df_mz.group_by("nucleoside")
        .map_groups(lambda x: pl.DataFrame({"nucleoside": x["nucleoside"][0],
                                            "cluster_score": cluster_score(x["scan_time"]),
                                            "count": len(x["nucleoside"])
                                            })
                    )
                )

    #Take mass error that is less than PPM_TOLERANCE and flatten list of candidate singletons
    df_matches = (
        df_mz_agg.filter(pl.col("cluster_score") >= 0)
                 .select(["nucleoside", "count", "cluster_score"])
                 .sort("count", descending = True)
                 #.explode("nucleoside")
                 )
    
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
    for _ in tqdm.tqdm(range(len(raw_file_read)-1), desc = "Matching singletons only (no deisotoping)"):
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