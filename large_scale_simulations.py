import argparse
import subprocess
import uuid
from time import time
import numpy as np
import hail as hl
import pandas as pd
from hail.utils import hadoop_exists

# parameters
h2g = 0.5
shoech_alpha = -0.45  # estimated alpha for Height from Shoech et al.
min_maf = 0.001  # MAF threshold for shoech equation
n = 150000  # simulation sample size
frac_causal = 0.01
path_pref = "" # path to store simulated data
tmp_path = "" # directory to store tmp files
bgen_imputed_path = "" # path to .bgen files containing all imputed variants
sample_file_path = "" # .sample file corresponding to the .bgen files in bgen_imputed_path
gwas_sample_path = "" # hail table containing QC'ed samples used in GWAS
variant_qc_path = "" # hail table containing QC'ed variants
phenotype_file_name = "" # tab-delimited phenotype file containing Height and FID columns
phenotype_file_name_ht = "" # hail table phenotype file containing same information as phenotype_file_name
gwas_cov = "" # hail table containing GWAS covariates
downsample_incl_path = "" # path to downsampled individual ID file

# per chrom h2g approximately proportional to chrom length
chr_lengths = np.array([
    248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717,
    133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285,
    58617616, 64444167, 46709983, 50818468, 156040895  # last is X chromosome
])
h2g_per_chrom = chr_lengths / chr_lengths.sum() * h2g


def checkpoint_tmp(hail_obj, tmppath=tmp_path, tmpname=None, overwrite=True):
    if tmpname is None:
        tmpname = str(uuid.uuid4())
    return hail_obj.checkpoint(tmppath + tmpname, overwrite=overwrite)


def index_import_bgen(bgen_path, *args, **kwargs):
    """Wrapper function to index bgen before importing if index file doesn't already exist"""
    if "index_file_map" in kwargs.keys():
        return hl.import_bgen(bgen_path, *args, **kwargs)
    if not hl.hadoop_exists(bgen_path.replace(".bgen", ".idx2")):
        print("Indexing bgen file before importing...")
        hl.index_bgen(bgen_path)
    return hl.import_bgen(bgen_path, *args, **kwargs)


def write_mt(chrom, overwrite):

    if hadoop_exists(f"{path_pref}/ukbb.150k.chr{chrom}.mt") and not overwrite:
        print(
            f"INFO: {path_pref}/ukbb.150k.chr{chrom}.mt already exists. "
            f"If you wish to overwrite it, please pass the --overwrite flag."
        )
        return

    else:
        print(f"INFO: Chromosome {chrom} - writing matrix table...")
        mt = hl.import_bgen(
            f"{bgen_imputed_path}/ukb_imp_chr{chrom}_v3.bgen",
            entry_fields=["dosage", "GP"],
            sample_file=f"{sample_file_path}"
        )

        # filter to gwas samples
        gwas_samples = hl.read_table(f"{gwas_sample_path}")
        mt = mt.filter_cols(hl.is_defined(gwas_samples[mt.s]))

        # apply variant_qc (for simulation): MAC > 10, HWE_P > 1e-10, info > 0.2
        variant_qc = hl.read_table(f"{variant_qc_path}")
        mt = mt.annotate_rows(variant_qc=variant_qc[mt.locus, mt.alleles].variant_qc)
        mt = mt.filter_rows(
            (mt.variant_qc.AC[0] > 10) &
            (mt.variant_qc.AC[1] > 10) &
            (mt.variant_qc.p_value_hwe > 1e-10)
        )

        mfi_ht = hl.import_table(f"{bgen_imputed_path}/ukb_mfi_chr{chrom}_v3.txt",
                                 impute=True,
                                 no_header=True)
        mfi_ht = mfi_ht.select(locus=hl.locus(str(chrom), mfi_ht.f2), alleles=[mfi_ht.f3, mfi_ht.f4], info=mfi_ht.f7)
        mfi_ht = mfi_ht.key_by(mfi_ht.locus, mfi_ht.alleles)

        mt = mt.annotate_rows(info=mfi_ht[mt.locus, mt.alleles].info)
        mt = mt.filter_rows(mt.info > 0.2)
        mt = mt.drop(mt.variant_qc, mt.info)

        # randomly draw probabilistic random genotypes based on GP - pGT is the "true" simulated genotype
        mt = mt.annotate_entries(u=hl.rand_unif(0, 1))
        mt = mt.annotate_entries(pGT=hl.int(hl.ceil(mt.u - mt.GP[0]) + hl.ceil(mt.u - mt.GP[0] - mt.GP[1])))
        mt = mt.drop(mt.u, mt.GP)

    # TODO: really this code block needs to be moved into the else clause
    #  temporarily here because I'm reading from bgens which were still at 360k in the previous version of this pipeline

    # downsample genotype matrix to n individuals - relies on all matrices having same order of samples (which they do)
    if n < mt.count_cols():
        mt = mt.head(None, n)
    else:
        print(f"WARNING: desired # of samples {n} > # of samples available {mt.count_cols()}. Not downsampling.")

    # annotate genotype means and std deviations
    stats = hl.agg.stats(mt.pGT)
    mt = mt.annotate_rows(mean_pGT=stats["mean"], stdev_pGT=stats["stdev"])

    # filter out monomorphic snps
    mt = mt.filter_rows(mt.stdev_pGT == 0, keep=False)

    # end TODO

    # write matrix
    mt = mt.checkpoint(f"{path_pref}/ukbb.150k.chr{chrom}.mt", overwrite=overwrite)
    mt.describe()
    print(f"INFO: wrote MatrixTable of size: {mt.count()}")


# cf. https://github.com/Nealelab/ukb_common/blob/master/saige/extract_vcf_from_mt.py
def gt_to_gp(mt, gt_location: str = "GT", gp_location: str = "GP"):
    return mt.annotate_entries(
        **{
            gp_location:
                hl.or_missing(
                    hl.is_defined(mt[gt_location]),
                    hl.map(lambda i: hl.if_else(mt[gt_location] == i, 1.0, 0.0),
                           hl.range(0, hl.triangle(hl.len(mt.alleles)))))
        })


def write_bgen(chrom):
    mt = hl.read_matrix_table(f"{path_pref}/ukbb.150k.chr{chrom}.mt")
    mt = gt_to_gp(mt, "pGT", "GP")
    hl.export_bgen(mt, f"{path_pref}/ukbb.150k.chr{chrom}", gp=mt.GP, varid=mt.rsid)


def sample_causal(df):
    """Sample a causal SNP and sample its effect from a dataframe representing a credible set in a single region"""

    # use normalized alphas to select causal SNP
    alphas = df["alpha"].to_numpy()
    alphas = alphas / alphas.sum()
    causal_idx = np.random.choice(np.arange(len(df)), p=alphas)

    # sample effect size from normal distribution with posterior mean and variance conditional on inclusion
    mean = df.iloc[causal_idx]["cond_mean"]
    sd = df.iloc[causal_idx]["cond_sd"]
    causal_effect = np.random.normal(loc=mean, scale=sd)
    df.iloc[causal_idx, df.columns.get_loc("large_beta")] = causal_effect
    df.iloc[causal_idx, df.columns.get_loc("large_gamma")] = True

    # return only the causal SNP as a single-row dataframe
    return df.iloc[[causal_idx]]


def sample_effects(chrom, sim_start, sim_end, overwrite, threshold_maf, standardize, h2g_ratio, scale):

    if hadoop_exists(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{sim_start}-{sim_end}.effect.ht") and not overwrite:
        print(
            f"INFO: {path_pref}/ukbb.150k.chr{chrom}.sim_{sim_start}-{sim_end}.effect.ht already exists. "
            f"If you wish to overwrite it, please pass the --overwrite flag."
        )
        return

    if h2g_ratio < 0:
        print(
            "WARNING: h2g ratio is not set or is invalid (negative). "
            "Sampling large effects from each SuSiE credible set"
        )

    print(f"INFO: Chromosome {chrom} - sampling effects...")

    # load genotype matrix
    mt = hl.read_matrix_table(f"{path_pref}/ukbb.150k.chr{chrom}.mt")

    if not hadoop_exists(f"{path_pref}/ukbb.150k.chr{chrom}.snps.ht") or overwrite:
        mt = mt.annotate_rows(AF=mt.mean_pGT / 2)
        if threshold_maf:
            mt = mt.annotate_rows(p_tilda=hl.min(hl.max(mt.AF, min_maf), 1 - min_maf))
        snps_ht = mt.rows()

        # load qc info about which SNPs are used in the GWAS -
        # used later to determine which simulated causal SNPs will be removed by QC
        qc_mt = hl.import_bgen(
            f"gs://ukbb-bolt/ukb_imp_qc/ukb_imp_qc_chr{chrom}_v3.bgen",
            entry_fields=['dosage', 'GP'],
            sample_file="gs://ukbb-bolt/ukb31063.sample",
            n_partitions=5000
        )
        gwas_snps = qc_mt.rows()
        gwas_snps = gwas_snps.annotate(in_GWAS=True)
        snps_ht = snps_ht.annotate(in_GWAS=hl.coalesce(gwas_snps[snps_ht.key].in_GWAS, False))
        snps_ht.write(f"{path_pref}/ukbb.150k.chr{chrom}.snps.ht", overwrite=overwrite)

    # read pandas df with snps in credible sets from UKBB Height
    df = pd.read_csv(
        f"{path_pref}/susie_posteriors_exc_high_chisq.txt",
        sep="\s+",
        dtype={"position": np.int32, "cond_mean": np.float64, "cond_sd": np.float64, "alpha": np.float64}
    )
    df["alleles"] = df[["allele1", "allele2"]].values.tolist()
    df = df.drop(["allele1", "allele2"], axis="columns")
    cs_df = df[df.chromosome == f"{chrom}"].copy()
    cs_df["large_beta"] = 0
    cs_df["large_gamma"] = False

    for i in range(sim_start, sim_end + 1):

        print(f"INFO: Simulation #{i}")

        # if we've already simulated beta, load file and continue
        if hadoop_exists(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{i}.true_beta.ht") and not overwrite:
            ht = hl.read_table(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{i}.true_beta.ht")
            h2g_large_sim = ht.h2g_large.collect()[0],
            h2g_small_sim = ht.emp_h2g_small.collect()[0],
            print(f"INFO: h2g large: {h2g_large_sim}   h2g small: {h2g_small_sim}")
            mt = mt.annotate_rows(**{
                f"beta_{i}": ht[mt.locus, mt.alleles].beta
            })
            mt = mt.annotate_globals(**{f"h2g_sim_{i}": h2g_large_sim + h2g_small_sim})
            continue

        # sample a causal SNP from each un-pruned CS in each region
        sampled_causals_df = (
            cs_df.groupby(["region", "cs"])
                 .apply(sample_causal)
                 .reset_index(drop=True)
        )

        # the default option - sample a "large" effect from every susie credible set that isn't pruned out
        if h2g_ratio < 0:
            sampled_causals_df["large_beta"] = sampled_causals_df["large_beta"] * scale
            h2g_large = (sampled_causals_df.large_beta.to_numpy() ** 2).sum()

        # no "large" effects sampled from susie posterior
        elif h2g_ratio == 0:
            h2g_large = 0
            # throw out sampled "large" effects, keep one placeholder and set effect to zero
            # (this facilitates later table joining)
            sampled_causals_df = sampled_causals_df.iloc[[0]].copy()
            sampled_causals_df["large_beta"] = 0
            sampled_causals_df["large_gamma"] = False

        # randomly sub-sample "large" effects from posterior to achieve desired heritability ratio
        else:
            target_h2g_large = h2g_per_chrom[chrom - 1] * (h2g_ratio / (1 + h2g_ratio))
            h2g_large = 0
            sub_sampled_causals_df = pd.DataFrame()

            while h2g_large < target_h2g_large and len(sampled_causals_df) > 0:
                idx = np.random.choice(len(sampled_causals_df))
                beta = sampled_causals_df.iloc[idx]["large_beta"]
                h2g_large += beta ** 2
                sub_sampled_causals_df = sub_sampled_causals_df.append(sampled_causals_df.iloc[[idx]])
                sampled_causals_df = sampled_causals_df.drop(idx, axis="index").reset_index(drop=True)

            sampled_causals_df = sub_sampled_causals_df

        # convert back to hail table
        large_ht = hl.Table.from_pandas(sampled_causals_df)

        # re-combine fields into locus and re-key by locus and alleles
        large_ht = large_ht.transmute(
            locus=hl.locus(
                large_ht["chromosome"],
                hl.int32(large_ht["position"]),
                reference_genome="GRCh37"
            )
        )
        large_ht = large_ht.key_by("locus", "alleles")
        large_ht = checkpoint_tmp(large_ht)

        # join tables
        snps_ht = hl.read_table(f"{path_pref}/ukbb.150k.chr{chrom}.snps.ht")
        joined = large_ht[snps_ht.key]
        snps_ht = snps_ht.annotate(
            large_gamma=hl.coalesce(joined.large_gamma, False),
            large_beta=hl.coalesce(joined.large_beta, 0)
        )

        if not standardize:
            # susie posteriors are on per-normalized-genotype scale - convert betas to per-allele scale
            snps_ht = snps_ht.annotate(large_beta=snps_ht.large_beta / snps_ht.stdev_pGT)
        snps_ht = checkpoint_tmp(snps_ht)

        # small effects
        if h2g_ratio > 0:
            target_h2g_small = h2g_large / h2g_ratio
        else:
            target_h2g_small = max(h2g_per_chrom[chrom - 1] - h2g_large, 0)

        if target_h2g_small == 0:
            snps_ht = snps_ht.annotate(
                small_gamma=False,
                small_beta=0.0
            )
            sigma2 = "NA"
        else:
            snps_ht = snps_ht.annotate(
                # don't overwrite existing large effects
                small_gamma=((hl.rand_unif(0, 1) < frac_causal) & ~snps_ht.large_gamma)
            )
            small_ht = snps_ht.filter(snps_ht.small_gamma)

            # scale variances so that overall h2g is as desired
            p = small_ht.p_tilda if threshold_maf else small_ht.AF
            sigma2 = target_h2g_small / small_ht.aggregate(hl.agg.sum(
                (small_ht.stdev_pGT ** 2) * ((2 * p * (1 - p)) ** shoech_alpha)
            ))
            small_beta_sd = hl.sqrt(sigma2 * (2 * p * (1 - p)) ** shoech_alpha)
            if standardize:
                small_beta_sd = small_beta_sd * small_ht.stdev_pGT
            small_ht = small_ht.annotate(small_beta=hl.rand_norm(mean=0, sd=small_beta_sd))

            # join tables
            joined = small_ht[snps_ht.key]
            snps_ht = snps_ht.annotate(
                small_gamma=hl.coalesce(joined.small_gamma, False),
                small_beta=hl.coalesce(joined.small_beta, 0.0)
            )

        # separate small effects into those that will/won't be removed by QC
        snps_ht = snps_ht.annotate(
            small_included_beta=hl.if_else(snps_ht.small_gamma & snps_ht.in_GWAS, snps_ht.small_beta, 0.0),
            small_missing_beta=hl.if_else(snps_ht.small_gamma & (~snps_ht.in_GWAS), snps_ht.small_beta, 0.0)
        )

        # compute and annotate components of heritability
        h2g_small_included = snps_ht.aggregate(
            hl.if_else(
                standardize,
                hl.agg.sum(snps_ht.small_included_beta ** 2),
                hl.agg.sum((snps_ht.stdev_pGT ** 2) * (snps_ht.small_included_beta ** 2))
            )
        )
        h2g_small_missing = snps_ht.aggregate(
            hl.if_else(
                standardize,
                hl.agg.sum(snps_ht.small_missing_beta ** 2),
                hl.agg.sum((snps_ht.stdev_pGT ** 2) * (snps_ht.small_missing_beta ** 2))
            )
        )
        snps_ht = snps_ht.annotate_globals(
            sigma2=sigma2,  # this is the coefficient for Shoech variance
            h2g_large=h2g_large,
            h2g_small_included=h2g_small_included,
            h2g_small_missing=h2g_small_missing,
            n_causal_variants=snps_ht.aggregate(hl.agg.sum(snps_ht.small_gamma | snps_ht.large_gamma))
        )
        snps_ht = snps_ht.checkpoint(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{i}.true_beta.ht", overwrite=overwrite)
        snps_ht.export(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{i}.true_beta.txt.bgz")
        print(
            f"INFO: h2g large: {h2g_large}   "
            f"h2g small (included): {h2g_small_included}   "
            f"h2g small (missing): {h2g_small_missing}   "
        )

        # compute total beta
        snps_ht = snps_ht.annotate(total_beta=snps_ht.small_beta + snps_ht.large_beta)
        joined = snps_ht[mt.locus, mt.alleles]
        mt = mt.annotate_rows(**{
            f"beta_total_{i}": joined.total_beta,
            f"beta_large_{i}": joined.large_beta,
            f"beta_small_included_{i}": joined.small_included_beta,
            f"beta_small_missing_{i}": joined.small_missing_beta
        })
        mt = mt.annotate_globals(**{
            f"h2g_total_sim_{i}": h2g_large + h2g_small_included + h2g_small_missing,
            f"h2g_large_sim_{i}": h2g_large,
            f"h2g_small_included_sim_{i}": h2g_small_included,
            f"h2g_small_missing_sim_{i}": h2g_small_missing,
        })
        mt = checkpoint_tmp(mt)

    # compute total SNP effects (X'beta)
    if standardize:
        mt = mt.annotate_entries(pGT_scaled=(mt.pGT - mt.mean_pGT) / mt.stdev_pGT)
        gt = mt.pGT_scaled
    else:
        gt = mt.pGT
    total = {f"total_effect_sim_{i}": hl.agg.sum(mt[f"beta_total_{i}"] * gt) for i in range(sim_start, sim_end + 1)}
    large = {f"large_effect_sim_{i}": hl.agg.sum(mt[f"beta_large_{i}"] * gt) for i in range(sim_start, sim_end + 1)}
    small_included = {
        f"small_included_effect_sim_{i}": hl.agg.sum(mt[f"beta_small_included_{i}"] * gt)
        for i in range(sim_start, sim_end + 1)
    }
    small_missing = {
        f"small_missing_effect_sim_{i}": hl.agg.sum(mt[f"beta_small_missing_{i}"] * gt)
        for i in range(sim_start, sim_end + 1)
    }
    mt = mt.select_cols(**total, **large, **small_included, **small_missing)
    mt.cols().write(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{sim_start}-{sim_end}.effect.ht", overwrite=overwrite)


def compute_true_pheno(sim_start, sim_end, overwrite):

    # TODO: effects of PCs are on standardized scale, so if standardize=False, these will be wrong
    #  address this by passing standardize here too, and rescaling height phenotype

    if hadoop_exists(f"{path_pref}/ukbb.150k.sim_{sim_start}-{sim_end}.pheno.ht") and not overwrite:
        print(
            f"INFO: Simulated phenotypes already exists for simulations {sim_start}-{sim_end} at "
            f"{path_pref}/ukbb.150k.sim_{sim_start}-{sim_end}.pheno.ht. "
            f"If you wish to overwrite them please pass the --overwrite flag."
        )
        return

    print(f"INFO: Computing phenotypes...")

    if not all([
        hadoop_exists(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{sim_start}-{sim_end}.effect.ht") for chrom in range(1, 23)
    ]):
        print(
            f"ERROR: no effect table found for some chromosomes for simulations {sim_start}-{sim_end}. "
            f"Run sample_effects() for all chromosomes first."
        )
        return

    # sum up chromosome effects
    pheno_ht = None

    # join all effect tables
    for chrom in range(1, 23):

        effect_ht = hl.read_table(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{sim_start}-{sim_end}.effect.ht")
        if pheno_ht is None:
            pheno_ht = effect_ht
        else:
            pheno_ht = pheno_ht.join(effect_ht, how="inner")
    assert pheno_ht.count() == n  # i.e. check no individuals were lost

    # sum chromosome effect components for each simulation
    rfs = pheno_ht.row.keys()
    pheno_ht = pheno_ht.transmute(**{
        f"sum_{comp}_effects_sim_{i}": hl.sum([pheno_ht[rf] for rf in rfs if rf.startswith(f"{comp}_effect_sim_{i}")])
        for i in range(sim_start, sim_end + 1)
        for comp in ["total", "large", "small_included", "small_missing"]
    })

    # sum chromosome h2g components for each simulation
    keys = pheno_ht.globals.keys()
    pheno_ht = pheno_ht.transmute_globals(**{
        f"h2g_{comp}_sim_{i}": hl.sum([pheno_ht[k] for k in keys if k.startswith(f"h2g_{comp}_sim_{i}")])
        for i in range(sim_start, sim_end + 1)
        for comp in ["total", "large", "small_included", "small_missing"]
    })

    # compute covariate coefficients for UKBB 360k PCs
    height = hl.Table.from_pandas(
        pd.read_csv(f"{phenotype_file_name}", sep="\t", usecols=["FID", "Height"])
    )
    height = height.annotate(FID=hl.str(height.FID))
    height = height.key_by("FID")
    cov = hl.read_table(f"{gwas_cov}")

    # only use PCs as covariates
    cov = cov.select(*[f"PC{i}" for i in range(1, 21)])
    cov = cov.annotate(
        height=height[cov.key].Height
    )
    cov = cov.filter(hl.is_nan(cov.height), keep=False)
    coeffs = cov.aggregate(hl.agg.linreg(
        cov.height,
        [1] + [cov[field] for field in cov.row_value.keys() if field != "height"]
    ))

    # compute total sum of covariate effects
    effects = [beta * cov[field] for beta, field in zip(coeffs.beta[1:], [f"PC{i}" for i in range(1, 21)])]
    cov = cov.annotate(
        effect=hl.fold(
            lambda a, b: a + b,
            0,
            effects
        )
    )
    cov = checkpoint_tmp(cov)

    # add covariates - no need to label with simulation index i as this will be the same for all simulations
    pheno_ht = pheno_ht.annotate(cov_effect=cov[pheno_ht.key].effect)
    pheno_ht = pheno_ht.filter(hl.is_missing(pheno_ht.cov_effect), keep=False)
    # some samples will be removed because they have nan values for height

    # add gaussian iid noise
    pheno_ht = pheno_ht.annotate(**{
        f"e_sim_{i}": hl.rand_norm(0, hl.sqrt(1 - pheno_ht[f"h2g_total_sim_{i}"]))
        for i in range(sim_start, sim_end + 1)
    })

    # combine genetic effects + noise + covariates -> overall phenotype
    pheno_ht = pheno_ht.annotate(**{
        f"pheno_sim_{i}": pheno_ht[f"sum_total_effects_sim_{i}"] + pheno_ht[f"e_sim_{i}"] + pheno_ht.cov_effect
        for i in range(sim_start, sim_end + 1)
    })
    pheno_ht.write(f"{path_pref}/ukbb.150k.sim_{sim_start}-{sim_end}.pheno.ht", overwrite=overwrite)


def clean():
    """Remove .mt files (recoverable from bgen files) and .snps.ht files for all chromosomes."""

    for chrom in range(1, 23):

        # remove snps_ht
        ret = subprocess.run(["gsutil", "-m", "rm", "-r", f"{path_pref}/ukbb.150k.chr{chrom}.snps.ht/"])
        if ret.returncode != 0:
            print(f"WARNING: unable to delete {path_pref}/ukbb.150k.chr{chrom}.snps.ht/")



# this method may not work with updated pipeline since it has barely been modified from Masa's version.
def run_gwas(chrom, sim_start, sim_end):
    if args.true_geno:
        prefix = f'{path_pref}/true_geno'
        x = 'pGT'
    else:
        prefix = f'{path_pref}/simulation'
        x = 'dosage'

    if args.down_sample is not None:
        incl = f'{downsample_incl_path}/ukb31063.{args.down_sample}.incl'
        prefix = f'{path_pref}/downsample_{args.down_sample}'
    else:
        incl = None

    if not hadoop_exists(f'{prefix}/ukb31063.chr{chrom}.sim_{sim_start}-{sim_end}.gwas_result.ht'):
        mt = hl.read_matrix_table(f"{path_pref}/ukbb.150k.chr{chrom}.mt)

        phenos = hl.read_table(f'{phenotype_file_name_ht}')
        mt = mt.annotate_cols(**phenos[mt.s])
        covariates = hl.read_table(f'{gwas_cov}')
        mt = mt.annotate_cols(**covariates[mt.s])

        if incl is not None:
            incl = hl.import_table(incl, no_header=True).rename({'f0': 's'})
            incl = incl.aggregate(hl.agg.collect_as_set(incl.s), _localize=False)
            mt = mt.filter_cols(incl.contains(mt.s))

        mfi_ht = hl.import_table(f'{bgen_imputed_path}/ukb_mfi_chr{chrom}_v3.txt',
                                 impute=True,
                                 no_header=True)
        mfi_ht = mfi_ht.select(locus=hl.locus(str(chrom), mfi_ht.f2), alleles=[mfi_ht.f3, mfi_ht.f4], info=mfi_ht.f7)
        mfi_ht = mfi_ht.key_by(mfi_ht.locus, mfi_ht.alleles)

        mt = mt.annotate_rows(v=hl.variant_str(mt.locus, mt.alleles),
                              maf=hl.agg.mean(mt.dosage) / 2,
                              info=mfi_ht[mt.locus, mt.alleles].info)
        mt = mt.annotate_rows(maf=hl.cond(mt.maf <= 0.5, mt.maf, 1.0 - mt.maf))
        # mt = checkpoint_tmp(mt)

        result_ht = hl.linear_regression_rows(y=[mt[f'sim_{i}'] for i in range(sim_start, sim_end + 1)],
                                              x=mt[x],
                                              covariates=[1] + [mt['PC' + str(i)] for i in range(1, 21)],
                                              pass_through=['v', 'rsid', 'maf', 'info'])
        result_ht = result_ht.checkpoint(f'{prefix}/ukb31063.chr{chrom}.sim_{sim_start}-{sim_end}.gwas_result.ht',
                                         overwrite=args.overwrite)
    else:
        result_ht = hl.read_table(f'{prefix}/ukb31063.chr{chrom}.sim_{sim_start}-{sim_end}.gwas_result.ht',)

    result_ht = result_ht.key_by()
    for i in range(sim_start, sim_end + 1):
        j = i - sim_start
        ht = hl.read_table(f"{path_pref}/ukbb.150k.chr{chrom}.sim_{i}.true_beta.ht")
        result_annotated = result_ht.select(v=result_ht.v,
                                            rsid=result_ht.rsid,
                                            chromosome=result_ht.locus.contig,
                                            position=result_ht.locus.position,
                                            allele1=result_ht.alleles[0],
                                            allele2=result_ht.alleles[1],
                                            maf=result_ht.maf,
                                            info=result_ht.info,
                                            beta=result_ht.beta[j],
                                            se=result_ht.standard_error[j],
                                            p=result_ht.p_value[j],
                                            gamma=ht[result_ht.locus, result_ht.alleles].gamma,
                                            true_beta=ht[result_ht.locus, result_ht.alleles].beta)
        result_annotated.export(f'{prefix}/ukb31063.chr{chrom}.sim{i}.txt.bgz')


def main(args):

    start = time()

    if args.write_mt:
        write_mt(args.chromosome, args.overwrite)
    if args.write_bgen:
        write_bgen(args.chromosome)
    if args.sample_effects:
        sample_effects(
            args.chromosome,
            args.sim_start,
            args.sim_end,
            args.overwrite,
            args.threshold_maf,
            args.standardize,
            args.h2g_ratio,
            args.scale
        )
    if args.compute_true_pheno:
        compute_true_pheno(args.sim_start, args.sim_end, args.overwrite)
    if args.run_gwas:
        run_gwas(args.chromosome, args.sim_start, args.sim_end)
    if args.clean:
        clean()

    print(f"INFO: Chromosome {args.chromosome} simulation completed in {time() - start} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-mt', action='store_true')
    parser.add_argument('--write-bgen', action='store_true')
    parser.add_argument('--sample-effects', action='store_true')
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--compute-true-pheno', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--run-gwas', action='store_true')
    parser.add_argument('--true-geno', action='store_true')
    parser.add_argument('--down-sample', type=str)
    parser.add_argument('--chromosome', type=int, default=1)
    parser.add_argument('--sim-start', type=int, default=1)
    parser.add_argument('--sim-end', type=int, default=10)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--h2g-ratio', type=float, default=-1,
        help="the ratio of large effect heritability to small effect heritability"
    )
    parser.add_argument(
        "--scale-large-effects", type=float, dest="scale", default=1,
        help="a float by which to multiply the large effects"
    )
    parser.add_argument(
        '--threshold-maf', action='store_true',
        help="threshold MAF for purposes of sampling effect sizes (to avoid extreme effect sizes for rare SNPs)"
    )
    args = parser.parse_args()

    if args.h2g_ratio >= 0 and args.scale != 1:
        raise ValueError(f"ERROR: --h2g-ratio and --scale-large-effects cannot be specified together.")

    main(args)
