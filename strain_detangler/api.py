
import numpy as np
import pandas as pd

from csv import writer
from gensim.models.ldamodel import LdaModel
from os.path import basename, dirname, join, exists
from os import makedirs
from regex import compile
from scipy.spatial.distance import cdist
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import matthews_corrcoef


def build_pangenome(logger=lambda x: None):
    """Return something."""
    logger('Building a pangenome')
    pass


"""
Workflow functions to generate appropriately named files and directories.  
"""

def prefix(filename, trim):
    full = basename(filename).split('.')  # Cuts off the information and suffix.
    edit = full[:-trim]
    return '.'.join([str(i) for i in edit])


def check_make(output, subdir):
    outdir = join(output, subdir)
    if not exists(outdir):
        makedirs(outdir)
    return outdir


def python_sed(in_file, out_file):
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        list(fout.write(line.replace('lcl|', '')) for line in fin)


"""
Utility functions to i) generate sample-SNP composition matrices from unmapped sequencing reads and 
                     ii) simultaneously infer a) the SNP composition of strains and 
                                              b) the strain compositions of multiple samples 
                         at once from what is essentially pileup information summarized in (i).
"""


def entropy_reduce_position_matrix(original, r, metric, min_fill=2, centroids=None, logger=None):
    # Adapted from DD's gimmebio.stat-strains.api.
    """Return an Entropy Reduced version of the input matrix.

    Return a pandas dataframe with a subset of columns from the
    original dataframe with the guarantee that every column in
    the original dataframe is within `r` of at least one column
    in the reduced frame*.

    * Exclude all columns with sum == 0

    Optionally pass a logger function which will get
    (num_centroids, num_columns_processed) pairs
    """
    if not centroids:
        centroids = list()
    for i, (col_name, col) in enumerate(original.iteritems()):
        if logger:
            logger(len(centroids), i)
        if (col > 0).sum() < min_fill:
            continue
        if len(centroids) == 0:
            centroids.append(col_name)
            continue
        d = min(cdist(
            pd.DataFrame(original[centroids]).T,
            pd.DataFrame(original[col_name]).T,
            metric=metric
        ))
        if d > r:
            centroids.append(col_name)
    return original[centroids]


def parse_pileup_bases(ref_base, read_bases, read_quals):  # Adapted from DESMAN's pileup_to_freq_table.py.
    nucleotides = ['A', 'T', 'C', 'G', 'a', 't', 'c', 'g']
    min_base_q = 20 + 33  # Phred scale
    indel_pattern = compile('\d+')
    counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}

    b_index = -1
    q_index = -1
    while b_index < len(read_bases) - 1:
        b_index += 1
        i = read_bases[b_index]
        if i == '.' or i == ',':  # A (reverse complement) match.
            q_index += 1
            if ord(read_quals[q_index]) >= min_base_q:
                counts[ref_base] += 1
            continue
        elif i == '^':  # The start of a new read.
            b_index += 1  # Skip this and the next character. '^' indicates first base in a read.
            # The ASCII of the character following `^' minus 33 gives the mapping quality. Usually low.
            continue
        elif i == '+' or i == '-':  # Indel.
            m = indel_pattern.match(read_bases, b_index + 1)
            b_index += int(m.group(0)) + len(m.group(0)) + 1
            continue
        elif i in nucleotides:  # SNP.
            q_index += 1
            if ord(read_quals[q_index]) >= min_base_q:
                counts[i.upper()] += 1
    return [str(counts['A']), str(counts['C']), str(counts['G']), str(counts['T'])]


def write_to_gene_file(outdir, curr_gene, last_pos, pinfo):
    gene_len = (len(open(join(outdir, curr_gene + '.counts.csv'), 'r').readline().strip().split(
        ",")) - 1) / 4  # The number of bases in the scaffold.
    if (gene_len - last_pos) != 0:
        for p in range(int(gene_len - last_pos)):
            # If not ending at the end of the scaffold, fill in missed columns with 0.
            pinfo += ['0', '0', '0', '0']
    with open(join(outdir, curr_gene + '.counts.csv'), 'a+') as gf:
        writer(gf).writerow(pinfo)


def concat_matrices(file_list, logger, outdir=None, fname=None):
    tbls = []
    with open(file_list) as fl:
        for i, line in enumerate(fl):
            tbl = pd.read_csv(line.strip(), header=0, index_col=0)
            if tbl.shape[1] != 0:
                print(tbl.shape)
                tbl.columns = [str(i) + '-' + s for s in tbl.columns.values]
                tbls.append(tbl)
                if logger: 
                    logger(i)
    dframe = pd.concat(tbls, axis=1, sort=True)
    dframe = dframe.fillna(0).astype(int)  # Replaces NA (unfilled positions) with 0.

    summary = summarize_df(dframe)
    # dframe.to_csv(path_or_buf=join('gstd.csv'), index=True) # TODO remove
    if outdir and fname:
        dframe.to_csv(path_or_buf=join(outdir, fname + '.merged.csv'), index=True)
    return dframe, summary


def summarize_df(dframe):
    col_cts = dframe.sum(axis=0)
    num_pos = np.count_nonzero(col_cts)
    tot_ct = col_cts.sum()
    num_samples = len(dframe.index.values)
    return [num_pos, tot_ct, num_samples]


def remove_uninformative_snps(dframe, bf, outdir=None, fname=None):
    file_ct = len(dframe.values)
    high_band = float(bf) * file_ct  # Removes positions that are not informative. This many positions or fewer are 0.
    low_band = file_ct - high_band  # Removes positions that may be sequencing errors. This many positions or more are 0.

    too_low = dframe.columns[(dframe == 0).sum() > low_band]
    dframe.drop(too_low, axis=1, inplace=True)
    too_high = dframe.columns[(dframe == 0).sum() < high_band]
    dframe.drop(too_high, axis=1, inplace=True)

    summary = summarize_df(dframe)
    if outdir and fname:
        dframe.to_csv(path_or_buf=outdir + "/" + fname + ".filtered.csv", index=True)
    return dframe, summary, [len(too_low), len(too_high)]


def pandas_as_corpus(dframe):
    out = []
    for rowname, row in dframe.iterrows():
        row = [(i, val) for i, val in enumerate(row) if val > 0]
        if row:
            out.append(row)
    return out


def topic_word_matrix(model, outdir, fname, logger):
    # Making the topic-word matrix
    tw_list = model.show_topics(formatted=False)  # list of {str, tuple of (str, float)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in tw_list]
    # Converts list of tuple of (str,list[tuple]) to list of tuple of (str,list)
    with open(join(outdir, fname + '.gensim_topwords.csv'), "w") as tfile:
        tw_writer = writer(tfile)
        for tw in tw_list:  # tw = tuple{str, list[tuple{str,float}]}
            out_str = [tw[0]]  # [topic_name]
            for wd in tw[1]:  # wd = tuple{str,float}
                out_str.append(wd[0])  # [topic_name,word,word...]
            tw_writer.writerow(out_str)
    logger('Saved the topic-word matrix')


def document_topic_matrix(model, corpus, row_names, outdir, fname, logger):
    # Making the document-topic matrix
    tbl = {}
    for i, doc in enumerate(corpus):
        tbl[i] = {key: val for key, val in model[doc]}
    tbl = pd.DataFrame.from_dict(tbl, orient='index').fillna(0)
    tbl.index = row_names
    tbl.to_csv(join(outdir, fname + '.gensim_doctopic.csv'))
    logger('Saved the document-topic matrix')


def lda_gensim(dframe, output, fname, topics, logger):  # An adaptation of DD's run_lda.py.
    # Data preparation for LDA
    id2word = {i: taxa_name for i, taxa_name in enumerate(list(dframe.columns))}
    corpus = pandas_as_corpus(dframe)
    logger('Finished loading id2word and dataframe as corpus')

    # Running the LDA and pickling
    lda10 = LdaModel(corpus, id2word=id2word, num_topics=topics, iterations=1000, passes=10)
    logger('Finished training LDA model, now pickling')
    lda10.save(join(output, fname + '.gensim_model.pkl'))

    topic_word_matrix(lda10, output, fname, logger)
    document_topic_matrix(lda10, corpus, dframe.index, output, fname, logger)
    logger('Finished outputting the factor matrices')


"""
Utility functions to evaluate the accuracy of inferred strain and sample composition 
by comparing it with provided gold-standard information. 
"""


def sort_snp_by_pos(snp_set):
    # Reorder the SNPs extracted from an LDA strain composition file by gene and position within gene.
    alpha_sort = sorted(snp_set)
    alpha_sort_lst = []
    for s in alpha_sort:
        alpha_sort_lst.append(s.split('-'))
    alpha_sort_df = pd.DataFrame.from_records(alpha_sort_lst)
    alpha_sort_df[[0,1]] = alpha_sort_df[[0,1]].apply(pd.to_numeric)
    num_sort_df = alpha_sort_df.sort_values([0,1], ascending = (True,True))
    num_sort_lst = num_sort_df.values.tolist()
    out_list = []
    for n in num_sort_lst:
        out_list.append('-'.join(str(i) for i in n))
    return out_list


def make_strains_lda(topic_wrd, infer_freq):
    # infer_snp_raw = make_strains_lda(topic_wrd, infer_freq)
    # Subset the SNP compositions of all strains to only the strains actually present in the samples.
    inc_strains_ind = infer_freq.columns.values
    inc_strains_dct = {}
    snp_set = set()
    with open(topic_wrd, 'r') as tf:
        for row in tf: # row: strain_index,gene-pos-allele,gene-pos-allele,...
            strain = row.strip().split(',')
            strain_index = strain[0]
            if strain_index in inc_strains_ind:
                strain_comp = strain[1:] # [gene-pos-allele]
                inc_strains_dct[strain_index] = strain_comp
                snp_set.update(strain_comp)

    strain_lst = []
    sorted_snp_lst = sort_snp_by_pos(snp_set)
    for snp in sorted_snp_lst:
        snp_column = [] # List of the SNP composition of each strain, convert to dataframe later
        for strain, alleles in inc_strains_dct.items():  # Check if each strain contains the current SNP
            if snp in alleles: 
                snp_column.append(1)
            else:
                snp_column.append(0)
        strain_lst.append(snp_column)
    strain_df = pd.DataFrame(strain_lst, index = sorted_snp_lst, columns = inc_strains_dct.keys()).T
    # strain_df.to_csv('lda.csv', header = True, index = True) # TODO remove
    return strain_df


def deduplicate_rows(snp_df, freq_df):
    # Subfunction to take out duplicated strains.
    snp_dedup = snp_df.drop_duplicates(keep='first') # Make a dataframe without duplicates 

    all_indices = snp_df.index.values
    kept_indices = snp_dedup.index.values
    removed_indices = [i for i in snp_df.index.values if i not in kept_indices] # Find out which strains (indices) were dropped
    removed_kept = {} # {removed_strain_index : kept_strain_index}
    for i in removed_indices: # Compare removed strain allele composition to kept strain composition
        snp_composition = snp_df.loc[i].tolist()
        for j in range(len(snp_dedup.index)):
            if snp_composition == snp_dedup.iloc[j].tolist():
                removed_kept[i] = kept_indices[j] 
                break # Found the match, go through the next removed strain. 

    freq_dct = freq_df.to_dict() # {all_strain_index : {sample_index: frequency}}
    for i in all_indices:
        if i in removed_indices: 
            j = removed_kept[i] # Kept strain j identical to removed strain i
            new_j = {key: freq_dct[i].get(key, 0) + freq_dct[j].get(key, 0) for key in set(freq_dct[i]) | set(freq_dct[j])}
            freq_dct[j] = new_j # Set frequencies in sample to the sum of removed i and kept j
            freq_dct.pop(i, None) # Remove strain i from dictionary.

    return snp_dedup, pd.DataFrame.from_dict(freq_dct)


def intersect_snps(infer_snp_raw, gstd_snp_raw, infer_freq):
    # infer_snp_final, gstd_std_final, infer_freq_final = intersect_snps(infer_snp_raw, gstd_snp_raw, infer_freq)
    # Take the intersection of the strain composition dataframes and return subsets containing only the SNPs they share.
    infer_snps = infer_snp_raw.columns.values
    gstd_snps = gstd_snp_raw.columns.values
    shared_snps = [s for s in infer_snps if s in gstd_snps]
    infer_snps_shared = infer_snp_raw.loc[:, shared_snps]
    gstd_snps_shared = gstd_snp_raw.loc[:, shared_snps]

    infer_snps_dedup, infer_freq_dedup = deduplicate_rows(infer_snps_shared, infer_freq)

    false_pos_snps = len([s for s in infer_snps if s not in gstd_snps])
    false_neg_snps = len([s for s in gstd_snps if s not in infer_snps])
    print(f'{len(shared_snps)} TP, {false_pos_snps} FP, {false_neg_snps} FN SNPs') # TODO Fix or update to logger

    return infer_snps_dedup, gstd_snps_shared, infer_freq_dedup


def make_pool(snp_df, freq_df, p, f):
    # infer_snp_pool, infer_freq_pool = make_pool(infer_snp_final, infer_freq_final, p, self.frequency_cutoff) 
    # Return only the frequencies and compositions of strains that are present in the pool.
    tmp = freq_df.iloc[[p]] 
    freq_pool = tmp.loc[:, (tmp >= f).any(axis = 0)]
    kept_indices = freq_pool.columns.values
    snp_pool = snp_df.loc[kept_indices]
    return snp_pool, freq_pool


def pairwise_manhattan(infer_strains, gstd_strains, t):
    # Calculate the Manhattan distance between the inferred and gold-standard strains and return pairs of inferred-gold standard strains that are the most similar, with a minimum threshold of t. 
    # paired_strains, min_strain_dist = pairwise_manhattan(infer_snp_pool, gstd_snp_pool, t) 
    md = pd.DataFrame(manhattan_distances(gstd_strains, infer_strains), index = gstd_strains.index.values, columns = infer_strains.index.values) # output_df = [x = rows = gstd, y = columns = infer]
    min_dist = md.min(axis = 1)
    min_strain = md.idxmin(axis = 1)
    gstd_index = md.index.values
    closest_inferred = set(infer_strains.index.values) # Closest inferred strain to the gold-standard    

    paired_actual = []
    paired_predict = []

    threshold = t * len(gstd_strains.columns)
    for i in range(len(min_dist)):
        if min_dist[i] < threshold:    # Successfully inferred gold-standard strain
            paired_actual.append(gstd_index[i]) 
            paired_predict.append(min_strain[i]) 
            closest_inferred.discard(min_strain[i])
        else:                           # Failed to infer gold-standard strain
            paired_actual.append(gstd_index[i]) 
            paired_predict.append(None) 
    for i in closest_inferred:          # False positive inferred strains
        paired_actual.append(None) 
        paired_predict.append(i)         

    return paired_actual, paired_predict, min_dist


def mcc_calculator(paired_actual, paired_predict):
    # Calculate the Matthews correlation coefficient between the actual strain composition of a pool and the inferred strain composition, as well as the number of strains 
    # mcc, num_success = mcc_calculator(paired_actual, paired_predict)
    f = lambda l: [(lambda x: 1 if x != None else -1)(x) for x in l]
    binary_actual = f(paired_actual)
    binary_predict = f(paired_predict)
    num_success = 0
    for i in range(len(binary_actual)):
        if (binary_actual[i] == binary_predict[i]):
            num_success += 1
    return matthews_corrcoef(binary_actual, binary_predict), num_success


def jsd_calculator(id_actual, id_predict, gstd_freq, infer_freq):    
    # Makes pairs of within-pool frequencies from pairs of gold-standard and inferred strains, then calculate JSD. 
    # jsd = jsd_calculator(paired_actual, paired_predict, gstd_freq_pool, infer_freq_pool)
    freq_actual = []
    gstd_freq_cols = gstd_freq.columns.values
    pool_index = gstd_freq.index.values[0]
    for i in id_actual:
        if i in gstd_freq_cols: 
            freq_actual.append(gstd_freq.loc[pool_index,i]) 
        else:
            freq_actual.append(0.0)
    freq_predict = []
    infer_freq_cols = infer_freq.columns.values
    for i in id_predict:
        if i in infer_freq_cols: 
            freq_predict.append(infer_freq.loc[pool_index,i])
        else:
            freq_predict.append(0.0)

    return jensenshannon(freq_actual, freq_predict)

    
















