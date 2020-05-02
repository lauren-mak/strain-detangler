
import numpy as np
import pandas as pd

from csv import writer
from gensim.models.ldamodel import LdaModel
from os.path import basename, dirname, join, exists
from os import makedirs
from regex import compile
from scipy.spatial.distance import cdist


def build_pangenome(logger=lambda x: None):
    """Return something."""
    logger('Building a pangenome')
    pass


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


def prefix(filename):
    lst = basename(filename).split('.')  # Cuts off the information and suffix.
    if len(lst) > 2:
        lst = lst[:-2]
    else:
        lst = lst[:-1]
    return '.'.join([str(i) for i in lst])


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
            tbl.columns = [str(i) + "-" + s for s in tbl.columns.values]
            tbls.append(tbl)
            logger(i)
    dframe = pd.concat(tbls, axis=1, sort=True)
    dframe = dframe.fillna(0).astype(int)  # Replaces NA (unfilled positions) with 0.

    summary = summarize_df(dframe)
    if outdir and fname:
        dframe.to_csv(path_or_buf=outdir + "/" + fname + ".merged.csv", index=True)
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
    outdir = check_directory(output, 'model/')
    lda10.save(join(outdir, fname + '.gensim_model.pkl'))

    topic_word_matrix(lda10, outdir, fname, logger)
    document_topic_matrix(lda10, corpus, dframe.index, outdir, fname, logger)
    logger('Finished outputting the factor matrices')

