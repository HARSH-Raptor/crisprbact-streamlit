import re
import pandas as pd

from crisprbact.utils import rev_comp
from itertools import chain

FEATURE_TYPES = ["CDS", "ncRNA", "rRNA", "tRNA"]


def compute_off_target_df(guide, seed_size, records, feature_df):
    """Returns a pandas DataFrame with data about the identified off-targets.
    The features column contains a list of biopython SeqFeature objects that overlap
    with the off-target"""
    offs_df = get_off_target_pos(guide, records, seed_size)
    if offs_df is not None:
        offs_df["features"] = [
            get_pos_features(off.recid, off.pampos, feature_df)
            for i, off in offs_df.iterrows()
        ]
        return offs_df
    else:
        return None


def get_off_target_pos(guide, records, seed_size):
    if records is not None:
        offs = iter([])
        for rec in records:
            sequence_str = str(rec.seq).upper()
            seqid = rec.id
            offs_plus = re.finditer(guide[-seed_size:] + "[ATGC]GG", sequence_str)
            offs_plus = gen_extract_off_target_strand_plus(
                offs_plus, seqid, sequence_str, guide, seed_size
            )

            # - ori
            offs_minus = re.finditer(
                "CC[ATGC]" + rev_comp(guide[-seed_size:]), sequence_str
            )
            offs_minus = gen_extract_off_target_strand_minus(
                offs_minus, seqid, sequence_str, guide, seed_size
            )
            offs = chain(offs, offs_plus, offs_minus)
        offs = list(offs)
        if not offs:
            return None
        offs_dict = dict(
            zip(
                [
                    "start",
                    "end",
                    "pampos",
                    "strand",
                    "recid",
                    "longest_perfect_match",
                    "pam_seq",
                ],
                zip(*offs),
            )
        )
        return pd.DataFrame(offs_dict)
    else:
        return None


def get_pos_features(recid, position, f_df):
    if len(f_df) > 0:
        feature_at_pos = f_df[
            (f_df.start < position) & (f_df.end > position) & (f_df.recid == recid)
        ]
        return feature_at_pos.feature.values
    else:
        return []


def gen_extract_off_target_strand_plus(
    off_target_matches, seqid, sequence_string, guide, seed_size
):
    for match in off_target_matches:
        guide_subseq = guide[: 20 - seed_size][::-1]
        # extract part of the sequence
        seq_extension_len = len(guide) - seed_size
        start_pos_seq = match.start() - seq_extension_len
        end_pos_seq = match.end() - seed_size - 3
        sub_sequence_to_match = sequence_string[start_pos_seq:end_pos_seq][::-1]
        matching_chars = list(common_start(sub_sequence_to_match, guide_subseq))
        matching_substr = "".join(matching_chars)
        longest_perfect_match = matching_substr[::-1] + match.group(0)[:-3]
        yield match.span() + (
            match.end(),
            "+",
            seqid,
            longest_perfect_match,
            match.group(0)[-3:],
        )


def common_start(seq1, seq2):
    for a, b in zip(seq1, seq2):
        if a == b:
            yield a
        else:
            return


def gen_extract_off_target_strand_minus(
    off_target_matches, seqid, sequence_string, guide, seed_size
):
    for match in off_target_matches:
        # Extract the sequence.
        # since rev_comp, extend the sequence to the end
        seq_extension_len = len(guide) - seed_size
        end_pos_seq = match.end() + seq_extension_len
        start_pos_seq = match.start() + seed_size + 3
        sub_sequence_to_match = sequence_string[start_pos_seq:end_pos_seq]

        # Extract part of the guide to match
        guide_subseq = rev_comp(guide[: 20 - seed_size])
        matching_chars = list(common_start(sub_sequence_to_match, guide_subseq))
        matching_substr = "".join(matching_chars)
        longest_perfect_match = match.group(0)[3:] + matching_substr

        yield match.span() + (
            match.start(),
            "-",
            seqid,
            longest_perfect_match,
            match.group(0)[:3],
        )


def extract_records(genome):
    records = list(genome)
    if records:
        return records
    else:
        return None


def extract_features(recs):
    if recs is not None:
        f_list = []
        for rec in recs:
            for f in rec.features:
                if f.type in FEATURE_TYPES:
                    f_list.append(
                        (
                            int(f.location.start),
                            int(f.location.end),
                            f.location.strand,
                            f.type,
                            f,
                            rec.id,
                        )
                    )
        f_dict = dict(
            zip(
                ["start", "end", "strand", "type", "feature", "recid"],
                zip(*f_list),
            )
        )
        return pd.DataFrame(f_dict)
    else:
        return None
