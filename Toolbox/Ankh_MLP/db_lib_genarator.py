import pandas as pd
import re
import itertools
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Construct double mutants library from high-scoring single mutations" )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to single mutation file."
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output double mutant file."
    )
    args = parser.parse_args()

    infile = args.input
    outfile = args.output

    df = pd.read_csv(infile, sep="\t")
    top100 = df.nlargest(100, "prediction")["mutant"].tolist()
    print(top100)
    all_singles = df["mutant"].tolist()

    def get_pos(m):
        return int(re.search(r"\d+", m).group())

    pairs = set()
    for m1, m2 in itertools.product(top100, all_singles):
        p1, p2 = get_pos(m1), get_pos(m2)
        if m1 == m2 or p1 == p2:
            continue
        if p1 < p2:
            pairs.add(f"{m1};{m2}")
        else:
            pairs.add(f"{m2};{m1}")

    out_df = pd.DataFrame({"mutant": list(pairs)})
    out_df['score'] = 1
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'{len(out_df)} of double mutants have been put into the library')

    out_df.to_csv(outfile, sep="\t", index=False)

if __name__ == "__main__":
    main()
