import csv
import pandas as pd


# Parse a mutation string into a dictionary with position as keys.
def parse_mutation(mutation):
    mutations = mutation.split(';')
    mutation_dict = {}
    for mut in mutations:
        pos = int(mut[1:-1])
        mutation_dict[pos] = mut
    return mutation_dict


def combine_mutations(top_mut, single_mutations):
    combined_mutations = []
    for mut in top_mut:
        mut_dict = parse_mutation(mut)
        for single in single_mutations:
            single_pos = int(single[1:-1])
            if single_pos in mut_dict:
                continue
            mut_dict[single_pos] = single
            sorted_mutations = ';'.join(mut_dict[pos] for pos in sorted(mut_dict))
            combined_mutations.append(sorted_mutations)
            del mut_dict[single_pos]
    return combined_mutations


# read in double mutations
double_mutations = []
with open('./GFP_AEQVI/GFP_AEQVI_t100.tsv', 'r') as top_file:
    reader = csv.reader(top_file, delimiter='\t')
    # next(reader)
    for row in reader:
        double_mutations.append(row[0])
print(double_mutations[0])

# read in single mutations
single_mutations = []
with open('../../../Data/Protein_Info/lib/GFP_AEQVI-sin_lib.tsv', 'r') as single_file:
    reader = csv.reader(single_file, delimiter='\t')
    next(reader)    # Skip the header row
    for row in reader:
        single_mutations.append(row[0])
print(single_mutations[0])

combined_mutations = combine_mutations(double_mutations, single_mutations)
df = pd.DataFrame(combined_mutations, columns=['mutant'])
df['score'] = 1
df['cluster'] = 0
# Remove duplications:
df = df.drop_duplicates(subset='mutant', keep='first')

# Save to TSV file
df.to_csv('./GFP_AEQVI/L2-t100.tsv', sep='\t', index=False)

print("TSV file created successfully.")
