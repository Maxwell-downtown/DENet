import re

target = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        
outfile = './warehouse/GFP_AEQVI_DEh10k.psc'
file_names = ['./warehouse/GFP_AEQVI_t10k.tsv']
num = 10000

        
def convert(muta, seq):
    ss = seq
    mut_count = muta.count(';')
    mut_count += 1
    sites = [int(s) for s in re.findall('[0-9]+', muta)]
    types = re.findall('[a-zA-Z]+', muta)
    sequence = list(ss)
    for m in range(0, mut_count):
        count = sites[m]
        cor = count - 1
        sequence[cor] = types[int(2*m+1)]
    sequence = "".join([str(item) for item in sequence])
    return sequence

combined_data = set()
for file_name in file_names:
    with open(file_name, 'r') as fin:
        next(fin) # skip header line
        i = 0
        for lines in fin:
            if i < num:
                words = lines.split()
                mutation_type = words[0]
                combined_data.add(mutation_type)
                i += 1
            else:
                break

print(f'Total number of mutants is: {len(combined_data)}')
file = open(outfile, "w+")

# Enble sequence filtering for high-order mutants
combined_data = [m for m in combined_data if ';' in m]

print(f'Total number of high-order mutants is {len(combined_data)}')
for i in range(0, len(combined_data)):
    seq = convert(combined_data[i], target)
    file.write(seq + '\n')
file.close()



