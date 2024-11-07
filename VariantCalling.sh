#!/bin/bash

# Define directories
fqdir=..
refdir=~/Mus_musculus/UCSC/mm10/Sequence/Bowtie2Index/mm10
ref_fa=./mm10.fa

# Index the reference genome if not already indexed
if [ ! -f "${ref_fa}.fai" ]; then
    samtools faidx "${ref_fa}"
fi

# Loop through forward read files
for r1 in "${fqdir}"/*for_paired.fq.gz; do
    echo "Processing ${r1}"
    sample_name=$(basename "${r1}" _for_paired.fq.gz)
    r2="${fqdir}/${sample_name}_rev_paired.fq.gz"

    if [[ -f "${r2}" ]]; then
        # Align reads with bowtie2
        bowtie2 -p 12 --very-sensitive -x "${refdir}" -1 "${r1}" -2 "${r2}" -S "${sample_name}.sam"
    else
        echo "Error: Reverse read file ${r2} not found."
        continue
    fi
done

# Process SAM files
for samfile in *.sam; do
    echo "Processing ${samfile}"
    sample_name="${samfile%.sam}"
    bamfile="${sample_name}.bam"
    sorted_bam="${sample_name}-sorted.bam"

    samtools view -bS "${samfile}" > "${bamfile}"
    samtools sort -o "${sorted_bam}" "${bamfile}"
    samtools index "${sorted_bam}"
    bcftools mpileup -Ou -f "${ref_fa}" "${sorted_bam}" | bcftools call -mv -Oz -o "${sample_name}.vcf.gz"
    bcftools index "${sample_name}.vcf.gz"
done
	
