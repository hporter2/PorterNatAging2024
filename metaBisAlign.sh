#!/bin/bash

##Script for bisulfite/pseudomethylation alignments
#Directory containing the FASTQ files
FASTQ_DIR="/path/to/fastqs/"
#Directory to store the output
OUTPUT_DIR="/path/for/output"
#Reference genome
REFERENCE_GENOME="/path/to/bisulfiteGenome/"

#Iterate over all *_1.fastq files in the FASTQ_DIR
for FASTQ1 in "${FASTQ_DIR}"/*_1.fastq; do
    # erive the corresponding FASTQ2 file name
    FASTQ2="${FASTQ1/_1.fastq/_2.fastq}"

    #Check if the FASTQ2 file exists
    if [[ -f "${FASTQ2}" ]]; then
        #Derive job name from FASTQ1 filename
        JOB_NAME=$(basename "${FASTQ1}" "_1.fastq")

        #Create SLURM script
        SLURM_SCRIPT="${OUTPUT_DIR}/${JOB_NAME}.sh"
        echo "#!/bin/bash" > "${SLURM_SCRIPT}"
        echo "#SBATCH --nodes 1"  >> "${SLURM_SCRIPT}"
        echo "#SBATCH -c 48"  >> "${SLURM_SCRIPT}"
        echo "#SBATCH -p serial"  >> "${SLURM_SCRIPT}"
        echo "#SBATCH --job-name=${JOB_NAME}" >> "${SLURM_SCRIPT}"
        echo "#SBATCH --output=${OUTPUT_DIR}/${JOB_NAME}.out" >> "${SLURM_SCRIPT}"
        echo "#SBATCH --error=${OUTPUT_DIR}/${JOB_NAME}.err" >> "${SLURM_SCRIPT}"
        echo "#SBATCH --time=24:00:00" >> "${SLURM_SCRIPT}"
        echo "#SBATCH --mem=180G" >> "${SLURM_SCRIPT}"
        echo "" >> "${SLURM_SCRIPT}"
        echo "ml bismark" >> "${SLURM_SCRIPT}"
        echo "ml bedtools" >> "${SLURM_SCRIPT}"
        echo "ml bowtie2" >> "${SLURM_SCRIPT}"
        echo "" >> "${SLURM_SCRIPT}"
        echo "FASTQ1=${FASTQ1}" >> "${SLURM_SCRIPT}"
        echo "FASTQ2=${FASTQ2}" >> "${SLURM_SCRIPT}"
        echo "OUTPUT=${OUTPUT_DIR}" >> "${SLURM_SCRIPT}"
        echo "" >> "${SLURM_SCRIPT}"
        echo "bismark --bowtie2 -N 1 -L 20 --multicore 10 --genome_folder ${REFERENCE_GENOME} -1 \$FASTQ1 -2 \$FASTQ2  -o \$OUTPUT" >> "${SLURM_SCRIPT}"

        #Submit job
        sbatch "${SLURM_SCRIPT}"

        #Remove script after running
        rm "${SLURM_SCRIPT}"
    else
        echo "Warning: No pair found for ${FASTQ1}"
    fi
done