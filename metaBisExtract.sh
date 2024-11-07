#!/bin/bash

### Scripts to run after completing bisulfite alignment with bismark.
#Directory containing the FASTQ files
BAM_DIR="/path/to/bam_dir/"
# Directory to store the output
OUTPUT_DIR="/path/to/out_dir/"


#Iterate over all *_1.fastq files in the FASTQ_DIR
for BAM in "${BAM_DIR}"/*.bam; do

    JOB_NAME=$(basename "${BAM}" ".bam")

    # Create SLURM script content
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
    echo "BAM=${BAM}" >> "${SLURM_SCRIPT}"
    echo "OUTPUT=${OUTPUT_DIR}" >> "${SLURM_SCRIPT}"
    echo "" >> "${SLURM_SCRIPT}"
    echo "bismark_methylation_extractor -p --bedGraph --zero_based --multicore 10 -o \$OUTPUT \$BAM" >> "${SLURM_SCRIPT}"

    #Submit the SLURM job
    sbatch "${SLURM_SCRIPT}"

    #Remove the SLURM script after submission
    rm "${SLURM_SCRIPT}"

done