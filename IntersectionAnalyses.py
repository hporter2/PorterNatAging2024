import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pybedtools as pb
import os
from statsmodels.stats import contingency_tables

## this code was used to generate Figure 5A
def site2proximity(bedfile, outfile, proximity_bp_window = 25):
    """
    function for converting site-level bed outputs into the local context for other analyses.
    tested for input bedfiles of type bed4, but should work with larger bed formats
    """
    tempbed = pd.read_csv(bedfile, sep = '\t')
    #move start and end coordinates by proximity_bp_window
    tempbed.iloc[:,1] = tempbed.iloc[:,1] - 25
    tempbed.iloc[:,2] = tempbed.iloc[:,2] + 25
    tempbed.to_csv(outfile, sep = '\t', header = False, index = False)
    return None

def arrayCGtoPosition(newsites_list, illumina_annot = annot, target_columns = ['CHR', 'MAPINFO']):
    '''
    Intersects lists of sites from 
    methylation array and converts to positions
    for comparison to sequencing-based datasets. 
    '''
    tempdf = illumina_annot.loc[illumina_annot.index.intersection(newsites_list),target_columns]
    print('Array dimensions after intersection')
    display(tempdf.shape)
    tempdf.iloc[:,0] = tempdf.iloc[:,0].astype(int)
    tempdf.iloc[:,1] = tempdf.iloc[:,1].astype(int)
    tempdf.columns = ['chr', 'start']
    tempdf['chr'] = ('chr' + tempdf['chr'].astype(str))
    tempdf['pos'] = tempdf['chr'] + '.' + tempdf['start'].astype(str)
    tempdf['name'] = tempdf.index
    tempdf.index = tempdf['pos']
    tempdf.pop('pos')
    return tempdf

def positionToBED(position_df, bed_file):
    '''
    Converts position df to .bed format for other comparisons. 
    '''
#     assert 
    tempdf = position_df.copy()
    tempdf['end'] = tempdf['start'] + 1
    tempdf = tempdf.loc[:,['chr', 'start', 'end', 'name']]
    tempdf.to_csv(bed_file, sep = '\t', header = None, index = None)
    return None

def makeEnrichDF(bed_list, enrichAgainst = 'aqtl_nodupe_38to19.bed'):
    '''
    Intersect a list of bed files against a common target for future enrichment analysis
    '''
    enrichdict = {}
    for file in bed_list:
        enrichdict[file] = [pb.BedTool(file).count(),
                           pb.BedTool(enrichAgainst).intersect(pb.BedTool(file), u = True).count()]
        enrichdf = pd.DataFrame(enrichdict).T
        enrichdf.columns = ['Total', 'InFeature']
        enrichdf['OutFeature'] = enrichdf['Total'] - enrichdf['InFeature']
    return enrichdf

def computeOddsRatio(enrichdf, userows, background_index = '450kbackground_fixed.bed'):
    '''
    Restructure enrichdf counts into contingency table for statistical analysis
    '''
    valdict = {}
    for i in userows:
        table = np.asarray([[enrichdf.loc[i,['InFeature']],
                             enrichdf.loc[background_index,['InFeature']]],
                            [enrichdf.loc[i, ['OutFeature']],
                             enrichdf.loc[background_index, ['OutFeature']]]]).squeeze()
        tempt = contingency_tables.Table2x2(table)
        valdict[i] = [tempt.log_oddsratio, 
                      tempt.log_oddsratio_pvalue(),
                      tempt.log_oddsratio_confint()]
    outvaldf = pd.DataFrame(valdict)
    return outvaldf


def metaOddsRatio(enrichdf):
    '''
    Split enrichment dataframe into the relevant subframes for each clock
    to prevent comparing against incorrect background sets
    '''
    pacesub = enrichdf.loc[enrichdf.index.str.contains('dunedin')]
    horsub = enrichdf.loc[enrichdf.index.str.contains('horv')]
    restsub = enrichdf.loc[~(enrichdf.index.str.contains('dunedin') | enrichdf.index.str.contains('horv'))]
    pacedun = computeOddsRatio(pacesub, pacesub.index, 'dunedinPACE_background.bed')
    hordun = computeOddsRatio(horsub, horsub.index, 'horvathback_38liftover.bed')
    restdun = computeOddsRatio(restsub, restsub.index, '450kbackground_fixed.bed')
    return pacedun.merge(hordun, left_index = True, right_index = True).merge(restdun, left_index = True, right_index = True)

def prepPlotData(dflist, grouplist, usecols):
    '''
    Reformat data into format amenable to seaborn barplots.
    Extra saving of the upper and lower bound confidence intervals enables
    showing error term. 
    '''
    plot_data = []
    for df, group in zip(dflist, grouplist):
        for col in usecols:
            OR = df[col][0]
            cil, ciu = df[col][2]
            plot_data.append({
                'group' : group,
                'var' : col,
                'or' : OR,
                'cil' : cil,
                'ciu' : ciu,
                'citup' : df[col][2]
            })
    return pd.DataFrame(plot_data)
    
## preprocessing annotation data from Stefansson et al., Nature Genetics (2024)    
## Data were obtained from the supplemental records provided upon request
## The data were lifted over from GRCh38 to hg19 in the UCSC web interface 
## removing duplicates since each QTL has one record per CpG and each CpG could have multiple controlling QTLs
rawinput = pd.read_csv('./aqtl_cpgs_38to19.bed', sep = '\t', header = None)
rawinput['pos'] = rawinput.iloc[:,0] + '.' + rawinput.iloc[:,1].astype(str)
rawinput.index = rawinput['pos']
rawinput.pop('pos')
rawinput = rawinput[~rawinput.index.duplicated(keep='first')]
rawinput.to_csv('aqtl_cpgs_38to19_nodupes.bed', sep = '\t', header = None, index = None)

rawinput = pd.read_csv('./aqtl_38to19.bed', sep = '\t', header = None)
rawinput['pos'] = rawinput.iloc[:,0] + '.' + rawinput.iloc[:,1].astype(str)
rawinput.index = rawinput['pos']
rawinput.pop('pos')
rawinput = rawinput[~rawinput.index.duplicated(keep='first')]
rawinput.to_csv('aqtl_nodupe_38to19.bed', sep = '\t', header = None, index = None)

site2proximity('./aqtl_cpgs_38to19_nodupes.bed', 'aqtl_cpgs_38to19_nodupes_prox25bp.bed')
site2proximity('./aqtl_nodupe_38to19.bed', 'aqtl_nodupe_near_38to19.bed')
## running the four enrichment sets against the ASM-QTLs and their affected CpGs from 
aqtl_direct = makeEnrichDF(bed_list2, 'aqtl_nodupe_38to19.bed')
aqtl_near = makeEnrichDF(bed_list2, 'aqtl_nodupe_near_38to19.bed')
acpg_direct = makeEnrichDF(bed_list2, 'aqtl_cpgs_38to19_nodupes.bed')
acpg_near = makeEnrichDF(bed_list2, 'aqtl_cpgs_38to19_nodupes_prox25bp.bed')

adr = metaOddsRatio(aqtl_direct)
anr = metaOddsRatio(aqtl_near)
cdr = metaOddsRatio(acpg_direct)
cnr = metaOddsRatio(acpg_near)

tpd = prepPlotData([adr, anr, cdr, cnr],
                   ['aqtl_direct', 'aqtl_near', 'acpg_direct', 'acpg_near'],
                  ['dunedinPACE_sites.bed', 'horvath_sites.bed',
                  'phenoage_sites.bed'])

plt.figure(figsize = (12, 6))
sns.barplot(x = 'var', y = 'or', hue='group', data = tpd, ci = None)
sns.boxplot(data = tpd, x = 'var', y = 'cil', hue = 'group')
sns.boxplot(data = tpd, x = 'var', y = 'ciu', hue = 'group')
plt.title('OddsRatio with 95% CI')
plt.legend()
plt.savefig('ASMQTL_enrichments.eps', format = 'eps')
plt.show()


