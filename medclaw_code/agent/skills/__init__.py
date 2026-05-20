from .pubmed_search import pubmed_search
from .drug_lookup import drug_lookup
from .clinical_trial_search import search_clinical_trials
from .variant_query import query_gene_variant
from .disease_info import get_disease_info
from .drug_interaction import check_drug_interaction
from .gene_info import lookup_gene_info
from .medical_calculator import medical_calculator

ALL_SKILLS = [
    pubmed_search,
    drug_lookup,
    search_clinical_trials,
    query_gene_variant,
    get_disease_info,
    check_drug_interaction,
    lookup_gene_info,
    medical_calculator,
]
