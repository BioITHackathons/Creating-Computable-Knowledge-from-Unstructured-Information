import argparse
import pandas as pd
import itertools
import json

from nltk import tokenize
from tqdm import tqdm

MAX_CONTEXT_LENGTH = 500


def load_abstracs(path):
    abs_df = pd.read_csv(path, sep="\t", names=["abstract_id", "title", "abstract"])

    return abs_df

def load_entities(path):
    entity_df = pd.read_csv(path, sep="\t", names=["abstract_id",
                                                   "entity_number", 
                                                   "entity_type",
                                                   "start_offset",
                                                   "end_offset",
                                                   "entity_string"
                                                   ])
    return entity_df

def load_relations(path):
    relation_df = pd.read_csv(path, sep="\t", names=["abstract_id", 
                                                     "drug_relation", 
                                                     "arg1",
                                                     "arg2"
                                                     ])

    return relation_df



def make_entity_pairs(entities):

    # Only pair chemicals with genes, not other chemicals
    chemicals = entities.loc[entities["entity_type"] == "CHEMICAL"]
    genes_y = entities.loc[entities["entity_type"] == "GENE-Y"]
    genes_n = entities.loc[entities["entity_type"] == "GENE-N"]
    genes_plain = entities.loc[entities["entity_type"] == "GENE"]
    genes = pd.concat([genes_y, genes_n, genes_plain])

    # Pair off by entity number, chemical always goes first
    chemical_ids = chemicals["entity_number"].tolist()
    gene_ids = genes["entity_number"].tolist()
    entity_pairs = list(itertools.product(chemical_ids, gene_ids))

    return chemicals, genes, entity_pairs



def parse_relations(relations):
    relation_dict = {}

    # Remove "Arg1" and "Arg2" from format
    for index, row in relations.iterrows():
        relation_type = row["drug_relation"]
        arg1 = row["arg1"].strip()
        arg1 = arg1.replace("Arg1:", "")
        arg2 = row["arg2"].strip()
        arg2 = arg2.replace("Arg2:", "")

        # Keys are (arg1, arg2) tuples, values are relation strings
        relation_dict[(arg1, arg2)] = relation_type.lower()

    return relation_dict


def get_pair_strings(pair, chemicals, genes):
    chem_id, gene_id = pair
    chem_string = chemicals.loc[chemicals["entity_number"] == chem_id, "entity_string"].item()
    gene_string = genes.loc[genes["entity_number"] == gene_id, "entity_string"].item()

    return chem_string, gene_string



def get_pair_context(pair, chemicals, genes, title, abstract):
    # Want to get chunck of abstract that has both the gene and the chemical
    title_length = len(title) + 1
    chem_id, gene_id = pair

    # Need to find min and max offsets encompassing both the gene and chemical
    chem_start = chemicals.loc[chemicals["entity_number"] == chem_id, "start_offset"].item()
    chem_end = chemicals.loc[chemicals["entity_number"] == chem_id, "end_offset"].item()
    gene_start = genes.loc[genes["entity_number"] == gene_id, "start_offset"].item()
    gene_end = genes.loc[genes["entity_number"] == gene_id, "end_offset"].item()

    entity_span_start = min(chem_start, gene_start) - title_length
    entity_span_end = max(chem_end, gene_end) - title_length

    # split abstact on sentences
    sentences = tokenize.sent_tokenize(abstract)

    context = ""
    for sent in sentences:
        # get span of each abstract sentence
        sent_span_start = abstract.find(sent)
        sent_span_end = sent_span_start + len(sent)

        # Context should only be one sentence
        if sent_span_start <= entity_span_start and sent_span_end >= entity_span_end:
            context = sent
            break
        else:
            continue

    return context


def format_yes_no(chem_string, gene_string, context, has_relation):
    text = f'is there a relationship between "{chem_string}" and "{gene_string}"'
    text += f' from context "{context}"? answer: '
    
    if has_relation:
        text += '"yes"'
    else:
        text += '"no"'

    return text


def format_relation_pred(chem_string, gene_string, context, answer):
    text = f'predict the relationship between "{chem_string}" and "{gene_string}"'
    text += f' in context "{context}". answer: "{answer}"'

    return text

def format_relation_json(chem_string, gene_string, context, answer):
    if type(chem_string) == type(gene_string) == str:
        example_data = {
            "entities": chem_string + ", " + gene_string,
            "context": context,
            "relation": answer
        }
        
        return example_data


def convert_to_ptuning_format(abs_df, entity_df, relation_df):
    lines = []

    # Get title and abstract text for each abstract
    for index, row in tqdm(abs_df.iterrows(), total=len(abs_df)):
        abstract_id = row["abstract_id"]
        title = row["title"]
        abstract = row["abstract"]

        # Get entities for this abstract
        entities = entity_df.loc[entity_df["abstract_id"] == abstract_id]
        chemicals, genes, entity_pairs = make_entity_pairs(entities)

        # Get and parse relations for this abstract
        relations = relation_df.loc[relation_df["abstract_id"] == abstract_id]
        relation_dict = parse_relations(relations)
        
        # Want one half of the negative pair must not be apart of a relation
        entities_with_relations = []
        for pair in relation_dict.keys():
            chem_string, gene_string = get_pair_strings(pair, chemicals, genes)
            entities_with_relations.extend([chem_string, gene_string])
        entities_with_relations = set(entities_with_relations)


        # Remove positive pairs from list of entities to get negative pairs only
        negative_entity_pairs = []
        for pair in entity_pairs:
            chem_string, gene_string = get_pair_strings(pair, chemicals, genes)
            
            if pair in relation_dict:
                continue
            if chem_string and gene_string in entities_with_relations:
                continue

            negative_entity_pairs.append(pair)

        # Make all postive input examples
        for pair in relation_dict.keys():
            chem_string, gene_string = get_pair_strings(pair, chemicals, genes)
            context = get_pair_context(pair, chemicals, genes, title, abstract)

            if len(context) > len(str(chem_string)) + len(gene_string):
            
                # Make yes/no question
                # yes_no_line = format_yes_no(chem_string, gene_string, context, has_relation=True)
                # lines.append(yes_no_line)

                # Make pred relation prompt
                answer = relation_dict[pair]
                #pred_relation_prompt = format_relation_pred(chem_string, gene_string, context, answer)
                relation_json = format_relation_json(chem_string, gene_string, context, answer)
                
                if relation_json is not None:
                    lines.append(relation_json)

        neg_count = 0
        total_pos = len(relation_dict.keys())

        # Make all yes/no questions for negative relation examples
        for pair in negative_entity_pairs:
            if neg_count >= total_pos:
                break

            chem_string, gene_string = get_pair_strings(pair, chemicals, genes)
            context = get_pair_context(pair, chemicals, genes, title, abstract)

            if len(context) > len(str(chem_string)) + len(gene_string):
                #yes_no_line = format_yes_no(chem_string, gene_string, context, has_relation=False)
                #lines.append(yes_no_line)
                relation_json = format_relation_json(chem_string, gene_string, context, answer)
                
                if relation_json is not None:
                    lines.append(relation_json)
                    neg_count += 1

    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for converting track1 data to v3 format')
    parser.add_argument('--for-eval', action='store_true')
    parser.add_argument('--abstracs', type=str, default='drugprot_training_abstracs.tsv')
    parser.add_argument('--entities', type=str, default='drugprot_training_entities.tsv')
    parser.add_argument('--relations', type=str, default='drugprot_training_relations.tsv')
    parser.add_argument('--save-path', type=str, default='relation_extraction_training_data.jsonl')
    args = parser.parse_args()

    abs_df = load_abstracs(args.abstracs)
    entity_df = load_entities(args.entities)
    relation_df = load_relations(args.relations)

    lines = convert_to_ptuning_format(abs_df, entity_df, relation_df)

    save_file = open(args.save_path, "w")

    for line in lines:
        save_file.write(json.dumps(line) + "\n")

    save_file.close()
    




