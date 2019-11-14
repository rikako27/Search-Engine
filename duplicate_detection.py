# Shingling
# - extracts the set all substrings of length k from a text
# Jaccard similarity
# - simply measure similarity of sets

# Threshold > 0.9, one of urls will be removed
# Note: if two documents has low information (small data),
# they are likely to generate high threshold.
from pathlib import Path
import os
import json
import re
import pickle

SHINGLE_SIZE = 5;

def get_shingles(text, size=SHINGLE_SIZE):
    shingles = set()
    for i in range(0, len(text) - size + 1):
        shingles.add(''.join(text[i:i+size]))
    return shingles

def jaccard(s1, s2):
    #SIM(A, B) = |A ∩ B| / |A ∪ B|
    # return the float number between 0 (no common) and 1 (identical)
    x = len(s1.intersection(s2))
    y = len(s1.union(s2))
    return x / float(y)

def tokenize(text):
    regex = '[a-zA-Z0-9]+'
    tokens = re.findall(regex, text.lower())
    return tokens

if __name__ == "__main__":
    set_url_removed = set()
    path_to_db = Path("../DEV")
    #path_to_db = Path("/home/lopes/Datasets/IR/DEV")

    rf = open("result.txt", "w")
    for dir in path_to_db.iterdir():
        file_list = [f for f in dir.iterdir() if f.is_file]
        for file_i in range(0, len(file_list) - 1):
            file1 = file_list[file_i]
            with open(file1, 'r', encoding="ascii", errors="ignore") as f1:
                parsed_json = json.load(f1)
                content1 = parsed_json['content']
                url1 = parsed_json['url']
                if url1 in set_url_removed:
                    continue
                tokens1 = tokenize(content1)

                for file_j in range(file_i + 1, len(file_list)):
                    file2 = file_list[file_j]

                    with open(file2, 'r', encoding="ascii", errors="ignore") as f2:
                        parsed_json = json.load(f2)
                        content2 = parsed_json['content']
                        url2 = parsed_json['url']
                        if url2 in set_url_removed:
                            continue
                        tokens2 = tokenize(content2)


                    shingles1 = get_shingles(tokens1)
                    shingles2 = get_shingles(tokens2)

                    similarities = jaccard(shingles1, shingles2)
                    if similarities > 0.9:
                        rf.write("%f\n" %similarities)
                        rf.write("%s and %s\n" % (url1, url2))

                        # add the url that has less data will be added to set_url_removed
                        len_file1 = len(tokens1)
                        len_file2 = len(tokens2)
                        if (len_file1 < len_file2):
                            set_url_removed.add(url1)
                        else:
                            set_url_removed.add(url2)
    rf.close()

    with open("database/removed_urls.txt", "wb") as fp:
        pickle.dump(set_url_removed, fp)
