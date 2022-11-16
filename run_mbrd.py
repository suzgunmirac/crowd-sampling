import argparse
import json
from collections import Counter

import numpy as np
import datasets
from tqdm import trange
import torch

def main():
    """
    Example:
        python run_mbrd.py \
            --path outputs/bigbench/bigbench_ipa_codex_original_N16.json \
            --mode bertscore \
            --save_path outputs/bigbench/bigbench_ipa_codex_bertscore_N16.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='outputs/bigbench/bigbench_ipa_codex_original_N16.json')
    parser.add_argument('--mode', type=str, default='bertscore')
    parser.add_argument('--save_path', type=str, default='outputs/bigbench/bigbench_ipa_codex_bertscore_N16.json')
    args = parser.parse_args()


    # Load Metric
    if args.mode == 'mbrd':
        bleurt = datasets.load_metric("bleurt")
    elif args.mode == 'bertscore':
        bertscore = datasets.load_metric("bertscore")
    elif args.mode == 'lcs':
        # pip3 install -i https://test.pypi.org/simple/ string2string==0.0.3
        from string2string.edit_distance import EditDistAlgs
        algs_unit = EditDistAlgs()

    path = args.path
    mode = args.mode
    save_path = args.save_path

    # Read
    file = open(path, 'r').read()
    data = json.loads(file)
    outputs = data['extracted_outputs']
    if 'inputs' in data:  # check
        assert len(data['extracted_outputs']) == len(data['inputs'])

    # Loop over Outputs
    new_data = data.copy()
    new_data['indices'] = [0 for _ in range(len(outputs))]
    for i in trange(len(outputs)):
        n = len(outputs[i])
        matrix = np.zeros((n, n))

        # MBR with BLEURT
        if mode == 'bleurt':
            for j1, cand1 in enumerate(outputs[i]):
                for j2, cand2 in enumerate(outputs[i]):
                    with torch.inference_mode():
                        score = bleurt.compute(predictions=[cand1], references=[cand2])['scores'][0]
                        matrix[j1][j2] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)

        # MBRD with BERTScore
        elif mode == 'bertscore':
            for j1, cand1 in enumerate(outputs[i]):
                for j2, cand2 in enumerate(outputs[i]):
                    with torch.inference_mode():
                        score = bertscore.compute(predictions=[cand1], references=[cand2], lang='en')['f1'][0]
                        matrix[j1][j2] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)
        
        # Majority Voting
        elif mode == 'majority':
            counter = Counter(outputs[i])
            txt, count = counter.most_common(1)[0]
            if count > 1:
                index = outputs[i].index(txt)
            else:
                index = 0

        # LCS (Longest Common Substring)
        elif mode == 'lcs':
            for j1, cand1 in enumerate(outputs[i]):
                cand1_split = cand1.split(' ')
                for j2, cand2 in enumerate(outputs[i]):
                    cand2_split = cand2.split(' ')
                    max_length = max(len(cand1_split), len(cand2_split))
                    dist, _ = algs_unit.longest_common_subsequence(
                        cand1_split,
                        cand2_split,
                        printBacktrack=False,
                        boolListOfList=True
                        )
                    score = dist / max_length
                    matrix[j1][j2] = score
            matrix = np.sum(matrix, axis=1)
            index = np.argmax(matrix)
        
        # Choose First / Sample Once
        elif mode == 'sample_once':
            index = 0

        # Random Choice
        elif mode == 'random':
            index = np.random.randint(0, n)
        
        # Not Implemented Yet...
        else:
            raise NotImplementedError()
        
        new_data['indices'][i] = int(index)
        new_data['extracted_outputs'][i] = [outputs[i][index]]

    # Save
    with open(save_path, 'w') as f:
        json.dump(new_data, f)

if __name__ == "__main__":
    main()