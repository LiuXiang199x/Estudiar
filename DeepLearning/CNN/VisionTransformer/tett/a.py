import numpy as np
import csv

def run_(file_pth):
    f = open(file_pth)
    f = csv.DictReader(f)
    wordsCounterList = {}

    for row in f:
        phrase_ = row["Details"].lower()
        words_lst = phrase_.replace(",","").split(" ")
        for word in words_lst:
            if word not in wordsCounterList.keys():
                wordsCounterList[word] = 1
            else:
                wordsCounterList[word] += 1
    
    result_ = {}
    for key_name in wordsCounterList.keys():
        if wordsCounterList[key_name] > 1:
            result_[key_name] = wordsCounterList[key_name]
    # print(wordsCounterList)
    result_ = sorted(result_.items(),key = lambda result_:result_[1],reverse= True)
    return result_

if __name__ == "__main__":
    csv_path = "/home/marco/下载/facebook-contest_export.csv"
    
    final_output = run_(csv_path)
    print(final_output)