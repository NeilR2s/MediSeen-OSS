import fire
import os
import lmdb
import cv2
import csv
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0: return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(imagePathOnly, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset directly from a CSV file.
    ARGS:
        imagePathOnly : FOLDER path where images are stored
        gtFile        : path to the CSV file (e.g., train.csv)
        outputPath    : LMDB output path
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    datalist = []
    with open(gtFile, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for i, row in enumerate(reader):
            if len(row) != 2:
                print(f"  - Skipping line #{i+2} in CSV: Expected 2 columns, found {len(row)}. Content: {row}")
                continue
            filename, label = row
            if not filename or not label:
                print(f"  - Skipping line #{i+2} in CSV: Contains empty filename or label. Content: {row}")
                continue
            datalist.append((filename, label))

    nSamples = len(datalist)
    for i in range(nSamples):
        filename, label = datalist[i]
        imagePath = os.path.join(imagePathOnly, filename)

        if not os.path.exists(imagePath):
            print(f"Image not found, skipping: {imagePath}")
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        
        if checkValid and not checkImageIsValid(imageBin):
            print(f'Invalid image, skipping: {imagePath}')
            continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'Written {cnt} / {nSamples}')
        cnt += 1
    
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples at: {outputPath}')

if __name__ == '__main__':
    fire.Fire({
        'create': createDataset
    })  