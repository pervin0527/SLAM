import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer

def extract_local_features(image):
    """
    local feature 검출 함수.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors


def create_visual_vocabulary(desc_list, n_clusters=50):
    """
    k-means 클러스터링으로 vocabulary 구축.
    """
    descriptors = np.vstack(desc_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)
    
    return kmeans.cluster_centers_


def compute_bow_histogram(descriptors, vocabulary):
    histogram = np.zeros(len(vocabulary)) ## n_cluster 차원의 벡터 생성. (n_clusters, )차원의 벡터

    for desc in descriptors:
        distances = np.linalg.norm(vocabulary - desc, axis=1) ## desc와 Visual Vocabulary의 모든 Code Word 간의 거리를 계산.
        closest_word = np.argmin(distances) ## 거리가 가장 가까운 code word 선택.
        histogram[closest_word] += 1 ## 등장횟수 1 증가.

    return histogram


def apply_tfidf(histograms):
    transformer = TfidfTransformer()
    tfidf_histograms = transformer.fit_transform(histograms).toarray()

    return tfidf_histograms


def compare_histograms(hist1, hist2):
    """
    두 개의 히스토그램간 코사인 유사도를 계산한다.
    """
    similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
    return similarity


def main():
    """1단계 : 파일 로드"""
    num_frames = 10
    data_path = "/home/pervinco/Datasets/KITTI/dataset/sequences/00"
    left_img_path = os.path.join(data_path, "image_0")
    right_img_path = os.path.join(data_path, "image_1")

    right_image_files = sorted([os.path.join(right_img_path, f) for f in os.listdir(right_img_path) if f.endswith(".png")])[:num_frames]
    left_image_files = sorted([os.path.join(left_img_path, f) for f in os.listdir(left_img_path) if f.endswith(".png")])[:num_frames]
    print(len(left_image_files), len(right_image_files))
    
    db_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in right_image_files]
    query_images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in left_image_files]

    """2단계 : db_images의 특징(local feature) 추출."""
    db_desc_list = []
    for idx, image in enumerate(db_images):
        ## local feature 검출
        keypoints, descriptors = extract_local_features(image)
        
        if idx == 0:
            print(f"keypoint 개수 : {len(keypoints)}")
            print(f"descriptor 벡터 차원 : {descriptors.shape}")
        
        if descriptors is not None:
            db_desc_list.append(descriptors)
    
    print(f"전체 desc 크기 : {len(db_desc_list)}")

    
    """3단계 : db 이미지에서 추출한 local feature들로 vocabulary 구축."""
    vocabulary = create_visual_vocabulary(db_desc_list, n_clusters=50)
    print(f"단어사전 크기 : {vocabulary.shape}") ## (n_clusters, desc_vec_dim)

    """4단계 : bag of words 히스토그램 생성."""
    db_histograms = [] ## (n_cluster) 차원의 히스토그램이 desc개만큼 구해져 리스트에 저장된다.
    for descriptors in db_desc_list:
        hist = compute_bow_histogram(descriptors, vocabulary)
        db_histograms.append(hist)

    db_histograms = np.array(db_histograms)
    print(f"DB 전체 히스토그램 : {db_histograms.shape}")

    """5단계 : Re-weighting을 적용."""
    db_tfidf_hist = apply_tfidf(db_histograms)
    print(f"DB tf-idf 히스토그램 : {db_tfidf_hist.shape}")

    """6단계 : query 이미지와 비교."""
    for idx, query_image in enumerate(query_images):
        ## query 이미지에 대한 local feature 추출.
        keypoints, query_descriptors = extract_local_features(query_image)

        if query_descriptors is None:
            print(f"Query image {idx} has no descriptors.")
            continue

        ## query 이미지에 대한 histogram 생성 및 tf-idf re-weighting
        query_histogram = compute_bow_histogram(query_descriptors, vocabulary)
        tfidf_query_histogram = apply_tfidf([query_histogram])[0]

        ## db에 저장된 이미지들의 히스토그램과 쿼리 이미지 히스토그램을 비교
        similarities = []
        for db_idx, db_histogram in enumerate(db_tfidf_hist):
            ## 코사인 유사도 계산
            similarity = compare_histograms(tfidf_query_histogram, db_histogram)
            similarities.append((db_idx, similarity))

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        print(f"Query image {idx} matches:")
        for match_idx, sim in similarities[:3]:
            print(f"  Database image {match_idx} with similarity {sim:.2f}")

if __name__ == "__main__":
    main()
