import pytest
import time
import unittest
import warnings


import numpy as np

from matrix_factorization import MatrixFactorization as mf

warnings.filterwarnings('ignore')


@pytest.fixture
def load_dataset():
    import pandas as pd
    df = pd.read_csv('../../../data/external/movielens/ratings.csv')
    return df

def evaluate(P, Q, R, P_new, Q_new):
    print('========================================')
    print('Old P: {}'.format(P))
    print('Old Q: {}'.format(Q))
    print()

    print('========================================')
    print('New P: {}'.format(P_new))
    print('New Q: {}'.format(Q_new))
    print('----------------------------------------')
    print(R)
    print(np.round(np.dot(P, Q.T)))

def test_matrix_factorization():
    print('mf test.............')
    since = time.time()

    # 潜在因子数
    K = 2
    # m x n のレーティング行列
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ])
    m, n = R.shape
    # m x K のユーザ行列P
    P = np.random.rand(m, K)
    # n x K のアイテム行列Q
    Q = np.random.rand(n, K)

    P_new, Q_new = mf.run(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.2, threshold=0.001)
    evaluate(P, Q, R, P_new, Q_new)
    assert False

def test_movielens(load_dataset):
    print('mf test.............')

    # Load dataset
    df_rating = load_dataset
    # print(df_rating.head())

    m = max(np.unique(df_rating.userId))
    n = max(np.unique(df_rating.movieId))
    # m = len(np.unique(df_rating.userId))
    # n = len(np.unique(df_rating.movieId))
    
    print('m x n: {} x {}'.format(m, n))

    R = np.zeros([m, n])

    print(type(df_rating.userId[0]))
    print(type(df_rating.movieId[0]))
    print(R[df_rating.userId[0]][df_rating.movieId[0]])
    print(np.unique(df_rating.rating))
    print(sum(df_rating.rating.isnull()))


    for i, s in df_rating.iterrows():
        if s.rating == 0: continue

        u_id = int(s.userId)
        m_id = int(s.movieId)

        R[u_id - 1][m_id - 1] = s.rating
    
    print(R)

    # 潜在因子数
    K = 2
    # m x n のレーティング行列
    m, n = df_rating[['userId', 'movieId']].shape
    print(m, n)
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ])
    m, n = R.shape
    # m x K のユーザ行列P
    P = np.random.rand(m, K)
    # n x K のアイテム行列Q
    Q = np.random.rand(n, K)

    P_new, Q_new = mf.run(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.2, threshold=0.001)
    evaluate(P, Q, R, P_new, Q_new)
            
    assert False