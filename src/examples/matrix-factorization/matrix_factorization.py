import numpy as np


class MatrixFactorization:
    
    @classmethod
    def run(cls, R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
        """
        Args:
            R (numpy.ndarray): Rating
            P (numpy.ndarray): m x K のユーザ行列
            Q (numpy.ndarray): n x K のアイテム行列
            K (int): Latent factor(潜在変数)数
            alpha (flaot): 学習率
            beta (float): 正則化項の学習率
        """
        m, n = R.shape
        Q = Q.T

        for step in range(steps):
            for i in range(m):
                for j in range(n):
                    if R[i][j] == 0:
                        continue

                    # 各レーティングの誤差算出
                    err = cls._calc_rating_err(R[i][j], P[i, :], Q[:, j])
                    
                    # P, Q更新
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * err * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * err * P[i][k] - beta * Q[k][j])
            
            # 全体の誤差算出
            e = cls._calc_error(R, P, Q, m, n, beta)
            if e < threshold:
                break
        return P, Q

    @classmethod
    def _calc_error(cls, R, P, Q, m, n, beta):
        """レーティング行列Rと推定レーティング行列との差を算出する
        
        Args:
            R (numpy.ndarray): 評価値行列
            P (numpy.ndarray): ユーザ行列 K x m
            Q (numpy.ndarray): アイテム行列 K x n
            beta (flaot): Learning rate
        """
        err = 0.0
        for i in range(m):
            for j in range(n):
                current_rating = R[i][j]
                if current_rating == 0:
                    continue
                
                # 行列全体の二乗誤差の和
                err += pow(cls._calc_rating_err(current_rating, P[i, :], Q[:, j]), 2)
        
        # L2 regularization
        l2_term = (beta / 2) * (np.linalg.norm(P) + np.linalg.norm(Q))
        err += l2_term
        return err

    @classmethod
    def _calc_rating_err(cls, r, p, q):
        """実際の評価値と内積との差を算出する

        Args:
            r (int): 実際の評価値
            p (numpy.ndarray): m列のユーザ行列
            q (numpy.ndarray): n列のアイテム行列
        Return:
            (int) 評価値との誤差
        """
        return r - np.dot(p, q)


if __name__ == '__main__':
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

    print('========================================')
    print('Old P: {}'.format(P))
    print('Old Q: {}'.format(Q))
    print()

    P_new, Q_new = MatrixFactorization.run(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.2, threshold=0.001)
    print('========================================')
    print('New P: {}'.format(P_new))
    print('New Q: {}'.format(Q_new))
    print('----------------------------------------')
    print(R)
    print(np.round(np.dot(P, Q.T)))