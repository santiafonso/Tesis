#pragma once
#include <algorithm>
#include <vector>

// Fenwick tree (Binary Indexed Tree) para KMC:
//   Actualizar(i, val)  — O(log N)  punto i al valor val
//   TotalAcumulado()    — O(1)      suma total
//   Buscar(umbral)      — O(log N)  menor índice cuya suma acumulada >= umbral
//   Preparar(arr)       — O(N)      reconstrucción completa (solo al inicio)

template<typename T>
class Acumulador {
public:
    explicit Acumulador(int N_)
        : N_(N_), tree_(N_ + 1, T(0)), vals_(N_, T(0)), total_(T(0))
    {
        highest_bit_ = 1;
        while (highest_bit_ <= N_) highest_bit_ <<= 1;
        highest_bit_ >>= 1;
    }

    // Reconstrucción O(N) — usar una sola vez al inicio de cada semiciclo
    void Preparar(const T* valores) {
        total_ = T(0);
        for (int i = 0; i < N_; ++i) { vals_[i] = valores[i]; total_ += valores[i]; }
        std::fill(tree_.begin(), tree_.end(), T(0));
        for (int i = 1; i <= N_; ++i) {
            tree_[i] += vals_[i - 1];
            int j = i + (i & (-i));
            if (j <= N_) tree_[j] += tree_[i];
        }
    }

    // Actualización incremental O(log N)
    void Actualizar(int i, T new_val) {
        T delta = new_val - vals_[i];
        total_ += delta;
        vals_[i] = new_val;
        if (delta != T(0)) {
            for (int x = i + 1; x <= N_; x += x & (-x))
                tree_[x] += delta;
        }
    }

    T TotalAcumulado() const { return total_; }

    // Menor índice (0-based) donde la suma acumulada supera umbral — O(log N)
    int Buscar(T umbral) const {
        if (umbral <= T(0)) return -1;
        int pos = 0;
        T rem = umbral;
        for (int pw = highest_bit_; pw > 0; pw >>= 1) {
            if (pos + pw <= N_ && tree_[pos + pw] < rem) {
                pos += pw;
                rem -= tree_[pos];
            }
        }
        return (pos < N_) ? pos : -1;
    }

private:
    int N_;
    int highest_bit_;
    T total_;
    std::vector<T> tree_;
    std::vector<T> vals_;
};
