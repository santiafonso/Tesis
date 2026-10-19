#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

template<typename T>
class Acumulador {
public:
    Acumulador(int N_):
        N(N_),
        largoSegmento(static_cast<int>(ceil(sqrt(N)))),
        nSegmentos((N + largoSegmento - 1) / largoSegmento),
        segmentos(N, static_cast<T>(0)),
        subtotales(nSegmentos, static_cast<T>(0)),
        acumulados(nSegmentos + 1, static_cast<T>(0)) {}

    void Preparar(const T* valores) {
#if defined(_OPENMP)
        #pragma omp parallel for
#endif
        for (int seg = 0; seg < nSegmentos; ++seg) {
            int inicioSeg = seg * largoSegmento;
            int finSeg = std::min((seg+1) * largoSegmento, N);
            int elementos = finSeg - inicioSeg;
            T totalSegmento = Scan(&valores[inicioSeg], elementos, &segmentos[inicioSeg]);
            subtotales[seg] = totalSegmento;
        }
        Scan(&subtotales[0], nSegmentos, &acumulados[1]);
    }

    T TotalAcumulado() const {
        return acumulados[nSegmentos];
    }

    T Valor(int i) const {
        if (i < 0 || i >= N) {
            return 0;
        }
        return acumulados[i / largoSegmento] + segmentos[i];
    }

    int Buscar(T umbral) const {
        if (umbral < static_cast<T>(0)) {
            return -1;
        }
        // buscar segmento
        int seg;
        for (seg = 0; (seg < nSegmentos) && (acumulados[seg+1] < umbral); ++seg);
        if (seg == nSegmentos) {
            return -1;
        }
        // buscar elemento
        T umbralLocal = umbral - acumulados[seg];
        int elem;
        for (elem = 0; segmentos[seg * largoSegmento + elem] < umbralLocal; ++elem);
        return seg * largoSegmento + elem;
    }

private:
    Acumulador() = delete;
    int N;
    int largoSegmento;
    int nSegmentos;
    std::vector<T> segmentos;
    std::vector<T> subtotales;
    std::vector<T> acumulados;

    template<typename TT>
    inline TT Scan(const TT* entrada, int elementos, TT* salida) {
        TT suma = static_cast<T>(0);
        for (int i = 0; i < elementos; ++i) {
            suma += entrada[i];
            salida[i] = suma;
        }
        return suma;
    }

#ifdef __AVX2__
    // Especialización AVX2 para double
    inline double Scan(const double* entrada, int elementos, double* salida) {
        __m256d sumav = _mm256_setzero_pd();
        int i;
        for (i = 0; i < elementos - elementos % 4; i += 4) {
            __m256d d_c_b_a = _mm256_loadu_pd(&entrada[i]);
            __m256d c_x_a_x = _mm256_castsi256_pd(_mm256_bslli_epi128(_mm256_castpd_si256(d_c_b_a), 8));
            __m256d cd_c_ab_a = _mm256_add_pd(d_c_b_a, c_x_a_x);
            __m256d x_cd_x_ab = _mm256_castsi256_pd(_mm256_srli_si256(_mm256_castpd_si256(cd_c_ab_a), 8));
            __m256d ab_ab_x_x = _mm256_permute4x64_pd(x_cd_x_ab, _MM_SHUFFLE(0, 0, 1, 1));
            __m256d scan = _mm256_add_pd(cd_c_ab_a, ab_ab_x_x);
            __m256d resultado = _mm256_add_pd(sumav, scan);
            _mm256_storeu_pd(&salida[i], resultado);
            sumav = _mm256_permute4x64_pd(resultado, _MM_SHUFFLE(3, 3, 3, 3));
        }
        double suma = _mm_cvtsd_f64(_mm256_castpd256_pd128(sumav));
        for (; i < elementos; ++i) {
            suma += entrada[i];
            salida[i] = suma;
        }
        return suma;
    }
#endif
};