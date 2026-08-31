///-------------------------------------------------------------------------------------///
///--- KMC galvanostatico — sin clusters, xi y el parametrizables via argv -------------///
///  Uso: ./ejecutable <xi> <el> <numValue>                                             ///
///  Optimizaciones:                                                                    ///
///  - acumulador_claude.h: Fenwick tree O(log N) en vez de scan O(N) por paso         ///
///  - Preparar() solo al inicio; VelocidadesAds/Dif usan updates incrementales        ///
///  - potencial() lazy: solo se recalcula cuando cambia la superficie                  ///
///  - has_surface_neighbor[] precomputado para decidir si recalcular potencial         ///
///  - Sin clusters(): elimina BFS O(N) y acumuladores PNN2/NN2 por paso               ///
///-------------------------------------------------------------------------------------///

#include "acumulador_claude.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <vector>
#include <string>

#ifdef TIMING
#include <chrono>
#endif

/// DEFINO CONSTANTES
static constexpr int PASTEMP = 1;
static constexpr double NPTS = PASTEMP;
static constexpr int Nbins = 200;
static constexpr int NMUESTRAS = 1;
static constexpr int frames = 0;

/// Galvanostatic parameters — leidos de argv en main()
static double xi_val, el_val;
static double Chi, Ele;
static constexpr double dif_coeff = 3.5e-12;

/// Cell and energy parameters
static constexpr double Ncel_x = 40;
static constexpr double Ncel_y = 20;
static constexpr double Ncel_z = Ncel_x;
static constexpr int Ncel_y_int = Ncel_y;
static constexpr double Nsitu = 1;
static constexpr int Npt = Nsitu * Ncel_x * Ncel_y * Ncel_z;
static constexpr double Np = Npt;
static constexpr double ac = 1.0;
static constexpr double Lx = ac * Ncel_x;
static constexpr double Ly = ac * Ncel_y;
static constexpr double Lz = ac * Ncel_z;
static constexpr int Nvec1 = 6;
static constexpr int Nvec2 = 0;
static constexpr int Nvec3 = 0;
static constexpr double rcorte1 = ac;
static constexpr double rcorte2 = 0.0;
static constexpr double rcorte3 = 0.0;
static constexpr int Nvec = 6;
static constexpr int SinVec = 1;
static constexpr int Nvecij = 10;
static constexpr double Area = Lx * Lz * 1.0e-16;
static constexpr int plano = Nsitu * Ncel_x * Ncel_z;
static constexpr double ce = 1.60217663e-19;
static constexpr double Qmax = Np * ce;
static constexpr double BK = 0.00008617385 * 1000.0;
static constexpr double T = 298.0;
static constexpr double kT = BK * T;
static constexpr double g_pot = -4.0;
static constexpr double J1 = (g_pot / 6.0) * kT;
double J2 = 0.0;

/// Galvanostatic constants — calculados en main() a partir de xi_val, el_val
static double Crate, ic, it, k0;
static constexpr double ds = 1e-8 * ac;
static constexpr double E0 = 0.0;

/// Kinetic parameters
static constexpr double doskT = 2.0 * BK * T;
static constexpr double kdif = 1.0e13;
static constexpr double kads = 1.0e13;
static constexpr double kdes = 1.0e13;
static constexpr double Height_dif =
    -log(dif_coeff / (kdif * ac * ac * 1e-16)) * kT;
static constexpr double Eadif = Height_dif / kT;
static double Height_k0, Eaads;   // dependen de k0 → calculados en main()
static constexpr int Nevento = 8;

/// Galvanostatic — dt y pasobin calculados en main() a partir de ic
static double dt, pasobin, timestep;
static constexpr double muoff = 0.149 * 1000.0;
static double mui;

static double KADS_FACTOR;
static double KDIF_FACTOR;
static constexpr double INV_DOSKT = 1.0 / doskT;

#define lee_info_en   "CS-40x20x40.xyz"
#define grabaver_en   "ver-kmc.dat"
#define grabaparam_en "parametros.dat"

// Prefijos de archivos de salida — construidos en main() con xi y el
static std::string grabadif_en;
static std::string grabavmd_en;

// Convierte string de parametro a formato seguro para nombre de archivo:
// "-1.5" -> "m1p5",  "0.0" -> "0p0"
static std::string sanitize(const std::string &s) {
  std::string r = s;
  for (char &c : r) {
    if (c == '-') c = 'm';
    else if (c == '.') c = 'p';
  }
  return r;
}

#define ULONGMAX 4294917238.0

// GENERADOR RANDOM
static unsigned int seed[256];
static unsigned int r;
static unsigned char irr;

inline double randomm(void) {
  return (double)(r = seed[irr++] += seed[r >> 24]) / ULONGMAX;
}

inline double randomm_01abierto(void) {
  double result;
  do {
    result = randomm();
  } while ((result <= 0.0) || (result >= 1.0));
  return result;
}

/// VARIABLES GLOBALES
static int j, n, Time_i;
static int Na, Nd, cont2, numvec[Npt][Nevento], numvec2[Npt][Nvec], Nconst,
    Npart, numven[Npt][Nvec1], numven2[Npt][Nvec2], N1[Npt], N2[Npt], N3[Npt],
    ttt;
static int numven3[Npt][Nvec3], surface[plano], bottom[plano],
    bulk[Npt - 2 * plano];
static std::vector<int> NNi_data;
static double dx, dy, dz, CORD[Npt][3], Evento[Npt][Nevento], ti, HI, R22;
static double Tiempo, Ft, Energia[Npt][Nevento];
static double TiempoBin[Nbins + 1][2], En, Timetotal, tita2, tita[Nbins],
    TiempoMedio[Nbins];
static double Nint, Ndint, Ndif, En2, sumaI, It, corriente, CB, mayor, menor,
    MU, counter, nt, TIME, MUi, MUs[Nbins];
static int Ocup[Npt];
static int8_t sitio_tipo[Npt];
static bool has_surface_neighbor[Npt];
static int numValue_global = 0;
static int Ntita[Nbins][Ncel_y_int];

static Acumulador<double>* g_acumulador = nullptr;
static bool g_need_potencial = true;

/// DECLARACIONES DE FUNCIONES
void Caja();
void CajaC();
static void Proceso();
void Vecinos();
static void Velocidades();
static void VelocidadesAds(int ii);
static void VelocidadesDif(int ii, int kk);
void Inicializa_generador();
void Graba();
void Vmd(int numValue);
void Inicial(int numValue);
void Inicial2();
void Difusion();
void GrabaDif(int i, int numValue);
void GrabaVer();
void Carga();
void potencial();

static void Actualizar(int i, int m, double Esitio);
static double CalcularEsitio(int i);
static double CalcularEf(int i, int j);

static FILE *AbrirArchivo(const char *path, const char *modo) {
  FILE *rv = fopen(path, modo);
  if (rv == nullptr) {
    fprintf(stderr, "Error abriendo archivo %s: %s\n", path, strerror(errno));
    exit(1);
  }
  return rv;
}

template <typename T>
static void LeerUnDato(FILE *archivo, const char *fmt, T *salida) {
  int rv = fscanf(archivo, fmt, salida);
  if (rv < 1) {
    fprintf(stderr, "Error leyendo valor formato %s.\n", fmt);
    exit(1);
  }
}

// Wrappers que mantienen el Fenwick tree sincronizado con Evento[][]
inline void ClearEvento(int i) {
  for (int jj = 0; jj < Nevento; ++jj) {
    if (Evento[i][jj] != 0.0) {
      Evento[i][jj] = 0.0;
      g_acumulador->Actualizar(i * Nevento + jj, 0.0);
    }
  }
}

inline void SetEvento(int i, int jj, double val) {
  Evento[i][jj] = val;
  g_acumulador->Actualizar(i * Nevento + jj, val);
}

// -------------------------------------------------------------------------
int main(int argc, char *argv[]) {

  if (argc < 4) {
    std::cerr << "Uso: " << argv[0] << " <xi> <el> <numValue>" << std::endl;
    return 1;
  }

  // --- Leer parametros de la linea de comandos ---
  xi_val = std::stod(argv[1]);
  el_val = std::stod(argv[2]);
  int numValue = std::stoi(argv[3]);
  numValue_global = numValue;

  // --- Calcular cantidades derivadas ---
  Chi       = std::pow(10.0, xi_val);
  Ele       = std::pow(10.0, el_val);
  Crate     = dif_coeff * Ele * 3600.0 / (Ly * Ly * 1e-16);
  ic        = -Crate * Qmax / (Area * 3.6);
  it        = ic * Area / (1000.0 * ce);
  k0        = Chi * std::sqrt(dif_coeff * Crate / 3600.0);
  Height_k0 = -std::log(k0 / (kads * ds)) * kT;
  Eaads     = Height_k0 / kT;
  dt        = std::abs(1000.0 * Qmax / (ic * Area));
  pasobin   = dt / Nbins;
  timestep  = pasobin;

  KADS_FACTOR = kads * std::exp(-Eaads);
  KDIF_FACTOR = kdif * std::exp(-Eadif);

  // --- Construir prefijos de archivos de salida ---
  std::string xi_tag = sanitize(argv[1]);
  std::string el_tag = sanitize(argv[2]);
  grabadif_en = "datos-40x20x40-xi" + xi_tag + "-el" + el_tag + "-g-4-";
  grabavmd_en = "vmd-40x20x40-xi"   + xi_tag + "-el" + el_tag + "-g-4-";

  // --- Registro de parametros ---
  FILE *archivo = AbrirArchivo(grabaparam_en, "a");
  fprintf(archivo,
          "xi=%s el=%s Chi=%g Ele=%g g=%g Ncel_x=%g Ncel_y=%g Ncel_z=%g "
          "muoff=%g NTHREADS=%d numValue=%d\n",
          argv[1], argv[2], Chi, Ele, (double)g_pot,
          (double)Ncel_x, (double)Ncel_y, (double)Ncel_z,
          (double)muoff, omp_get_max_threads(), numValue);
  fclose(archivo);

  Caja();
  Vecinos();
  Inicial(numValue);

  g_acumulador = new Acumulador<double>(Npt * Nevento);
  omp_set_num_threads(omp_get_max_threads());

#ifdef TIMING
  auto start = std::chrono::steady_clock::now();
#endif

  for (int NM = 0; NM < NMUESTRAS; NM++) {
    Inicial2();
    int Kk = 0;
    double nn = 1.0;
    Tiempo = 0.0;
    timestep = pasobin;
    ttt = 0;
    Time_i = 0;
    counter = 0.0;
    MUi = 0.0;
    mui = MU = muoff - 1.0;

    CB = kads * exp(-Eaads);
    potencial();
    g_need_potencial = false;
    Velocidades();

    while (mui < muoff) {
      Kk++;
      if (g_need_potencial) { potencial(); g_need_potencial = false; }
      Proceso();

#ifdef TIMING
      if (Kk % 1000 == 0) {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        start = end;
      }
#endif

      if (Tiempo > timestep) {
        MU = MUi / counter;
        MUs[Time_i] /= TiempoBin[Time_i][1];
        tita[Time_i] /= TiempoBin[Time_i][1];

        if (TiempoBin[Time_i][1] == 0) {
          tita[Time_i] = 0.0;
        }

        GrabaDif(Time_i, numValue);
        Vmd(numValue);
        Time_i++;
        timestep += pasobin;
        counter = MUi = 0.0;
      }
    }
    Vmd(numValue);
  }

  return 0;
}

// -------------------------------------------------------------------------
// PROCESO — sin Preparar() por paso; el Fenwick tree se mantiene incremental
// -------------------------------------------------------------------------
static void Proceso() {
  double sumaV = g_acumulador->TotalAcumulado();

  double R;
  int indice = -1;
  do {
    R = randomm_01abierto() * sumaV;
    indice = g_acumulador->Buscar(R);
  } while (indice < 0);

  int ii = indice / Nevento;
  int kk = indice % Nevento;

  double Rdos = randomm_01abierto();
  ti = -log(Rdos) / sumaV;
  Tiempo += ti;

  switch (kk) {
  case Nevento - 2: { Ocup[ii] = true;  Nconst++; Nint++;  } break;
  case Nevento - 1: { Ocup[ii] = false; Nconst--; Ndint++; } break;
  default:          { Ocup[ii] = false; Ocup[numvec[ii][kk]] = true; Ndif++; } break;
  }

  // potencial() solo se recalcula cuando cambia el estado superficial
  if (kk == Nevento - 2 || kk == Nevento - 1) {
    g_need_potencial = true;
  } else {
    int dest = numvec[ii][kk];
    if (sitio_tipo[ii] == 0 || has_surface_neighbor[ii] ||
        sitio_tipo[dest] == 0 || has_surface_neighbor[dest])
      g_need_potencial = true;
  }

  int dd = static_cast<int>((Tiempo - TiempoBin[0][0]) / pasobin);

  if (dd >= 0 && dd < Nbins) {
    TiempoBin[dd][1]++;
    MUs[dd] += mui;
    tita[dd] += Nconst;

  }

  MUi += mui;
  if (kk == Nevento - 2) Na++;
  if (kk == Nevento - 1) Nd++;
  counter++;

  sumaI = 0.0;
  switch (kk) {
  case Nevento - 2: { VelocidadesAds(ii); } break;
  case Nevento - 1: { VelocidadesAds(ii); } break;
  default:          { VelocidadesDif(ii, kk); } break;
  }
}

// -------------------------------------------------------------------------
void Inicial(int numValue) {
  double a;

  Tiempo = ti = sumaI = corriente = HI = Npart = R22 = 0.0;

  Inicializa_generador();
  Na = Nd = 0;

  std::string nombreArchivo = grabadif_en + std::to_string(numValue) + ".dat";
  FILE *archivo = AbrirArchivo(nombreArchivo.c_str(), "a");
  fprintf(archivo, "SoC E[V] Tiempo logChi logEle\n");
  fclose(archivo);

  a = 0.0;
  for (int i = 0; i <= Nbins; i++) {
    TiempoBin[i][0] = a;
    a += pasobin;
  }

  for (int i = 0; i < Nbins; i++) {
    tita[i] = TiempoMedio[i] = MUs[i] = 0.0;
    TiempoBin[i][1] = 0.0;
    std::fill_n(Ntita[i], Ncel_y_int, 0);
  }

  for (int i = 0; i < Nbins; i++) {
    TiempoMedio[i] = TiempoBin[i][0] + ((TiempoBin[i + 1][0] - TiempoBin[i][0]) / 2);
  }
}

// -------------------------------------------------------------------------
void Inicial2() {
  Tiempo = ti = sumaI = corriente = Npart = R22 = Nconst = HI = 0.0;

  for (int i = 0; i < Npt; i++) Ocup[i] = false;
  for (int i = 0; i < Npt; i++)
    for (int jj = 0; jj < Nevento; jj++)
      Evento[i][jj] = Energia[i][jj] = 0.0;

  CB = kads * exp(-Eaads);
  Velocidades();
}


// -------------------------------------------------------------------------
// POTENCIAL (metodo de Brent)
// -------------------------------------------------------------------------
void potencial() {
  int Nit, n1, NI, NR;
  double CA, muui, CC, CD, CE, CF, hola, CG, CH, Esitio, Ef, Error, ErrorIt, a,
      b, bb, c, d, fa, fb, fc, s, fs, fbb, x0, xi, fx, dfx, div, fx1, div1, x1,
      xinf, xsup, fxi, divi;
  int mflag, ij;
  CD = CC = 0.0;

  for (int i = 0; i < plano; i++) {
    Esitio = Ef = 0.0;
    ij = surface[i];
    switch (Ocup[ij]) {
    case 1: {
      for (n = 0; n < N1[ij]; n++) { j = numven[ij][n]; if (Ocup[j] == 1) Esitio += J1; }
      Esitio += E0;
      CC += exp(Esitio * INV_DOSKT);
    } break;
    case 0: {
      for (n = 0; n < N1[ij]; n++) { j = numven[ij][n]; if (Ocup[j] == 1) Esitio += J1; }
      Esitio += E0;
      CD += exp(-Esitio * INV_DOSKT);
    } break;
    }
  }

  CE = CB * CC;
  CF = CB * CD;

  muui = mui;
  hola = 0.0;
  Error = 1.0e-10;
  ErrorIt = Error + 1.0;
VV:
  a = -2500.0 - hola;
  b =  2000.0 + hola;
  fa = it - (CE * exp(-a * INV_DOSKT) - CF * exp(a * INV_DOSKT));
  fb = it - (CE * exp(-b * INV_DOSKT) - CF * exp(b * INV_DOSKT));
  if ((fa * fb) >= 0.0) { hola++; goto VV; }
  if (fabs(fa) < fabs(fb)) {
    bb = b; b = a; a = bb;
    fbb = fb; fb = fa; fa = fbb;
  }
  c = a; fc = fa; fs = 2.0; s = 0.0; d = 0.0; mflag = 1;

  while (ErrorIt > Error) {
    if (fa != fc && fb != fc) {
      s = (a * fb * fc / ((fa - fb) * (fa - fc))) +
          (b * fa * fc / ((fb - fa) * (fb - fc))) +
          (c * fa * fb / ((fc - fa) * (fc - fb)));
    } else {
      s = b - (fb * (b - a) / (fb - fa));
    }
    if (((s < ((3.0 * (a + b)) * 0.25)) || (s > b)) ||
        (mflag == 1 && (fabs(s - b) >= (fabs(b - c) * 0.5))) ||
        (mflag == 0 && (fabs(s - b) >= (fabs(c - d) * 0.5))) ||
        (mflag == 1 && (fabs(b - c) < Error)) ||
        (mflag == 0 && (fabs(c - d) < Error))) {
      s = (a + b) * 0.5;
      mflag = 1;
    } else {
      mflag = 0;
    }
    fs = it - (CE * exp(-s * INV_DOSKT) - CF * exp(s * INV_DOSKT));
    d = c; c = b; fc = fb;
    if ((fa * fs) < 0.0) { b = s; fb = fs; } else { a = s; fa = fs; }
    if (fabs(fa) < fabs(fb)) {
      bb = b; b = a; a = bb;
      fbb = fb; fb = fa; fa = fbb;
    }
    ErrorIt = fabs(b - a);
  }
  mui = s;
}

// -------------------------------------------------------------------------
// VECINOS
// -------------------------------------------------------------------------
void Vecinos() {
  int n, n1, n2, n3, n4;
  double r2, dx2, dy2;
  int Nvec = 6;
  mayor = 0.0;
  menor = 1e5;

  for (int i = 0; i < Npt; i++) {
    if (CORD[i][1] > mayor) mayor = CORD[i][1];
    if (CORD[i][1] < menor) menor = CORD[i][1];
    for (int jj = 0; jj < Nvec; jj++) numvec[i][jj] = -1;
  }

  n = n1 = n2 = 0;
  for (int i = 0; i < Npt; i++) {
    if (CORD[i][1] < menor + 0.1) { surface[n++] = i; }
    if (CORD[i][1] > mayor - 0.1) { bottom[n1++] = i; }
    if ((CORD[i][1] < mayor) && (CORD[i][1] > menor)) { bulk[n2++] = i; }
  }

  for (int i = 0; i < Npt; i++) {
    if      (CORD[i][1] < menor + 0.1) sitio_tipo[i] = 0;
    else if (CORD[i][1] > mayor - 0.1) sitio_tipo[i] = 2;
    else                               sitio_tipo[i] = 1;
  }

  auto BuildNeighbors = [&](int site) {
    int count = 0;
    N1[site] = 0;
    for (int jj = 0; jj < Npt; jj++) {
      dx = fabs(CORD[site][0] - CORD[jj][0]);
      dy = fabs(CORD[site][1] - CORD[jj][1]);
      dz = fabs(CORD[site][2] - CORD[jj][2]);
      if (dz > 0.5 * Lz) dz = Lz - dz;
      if (dx > 0.5 * Lx) dx = Lx - dx;
      double radio = sqrt((dx * dx) + (dy * dy) + (dz * dz));
      if (site != jj && radio < rcorte1 + 0.1) {
        numvec[site][count] = jj;
        numven[site][N1[site]] = jj;
        count++;
        N1[site]++;
      }
    }
  };

  for (int i = 0; i < plano; i++)            BuildNeighbors(surface[i]);
  for (int i = 0; i < plano; i++)            BuildNeighbors(bottom[i]);
  for (int i = 0; i < Npt - 2 * plano; i++) BuildNeighbors(bulk[i]);

  for (int i = 0; i < Npt; i++) {
    has_surface_neighbor[i] = false;
    for (int k = 0; k < N1[i]; k++) {
      if (sitio_tipo[numven[i][k]] == 0) { has_surface_neighbor[i] = true; break; }
    }
  }
}

// -------------------------------------------------------------------------
static double CalcularEsitio(int i) {
  double Esitio = 0.0;
  const int n1 = N1[i];
  const int* __restrict__ vecinos = numven[i];
  for (int kn = 0; kn < n1; kn++) if (Ocup[vecinos[kn]]) Esitio += J1;
  return Esitio;
}

static double CalcularEf(int i, int jj) {
  double Ef = 0.0;
  const int n1 = N1[jj];
  const int* __restrict__ vecinos = numven[jj];
  for (int kn = 0; kn < n1; kn++) if (Ocup[vecinos[kn]]) Ef += J1;
  return Ef;
}

// Actualizar sincroniza Evento[][] y el Fenwick tree
static void Actualizar(int i, int m, double Esitio) {
  int jv = numvec[i][m];
  if (!Ocup[jv]) {
    double dE = CalcularEf(i, jv) - Esitio;
    Energia[i][m] = dE;
    Evento[i][m] = KDIF_FACTOR * exp(-dE * INV_DOSKT);
    g_acumulador->Actualizar(i * Nevento + m, Evento[i][m]);
  }
}

// -------------------------------------------------------------------------
static void Velocidades() {
  double sumaI = 0.0;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < Npt; i++)
    for (int jj = 0; jj < Nevento; jj++)
      Evento[i][jj] = Energia[i][jj] = 0.0;

  for (int i = 0; i < plano; i++) {
    int s = surface[i];
    double Esitio = CalcularEsitio(s);
    if (Ocup[s]) {
      for (int jj = 0; jj < Nvec - SinVec; jj++) Actualizar(s, jj, Esitio);
      double E_des = -(Esitio + E0);
      Energia[s][Nevento - 1] = E_des;
      Evento[s][Nevento - 1] = KADS_FACTOR * exp(-(E_des + mui) * INV_DOSKT);
      sumaI += Evento[s][Nevento - 1];
    } else {
      double E_ads = Esitio + E0;
      Energia[s][Nevento - 2] = E_ads;
      Evento[s][Nevento - 2] = KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT);
      sumaI += Evento[s][Nevento - 2];
    }
  }

  for (int i = 0; i < plano; i++) {
    int b = bottom[i];
    double Esitio = CalcularEsitio(b);
    if (Ocup[b]) for (int jj = 0; jj < Nvec - SinVec; jj++) Actualizar(b, jj, Esitio);
  }

  for (int i = 0; i < Npt - 2 * plano; i++) {
    int bk = bulk[i];
    double Esitio = CalcularEsitio(bk);
    if (Ocup[bk]) for (int jj = 0; jj < Nvec; jj++) Actualizar(bk, jj, Esitio);
  }
  g_acumulador->Preparar(&Evento[0][0]);
  g_need_potencial = true;
}

// -------------------------------------------------------------------------
static void VelocidadesAds(int ii) {
  int i = ii;
  ClearEvento(i);
  std::fill_n(Energia[i], Nevento, 0.0);
  double Esitio = CalcularEsitio(i);
  if (Ocup[i]) {
    for (int jj = 0; jj < Nvec - SinVec; jj++) Actualizar(i, jj, Esitio);
    double E_des = -(Esitio + E0);
    Energia[i][Nevento - 1] = E_des;
    SetEvento(i, Nevento - 1, KADS_FACTOR * exp(-(E_des + mui) * INV_DOSKT));
  } else {
    double E_ads = Esitio + E0;
    Energia[i][Nevento - 2] = E_ads;
    SetEvento(i, Nevento - 2, KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT));
  }

  for (int jj = 0; jj < Nvec - SinVec; jj++) {
    int iv = numvec[ii][jj];
    if (iv == -1) continue;
    ClearEvento(iv);
    std::fill_n(Energia[iv], Nevento, 0.0);
    int8_t tipo = sitio_tipo[iv];
    if (tipo == 0) {
      double Es = CalcularEsitio(iv);
      if (Ocup[iv]) {
        for (int gg = 0; gg < Nvec - SinVec; gg++) Actualizar(iv, gg, Es);
        double E_des = -(Es + E0);
        Energia[iv][Nevento - 1] = E_des;
        SetEvento(iv, Nevento - 1, KADS_FACTOR * exp(-(E_des + mui) * INV_DOSKT));
      } else {
        double E_ads = Es + E0;
        Energia[iv][Nevento - 2] = E_ads;
        SetEvento(iv, Nevento - 2, KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT));
      }
    } else {
      if (Ocup[iv]) {
        double Es = CalcularEsitio(iv);
        for (int gg = 0; gg < Nvec; gg++) Actualizar(iv, gg, Es);
      }
    }
  }
}

// -------------------------------------------------------------------------
static void VelocidadesDif(int ii, int kk) {
  int i = ii;
  double Esitio = CalcularEsitio(i);
  ClearEvento(i);
  std::fill_n(Energia[i], Nevento, 0.0);
  if (sitio_tipo[i] == 0) {
    double E_ads = Esitio + E0;
    Energia[i][Nevento - 2] = E_ads;
    SetEvento(i, Nevento - 2, KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT));
  }

  for (int jji = 0; jji < Nvec; jji++) {
    int iii = numvec[ii][jji];
    if (iii == -1) continue;
    ClearEvento(iii);
    std::fill_n(Energia[iii], Nevento, 0.0);
    double Es = CalcularEsitio(iii);
    int8_t tipo = sitio_tipo[iii];
    if (tipo == 0) {
      if (Ocup[iii]) {
        for (int gg = 0; gg < Nvec - SinVec; gg++) Actualizar(iii, gg, Es);
        double E_des = -(Es + E0);
        Energia[iii][Nevento - 1] = E_des;
        SetEvento(iii, Nevento - 1, KADS_FACTOR * exp(-(E_des + mui) * INV_DOSKT));
      } else {
        double E_ads = Es + E0;
        Energia[iii][Nevento - 2] = E_ads;
        SetEvento(iii, Nevento - 2, KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT));
      }
    } else if (tipo == 2) {
      if (Ocup[iii]) for (int gg = 0; gg < Nvec - SinVec; gg++) Actualizar(iii, gg, Es);
    } else {
      if (Ocup[iii]) for (int gg = 0; gg < Nvec; gg++) Actualizar(iii, gg, Es);
    }
  }

  for (int ijj = 0; ijj < Nvec; ijj++) {
    int iiii = numvec[numvec[ii][kk]][ijj];
    if (iiii == -1) continue;
    double Es = CalcularEsitio(iiii);
    ClearEvento(iiii);
    std::fill_n(Energia[iiii], Nevento, 0.0);
    int8_t tipo = sitio_tipo[iiii];
    if (tipo == 0) {
      if (Ocup[iiii]) {
        for (int ggg = 0; ggg < Nvec - SinVec; ggg++) Actualizar(iiii, ggg, Es);
        double E_des = -(Es + E0);
        Energia[iiii][Nevento - 1] = E_des;
        SetEvento(iiii, Nevento - 1, KADS_FACTOR * exp(-(E_des + mui) * INV_DOSKT));
      } else {
        double E_ads = Es + E0;
        Energia[iiii][Nevento - 2] = E_ads;
        SetEvento(iiii, Nevento - 2, KADS_FACTOR * exp((mui - E_ads) * INV_DOSKT));
      }
    } else if (tipo == 2) {
      if (Ocup[iiii]) for (int ggg = 0; ggg < Nvec - SinVec; ggg++) Actualizar(iiii, ggg, Es);
    } else {
      if (Ocup[iiii]) for (int ggg = 0; ggg < Nvec; ggg++) Actualizar(iiii, ggg, Es);
    }
  }
}

// -------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------
void GrabaDif(int i, int numValue) {
  std::string nombreArchivo = grabadif_en + std::to_string(numValue) + ".dat";
  FILE *archivo = AbrirArchivo(nombreArchivo.c_str(), "a");
  fprintf(archivo, "%f %f %f %f %f\n", (tita[i] / (Npt)),
          (float)(-MUs[i] * 1.0e-3), (float)(TiempoMedio[i]),
          log10(Chi), log10(Ele));
  fclose(archivo);
}

void GrabaVer() {
  float NNN = Nconst;
  FILE *archivo = AbrirArchivo(grabaver_en, "a");
  fprintf(archivo, "%f %f %f %f %f %f %f %f %f %f %f %d %d %d \n",
          (float)(Tiempo), (float)(-mui * 1e-3), (float)(NNN / Npt),
          (float)(ic), (float)(Nconst), (float)(Nint), (float)(Ndint),
          (float)(Ndif), (float)(HI), (float)(Eadif * kT), (float)(Eaads * kT),
          (int)(Lx * ac / 5.0), (int)(Ly * ac / 5.0), (int)(Npt * ac / 5.0));
  fclose(archivo);
}

void Inicializa_generador(void) {
#ifndef FIXEDSEED
  srand((unsigned)time(0));
#else
  srand(1);
#endif
  irr = 1;
  for (int i = 0; i < 256; ++i) seed[i] = rand();
  r = seed[0];
  for (int i = 0; i < 70000; ++i) r = seed[irr++] += seed[r >> 24];
}

void Vmd(int numValue) {
  std::string nombreVmd = grabavmd_en + std::to_string(numValue) + ".xyz";
  FILE *archivo1 = AbrirArchivo(nombreVmd.c_str(), "a");
  fprintf(archivo1, "%d \n", (int)(Npt));
  fprintf(archivo1, "\n");
  for (int i = 0; i < Npt; i++) {
    if (Ocup[i]) {
      if (CORD[i][1] == 0) {
        fprintf(archivo1, "O %4.5f %4.5f %4.5f\n",
                (float)(CORD[i][0]), (float)(CORD[i][1]), (float)(CORD[i][2]));
      } else {
        fprintf(archivo1, "Li %4.5f %4.5f %4.5f\n",
                (float)(CORD[i][0]), (float)(CORD[i][1]), (float)(CORD[i][2]));
      }
    } else {
      fprintf(archivo1, "C %4.5f %4.5f %4.5f\n",
              (float)(CORD[i][0]), (float)(CORD[i][1]), (float)(CORD[i][2]));
    }
  }
  fclose(archivo1);
}

void Carga() {
  int e1;
  int E;
  double B, C, D;
  char Li[4];

  FILE *archivo1 = AbrirArchivo(lee_info_en, "r");

  for (int jj = 0; jj < frames; jj++) {
    LeerUnDato(archivo1, "%d", &E);
    fscanf(archivo1, "\n");
    for (int ii = 0; ii < Npt; ii++) {
      LeerUnDato(archivo1, "%3s", Li);
      LeerUnDato(archivo1, "%le", &B);
      LeerUnDato(archivo1, "%le", &C);
      LeerUnDato(archivo1, "%le", &D);
    }
  }
  e1 = 0;
  LeerUnDato(archivo1, "%d", &E);
  fscanf(archivo1, "\n");
  for (int ii = 0; ii < Npt; ii++) {
    LeerUnDato(archivo1, "%3s", Li);
    LeerUnDato(archivo1, "%le", &B);
    LeerUnDato(archivo1, "%le", &C);
    LeerUnDato(archivo1, "%le", &D);
    if (D < 0.0) {
      Ocup[ii] = false;
    } else {
      Ocup[ii] = true;
      e1++;
    }
  }
  fclose(archivo1);
  Nconst = e1;
}

void Caja() {
  int e1, jj, ii;
  int E;
  double B, C, D;
  char Li;
  FILE *archivo1 = AbrirArchivo(lee_info_en, "r");
  fscanf(archivo1, "%d", &E);
  fscanf(archivo1, "\n");
  for (ii = 0; ii < Npt; ii++) {
    fscanf(archivo1, "%s", &Li);
    fscanf(archivo1, "%le", &B); CORD[ii][0] = B;
    fscanf(archivo1, "%le", &C); CORD[ii][1] = C;
    fscanf(archivo1, "%le", &D); CORD[ii][2] = D;
  }
  fclose(archivo1);
}
